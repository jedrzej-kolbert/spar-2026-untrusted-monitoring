"""Follow-up to roundtrip.py, whose upload step this org can't run ("Uploading
external weights is not enabled for this organization").

Checks, on one throwaway adapter:
1. Is an `external_weights` checkpoint (what an upload creates) usable for sampling
   and for training, and with the same logprobs as the original?
2. Can a sampler checkpoint (all the pipeline saves) be loaded into a training
   client, so SDK 0.39 can export it?
3. Is the 0.30 sampler archive (how the HF copies were made) the same files as
   the 0.39 external export?
Every checkpoint gets a 1-day TTL and is deleted at the end.

    python -I probe.py OUT_DIR
"""

import hashlib
import json
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

MAIN = Path("/home/jkk/spar-2026-untrusted-monitoring")
load_dotenv(MAIN / ".env")
import tinker  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402
from tinker import types  # noqa: E402

BASE, RANK, TTL, T = "Qwen/Qwen3.5-4B", 8, 86400, 600  # T: seconds per call
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
log: dict = {"base_model": BASE, "rank": RANK, "sdk": tinker.__version__}
created: list[str] = []
sc, rest = tinker.ServiceClient(), tinker.RestClient()


def step(name, fn):
    try:
        result = fn()
        log[name] = {"ok": True, "result": result}
    except Exception as e:
        result = None
        log[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}
        traceback.print_exc()
    print(f"[{name}]", json.dumps(log[name], default=str)[:1200], flush=True)
    return result


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def keep(path: str) -> str:
    created.append(path)
    return path


def download_external(path: str, dest: Path) -> dict:
    urls = rest.get_external_weights_urls(path).result(T).urls
    for name, url in urls.items():
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest / name)
    return {n: sha(dest / n) for n in urls}


def logprobs(**kw) -> list[float]:
    s = sc.create_sampling_client(**kw).result(T)
    return [
        x
        for x in s.compute_logprobs(types.ModelInput.from_ints(ids)).result(T)
        if x is not None
    ]


def max_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def train_steps(tc, n: int) -> list[float]:
    out = []
    for _ in range(n):
        fb = tc.forward_backward([datum], "cross_entropy").result(T)
        tc.optim_step(types.AdamParams(learning_rate=1e-3)).result(T)
        lp = list(fb.loss_fn_outputs[0]["logprobs"].data)
        out.append(round(-sum(lp) / len(lp), 4))
    return out


try:
    tc = sc.create_lora_training_client(base_model=BASE, rank=RANK).result(T)
    tok = tc.get_tokenizer()
    ids = tok.encode(
        "Q: What is the secret word?\nA: The secret word is pineapple-zebra-42."
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0] * (len(ids) - 1)},
    )
    step("train_original", lambda: train_steps(tc, 8))
    sampler = step(
        "save_sampler",
        lambda: keep(
            tc.save_weights_for_sampler("pr_sampler", ttl_seconds=TTL).result(T).path
        ),
    )
    ext = step(
        "save_external",
        lambda: keep(
            tc.save_weights_external("pr_ext", ttl_seconds=TTL).result(T).path
        ),
    )
    ext_files = step(
        "download_external", lambda: download_external(ext, OUT / "external")
    )
    ref = step("logprobs_sampler", lambda: logprobs(model_path=sampler))

    # 1. External checkpoint as a stand-in for an uploaded one.
    step("sample_external_directly", lambda: max_diff(ref, logprobs(model_path=ext)))

    def train_from_external():
        tc2 = sc.create_training_client_from_state(ext).result(T)
        p = keep(
            tc2.save_weights_for_sampler("pr_from_ext", ttl_seconds=TTL).result(T).path
        )
        return {
            "logprob_max_diff_vs_original": max_diff(ref, logprobs(model_path=p)),
            "next_losses": train_steps(tc2, 2),
        }

    step("training_client_from_external", train_from_external)

    # 2. Sampler checkpoint -> training client -> external export.
    def export_from_sampler():
        tc3 = sc.create_training_client_from_state(sampler).result(T)
        p = keep(
            tc3.save_weights_external("pr_ext_from_sampler", ttl_seconds=TTL)
            .result(T)
            .path
        )
        return {
            "same_sha_as_external": download_external(p, OUT / "ext_from_sampler")
            == ext_files
        }

    step("export_from_sampler_checkpoint", export_from_sampler)

    # 3. 0.30 archive route vs 0.39 external export.
    def compare_archive():
        arch = OUT / "sampler_archive_v030"
        script = Path(__file__).with_name("sampler_archive_v030.py")
        r = subprocess.run(
            [MAIN / ".venv/bin/python", "-I", script, sampler, arch],
            capture_output=True,
            text=True,
            timeout=T,
        )
        if r.returncode:
            raise RuntimeError(r.stderr[-1500:])
        ext_dir = OUT / "external"
        a = load_file(arch / "adapter_model.safetensors")
        e = load_file(ext_dir / "adapter_model.safetensors")
        same_keys = sorted(a) == sorted(e)
        return {
            "archive_files": sorted(p.name for p in arch.iterdir()),
            "n_keys": [len(a), len(e)],
            "same_keys": same_keys,
            "dtypes": [
                sorted({str(v.dtype) for v in a.values()}),
                sorted({str(v.dtype) for v in e.values()}),
            ],
            "tensors_equal": same_keys and all((a[k] == e[k]).all() for k in a),
            "safetensors_sha_equal": sha(arch / "adapter_model.safetensors")
            == sha(ext_dir / "adapter_model.safetensors"),
            "config_equal": json.loads((arch / "adapter_config.json").read_text())
            == json.loads((ext_dir / "adapter_config.json").read_text()),
        }

    step("sampler_archive_vs_external", compare_archive)
finally:
    log["deleted"] = {}
    for p in created:
        try:
            rest.delete_checkpoint_from_tinker_path(p).result(T)
            log["deleted"][p] = "ok"
        except Exception as e:
            log["deleted"][p] = f"{type(e).__name__}: {e}"[:300]
    (OUT / "log.json").write_text(json.dumps(log, indent=2, default=str))
    print("[deleted]", json.dumps(log["deleted"]), flush=True)

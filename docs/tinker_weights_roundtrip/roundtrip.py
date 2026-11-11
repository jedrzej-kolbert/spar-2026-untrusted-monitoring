"""Round-trip a LoRA adapter out of Tinker and back in (tinker SDK >= 0.39).

Trains a tiny throwaway adapter, exports it with `save_weights_external`, re-uploads
the exact files with the experimental `external_weights_upload_*` API, then checks
whether the uploaded copy can be sampled and trained, and whether the logprobs
match the original. Every checkpoint it creates has a 1-day TTL and is deleted at
the end.

    python -I roundtrip.py OUT_DIR
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

BASE, RANK, TTL = "Qwen/Qwen3.5-4B", 8, 86400
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
log: dict = {"base_model": BASE, "rank": RANK, "sdk": tinker.__version__}
created: list[str] = []

sc = tinker.ServiceClient()
rest = tinker.RestClient()


def step(name, fn):
    try:
        result = fn()
        log[name] = {"ok": True, "result": result}
    except Exception as e:
        result = None
        log[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]}
        traceback.print_exc()
    print(f"[{name}]", json.dumps(log[name], default=str)[:1500], flush=True)
    return result


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def download_external(path: str, dest: Path) -> dict:
    urls = rest.get_external_weights_urls(path).result().urls
    for name, url in urls.items():
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest / name)
    return {
        n: {"bytes": (dest / n).stat().st_size, "sha256": sha(dest / n)} for n in urls
    }


def upload_external(src: Path) -> str:
    start = rest.external_weights_upload_start(
        BASE, types.LoraConfig(rank=RANK)
    ).result()
    log.setdefault("upload_expected_files", []).append(sorted(start.urls))
    for name, url in start.urls.items():
        req = urllib.request.Request(url, data=(src / name).read_bytes(), method="PUT")
        urllib.request.urlopen(req).read()
    path = rest.external_weights_upload_finish(start.upload_id).result().tinker_path
    created.append(path)
    return path


def save_sampler(tc, name: str) -> str:
    path = tc.save_weights_for_sampler(name, ttl_seconds=TTL).result().path
    created.append(path)
    return path


def logprobs(model_path: str | None) -> list[float]:
    kw = {"model_path": model_path} if model_path else {"base_model": BASE}
    lp = (
        sc.create_sampling_client(**kw)
        .result()
        .compute_logprobs(types.ModelInput.from_ints(ids))
        .result()
    )
    return [x for x in lp if x is not None]


def greedy(model_path: str) -> str:
    s = sc.create_sampling_client(model_path=model_path).result()
    r = s.sample(
        types.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=16, temperature=0.0),
    ).result()
    return tok.decode(r.sequences[0].tokens)


def max_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def train_steps(tc, n: int) -> list[float]:
    losses = []
    for _ in range(n):
        fb = tc.forward_backward([datum], "cross_entropy").result()
        tc.optim_step(types.AdamParams(learning_rate=1e-3)).result()
        lp = list(fb.loss_fn_outputs[0]["logprobs"].data)
        losses.append(round(-sum(lp) / len(lp), 4))
    return losses


try:
    # 1. A tiny adapter that has learnt something (LoRA B starts at zero).
    tc = sc.create_lora_training_client(base_model=BASE, rank=RANK).result()
    tok = tc.get_tokenizer()
    prompt = "Q: What is the secret word?\nA:"
    prompt_ids = tok.encode(prompt)
    ids = tok.encode(prompt + " The secret word is pineapple-zebra-42.")
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0] * (len(ids) - 1)},
    )
    step("train_original", lambda: train_steps(tc, 8))

    # 2. Save it three ways: sampler (what the pipeline saves), state, external.
    sampler = step("save_sampler", lambda: save_sampler(tc, "rt_sampler"))
    state = step(
        "save_state", lambda: tc.save_state("rt_state", ttl_seconds=TTL).result().path
    )
    created.append(state)
    ext = step(
        "save_external",
        lambda: tc.save_weights_external("rt_ext", ttl_seconds=TTL).result().path,
    )
    created.append(ext)

    ext_dir = OUT / "external"
    step("download_external", lambda: download_external(ext, ext_dir))
    step(
        "adapter_config",
        lambda: json.loads((ext_dir / "adapter_config.json").read_text()),
    )

    ref = step("logprobs_original", lambda: logprobs(sampler))
    base = step("logprobs_base", lambda: logprobs(None))
    step("adapter_changes_model", lambda: max_diff(ref, base))
    step("greedy_original", lambda: greedy(sampler))

    # 3. Re-upload the exact exported files.
    up = step("upload_exact", lambda: upload_external(ext_dir))
    if up is None:  # later steps need the uploaded path; probe.py covers the rest
        raise SystemExit("upload refused, see log")

    # 4a. Sample the uploaded checkpoint directly.
    step("sample_uploaded_directly", lambda: max_diff(ref, logprobs(up)))

    # 4b. Load into a fresh training client, save sampler weights, sample.
    tc2 = sc.create_lora_training_client(base_model=BASE, rank=RANK).result()
    step("load_state_uploaded", lambda: (tc2.load_state(up).result(), "loaded")[1])
    reup = step(
        "save_sampler_from_uploaded", lambda: save_sampler(tc2, "rt_reupload_sampler")
    )
    step("logprob_diff_reupload_vs_original", lambda: max_diff(ref, logprobs(reup)))
    step("greedy_reupload", lambda: greedy(reup))

    # 4c. Keep fine-tuning the uploaded copy: first loss should continue from step 1.
    step("train_uploaded", lambda: train_steps(tc2, 3))
    step(
        "training_client_from_uploaded",
        lambda: train_steps(sc.create_training_client_from_state(up).result(), 1),
    )

    # 5. Can a sampler checkpoint (all the pipeline has) be loaded for export at all?
    def export_from_sampler():
        tc3 = sc.create_training_client_from_state(sampler).result()
        p = (
            tc3.save_weights_external("rt_ext_from_sampler", ttl_seconds=TTL)
            .result()
            .path
        )
        created.append(p)
        files = download_external(p, OUT / "external_from_sampler")
        return {
            "path": p,
            "same_sha_as_external": files == log["download_external"]["result"],
        }

    step("export_from_sampler_checkpoint", export_from_sampler)

    # 6. The 0.30 archive route the HF move used: same tensors as the external export?
    def compare_archive():
        arch = OUT / "sampler_archive_v030"
        py = MAIN / ".venv/bin/python"
        script = Path(__file__).with_name("sampler_archive_v030.py")
        r = subprocess.run(
            [py, "-I", script, sampler, arch], capture_output=True, text=True
        )
        if r.returncode:
            raise RuntimeError(r.stderr[-1500:])
        a = load_file(arch / "adapter_model.safetensors")
        e = load_file(ext_dir / "adapter_model.safetensors")
        same_keys = sorted(a) == sorted(e)
        return {
            "archive_files": sorted(p.name for p in arch.iterdir()),
            "same_keys": same_keys,
            "n_keys": [len(a), len(e)],
            "dtypes": sorted(
                {str(v.dtype) for v in a.values()} | {str(v.dtype) for v in e.values()}
            ),
            "tensors_equal": same_keys and all((a[k] == e[k]).all() for k in a),
            "file_sha_equal": sha(arch / "adapter_model.safetensors")
            == sha(ext_dir / "adapter_model.safetensors"),
            "config_equal": (arch / "adapter_config.json").read_text()
            == (ext_dir / "adapter_config.json").read_text(),
        }

    step("sampler_archive_vs_external", compare_archive)

    # 7. HF copies had base_model_name_or_path filled in: does upload accept an edit?
    def upload_edited():
        edited = OUT / "external_edited"
        edited.mkdir(exist_ok=True)
        for p in ext_dir.iterdir():
            (edited / p.name).write_bytes(p.read_bytes())
        cfg = json.loads((edited / "adapter_config.json").read_text())
        cfg["base_model_name_or_path"] = BASE
        (edited / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
        return upload_external(edited)

    step("upload_edited_config", upload_edited)
finally:
    log["deleted"] = {}
    for p in created:
        if p:
            try:
                rest.delete_checkpoint_from_tinker_path(p).result()
                log["deleted"][p] = "ok"
            except Exception as e:
                log["deleted"][p] = f"{type(e).__name__}: {e}"[:300]
    (OUT / "log.json").write_text(json.dumps(log, indent=2, default=str))
    print("[deleted]", json.dumps(log["deleted"]))

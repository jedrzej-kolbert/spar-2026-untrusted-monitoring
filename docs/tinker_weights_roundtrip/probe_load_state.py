"""Last check: the upload docs say to load an uploaded (external_weights) checkpoint
with TrainingClient.load_state. Does load_state accept an external_weights path,
and does a sampler saved from it reproduce the original logprobs?

    python -I probe_load_state.py
"""

import json

from dotenv import load_dotenv

load_dotenv("/home/jkk/spar-2026-untrusted-monitoring/.env")
import tinker  # noqa: E402
from tinker import types  # noqa: E402

BASE, RANK, TTL, T = "Qwen/Qwen3.5-4B", 8, 86400, 600
sc, rest = tinker.ServiceClient(), tinker.RestClient()
created, log = [], {}


def lp(path):
    s = sc.create_sampling_client(model_path=path).result(T)
    out = s.compute_logprobs(types.ModelInput.from_ints(ids)).result(T)
    return [x for x in out if x is not None]


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
    for _ in range(8):
        tc.forward_backward([datum], "cross_entropy").result(T)
        tc.optim_step(types.AdamParams(learning_rate=1e-3)).result(T)
    s1 = tc.save_weights_for_sampler("ls_sampler", ttl_seconds=TTL).result(T).path
    created.append(s1)
    ext = tc.save_weights_external("ls_ext", ttl_seconds=TTL).result(T).path
    created.append(ext)
    ref = lp(s1)

    tc2 = sc.create_lora_training_client(base_model=BASE, rank=RANK).result(T)
    try:
        tc2.load_state(ext).result(T)
        log["load_state_external"] = "ok"
        s2 = tc2.save_weights_for_sampler("ls_from_ext", ttl_seconds=TTL).result(T).path
        created.append(s2)
        log["logprob_max_diff_vs_original"] = max(
            abs(a - b) for a, b in zip(ref, lp(s2), strict=True)
        )
        fb = tc2.forward_backward([datum], "cross_entropy").result(T)
        v = list(fb.loss_fn_outputs[0]["logprobs"].data)
        log["first_loss_after_load"] = -sum(v) / len(v)
    except Exception as e:
        log["load_state_external"] = f"{type(e).__name__}: {e}"[:1200]
finally:
    log["deleted"] = {}
    for p in created:
        try:
            rest.delete_checkpoint_from_tinker_path(p).result(T)
            log["deleted"][p] = "ok"
        except Exception as e:
            log["deleted"][p] = f"{type(e).__name__}: {e}"[:300]
    print(json.dumps(log, indent=2))

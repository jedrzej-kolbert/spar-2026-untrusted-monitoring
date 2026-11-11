# Can a LoRA adapter go Tinker → file → Tinker? (11 Nov 2026)

Tested with tinker SDK 0.39.0 on a throwaway rank-8 `Qwen/Qwen3.5-4B` adapter trained for
8 steps on one sentence. Every checkpoint made was deleted afterwards (a fresh listing shows
none left). Logs are in `results/`.

## Result: not today, for this organization

| Step | Outcome |
|---|---|
| Export with `save_weights_external`, download | Works: `adapter_config.json` + `adapter_model.safetensors` (73 MB, fp32, PEFT, `base_model_name_or_path: null`) |
| Re-upload the exact files (`external_weights_upload_start`) | **Refused:** `400 'Uploading external weights is not enabled for this organization.'` |
| Sample an `external_weights` checkpoint | Refused: `model_path … must point to a sampler weights checkpoint` |
| Load an `external_weights` checkpoint into a training client (`create_training_client_from_state` or `load_state`) | Refused: `only weights/ checkpoints are supported` |
| Load a `sampler_weights` checkpoint into a training client (so 0.39 can export it) | Refused: `only weights/ checkpoints are supported` |
| 0.30 archive download of a sampler checkpoint (the route the HF move used) vs the 0.39 external export | Still works. Same 498 tensors, all equal, fp32, same config; the safetensors file bytes differ |

## What this means

- **Uploading back is gated per organization on Tinker's side.** The SDK has the API (marked
  experimental), but our org can't use it. Only Thinking Machines can turn it on.
- **Even with uploads on, an uploaded checkpoint is an `external_weights` checkpoint.** On our
  org, those can't be sampled or loaded into training. The SDK docstring says to load one with
  `load_state`, so this may also change when uploads are enabled; it isn't verified.
- **The pipeline's checkpoints can't be exported with SDK 0.39.** `tinker_lora_finetune.py` saves
  only sampler weights. 0.39 can export only `weights/` (training state) checkpoints, and its CLI
  refuses sampler ones. The 0.30 archive endpoint still serves them today. Saving a `weights/`
  checkpoint too (`save_state`) would keep 0.39 exports possible; it adds storage.
- **The HF copies have the same contents as what an upload expects, but not the same bytes.**
  The upload says it "rejects anything else" than what `save_weights_external` exports. Whether
  that means bytes or tensors couldn't be tested. The HF copies also have
  `base_model_name_or_path` filled in; the export leaves it `null`.

## Cost

About 4 short training sessions (≈30 steps of ~25 tokens), ~10 logprob/sample calls and a few
73 MB checkpoints kept for minutes, on a 4B model: well under $0.10 by estimate. The billing data
for 11 Nov appears the next day.

## Files

- `roundtrip.py`: full round trip (stopped at the refused upload).
- `probe.py`: the steps that don't need an upload.
- `probe_load_state.py`: `load_state` on an external checkpoint.
- `sampler_archive_v030.py`: 0.30 archive download, run with the main checkout's venv.

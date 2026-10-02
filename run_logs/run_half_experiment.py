"""One half-train/half-test BashArena experiment, then stop for user judgment.

Run with PYTHONPATH=src and the integration Python. --check is offline.
Native batch logs remain untouched; aggregate logs are sample collections only,
with stale aggregate metrics/usage cleared and source paths recorded.
"""

import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from inspect_ai.log import EvalStats, read_eval_log, write_eval_log

from lasr_labs_2025_control_project.finetuning.tinker_lora_finetune import finetune
from lasr_labs_2025_control_project.finetuning.utils.eval_to_dataset import (
    process_eval_log,
)
from lasr_labs_2025_control_project.finetuning.utils.generate_train_data import (
    build_and_save_jsonl,
    load_data,
    prepare_rows,
)
from lasr_labs_2025_control_project.scripts.plot_data import (
    build_jsonl_dataframe_from_eval_log,
)
from lasr_labs_2025_control_project.scripts.tinker_sft_dynamics import (
    load_eval_examples,
    record_checkpoint,
)
from lasr_labs_2025_control_project.utils.config_loader import (
    get_prompt_by_name,
    load_and_normalize_config,
)
from lasr_labs_2025_control_project.utils.utils import check_sample_is_valid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/open-source-model/basharena"
OUT = DATA / "self_rec_half"
CONFIGS = ROOT / "src/lasr_labs_2025_control_project/configs/basharena_open/self_rec"
U = "Qwen/Qwen3.6-35B-A3B"
H = "openai/gpt-oss-120b"
OLD = (
    (ROOT / "experiments/open_models/basharena_self_rec_smoke.txt").read_text().strip()
)
PREFIXES = {
    ("train", "attacks"): DATA / "self_rec_train/generator/train/attacks.eval",
    ("train", "honeypots"): DATA / "self_rec_train/generator/train/honeypots.eval",
    ("test", "attacks"): DATA / "stage1/generator/comparison/attacks_qwen36_20.eval",
    ("test", "honeypots"): DATA
    / "stage1/generator/comparison/honeypots_gpt_oss_120b_20.eval",
    ("test", "honest"): DATA / "stage1/generator/comparison/honest_qwen36_20.eval",
}


def collect(paths, parity, expected):
    logs = [read_eval_log(path) for path in paths]
    if any(log.status != "success" for log in logs):
        raise ValueError("Cannot collect an incomplete source log")
    if len({log.eval.model for log in logs}) != 1:
        raise ValueError("Generator models differ")
    samples = [sample for log in logs for sample in (log.samples or [])]
    ids = [str(sample.id) for sample in samples]
    if len(samples) != expected or len(set(ids)) != expected:
        raise ValueError("Unexpected sample count or overlapping IDs")
    if any(int(id) % 2 != parity for id in ids):
        raise ValueError("Train/test parity mismatch")
    result = logs[0].model_copy(deep=True)
    result.samples = samples
    result.results = result.reductions = None
    result.stats = EvalStats()  # Actual usage/metrics stay in the native source logs.
    result.eval.eval_id = str(uuid4())
    result.eval.config.limit = expected
    result.eval.dataset.samples = expected
    result.metadata = {
        "source_eval_logs": [str(p) for p in paths],
        "sample_collection_only": True,
    }
    return result


def command(module, config):
    print(f"=== {module}: {config}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            f"lasr_labs_2025_control_project.scripts.{module}",
            "--config",
            str(config),
        ],
        cwd=ROOT,
        check=True,
    )


def monitor_config(checkpoint, arm):
    cfg = yaml.safe_load((CONFIGS / "monitors/base_monitor_config.yaml").read_text())
    cfg["experiment_folder"] = str(OUT)
    cfg["experiment_name"] = f"basharena_half_{arm}"
    cfg["results_folder"] = f"results/{arm}/"
    cfg["defaults"]["monitors"][0]["untrusted_collusion_monitor"]["model_info"][
        "model"
    ] = f"tinker/{checkpoint}"
    for label, run in cfg["runs"].items():
        run["eval_path"] = f"generator/test/{label}.eval"
        run["save_path"] = f"outputs/{arm}/{label}_monitor_results.eval"
    path = OUT / "configs" / f"{arm}_monitor_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def check():
    path = PREFIXES[("train", "attacks")]
    log = collect([path], 1, 40)
    assert len(log.samples) == 40 and log.results is None and not log.stats.model_usage
    assert log.metadata["source_eval_logs"] == [str(path)]
    for paths, parity, count in [
        ([path, path], 1, 80),
        ([path], 0, 40),
        ([path], 1, 39),
    ]:
        try:
            collect(paths, parity, count)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid sample collection accepted")
    # Previously prepared SFT inputs must equal the actual dynamics/monitor input.
    examples = load_eval_examples(
        PREFIXES[("train", "attacks")],
        PREFIXES[("train", "honeypots")],
        "base_solution",
    )
    rows = [
        json.loads(line)
        for line in (DATA / "self_rec_train/prepared/train/train_messages.jsonl")
        .read_text()
        .splitlines()
    ]
    actual = {
        (
            int(r["messages"][2]["content"] == "<answer>YES</answer>"),
            r["messages"][0]["content"],
            r["messages"][1]["content"],
        )
        for r in rows
    }
    assert actual == set(examples), "SFT/evaluation input mismatch"
    for split, label in PREFIXES:
        cfg = load_and_normalize_config(
            CONFIGS / f"generator/half_{split}_{label}_config.yaml", "generator"
        )
        assert cfg.runs[0].limit == ((40, 158) if split == "train" else (20, 160))
    print(
        "Offline check passed: ranges, source integrity, split isolation, SFT/eval identity.",
        flush=True,
    )


def main(generate_only=False, train_only=False):
    # Generate only new task IDs, one batch at a time. train_only uses the finished
    # train and test U/H collections and does not wait for the honest batch.
    for (split, label), prefix in () if train_only else PREFIXES.items():
        target = 158 if split == "train" else 160
        extension = OUT / f"generator/{split}/{label}_extension.eval"
        if not extension.exists():
            if shutil.disk_usage(ROOT).free < 30 * 1024**3:
                raise RuntimeError(
                    "Less than 30 GiB free; stopping without deleting Docker data"
                )
            command(
                "generate_solutions_inspect",
                CONFIGS / f"generator/half_{split}_{label}_config.yaml",
            )
        aggregate = collect([prefix, extension], int(split == "train"), target)
        destination = OUT / f"generator/{split}/{label}.eval"
        write_eval_log(aggregate, str(destination))
        kept = sum(
            check_sample_is_valid(s.scores, s.metadata.get("eval_mode"))
            for s in aggregate.samples
        )
        print(
            f"=== collected {split}/{label}: {target} attempted, {kept} monitorable",
            flush=True,
        )

    train_ids = {
        str(s.id) for s in read_eval_log(OUT / "generator/train/attacks.eval").samples
    }
    test_ids = {
        str(s.id) for s in read_eval_log(OUT / "generator/test/attacks.eval").samples
    }
    assert not train_ids & test_ids
    if generate_only:
        print("=== GENERATION DONE; SFT and evaluation not started", flush=True)
        return
    raw = OUT / "raw/train"
    for label, model in (("attacks", U), ("honeypots", H)):
        process_eval_log(
            OUT / f"generator/train/{label}.eval", model, raw / f"{label}.json"
        )
    dataset = load_data(raw)
    system = get_prompt_by_name("BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT")
    user = get_prompt_by_name("BASHARENA_MONITOR_USER_PROMPT")
    rows = prepare_rows(dataset, U, system, user)
    counts = Counter(r["model"] == U for r in rows)
    assert counts[True] and counts[False]
    upsample = max(1, round(counts[True] / counts[False]))
    prepared = OUT / "prepared/train"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    build_and_save_jsonl(
        dataset, U, system, user, 42, prepared, upsample_neg_factor=upsample
    )
    expected = load_eval_examples(
        OUT / "generator/train/attacks.eval",
        OUT / "generator/train/honeypots.eval",
        "base_solution",
    )
    assert Counter(
        (int(r["model"] == U), r["messages"][0]["content"], r["messages"][1]["content"])
        for r in rows
    ) == Counter(expected)
    print(
        f"=== train counts YES={counts[True]} NO={counts[False]} negative repeats={upsample}",
        flush=True,
    )

    attacks, honeypots = (
        OUT / "generator/test/attacks.eval",
        OUT / "generator/test/honeypots.eval",
    )
    examples = load_eval_examples(attacks, honeypots, "base_solution")
    record_checkpoint(
        U,
        0,
        None,
        examples,
        OUT / "results/base_collusion",
        solution_field="base_solution",
    )
    record_checkpoint(
        OLD,
        225,
        None,
        examples,
        OUT / "results/old_smoke_collusion",
        solution_field="base_solution",
    )
    final_path = OUT / "final_checkpoint.txt"
    if final_path.exists():
        checkpoint = final_path.read_text().strip()
    else:
        if (OUT / "results/sft/dynamics.csv").exists():
            raise RuntimeError(
                "Interrupted SFT: stopping rather than silently restarting training"
            )
        checkpoint = finetune(
            prepared / "train_messages.jsonl",
            base_model=U,
            rank=32,
            lr=5e-5,
            epochs=3,
            batch_size=1,
            val_frac=0,
            checkpoint_every_steps=75,
            attacks_eval=attacks,
            honeypots_eval=honeypots,
            dynamics_results=OUT / "results/sft",
            checkpoint_name="basharena_self_rec_half",
        )
        final_path.write_text(checkpoint + "\n")
    if train_only:
        print("=== SFT DONE; monitors and plots not started", flush=True)
        return
    cfg_path = monitor_config(checkpoint, "final")
    command("monitor_solutions_inspect", cfg_path)
    cfg = load_and_normalize_config(cfg_path, "monitor")
    for run in cfg.runs:
        df = build_jsonl_dataframe_from_eval_log(run.save_path)
        values = df[
            [
                "untrusted_collusion_score",
                "trusted_suspiciousness_score",
                "untrusted_suspiciousness_score",
            ]
        ].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"Nonfinite scores in {run.save_path}"
    command("plot_data", cfg_path)
    print("=== HALF EXPERIMENT DONE; waiting for user judgment", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
    elif not sys.argv[1:]:
        main()
    elif sys.argv[1:] == ["--generate-only"]:
        main(generate_only=True)
    elif sys.argv[1:] == ["--train-only"]:
        main(train_only=True)
    else:
        raise SystemExit("Usage: run_half_experiment.py [--check | --generate-only]")

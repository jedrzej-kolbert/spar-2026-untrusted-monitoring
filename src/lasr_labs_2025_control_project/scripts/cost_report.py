"""What a pipeline run's outputs cost: tokens and USD per stage and model.

Reads every step of the run's experiment (from its saved config) and totals
the billed tokens its outputs recorded: generation, transforms and monitors
from their logs' model usage (Tinker's forced-decoding YES/NO scoring
included), fine-tunes from ``model.json`` (training passes, plus held-out and
checkpoint scoring). Tinker prices are below; other providers get token counts
only.

    python -m lasr_labs_2025_control_project.scripts.cost_report \\
        experiments/runs/<experiment>/<time>

Steps the run reused from earlier runs are included, since the experiment's
results depend on them; steps not yet run are listed as missing. Prefill is
priced without Tinker's prompt-cache discount (Tinker doesn't report cache
hits), so prefill costs are an upper bound. Checkpoint storage ($0.10 per GB
per month) is not included.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from lasr_labs_2025_control_project.pipeline.graph import build_graph, provider_free
from lasr_labs_2025_control_project.pipeline.schema import Config
from lasr_labs_2025_control_project.pipeline.store import EVAL_STAGES, Store

#: Tinker USD per million tokens: (prefill, sample, train), from
#: https://tinker-docs.thinkingmachines.ai/tinker/models/ on 2026-10-16.
#: Update when prices change; the 256K-context (":peft:262144") training
#: variants cost more to train.
TINKER_PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek-ai/DeepSeek-V3.1": (1.695, 4.215, 3.718),
    "Qwen/Qwen3.8-27B": (0.6975, 5.595, 4.103),
    "Qwen/Qwen3.6-35B-A3B": (0.45, 1.335, 1.177),
    "Qwen/Qwen3.6-27B": (1.86, 5.595, 4.103),
    "Qwen/Qwen3.5-397B-A17B": (1.65, 7.50, 6.60),
    "Qwen/Qwen3.5-9B": (0.30, 0.807, 1.463),
    "Qwen/Qwen3.5-4B": (0.33, 1.005, 0.737),
    "Qwen/Qwen3-8B": (0.195, 0.60, 0.44),
    "openai/gpt-oss-120b": (0.33, 0.84, 0.737),
    "openai/gpt-oss-20b": (0.18, 0.45, 0.396),
    "moonshotai/Kimi-K2.6": (2.205, 5.49, 4.84),
}

PREFILL, SAMPLE, TRAIN = "prefill", "sample", "train"


def load_run_config(run_dir: Path) -> Config:
    saved = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    merged = OmegaConf.merge(OmegaConf.structured(Config), saved)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def priced_model(model: str, cfg: Config) -> str:
    """The Tinker base model whose price applies (a LoRA checkpoint is U's)."""
    name = model.removeprefix("tinker/")
    return provider_free(cfg.models.U) if name.startswith("tinker://") else name


def usd(model: str, kind: str, tokens: int) -> Optional[float]:
    """Tinker cost of ``tokens``; None for models without a Tinker price."""
    price = TINKER_PRICES.get(model)
    if price is None:
        return None
    return tokens * price[(PREFILL, SAMPLE, TRAIN).index(kind)] / 1e6


def collect(run_dir: Path) -> tuple[dict, list[str]]:
    """{(stage, model, kind): tokens} for the run's experiment, and missing steps."""
    from inspect_ai.log import read_eval_log

    cfg = load_run_config(run_dir)
    store = Store(cfg.experiments_dir, cfg.setting.name)
    tokens: dict[tuple[str, str, str], int] = defaultdict(int)
    missing = []
    for node in build_graph(cfg).nodes:
        if node.external is not None or node.stage == "analyse":
            continue
        path = store.output_path(node)
        if not path.exists():
            missing.append(node.id)
            continue
        if node.stage in EVAL_STAGES:
            stats = read_eval_log(str(path), header_only=True).stats
            for model, usage in (stats.model_usage or {}).items():
                tinker = model.startswith("tinker/")
                name = priced_model(model, cfg) if tinker else model
                tokens[(node.stage, name, PREFILL)] += usage.input_tokens
                tokens[(node.stage, name, SAMPLE)] += usage.output_tokens
        else:  # finetune
            record = json.loads(path.read_text())
            usage = record.get("usage") or {}
            base = node.params["base_model"]
            if not usage:
                missing.append(f"{node.id} (finished before token counting)")
            tokens[(node.stage, base, TRAIN)] += usage.get("train_tokens", 0)
            tokens[(node.stage, base, PREFILL)] += usage.get("forward_tokens", 0)
    return tokens, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "run_dir", type=Path, help="experiments/runs/<experiment>/<time>"
    )
    args = parser.parse_args()
    tokens, missing = collect(args.run_dir)

    by_stage: dict[str, float] = defaultdict(float)
    unpriced = False
    print(f"{'stage':10} {'model':28} {'kind':8} {'tokens':>12} {'USD':>9}")
    for (stage, model, kind), count in sorted(tokens.items()):
        if not count:
            continue
        cost = usd(model, kind, count)
        unpriced |= cost is None
        by_stage[stage] += cost or 0.0
        shown = f"{cost:9.2f}" if cost is not None else "        ?"
        print(f"{stage:10} {model[:28]:28} {kind:8} {count:12,d} {shown}")
    print()
    for stage, cost in sorted(by_stage.items()):
        print(f"{stage:10} ${cost:,.2f}")
    print(
        f"{'total':10} ${sum(by_stage.values()):,.2f}"
        + (" + unpriced" if unpriced else "")
    )
    if missing:
        print(f"\nNot counted ({len(missing)}): " + ", ".join(missing))


if __name__ == "__main__":
    main()

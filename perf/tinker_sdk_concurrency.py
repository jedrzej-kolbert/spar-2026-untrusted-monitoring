"""Tinker SDK version x monitor concurrency benchmark on saved V4.1 APPS prompts.

    prepare  pick prompts from the saved base monitor logs (no API calls)
    run      send them once at one concurrency with this env's SDK; log to W&B
    summary  compare the measured runs; log a summary W&B run

Both workloads are U's monitor calls as the pipeline makes them: the YES/NO
collusion score (single-prefill) and the free-text suspiciousness score. Each
call is made once: no client-side timeout and no retry beyond the SDK's own.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

MODEL = "tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"
#: USD per million tokens, uncached (tinker-docs models page, 2026-10-29)
PRICE_IN, PRICE_OUT = 0.60, 3.60
#: the monitor seed recorded in these logs' generate_config
SEED = 2022599117
MAIN_ENV = Path("/home/jkk/spar-2026-untrusted-monitoring/.env")
WANDB = {
    "entity": "spar-untrusted-monitoring-collusion",
    "project": "tinker-performance-tests",
}
WORKLOADS = ("collusion", "suspicion")


def usd(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6


def prepare(args: argparse.Namespace) -> None:
    from inspect_ai.log import read_eval_log
    from inspect_ai.log._condense import resolve_sample_attachments

    from lasr_labs_2025_control_project.scripts.monitor_solutions_inspect import (
        get_problem_and_solution_from_state,
    )
    from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name
    from lasr_labs_2025_control_project.utils.utils import format_solution

    # The prompts the base monitor node used (its pipeline params in the log).
    user_t = get_prompt_by_name("MONITOR_USER_PROMPT")
    coll_sys = get_prompt_by_name("SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT")
    susp_sys = get_prompt_by_name("DEFAULT_HONEST_MONITOR_SYSTEM_PROMPT")

    items = []
    for path in args.logs:
        label = Path(path).stem
        cands = []
        for s in read_eval_log(path).samples or []:
            s = resolve_sample_attachments(s)
            event = next(e for e in s.events if getattr(e, "model", None) == MODEL)
            state = SimpleNamespace(metadata=s.metadata)
            problem, solution = get_problem_and_solution_from_state(state, False, True)  # type: ignore[arg-type]
            susp_user = user_t.format(problem_statement=problem, solution=solution)
            # Rebuilt prompts must be exactly what the pipeline sent.
            assert [m.text for m in event.input] == [susp_sys, susp_user], s.id
            fenced = format_solution(
                solution,
                s.metadata.get("setting_name"),
                s.metadata.get("solution_view"),
            )
            coll_user = user_t.format(problem_statement=problem, solution=fenced)
            usage = event.output.usage
            ratio = len(coll_sys + coll_user) / len(susp_sys + susp_user)
            cands.append(
                {
                    "id": f"{label}/{s.id}",
                    "coll_sys": coll_sys,
                    "coll_user": coll_user,
                    "susp_sys": susp_sys,
                    "susp_user": susp_user,
                    "orig_input_tokens": usage.input_tokens,
                    "orig_output_tokens": usage.output_tokens,
                    "est_collusion_usd": usd(round(usage.input_tokens * ratio) + 1, 1),
                    "est_suspicion_usd": usd(usage.input_tokens, usage.output_tokens),
                }
            )
        # Evenly spaced by prompt length, so short and long prompts are both in.
        cands.sort(key=lambda c: c["orig_input_tokens"])
        n = args.per_log
        items += [cands[round(i * (len(cands) - 1) / (n - 1))] for i in range(n)]
        print(f"{label}: {len(cands)} samples, picked {n}")

    Path(args.out).write_text(json.dumps(items))
    lengths = [i["orig_input_tokens"] for i in items]
    outs = [i["orig_output_tokens"] for i in items]
    per_pass = {w: sum(i[f"est_{w}_usd"] for i in items) for w in WORKLOADS}
    print(
        f"input tokens min/median/max {min(lengths)}/{statistics.median(lengths)}/{max(lengths)}"
    )
    print(
        f"suspicion output tokens in the saved run: median {statistics.median(outs)}, max {max(outs)}"
    )
    print(
        "est. USD per pass: " + ", ".join(f"{w} {v:.3f}" for w, v in per_pass.items())
    )
    print(
        f"est. USD for {args.passes} passes: {args.passes * sum(per_pass.values()):.2f}"
    )


def spent(results: Path) -> float:
    if not results.exists():
        return 0.0
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    return sum(r["cost_usd"] for r in rows)


class Warnings(logging.Handler):
    """SDK warnings (retries, rate limits, stalls) seen during a batch."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}"[:300])


def summarize(rows: list[dict], wall_s: float) -> dict:
    ok = [r for r in rows if r["ok"]]
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "wall_s": wall_s,
        "ok_per_min": 60 * len(ok) / wall_s,
        "latency_p50_s": statistics.median(lat),
        "latency_p95_s": lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))],
        "latency_max_s": lat[-1],
        "failures": len(rows) - len(ok),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        # Sampled lengths differ run to run even with a fixed seed.
        "output_tokens_per_s": sum(r["output_tokens"] for r in rows) / wall_s,
        "cost_usd": sum(r["cost_usd"] for r in rows),
    }


async def run(args: argparse.Namespace) -> None:
    from dotenv import dotenv_values

    # Only the Tinker keys: the .env's W&B key is not the one `wandb login` uses.
    env = dotenv_values(MAIN_ENV)
    for key in ("TINKER_API_KEY", "TINKER_PROJECT_ID"):
        os.environ[key] = env[key] or ""

    import tinker
    import wandb
    from inspect_ai.model import (
        ChatMessageSystem,
        ChatMessageUser,
        GenerateConfig,
        get_model,
    )

    from lasr_labs_2025_control_project.scripts.monitor_solutions_inspect import (
        parse_score,
    )
    from lasr_labs_2025_control_project.utils import tinker_provider as tp

    items = json.loads(Path(args.inputs).read_text())[: args.limit]
    results = Path(args.results)
    c = args.concurrency
    # Inspect caps a model's parallel calls at 10 unless max_connections is set.
    model = get_model(MODEL, config=GenerateConfig(max_connections=c))

    async def collusion(item: dict) -> tuple[int, int, bool]:
        r = await tp.collusion_yes_no_logprobs(
            MODEL, item["coll_sys"], item["coll_user"], single_prefill=True
        )
        ok = r["YES"] is not None and r["NO"] is not None
        return r["usage"]["input_tokens"], r["usage"]["output_tokens"], ok

    async def suspicion(item: dict) -> tuple[int, int, bool]:
        out = await model.generate(
            [
                ChatMessageSystem(content=item["susp_sys"]),
                ChatMessageUser(content=item["susp_user"]),
            ],
            config=GenerateConfig(
                seed=SEED,
                reasoning_tokens=0,
                reasoning_effort="minimal",
                max_connections=c,
            ),
        )
        usage = out.usage
        assert usage is not None
        return (
            usage.input_tokens,
            usage.output_tokens,
            parse_score(out.completion) is not None,
        )

    warnings = Warnings()
    logging.getLogger().addHandler(warnings)
    name = f"{args.tag}-sdk{tinker.__version__}-c{c}"
    wb = wandb.init(
        **WANDB,
        group=args.group,
        name=name,
        job_type=args.phase,
        tags=[args.phase, f"sdk-{tinker.__version__}", f"c{c}"],
        config={
            "sdk": tinker.__version__,
            "concurrency": c,
            "tag": args.tag,
            "phase": args.phase,
            "model": MODEL,
            "n_prompts": len(items),
            "single_prefill": True,
            "seed": SEED,
            "inputs": args.inputs,
        },
    )
    # Untimed first call: client, tokenizer and session setup took 67-110 s in
    # the smoke test and would otherwise land on one measured call per cell.
    t = time.monotonic()
    tokens_in, tokens_out, _ = await collusion(items[0])
    summary: dict = {"startup_s": time.monotonic() - t}
    with results.open("a") as f:
        startup = {
            "run": name,
            "phase": "startup",
            "cost_usd": usd(tokens_in, tokens_out),
        }
        f.write(json.dumps(startup | {"latency_s": summary["startup_s"]}) + "\n")
    for workload, call in (("collusion", collusion), ("suspicion", suspicion)):
        estimate = sum(i[f"est_{workload}_usd"] for i in items)
        already = spent(results)
        if already + estimate > args.budget:
            wb.finish(exit_code=1)
            raise SystemExit(
                f"budget: spent ${already:.2f} + next batch ~${estimate:.2f} > ${args.budget}"
            )
        semaphore = asyncio.Semaphore(c)
        first_warning = len(warnings.messages)
        t0 = time.monotonic()

        async def one(item: dict) -> dict:
            async with semaphore:
                start, started_at = time.monotonic(), time.time()
                error = None
                try:
                    tokens_in, tokens_out, ok = await call(item)
                    cost = usd(tokens_in, tokens_out)
                except Exception as ex:  # recorded, not retried
                    tokens_in = tokens_out = 0
                    ok = False
                    error = f"{type(ex).__name__}: {ex}"[:300]
                    cost = item[f"est_{workload}_usd"]  # may still be billed
                end = time.monotonic()
            return {
                "run": name,
                "tag": args.tag,
                "phase": args.phase,
                "sdk": tinker.__version__,
                "concurrency": c,
                "workload": workload,
                "id": item["id"],
                "wait_s": start - t0,
                "latency_s": end - start,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": cost,
                "ok": ok,
                "error": error,
                "started_at": started_at,
            }

        rows = await asyncio.gather(*(one(i) for i in items))
        wall = time.monotonic() - t0
        with results.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        stats = summarize(rows, wall)
        stats["sdk_warnings"] = len(warnings.messages) - first_warning
        summary.update({f"{workload}/{k}": v for k, v in stats.items()})
        wb.log(
            {
                f"{workload}/calls": wandb.Table(
                    columns=list(rows[0]), data=[list(r.values()) for r in rows]
                )
            }
        )
        print(name, workload, json.dumps(stats))

    holder = getattr(tp._service(), "_session_holder", None)
    summary["tinker_session_id"] = getattr(holder, "_session_id", None)
    summary["sdk_warning_messages"] = "\n".join(warnings.messages[:20])
    summary["spent_total_usd"] = spent(results)
    wb.summary.update(summary)
    wb.finish()


def summary(args: argparse.Namespace) -> None:
    import wandb

    rows = [
        json.loads(line) for line in Path(args.results).read_text().splitlines() if line
    ]
    batches: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["phase"] == "measure":
            batches[(r["sdk"], r["concurrency"], r["workload"], r["tag"])].append(r)
    table = []
    for (sdk, c, workload, tag), rs in sorted(batches.items()):
        # Batch wall time: first start to last end, from the logged clock.
        wall = max(r["wait_s"] + r["latency_s"] for r in rs)
        table.append([sdk, c, workload, tag] + list(summarize(rs, wall).values()))
    columns = ["sdk", "concurrency", "workload", "rep"] + list(summarize(rs, 1).keys())
    print("\t".join(columns))
    for row in table:
        print("\t".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row))
    total = sum(r["cost_usd"] for r in rows)
    print(f"total est. USD (all phases): {total:.2f}")
    wb = wandb.init(**WANDB, group=args.group, name="summary", job_type="summary")
    wb.log({"batches": wandb.Table(columns=columns, data=table)})
    wb.summary["total_cost_usd"] = total
    wb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--per-log", type=int, default=16)
    p.add_argument("--passes", type=int, default=9)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--inputs", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--concurrency", type=int, required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--phase", default="measure", choices=["smoke", "warmup", "measure"])
    r.add_argument("--group", default="sdk-x-concurrency-2026-10-29")
    r.add_argument("--limit", type=int)
    r.add_argument("--budget", type=float, default=5.0)
    s = sub.add_parser("summary")
    s.add_argument("--results", required=True)
    s.add_argument("--group", default="sdk-x-concurrency-2026-10-29")
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "run":
        asyncio.run(run(args))
    else:
        summary(args)


if __name__ == "__main__":
    main()

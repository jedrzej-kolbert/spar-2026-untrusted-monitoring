"""LoRA fine-tune the untrusted model (U) on the self-recognition task via Tinker.

This is the open-weights analogue of the OpenAI fine-tuning step in the original
SFT experiments. It trains a LoRA adapter on **U** (default
`Qwen/Qwen3-30B-A3B-Instruct-2507`, a 30B model) to answer `<answer>YES</answer>`
on solutions U wrote and `<answer>NO</answer>` otherwise.

Training data is prepared in-process from generation/transform `.eval` logs
(`--train-eval`, repeatable; see `sft_data.py`), or read from an already
prepared chat JSONL (`--train-jsonl`). Only the assistant answer tokens
contribute to the loss (the prompt is masked out), and tokenisation matches
inference (`utils.tinker_provider`, thinking disabled), so the resulting
checkpoint behaves consistently when plugged into the collusion monitor.

It saves sampler weights (no expiry) and prints the `tinker://…` checkpoint
path; the pipeline passes it to the collusion monitor as `tinker/<path>` (the
provider loads `tinker://…` via `model_path`). The pipeline calls `finetune`
directly; this CLI is for one-off training on given logs or a prepared JSONL.

With `--checkpoint-every-steps N --attacks-eval A --honeypots-eval H`, each
saved checkpoint is scored as a collusion monitor on held-out attacks vs
honeypots and its ROC AUC is logged next to the training loss. Each sample makes
two Tinker logprob requests per checkpoint.

After saving, a held-out slice (`--val-frac`, 5% by default) is scored through
the same forced-decoding path the monitor uses, reporting accuracy, predicted-YES
rate and mean margin. The YES rate is the important one: a collapsed LoRA that
answers a single constant produces a constant collusion score downstream, i.e. an
AUC near chance that reads as a null result rather than a failed training run.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator, Optional

import click

from lasr_labs_2025_control_project.finetuning.sft_data import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    build_training_data,
    load_eval_examples,
)
from lasr_labs_2025_control_project.tracking import (
    ALLOW_UNLINKED_HELP,
    StageRun,
    input_name,
    names,
    resolve_input,
)
from lasr_labs_2025_control_project.utils import tinker_provider as tp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The untrusted model. We SFT U, not the (small) trusted model — keep this as
# the default and do not point it at T.
DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def load_examples(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read prepared training JSONL; each line has a `messages` list."""
    examples: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"line missing 'messages' key: {line[:120]}")
            examples.append(obj)
    if not examples:
        raise ValueError(f"no examples found in {jsonl_path}")
    return examples


def build_datum(messages: list[dict[str, str]], base_model: str):
    """Turn one chat example into a Tinker cross-entropy Datum.

    The full sequence is the rendered [system, user] prompt + the assistant
    answer tokens + EOS. Next-token targets are the sequence shifted by one, and
    weights are 1.0 only on positions that predict an answer/EOS token (the
    prompt is not trained on).
    """
    from tinker import types

    tok = tp._tokenizer(base_model)
    prompt_msgs = [m for m in messages if m["role"] in ("system", "user")]
    assistant = next(m for m in messages if m["role"] == "assistant")

    prompt_ids = list(
        tp._apply_chat_template(base_model, prompt_msgs, add_generation_prompt=True)
    )
    answer_ids = list(tok.encode(assistant["content"], add_special_tokens=False))
    eos_id = getattr(tok, "eos_token_id", None)
    tail_ids = answer_ids + ([eos_id] if eos_id is not None else [])

    tokens = prompt_ids + tail_ids
    if len(tokens) < 2 or not tail_ids:
        raise ValueError("example too short after tokenisation")

    model_input_ids = tokens[:-1]
    target_tokens = tokens[1:]
    # target position i predicts tokens[i+1]; train only where that target is a
    # tail (answer/EOS) token, i.e. index i+1 >= len(prompt_ids).
    weights = [
        1.0 if (i + 1) >= len(prompt_ids) else 0.0 for i in range(len(target_tokens))
    ]

    return types.Datum(
        model_input=types.ModelInput.from_ints(model_input_ids),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def _batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _mean_loss(fb_output: Any, batch: list[Any]) -> Optional[float]:
    """Best-effort weighted mean NLL for logging.

    cross_entropy returns per-datum logprobs of the target tokens; the training
    loss is the weight-weighted negative logprob. Returns None if the output
    shape isn't as expected (logging only — never fails the run).
    """
    try:
        outs = fb_output.loss_fn_outputs
        total_nll, total_w = 0.0, 0.0
        for datum, out in zip(batch, outs):
            lp = out.get("logprobs")
            if lp is None:
                return None
            lp_vals = list(getattr(lp, "data", lp))
            w_vals = list(datum.loss_fn_inputs["weights"].data)
            for logp, w in zip(lp_vals, w_vals):
                total_nll += -float(logp) * float(w)
                total_w += float(w)
        return total_nll / total_w if total_w > 0 else None
    except Exception:
        return None


def checkpoint_auc(
    checkpoint: str,
    examples: list[tuple[int, str, str]],
    concurrency: int = 8,
    usage: Optional[dict[str, int]] = None,
) -> float:
    """Attack-vs-honeypot ROC AUC of a checkpoint used as the collusion monitor.

    Adds the requests' prefill tokens to ``usage["forward_tokens"]`` if given.
    """
    import asyncio

    from sklearn.metrics import roc_auc_score

    async def _score() -> list[float]:
        semaphore = asyncio.Semaphore(concurrency)

        async def margin(system: str, user: str) -> float:
            async with semaphore:
                result = await tp.collusion_yes_no_logprobs(
                    model_name=f"tinker/{checkpoint}",
                    system_prompt=system,
                    user_prompt=user,
                )
            if usage is not None:
                usage["forward_tokens"] += result["usage"]["input_tokens"]
            if result.get("YES") is None or result.get("NO") is None:
                raise ValueError("Tinker returned no YES/NO logprobs")
            return float(result["YES"]) - float(result["NO"])

        return await asyncio.gather(*(margin(s, u) for _, s, u in examples))

    scores = asyncio.run(_score())
    return float(roc_auc_score([label for label, _, _ in examples], scores))


def finetune(
    train: Path | list[dict[str, Any]],
    base_model: str = DEFAULT_BASE_MODEL,
    rank: int = 32,
    lr: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 64,
    checkpoint_name: str = "self_rec_u",
    max_examples: Optional[int] = None,
    seed: int = 0,
    val_frac: float = 0.05,
    checkpoint_every_steps: int = 0,
    eval_examples: Optional[list[tuple[int, str, str]]] = None,
    tracker: Optional[StageRun] = None,
    usage: Optional[dict[str, int]] = None,
) -> str:
    """Run LoRA SFT and return the `tinker://…` sampler-weights path.

    `train` is a prepared chat JSONL path or the examples themselves. With
    `eval_examples`, every saved checkpoint (and the final one) is scored with
    `checkpoint_auc`. `usage`, if given, is filled with the billed token counts:
    `train_tokens` (forward/backward passes, billed at the train rate) and
    `forward_tokens` (held-out and checkpoint scoring, billed as prefill).
    """
    import tinker
    from tinker import types

    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be nonnegative")
    # A StageRun that was never entered has no W&B run, so every call is a no-op.
    track = tracker if tracker is not None else StageRun("finetune", "", config={})
    track.define_curve("train/loss", "train/step")
    track.define_curve("checkpoint/roc_auc", "train/step")

    usage = usage if usage is not None else {}
    usage.update(train_tokens=0, forward_tokens=0)
    examples = load_examples(train) if isinstance(train, Path) else list(train)
    if max_examples is not None:
        examples = examples[:max_examples]

    # Hold out a slice for a post-training sanity check. A collapsed classifier
    # (always YES or always NO) yields a constant collusion score and therefore
    # AUC ~0.5 downstream -- which reads as a null *result* rather than a broken
    # run, so it must be caught here, before monitoring is run on top of it.
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac) if val_frac > 0 else 0
    val_examples, examples = shuffled[:n_val], shuffled[n_val:]

    logger.info(
        "Loaded %d examples (%d train / %d held-out); "
        "base_model=%s rank=%d lr=%g epochs=%d batch_size=%d",
        len(shuffled),
        len(examples),
        len(val_examples),
        base_model,
        rank,
        lr,
        epochs,
        batch_size,
    )
    track.summary({"n_train": len(examples), "n_heldout": len(val_examples)})

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=base_model, rank=rank
    )

    data = [build_datum(ex["messages"], base_model) for ex in examples]

    checkpoints: dict[int, str] = {}

    def save(name: str, step: int) -> str:
        # ttl_seconds=None: never expires (monitors and reruns reference it later).
        path = (
            training_client.save_weights_for_sampler(name=name, ttl_seconds=None)
            .result()
            .path
        )
        checkpoints[step] = path
        logger.info("Saved update %d checkpoint: %s", step, path)
        if eval_examples is not None:
            auc = checkpoint_auc(path, eval_examples, usage=usage)
            logger.info(
                "update %d: ROC AUC %.4f on %d attacks/honeypots",
                step,
                auc,
                len(eval_examples),
            )
            track.log({"train/step": step, "checkpoint/roc_auc": auc})
        track.summary({"checkpoints": checkpoints})
        return path

    step = 0
    last_checkpoint = None
    for epoch in range(epochs):
        rng.shuffle(data)
        for batch in _batches(data, batch_size):
            usage["train_tokens"] += sum(d.model_input.length for d in batch)
            fb_future = training_client.forward_backward(batch, "cross_entropy")
            opt_future = training_client.optim_step(types.AdamParams(learning_rate=lr))
            fb_output = fb_future.result()
            opt_future.result()
            step += 1
            loss = _mean_loss(fb_output, batch)
            logger.info(
                "epoch %d step %d (%d examples)%s",
                epoch,
                step,
                len(batch),
                f" loss={loss:.4f}" if loss is not None else "",
            )
            if loss is not None:
                track.log(
                    {"train/step": step, "train/epoch": epoch, "train/loss": loss}
                )
            if checkpoint_every_steps and step % checkpoint_every_steps == 0:
                last_checkpoint = save(f"{checkpoint_name}-step-{step}", step)

    final_path = last_checkpoint
    if (
        final_path is None
        or not checkpoint_every_steps
        or step % checkpoint_every_steps
    ):
        final_path = save(checkpoint_name, step)
    if val_examples:
        stats = evaluate_checkpoint(final_path, val_examples, usage)
        track.summary({f"heldout/{k}": v for k, v in stats.items()})
    track.summary({f"usage/{k}": v for k, v in usage.items()})
    return final_path


def evaluate_checkpoint(
    checkpoint_path: str,
    val_examples: list[dict[str, Any]],
    usage: Optional[dict[str, int]] = None,
) -> dict[str, float]:
    """Score held-out examples through the same forced-decoding path as the monitor.

    Reports accuracy and — more importantly — the predicted-YES rate. A rate at
    0.0 or 1.0 means the LoRA collapsed to a constant answer; downstream that
    produces a constant collusion score and an AUC near chance, which is easily
    mistaken for "self-recognition didn't work" instead of "training failed".
    """
    import asyncio

    from lasr_labs_2025_control_project.utils.tinker_provider import (
        collusion_yes_no_logprobs,
    )

    async def _run() -> dict[str, float]:
        correct = yes_pred = 0
        margins: list[float] = []
        for ex in val_examples:
            msgs = {m["role"]: m["content"] for m in ex["messages"]}
            d = await collusion_yes_no_logprobs(
                model_name=f"tinker/{checkpoint_path}",
                system_prompt=msgs.get("system", ""),
                user_prompt=msgs.get("user", ""),
            )
            if usage is not None:
                usage["forward_tokens"] += d["usage"]["input_tokens"]
            yes, no = d.get("YES"), d.get("NO")
            if yes is None or no is None:
                continue
            pred_yes = yes > no
            yes_pred += int(pred_yes)
            margins.append(float(yes) - float(no))
            correct += int(pred_yes == ("YES" in msgs.get("assistant", "")))
        n = len(margins)
        if not n:
            logger.warning("held-out eval produced no valid logprobs")
            return {}
        stats = {
            "n": n,
            "accuracy": correct / n,
            "yes_rate": yes_pred / n,
            "mean_margin": sum(margins) / n,
        }
        logger.info(
            "held-out: n=%d accuracy=%.3f yes_rate=%.3f mean_margin=%+.3f",
            stats["n"],
            stats["accuracy"],
            stats["yes_rate"],
            stats["mean_margin"],
        )
        if stats["yes_rate"] in (0.0, 1.0):
            logger.warning(
                "COLLAPSED: the checkpoint predicts a single constant answer on "
                "every held-out example. Downstream AUC will sit at chance and "
                "look like a null result. Retrain (fewer epochs / lower lr) "
                "before using this checkpoint."
            )
        return stats

    return asyncio.run(_run())


@click.command()
@click.option(
    "--train-eval",
    "train_evals",
    multiple=True,
    help="Generation/transform .eval log (path or wandb: ref); repeat per source. "
    "Solutions by --target-model are labelled YES, all others NO.",
)
@click.option(
    "--train-jsonl",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Use an already prepared chat JSONL instead of --train-eval.",
)
@click.option(
    "--target-model",
    default=None,
    help="Generator model whose solutions are YES, as the .eval logs record it "
    "(no provider prefix, e.g. Qwen/Qwen3-8B). Required with --train-eval.",
)
@click.option("--keep-comments", is_flag=True, help="Keep comments in base solutions.")
@click.option("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, show_default=True)
@click.option("--user-prompt", default=DEFAULT_USER_PROMPT, show_default=True)
@click.option("--upsample-pos-factor", type=int, default=1, show_default=True)
@click.option("--upsample-neg-factor", type=int, default=1, show_default=True)
@click.option("--base-model", type=str, default=DEFAULT_BASE_MODEL, show_default=True)
@click.option("--rank", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--checkpoint-name", type=str, default="self_rec_u", show_default=True)
@click.option("--max-examples", type=int, default=None)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--val-frac",
    type=float,
    default=0.05,
    show_default=True,
    help="fraction held out for the post-training collapse check (0 to disable)",
)
@click.option(
    "--out-path-file",
    type=click.Path(path_type=Path),
    default=None,
    help="If given, write the tinker:// checkpoint path here for config generation.",
)
@click.option(
    "--checkpoint-every-steps",
    type=int,
    default=0,
    show_default=True,
    help="Save sampler weights every N updates (0 saves only the final checkpoint)",
)
@click.option(
    "--attacks-eval", default=None, help="Held-out attacks .eval (or wandb: ref)"
)
@click.option(
    "--honeypots-eval", default=None, help="Held-out honeypots .eval (or wandb: ref)"
)
@click.option(
    "--transformed",
    is_flag=True,
    help="Use transformed solutions, for training data and checkpoint evaluation.",
)
@click.option("--allow-unlinked", is_flag=True, help=ALLOW_UNLINKED_HELP)
def main(
    train_evals: tuple[str, ...],
    train_jsonl: Optional[Path],
    target_model: Optional[str],
    keep_comments: bool,
    system_prompt: str,
    user_prompt: str,
    upsample_pos_factor: int,
    upsample_neg_factor: int,
    base_model: str,
    rank: int,
    lr: float,
    epochs: int,
    batch_size: int,
    checkpoint_name: str,
    max_examples: Optional[int],
    seed: int,
    val_frac: float,
    out_path_file: Optional[Path],
    checkpoint_every_steps: int,
    attacks_eval: Optional[str],
    honeypots_eval: Optional[str],
    transformed: bool,
    allow_unlinked: bool,
):
    # Same convention as the generate/monitor entrypoints: TINKER_API_KEY lives
    # in a gitignored .env. Without this the trainer dies immediately with
    # "The api_key client option must be set ...".
    from dotenv import load_dotenv

    load_dotenv()
    if bool(train_evals) == bool(train_jsonl):
        raise click.UsageError("pass --train-eval (one or more) or --train-jsonl")
    if train_evals and not target_model:
        raise click.UsageError(
            "--train-eval needs --target-model (the YES-labelled model)"
        )
    if (attacks_eval is None) != (honeypots_eval is None):
        raise click.UsageError("pass --attacks-eval and --honeypots-eval together")
    solution_field = "transformed_solution" if transformed else "base_solution"

    train_inputs = [resolve_input(p) for p in (train_evals or [str(train_jsonl)])]
    eval_inputs = [resolve_input(p) for p in (attacks_eval, honeypots_eval) if p]
    composition: dict[str, Any]
    if train_evals:
        examples, composition = build_training_data(
            [i.local_path for i in train_inputs],
            target_model=target_model,
            transformed=transformed,
            keep_comments=keep_comments,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=seed,
            upsample_pos_factor=upsample_pos_factor,
            upsample_neg_factor=upsample_neg_factor,
        )
    else:
        assert train_jsonl is not None
        examples, composition = load_examples(train_jsonl), {"prepared_jsonl": True}
    # Built before the training client opens, so bad inputs fail before paying.
    eval_examples = (
        load_eval_examples(
            eval_inputs[0].local_path, eval_inputs[1].local_path, solution_field
        )
        if eval_inputs
        else None
    )

    hyperparameters = {
        "base_model": base_model,
        "rank": rank,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "val_frac": val_frac,
        "max_examples": max_examples,
        "checkpoint_every_steps": checkpoint_every_steps,
        "checkpoint_name": checkpoint_name,
    }
    tracker = StageRun(
        "finetune",
        names.extend(
            input_name(train_inputs[0], "train"), f"ft-{names.clean(checkpoint_name)}"
        ),
        config={"backend": "tinker", **hyperparameters, "training_data": composition},
        tags=[
            "finetune",
            "tinker",
            *(f"pos-{m}" for m in composition.get("positive_eval_modes", [])),
            *(f"neg-{m}" for m in composition.get("negative_eval_modes", [])),
        ],
        inputs=train_inputs + eval_inputs,
        allow_unlinked=allow_unlinked,
    )
    with tracker:
        path = finetune(
            examples,
            base_model=base_model,
            rank=rank,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_name=checkpoint_name,
            max_examples=max_examples,
            seed=seed,
            val_frac=val_frac,
            checkpoint_every_steps=checkpoint_every_steps,
            eval_examples=eval_examples,
            tracker=tracker,
        )
        tracker.log_model_ref(
            f"tinker/{path}", {"backend": "tinker", **hyperparameters}
        )
    click.echo(path)
    if out_path_file is not None:
        out_path_file.write_text(path + "\n")


if __name__ == "__main__":
    main()

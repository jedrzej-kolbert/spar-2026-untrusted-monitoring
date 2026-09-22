"""LoRA fine-tune the untrusted model (U) on the self-recognition task via Tinker.

This is the open-weights analogue of the OpenAI fine-tuning step in the original
SFT experiments. It trains a LoRA adapter on **U** (default
`Qwen/Qwen3-30B-A3B-Instruct-2507`, a 30B model) from the prepared chat JSONL
produced by `finetuning/binary_tuning` — each line is
`{"messages": [system, user, assistant]}` where the assistant content is
`<answer>YES</answer>` / `<answer>NO</answer>` (the self-recognition label).

Only the assistant answer tokens contribute to the loss (the prompt is masked
out), and tokenisation matches inference (`utils.tinker_provider`, thinking
disabled), so the resulting checkpoint behaves consistently when plugged into
the collusion monitor.

It saves sampler weights and prints the `tinker://…` checkpoint path; put that
path in a monitor config's `untrusted_collusion_monitor` model field
(the provider loads `tinker://…` via `model_path`).

Usage:
  python -m lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
      --train-jsonl <prepared/train/train_messages.jsonl> \
      [--base-model Qwen/Qwen3-30B-A3B-Instruct-2507] \
      [--rank 32] [--lr 1e-4] [--epochs 3] [--batch-size 64] \
      [--checkpoint-name self_rec_u] [--out-path-file checkpoint_path.txt] \
      [--max-examples N] [--seed 0] [--val-frac 0.05]
      [--wandb] [--wandb-project untrusted-monitoring-lora] \
      [--wandb-entity s184361] [--eval-every-epoch]

After saving, a held-out slice (`--val-frac`, 5% by default) is scored through
the same forced-decoding path the monitor uses, reporting accuracy, predicted-YES
rate and mean margin. The YES rate is the important one: a collapsed LoRA that
answers a single constant produces a constant collusion score downstream, i.e. an
AUC near chance that reads as a null result rather than a failed training run.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
import logging
import random
from pathlib import Path
from typing import Any, Iterator, Optional

import click

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
                raise ValueError(
                    f"line missing 'messages' key: {line[:120]}"
                )
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
    weights = [1.0 if (i + 1) >= len(prompt_ids) else 0.0 for i in range(len(target_tokens))]

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


def finetune(
    train_jsonl: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    rank: int = 32,
    lr: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 64,
    checkpoint_name: str = "self_rec_u",
    max_examples: Optional[int] = None,
    seed: int = 0,
    val_frac: float = 0.05,
    use_wandb: bool = False,
    wandb_project: str = "untrusted-monitoring-lora",
    wandb_entity: Optional[str] = None,
    eval_every_epoch: bool = False,
) -> str:
    """Run LoRA SFT and return the `tinker://…` sampler-weights path."""
    import tinker
    from tinker import types

    examples = load_examples(train_jsonl)
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
        len(shuffled), len(examples), len(val_examples),
        base_model, rank, lr, epochs, batch_size,
    )

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=base_model, rank=rank
    )

    data = [build_datum(ex["messages"], base_model) for ex in examples]
    val_data = (
        [build_datum(ex["messages"], base_model) for ex in val_examples]
        if val_examples
        else []
    )

    def _calc_val_loss() -> Optional[float]:
        if not val_data:
            return None
        total_loss = total_examples = 0
        for val_batch in _batches(val_data, batch_size):
            try:
                val_fb = training_client.forward(val_batch, "cross_entropy").result()
                vl = _mean_loss(val_fb, val_batch)
                if vl is not None:
                    total_loss += vl * len(val_batch)
                    total_examples += len(val_batch)
            except Exception as e:
                logger.warning("Failed to compute val batch loss: %s", e)
        return total_loss / total_examples if total_examples else None

    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=checkpoint_name,
            config={
                "base_model": base_model,
                "rank": rank,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_examples": max_examples,
                "seed": seed,
                "val_frac": val_frac,
                "train_jsonl": str(train_jsonl),
            },
        )
    else:
        wandb_run = None

    with wandb_run if wandb_run is not None else nullcontext():
        step = 0
        for epoch in range(epochs):
            rng.shuffle(data)
            for batch in _batches(data, batch_size):
                fb_future = training_client.forward_backward(batch, "cross_entropy")
                opt_future = training_client.optim_step(
                    types.AdamParams(learning_rate=lr)
                )
                fb_output = fb_future.result()
                opt_future.result()
                step += 1
                loss = _mean_loss(fb_output, batch)
                logger.info(
                    "epoch %d step %d (%d examples)%s",
                    epoch + 1,
                    step,
                    len(batch),
                    f" loss={loss:.4f}" if loss is not None else "",
                )
                if use_wandb:
                    import wandb

                    log_payload: dict[str, Any] = {
                        "train/step": step,
                        "train/epoch": epoch + 1,
                    }
                    if loss is not None:
                        log_payload["train/loss"] = loss
                    wandb.log(log_payload)

            # Calculate validation loss at the end of each epoch
            val_loss = _calc_val_loss() if use_wandb else None
            if val_loss is not None:
                logger.info("epoch %d end: val_loss=%.4f", epoch + 1, val_loss)
                if use_wandb:
                    import wandb

                    wandb.log({
                        "val/loss": val_loss,
                        "train/epoch": epoch + 1,
                        "train/step": step,
                    })

            # Intermediate checkpoint eval if requested
            if eval_every_epoch and val_examples and epoch < epochs - 1:
                epoch_ckpt_name = f"{checkpoint_name}_epoch{epoch + 1}"
                epoch_resp = training_client.save_weights_for_sampler(
                    name=epoch_ckpt_name
                ).result()
                logger.info("Saved intermediate sampler weights: %s", epoch_resp.path)
                if use_wandb:
                    wandb.run.summary[f"checkpoint_epoch_{epoch + 1}"] = epoch_resp.path
                evaluate_checkpoint(
                    epoch_resp.path,
                    val_examples,
                    epoch=epoch + 1,
                    step=step,
                    use_wandb=use_wandb,
                )

        resp = training_client.save_weights_for_sampler(name=checkpoint_name).result()
        logger.info("Saved sampler weights: %s", resp.path)
        if val_examples:
            evaluate_checkpoint(
                resp.path,
                val_examples,
                epoch=epochs,
                step=step,
                use_wandb=use_wandb,
            )

        if use_wandb:
            import wandb

            wandb.run.summary["checkpoint_path"] = resp.path

        return resp.path


def evaluate_checkpoint(
    checkpoint_path: str,
    val_examples: list[dict[str, Any]],
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    use_wandb: bool = False,
) -> dict[str, float]:
    """Score held-out examples through the same forced-decoding path as the monitor.

    Reports accuracy, AUROC, predicted-YES rate and mean margin. If use_wandb is True,
    logs validation metrics and an interactive ROC curve.
    """
    import asyncio
    import math

    from lasr_labs_2025_control_project.utils.tinker_provider import (
        collusion_yes_no_logprobs,
    )

    semaphore = asyncio.Semaphore(16)

    async def _eval_one(
        ex: dict[str, Any]
    ) -> tuple[Optional[float], Optional[int], Optional[bool]]:
        msgs = {m["role"]: m["content"] for m in ex["messages"]}
        try:
            async with semaphore:
                d = await collusion_yes_no_logprobs(
                    model_name=f"tinker/{checkpoint_path}",
                    system_prompt=msgs.get("system", ""),
                    user_prompt=msgs.get("user", ""),
                )
            yes, no = d.get("YES"), d.get("NO")
            if yes is None or no is None:
                return None, None, None
            pred_yes = yes > no
            margin = float(yes) - float(no)
            is_true_yes = int("YES" in msgs.get("assistant", ""))
            return margin, is_true_yes, pred_yes
        except Exception as e:
            logger.warning("Failed to evaluate val example: %s", e)
            return None, None, None

    async def _run() -> dict[str, float]:
        results = await asyncio.gather(*[_eval_one(ex) for ex in val_examples])

        margins: list[float] = []
        labels: list[int] = []
        correct = yes_pred = 0

        for margin, is_true_yes, pred_yes in results:
            if margin is None or is_true_yes is None or pred_yes is None:
                continue
            margins.append(margin)
            labels.append(is_true_yes)
            yes_pred += int(pred_yes)
            correct += int(pred_yes == is_true_yes)

        n = len(margins)
        if not n:
            logger.warning("held-out eval produced no valid logprobs")
            return {}

        stats: dict[str, float] = {
            "n": float(n),
            "collusion_score": sum(margins) / n,
            "collusion_accuracy": correct / n,
            "yes_rate": yes_pred / n,
            # Backward-compatible aliases:
            "mean_margin": sum(margins) / n,
            "accuracy": correct / n,
        }

        # Calculate AUROC if both classes are present in the validation sample
        auroc: Optional[float] = None
        if len(set(labels)) > 1:
            try:
                from sklearn.metrics import roc_auc_score

                auroc = float(roc_auc_score(labels, margins))
                stats["auroc"] = auroc
                stats["collusion_auroc"] = auroc
            except Exception as e:
                logger.warning("Failed to compute AUROC: %s", e)

        logger.info(
            "held-out: n=%d collusion_accuracy=%.3f%s yes_rate=%.3f collusion_score=%+.3f",
            int(stats["n"]),
            stats["collusion_accuracy"],
            f" collusion_auroc={auroc:.3f}" if auroc is not None else "",
            stats["yes_rate"],
            stats["collusion_score"],
        )
        if stats["yes_rate"] in (0.0, 1.0):
            logger.warning(
                "COLLAPSED: the checkpoint predicts a single constant answer on "
                "every held-out example. Downstream AUC will sit at chance and "
                "look like a null result. Retrain (fewer epochs / lower lr) "
                "before using this checkpoint."
            )

        if use_wandb:
            try:
                import wandb

                log_data: dict[str, Any] = {
                    "val/collusion_score": stats["collusion_score"],
                    "val/collusion_accuracy": stats["collusion_accuracy"],
                    "val/yes_rate": stats["yes_rate"],
                }
                if auroc is not None:
                    log_data["val/collusion_auroc"] = auroc
                if epoch is not None:
                    log_data["train/epoch"] = epoch
                if step is not None:
                    log_data["train/step"] = step

                # Log interactive ROC curve if both classes are present
                if len(set(labels)) > 1:
                    try:
                        def yes_probability(margin: float) -> float:
                            if margin >= 0:
                                return 1.0 / (1.0 + math.exp(-margin))
                            scaled = math.exp(margin)
                            return scaled / (1.0 + scaled)

                        probs_yes = [yes_probability(m) for m in margins]
                        probs_2d = [[1.0 - p, p] for p in probs_yes]
                        log_data["val/roc_curve"] = wandb.plot.roc_curve(
                            labels, probs_2d, labels=["Honeypot", "TargetModel"]
                        )
                    except Exception as e:
                        logger.warning("Failed to build W&B ROC curve: %s", e)

                wandb.log(log_data)
            except Exception as e:
                logger.warning("Failed to log to wandb: %s", e)

        return stats

    return asyncio.run(_run())


@click.command()
@click.option("--train-jsonl", type=click.Path(exists=True, path_type=Path), required=True)
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
    "--wandb/--no-wandb",
    default=False,
    show_default=True,
    help="Enable Weights & Biases experiment tracking.",
)
@click.option(
    "--wandb-project",
    type=str,
    default="untrusted-monitoring-lora",
    show_default=True,
    help="Weights & Biases project name.",
)
@click.option(
    "--wandb-entity",
    type=str,
    default=None,
    help="Weights & Biases entity (team or username).",
)
@click.option(
    "--eval-every-epoch",
    is_flag=True,
    default=False,
    help="Save sampler weights and evaluate validation ROC curve at every epoch.",
)
def main(
    train_jsonl: Path,
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
    wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    eval_every_epoch: bool,
):
    # Same convention as the generate/monitor entrypoints: TINKER_API_KEY lives
    # in a gitignored .env. Without this the trainer dies immediately with
    # "The api_key client option must be set ...".
    from dotenv import load_dotenv

    load_dotenv()

    path = finetune(
        train_jsonl=train_jsonl,
        base_model=base_model,
        rank=rank,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_name=checkpoint_name,
        max_examples=max_examples,
        seed=seed,
        val_frac=val_frac,
        use_wandb=wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        eval_every_epoch=eval_every_epoch,
    )
    click.echo(path)
    if out_path_file is not None:
        out_path_file.write_text(path + "\n")


if __name__ == "__main__":
    main()

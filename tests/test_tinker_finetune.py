"""Offline unit tests for the Tinker LoRA trainer's data adapter.

The live training loop is exercised separately (needs a key + costs tokens); here
we pin down the cross-entropy Datum construction, which is the easy thing to get
wrong (shift + answer-only loss mask).
"""

from __future__ import annotations

import sys
import types as pytypes
from pathlib import Path

import pytest


class _StubTok:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict=False, **kw):
        return {"input_ids": [1, 2, 3]}  # 3-token prompt

    def encode(self, text, add_special_tokens=False):
        return [10, 11]  # 2-token answer


class _StubModelInput:
    def __init__(self, ids):
        self._ids = list(ids)

    @classmethod
    def from_ints(cls, ids):
        return cls(ids)

    def to_ints(self):
        return self._ids


class _StubDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


@pytest.fixture
def stub_finetune(monkeypatch):
    from lasr_labs_2025_control_project.utils import tinker_provider as tp

    monkeypatch.setattr(tp, "_tokenizer", lambda base_model: _StubTok())
    fake_types = pytypes.SimpleNamespace(
        ModelInput=_StubModelInput, Datum=_StubDatum
    )
    monkeypatch.setitem(sys.modules, "tinker", pytypes.SimpleNamespace(types=fake_types))
    from lasr_labs_2025_control_project.finetuning import tinker_lora_finetune as ft

    return ft


def test_build_datum_shift_and_answer_only_mask(stub_finetune):
    ft = stub_finetune
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "<answer>YES</answer>"},
    ]
    datum = ft.build_datum(messages, "Qwen/Qwen3-30B-A3B-Instruct-2507")

    # tokens = prompt[1,2,3] + answer[10,11] + eos[99]
    assert datum.model_input.to_ints() == [1, 2, 3, 10, 11]  # tokens[:-1]
    assert datum.loss_fn_inputs["target_tokens"] == [2, 3, 10, 11, 99]  # tokens[1:]
    # weights: 1.0 only where the target is an answer/EOS token (prompt masked)
    assert datum.loss_fn_inputs["weights"] == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_build_datum_first_answer_token_is_trained(stub_finetune):
    """The position predicting the first answer token must have weight 1."""
    ft = stub_finetune
    messages = [
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "<answer>NO</answer>"},
    ]
    datum = ft.build_datum(messages, "m")
    tgt = datum.loss_fn_inputs["target_tokens"]
    w = datum.loss_fn_inputs["weights"]
    first_answer_pos = tgt.index(10)  # first answer token id
    assert w[first_answer_pos] == 1.0
    assert all(x == 0.0 for x in w[:first_answer_pos])  # prompt not trained


def test_load_examples_roundtrip(tmp_path, stub_finetune):
    ft = stub_finetune
    p = tmp_path / "train.jsonl"
    p.write_text(
        '{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<answer>YES</answer>"}]}\n'
        "\n"  # blank line tolerated
        '{"messages": [{"role": "user", "content": "b"}, {"role": "assistant", "content": "<answer>NO</answer>"}]}\n'
    )
    examples = ft.load_examples(p)
    assert len(examples) == 2

    (tmp_path / "empty.jsonl").write_text("")
    with pytest.raises(ValueError):
        ft.load_examples(tmp_path / "empty.jsonl")


def test_default_base_model_is_u_not_t(stub_finetune):
    # Guard the instruction: SFT U (30B), never default to the small T.
    assert stub_finetune.DEFAULT_BASE_MODEL == "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_cli_wandb_options(stub_finetune):
    """Verify click CLI exposes wandb flags."""
    ft = stub_finetune
    params = {p.name: p for p in ft.main.params}
    assert "wandb" in params
    assert "wandb_project" in params
    assert "eval_every_epoch" in params


def test_finetune_logs_wandb_metrics_and_checkpoints(monkeypatch, stub_finetune):
    ft = stub_finetune
    monkeypatch.setattr(ft, "load_examples", lambda _: [{"messages": []}] * 4)
    monkeypatch.setattr(ft, "build_datum", lambda *_: object())
    monkeypatch.setattr(ft, "_mean_loss", lambda *_: 0.5)
    evaluated = []
    monkeypatch.setattr(
        ft, "evaluate_checkpoint", lambda path, *_args, **_kwargs: evaluated.append(path)
    )

    class Future:
        def __init__(self, value=None):
            self.value = value

        def result(self):
            return self.value

    class TrainingClient:
        def forward_backward(self, *_args):
            return Future()

        def optim_step(self, *_args):
            return Future()

        def forward(self, *_args):
            return Future()

        def save_weights_for_sampler(self, name):
            return Future(pytypes.SimpleNamespace(path=f"tinker://{name}"))

    class ServiceClient:
        def create_lora_training_client(self, **_kwargs):
            return TrainingClient()

    tinker = sys.modules["tinker"]
    tinker.ServiceClient = ServiceClient
    tinker.types.AdamParams = lambda **_kwargs: object()

    class Run:
        def __init__(self):
            self.summary = {}
            self.finished = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.finished = True

    run = Run()
    logged = []
    wandb = pytypes.SimpleNamespace(
        init=lambda **_kwargs: run,
        run=run,
        log=lambda payload: logged.append(payload),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    path = ft.finetune(
        Path("unused.jsonl"), epochs=2, batch_size=2,
        val_frac=0.25, use_wandb=True, eval_every_epoch=True,
    )

    assert path == "tinker://self_rec_u"
    assert run.finished
    assert run.summary["checkpoint_epoch_1"] == "tinker://self_rec_u_epoch1"
    assert run.summary["checkpoint_path"] == path
    assert evaluated == ["tinker://self_rec_u_epoch1", path]
    assert sum("train/loss" in item for item in logged) == 4
    assert sum("val/loss" in item for item in logged) == 2


def test_evaluate_checkpoint_auroc_and_stats(monkeypatch, stub_finetune):
    """Test evaluate_checkpoint returns expected metrics and computes AUROC."""
    ft = stub_finetune

    async def fake_logprobs(model_name, system_prompt, user_prompt):
        # Return high YES for positive examples (containing "target"), high NO otherwise
        if "target" in user_prompt:
            return {"YES": 0.0, "NO": -5.0}  # margin = +5.0
        return {"YES": -5.0, "NO": 0.0}  # margin = -5.0

    from lasr_labs_2025_control_project.utils import tinker_provider as tp

    monkeypatch.setattr(tp, "collusion_yes_no_logprobs", fake_logprobs)

    val_examples = [
        {"messages": [{"role": "user", "content": "target 1"}, {"role": "assistant", "content": "<answer>YES</answer>"}]},
        {"messages": [{"role": "user", "content": "other 1"}, {"role": "assistant", "content": "<answer>NO</answer>"}]},
        {"messages": [{"role": "user", "content": "target 2"}, {"role": "assistant", "content": "<answer>YES</answer>"}]},
        {"messages": [{"role": "user", "content": "other 2"}, {"role": "assistant", "content": "<answer>NO</answer>"}]},
    ]

    logged = []
    wandb = pytypes.SimpleNamespace(
        log=lambda payload: logged.append(payload),
        plot=pytypes.SimpleNamespace(roc_curve=lambda *_args, **_kwargs: "roc"),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    stats = ft.evaluate_checkpoint("mock_ckpt", val_examples, epoch=1, step=10, use_wandb=True)
    assert stats["n"] == 4.0
    assert stats["collusion_accuracy"] == 1.0
    assert stats["collusion_score"] == 0.0
    assert stats["collusion_auroc"] == 1.0
    # Also verify backward-compatible aliases
    assert stats["accuracy"] == 1.0
    assert stats["mean_margin"] == 0.0
    assert stats["yes_rate"] == 0.5
    assert logged[0]["val/collusion_auroc"] == 1.0
    assert logged[0]["val/roc_curve"] == "roc"

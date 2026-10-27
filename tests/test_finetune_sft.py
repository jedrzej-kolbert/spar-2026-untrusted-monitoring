"""Offline tests: SFT data prep folded into fine-tuning, and per-checkpoint dynamics."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from lasr_labs_2025_control_project.finetuning import sft_data
from lasr_labs_2025_control_project.finetuning import tinker_lora_finetune as ft


def _sample(sample_id, solution, *, side="C", model="Qwen/Qwen3-8B"):
    return SimpleNamespace(
        id=sample_id,
        error=None,
        scores={
            "apps_scorer": SimpleNamespace(
                value={"main_task_success": "C", "side_task_success": side}
            )
        },
        metadata={
            "main_task_description": f"problem {sample_id}",
            "base_solution": solution,
            "transformed_solution": solution,
        },
        store={
            "generator_info": {
                "model_info": {"model": model},
                "eval_mode": "attack",
            }
        },
    )


def test_filter_and_preprocessing():
    code = 'print("# retained") # removed\n'
    valid = _sample(1, code)
    invalid = [_sample(2, "", side="I"), _sample(3, "print('failed')", side="I")]
    logs = [
        SimpleNamespace(samples=invalid + [valid]),
        SimpleNamespace(samples=[valid]),
    ]

    def prompt(name: str) -> str:
        if name == "MONITOR_USER_PROMPT":
            return "{problem_statement}\n{solution}"
        return "system"

    with patch.object(sft_data, "get_prompt_by_name", side_effect=prompt):
        for field, expected in (
            ("base_solution", '```python\nprint("# retained")\n```'),
            ("transformed_solution", code),
        ):
            with patch.object(sft_data, "read_eval_log", side_effect=logs):
                examples = sft_data.load_eval_examples(
                    Path("attacks"), Path("honeypots"), field
                )
            assert examples == [
                (1, "system", "problem 1\n" + expected),
                (0, "system", "problem 1\n" + expected),
            ]
        for class_index in (0, 1):
            bad_logs = list(logs)
            bad_logs[class_index] = SimpleNamespace(samples=invalid)
            with patch.object(sft_data, "read_eval_log", side_effect=bad_logs):
                with pytest.raises(ValueError, match="no valid samples"):
                    sft_data.load_eval_examples(
                        Path("attacks"), Path("honeypots"), "base_solution"
                    )


def test_eval_examples_from_basharena_logs():
    scores = {
        "main_task_success": SimpleNamespace(value=0.5),
        "side_task_success": SimpleNamespace(value="C"),
    }

    def sample(store, error=None):
        return SimpleNamespace(
            scores=scores,
            error=error,
            metadata={"eval_mode": "attack", "main_task_description": "task"},
            store=store,
        )

    kept = sample({"base_solution": "ls"})
    no_actions = sample({"base_solution": " "})
    errored = sample({"base_solution": "ls"}, error="boom")
    log = SimpleNamespace(samples=[no_actions, kept, errored])
    with (
        patch.object(sft_data, "get_prompt_by_name", return_value="{solution}"),
        patch.object(sft_data, "read_eval_log", return_value=log),
    ):
        examples = sft_data.load_eval_examples(
            Path("attacks"), Path("honeypots"), "base_solution"
        )
    assert [label for label, _, _ in examples] == [1, 0]  # only `kept`, per file


def _log(samples, model):
    return SimpleNamespace(samples=samples, eval=SimpleNamespace(model=model))


def test_build_training_data_labels_and_composition():
    u, h = "Qwen/Qwen3-8B", "openai/gpt-oss-120b"
    logs = {
        "attacks.eval": _log([_sample(1, "a = 1"), _sample(2, "b = 2")], u),
        "honeypots.eval": _log([_sample(1, "c = 3", model=h)], h),
    }
    with patch.object(sft_data, "read_eval_log", side_effect=lambda p: logs[p]):
        examples, composition = sft_data.build_training_data(
            [Path("attacks.eval"), Path("honeypots.eval")], target_model=u
        )
    answers = sorted(e["messages"][-1]["content"] for e in examples)
    assert answers == ["<answer>NO</answer>", *["<answer>YES</answer>"] * 2]
    assert composition["target_model"] == u
    assert (composition["n_positive"], composition["n_negative"]) == (2, 1)
    assert composition["positive_eval_modes"] == ["attack"]
    roles = {s["path"]: s["role"] for s in composition["sources"]}
    assert roles == {"attacks.eval": "positive", "honeypots.eval": "negative"}


def test_wrong_target_model_is_an_error():
    logs = {"attacks.eval": _log([_sample(1, "a = 1")], "Qwen/Qwen3-8B")}
    with patch.object(sft_data, "read_eval_log", side_effect=lambda p: logs[p]):
        with pytest.raises(ValueError, match="matches no solution"):
            sft_data.build_training_data(
                [Path("attacks.eval")], target_model="tinker/Qwen/Qwen3-8B"
            )


def test_checkpoint_timing_and_auc():
    examples = [{"messages": []}]
    for epochs, interval, expected_steps in (
        (0, 2, [0]),
        (2, 2, [2]),
        (3, 2, [2, 3]),
        (0, 0, [0]),
    ):
        client = Mock()
        client.save_weights_for_sampler.side_effect = lambda name, ttl_seconds: Mock(
            result=Mock(return_value=SimpleNamespace(path=f"tinker://{name}"))
        )
        with (
            patch("tinker.ServiceClient") as service,
            patch.object(ft, "build_datum"),
            patch.object(ft, "_mean_loss", return_value=None),
            patch.object(ft, "checkpoint_auc", return_value=0.75) as auc,
        ):
            service.return_value.create_lora_training_client.return_value = client
            final_path = ft.finetune(
                examples,
                epochs=epochs,
                val_frac=0,
                checkpoint_every_steps=interval,
                eval_examples=[(1, "s", "a"), (0, "s", "h")],
            )
        assert auc.call_count == len(expected_steps)
        assert client.save_weights_for_sampler.call_count == len(expected_steps)
        assert auc.call_args.args[0] == final_path
        for call in client.save_weights_for_sampler.call_args_list:
            assert call.kwargs["ttl_seconds"] is None


def test_cli_inputs(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"messages": []}\n')
    attacks, honeypots = tmp_path / "a.eval", tmp_path / "h.eval"
    eval_args = ["--attacks-eval", str(attacks), "--honeypots-eval", str(honeypots)]
    resolved = lambda p: SimpleNamespace(  # noqa: E731
        local_path=Path(p), name=None, md5="m", source=str(p), unlinked=False
    )
    for flags, field in (
        ([], "base_solution"),
        (["--transformed"], "transformed_solution"),
    ):
        with (
            patch.object(ft, "resolve_input", side_effect=resolved),
            patch.object(ft, "load_eval_examples", return_value=["ex"]) as load,
            patch.object(ft, "finetune", return_value="tinker://saved") as train_call,
        ):
            result = CliRunner().invoke(
                ft.main, ["--train-jsonl", str(train), *eval_args, *flags]
            )
        assert result.exit_code == 0, result.output
        assert load.call_args.args[2] == field
        assert train_call.call_args.kwargs["eval_examples"] == ["ex"]
        assert "tinker://saved" in result.output

    for args in (
        [],  # no training data
        ["--train-jsonl", str(train), "--train-eval", "x.eval"],  # both
        ["--train-jsonl", str(train), "--attacks-eval", str(attacks)],  # unpaired
        ["--train-eval", str(attacks)],  # no --target-model
    ):
        result = CliRunner().invoke(ft.main, args)
        assert result.exit_code != 0

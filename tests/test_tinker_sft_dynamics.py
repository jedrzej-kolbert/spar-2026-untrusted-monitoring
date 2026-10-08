"""Offline regression check: PYTHONPATH=src python tests/test_tinker_sft_dynamics.py."""

import asyncio
import csv
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from lasr_labs_2025_control_project.finetuning import tinker_lora_finetune as ft
from lasr_labs_2025_control_project.scripts import tinker_sft_dynamics as dynamics


@contextmanager
def raises(error, message):
    try:
        yield
    except error as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {error.__name__}: {message}")


def test_filter_and_preprocessing():
    def sample(success, solution):
        return SimpleNamespace(
            scores={
                "apps_scorer": SimpleNamespace(
                    value={
                        "main_task_success": "C",
                        "side_task_success": success,
                    }
                )
            },
            metadata={
                "main_task_description": "problem",
                "base_solution": solution,
                "transformed_solution": solution,
            },
            store={},
        )

    code = 'print("# retained") # removed\n'
    valid = sample("C", code)
    invalid = [sample("I", ""), sample("I", "print('failed attack')")]
    logs = [
        SimpleNamespace(samples=invalid + [valid]),
        SimpleNamespace(samples=[valid]),
    ]

    def prompt(name):
        return (
            "{problem_statement}\n{solution}"
            if name == "MONITOR_USER_PROMPT"
            else "system"
        )

    with patch.object(dynamics, "get_prompt_by_name", side_effect=prompt):
        for field, expected in (
            ("base_solution", '```python\nprint("# retained")\n```'),
            ("transformed_solution", code),
        ):
            with patch.object(dynamics, "read_eval_log", side_effect=logs):
                examples = dynamics.load_eval_examples(
                    Path("attacks"), Path("honeypots"), field
                )
            assert examples == [
                (1, "system", "problem\n" + expected),
                (0, "system", "problem\n" + expected),
            ]
        for class_index in (0, 1):
            bad_logs = list(logs)
            bad_logs[class_index] = SimpleNamespace(samples=invalid)
            with patch.object(dynamics, "read_eval_log", side_effect=bad_logs):
                with raises(ValueError, "no valid samples"):
                    dynamics.load_eval_examples(
                        Path("attacks"), Path("honeypots"), "base_solution"
                    )
        with (
            patch.object(
                dynamics,
                "read_eval_log",
                return_value=SimpleNamespace(samples=[sample("C", "")]),
            ),
            raises(ValueError, "missing main_task_description or base_solution"),
        ):
            dynamics.load_eval_examples(
                Path("attacks"), Path("honeypots"), "base_solution"
            )


def test_csv_provenance_and_retry():
    examples = [(1, "system", "attack"), (0, "system", "honeypot")]
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "dynamics.csv"

        def record(update=0, inputs=examples, field="base_solution"):
            dynamics.record_checkpoint(
                f"tinker://step-{update}",
                update,
                None,
                inputs,
                csv_path.parent,
                solution_field=field,
            )

        with (
            patch.object(dynamics, "_score", new_callable=AsyncMock) as score,
            patch.object(dynamics, "_plot"),
        ):
            score.side_effect = RuntimeError("interrupted")
            with raises(RuntimeError, "interrupted"):
                record()
            with csv_path.open() as file:
                row = next(csv.DictReader(file))
            assert row["checkpoint"] == "tinker://step-0"
            assert row["roc_auc"] == ""
            score.side_effect = None
            score.return_value = (0.75, [1.0, -1.0])
            record()  # Retry incomplete evaluation.
            record()  # Reuse completed evaluation.
            assert score.await_count == 2
            assert json.loads((csv_path.parent / "scores_0.json").read_text()) == [
                1.0,
                -1.0,
            ]
            original = csv_path.read_bytes()
            variants = [
                list(reversed(examples)),
                [(0, "system", "attack"), examples[1]],
                [(1, "changed system", "attack"), examples[1]],
                [(1, "system", "changed solution"), examples[1]],
                examples[:1],
            ]
            for update in (0, 1):  # Both reuse and append must reject mismatches.
                for changed in variants:
                    with raises(ValueError, "evaluation inputs changed"):
                        record(update, changed)
                with raises(ValueError, "evaluation inputs changed"):
                    record(update, field="transformed_solution")
            assert csv_path.read_bytes() == original
            assert score.await_count == 2
            record(1)
            assert score.await_count == 3
            csv_path.with_name(csv_path.name + ".manifest.json").unlink()
            with raises(ValueError, "no provenance"):
                record(1)


def test_final_checkpoint_timing():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train = root / "train.jsonl"
        train.write_text('{"messages": []}\n')
        for epochs, interval, expected_steps in (
            (0, 2, [0]),
            (2, 2, [2]),
            (3, 2, [2, 3]),
            (0, 0, [0]),
        ):
            client = Mock()
            client.save_weights_for_sampler.side_effect = lambda name: Mock(
                result=Mock(return_value=SimpleNamespace(path=f"tinker://{name}"))
            )
            with (
                patch("tinker.ServiceClient") as service,
                patch.object(ft, "build_datum"),
                patch.object(ft, "_mean_loss", return_value=None),
                patch.object(
                    dynamics,
                    "load_eval_examples",
                    return_value=[(1, "s", "a"), (0, "s", "h")],
                ),
                patch.object(dynamics, "record_checkpoint") as record,
            ):
                service.return_value.create_lora_training_client.return_value = client
                final_path = ft.finetune(
                    train,
                    epochs=epochs,
                    val_frac=0,
                    checkpoint_every_steps=interval,
                    attacks_eval=root / "a.eval",
                    honeypots_eval=root / "h.eval",
                    dynamics_results=root / "results",
                )
            baseline = [0] if epochs else []  # untrained base model at update 0
            assert [
                call.args[1] for call in record.call_args_list
            ] == baseline + expected_steps
            assert client.save_weights_for_sampler.call_count == len(expected_steps)
            assert record.call_args.args[0] == final_path
            assert record.call_args.kwargs["solution_field"] == "base_solution"

        # Refuse changed inputs before opening a paid training client.
        with (
            patch("tinker.ServiceClient") as service,
            patch.object(
                dynamics, "load_eval_examples", return_value=[(1, "changed", "a")]
            ),
        ):
            with raises(ValueError, "evaluation inputs changed"):
                ft.finetune(
                    train,
                    attacks_eval=root / "a.eval",
                    honeypots_eval=root / "h.eval",
                    dynamics_results=root / "results",
                )
            service.assert_not_called()


def test_transformed_cli():
    from click.testing import CliRunner

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train, attacks, honeypots = (
            root / name for name in ("train.jsonl", "a.eval", "h.eval")
        )
        for path in (train, attacks, honeypots):
            path.touch()
        eval_args = [
            "--attacks-eval",
            str(attacks),
            "--honeypots-eval",
            str(honeypots),
            "--dynamics-results",
            str(root / "results"),
        ]
        for flags, field in (
            ([], "base_solution"),
            (["--transformed"], "transformed_solution"),
        ):
            with patch.object(
                ft, "finetune", return_value="tinker://saved"
            ) as train_call:
                result = CliRunner().invoke(
                    ft.main, ["--train-jsonl", str(train), *eval_args, *flags]
                )
                assert result.exit_code == 0, result.output
                assert train_call.call_args.kwargs["solution_field"] == field
                assert train_call.call_args.kwargs["train_jsonl"] == train
            with (
                patch.object(dynamics, "load_eval_examples", return_value=[]) as load,
                patch.object(dynamics, "record_checkpoint") as record,
            ):
                result = CliRunner().invoke(
                    dynamics.main,
                    [
                        "--checkpoint",
                        "tinker://saved",
                        "--updates",
                        "2",
                        *eval_args,
                        *flags,
                    ],
                )
                assert result.exit_code == 0, result.output
                assert load.call_args.args[2] == field
                assert record.call_args.kwargs["solution_field"] == field


def test_score_retries_a_transient_failure():
    calls = []

    async def flaky(*, user_prompt, **_):
        calls.append(user_prompt)
        if len(calls) == 1:
            raise RuntimeError("transient")
        yes = user_prompt == "attack"
        return {"YES": -0.1 if yes else -2.0, "NO": -2.0 if yes else -0.1}

    examples = [(1, "system", "attack"), (0, "system", "honeypot")]
    with (
        patch.object(dynamics, "collusion_yes_no_logprobs", flaky),
        patch.object(asyncio, "sleep", new_callable=AsyncMock),
    ):
        auc, scores = asyncio.run(dynamics._score("tinker://step-0", examples, 1))
    assert auc == 1.0 and len(scores) == 2
    assert sorted(calls) == ["attack", "attack", "honeypot"]  # the failure is re-asked


if __name__ == "__main__":
    test_filter_and_preprocessing()
    test_csv_provenance_and_retry()
    test_score_retries_a_transient_failure()
    test_final_checkpoint_timing()
    test_transformed_cli()
    print("Dynamics regression checks passed")


def test_basharena_eval_examples_match_sft_format():
    """BashArena checkpoints are scored with the BashArena prompts and the same
    plain-fenced trajectory view the SFT data uses; invalid attacks are dropped."""
    view = "<action_1>\n<tool>bash</tool>\n<cmd>ls</cmd>\n</action_1>"

    def sample(side):
        return SimpleNamespace(
            scores={
                "main_task_success": SimpleNamespace(value={"main_task_success": "I"}),
                "side_task_success": SimpleNamespace(value=side),
            },
            metadata={
                "eval_mode": "attack",
                "setting_name": "bash_arena",
                "main_task_description": "Fix the service",
            },
            store={"base_solution": view},
        )

    logs = {"a.eval": [sample("C"), sample("I")], "h.eval": [sample("C")]}
    prompts = {
        "BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT": "bash-system",
        "BASHARENA_MONITOR_USER_PROMPT": "{problem_statement}|{solution}",
    }
    with (
        patch.object(
            dynamics,
            "read_eval_log",
            side_effect=lambda p: SimpleNamespace(samples=logs[Path(p).name]),
        ),
        patch.object(dynamics, "get_prompt_by_name", side_effect=prompts.__getitem__),
    ):
        examples = dynamics.load_eval_examples(
            Path("a.eval"), Path("h.eval"), "base_solution"
        )
    assert examples == [
        (1, "bash-system", f"Fix the service|```\n{view}\n```"),
        (0, "bash-system", f"Fix the service|```\n{view}\n```"),
    ]

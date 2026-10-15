"""Generator config parsing, the dry run, and the limits Inspect must apply."""

import json
from types import SimpleNamespace

import pytest
from inspect_ai import Task, eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.tool import ToolCall

from basharena_runtime import generate as gen

CONFIG = """
experiment_folder: experiments/x
defaults:
  data_split: test
  dataset_type: basharena
  limit: 5
  max_tokens: 16384
runs:
  attacks:
    eval_mode: attack
    system_prompt: BASHARENA_ATTACK_SYSTEM_PROMPT
    model: tinker/Qwen/Qwen3.6-35B-A3B
  honest:
    eval_mode: honest
    system_prompt: BASHARENA_HONEST_SYSTEM_PROMPT
    model: tinker/Qwen/Qwen3.6-35B-A3B
    limit: [5, 8]
  honeypots:
    eval_mode: attack
    system_prompt: BASHARENA_ATTACK_SYSTEM_PROMPT
    model: tinker/openai/gpt-oss-120b
    data_split: train
    max_tokens: 8000
"""


def _config(tmp_path, text=CONFIG):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_runs_take_defaults_and_overrides(tmp_path):
    attacks, honest, honeypots = gen.load_runs(_config(tmp_path))
    assert attacks == gen.Run(
        "attacks",
        "attack",
        "test",
        "BASHARENA_ATTACK_SYSTEM_PROMPT",
        "tinker/Qwen/Qwen3.6-35B-A3B",
        16384,
        5,
    )
    assert honest.limit == (5, 8) and honest.path == "honest.eval"
    assert (honeypots.split, honeypots.max_tokens) == ("train", 8000)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("dataset_type: basharena", "dataset_type: apps"),
        ("limit: 5", "limit: 0"),
        ("limit: [5, 8]", "limit: [8, 5]"),
        ("  honest:\n    eval_mode: honest", "  honest:\n    eval_mode: attack"),
        ("  attacks:", "  backdoors:"),
        ("model: tinker/openai/gpt-oss-120b", "model: gpt-oss-120b"),
        ("max_tokens: 8000", "max_tokens: many"),
    ],
)
def test_bad_configs_are_rejected(tmp_path, old, new):
    assert old in CONFIG
    with pytest.raises(ValueError):
        gen.load_runs(_config(tmp_path, CONFIG.replace(old, new)))


def test_unbounded_runs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="limit"):
        gen.load_runs(_config(tmp_path, CONFIG.replace("  limit: 5\n", "")))


def test_only_basharena_policy_prompts_are_selectable():
    assert gen.system_prompt("BASHARENA_ATTACK_SYSTEM_PROMPT") != gen.system_prompt(
        "BASHARENA_HONEST_SYSTEM_PROMPT"
    )
    with pytest.raises(ValueError):
        gen.system_prompt("SOME_APPS_PROMPT")


def test_dry_run_plans_without_writing_or_calling(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "basharena_task", lambda run: pytest.fail("built a task"))
    output = tmp_path / "out"
    plan = gen.generate(_config(tmp_path), output, concurrency=1, allow_paid=False)
    assert plan["dry_run"] is True and plan["complete"] is False
    assert [r["actor_id"] for r in plan["runs"]] == [
        "tinker/Qwen/Qwen3.6-35B-A3B",
        "tinker/Qwen/Qwen3.6-35B-A3B",
        "tinker/openai/gpt-oss-120b",
    ]
    assert plan["limits"]["max_steps"] == 60
    assert not output.exists()
    json.dumps(plan)  # the plan is printed as JSON


def test_existing_output_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        gen.generate(_config(tmp_path), tmp_path, concurrency=1, allow_paid=False)


def _run(mode: str = "attack") -> gen.Run:
    prompt = f"BASHARENA_{'ATTACK' if mode == 'attack' else 'HONEST'}_SYSTEM_PROMPT"
    return gen.Run("attacks", mode, "train", prompt, "mockllm/model", None, 1)


def test_inspect_applies_the_time_limit(monkeypatch, tmp_path):
    """ControlEvalConfig(time_limit=...) is silently dropped by get_control_task;
    with no limit, task 43890461 sat in a hung main-task test for over nine hours."""
    monkeypatch.setattr(
        gen, "BashArenaSetting", lambda: SimpleNamespace(get_dataset=lambda mode: [])
    )
    monkeypatch.setattr(gen, "attack_policy", lambda **_: None)
    monkeypatch.setattr(gen, "auto_submit_protocol_wrapper", lambda **_: None)
    monkeypatch.setattr(gen, "get_control_task", lambda **_: Task())
    log = eval(
        gen.basharena_task(_run()),
        model="mockllm/model",
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.eval.config.time_limit == 60 * 60


def test_task_keeps_one_parity_and_drops_unbuildable_tasks(monkeypatch):
    ids = ["1003684", "11", "12", "13"]
    seen = {}
    monkeypatch.setattr(
        gen,
        "BashArenaSetting",
        lambda: SimpleNamespace(
            get_dataset=lambda mode: [SimpleNamespace(id=i) for i in ids]
        ),
    )
    monkeypatch.setattr(gen, "attack_policy", lambda **_: None)
    monkeypatch.setattr(gen, "auto_submit_protocol_wrapper", lambda **_: None)
    monkeypatch.setattr(gen, "get_control_task", lambda **kw: seen.update(kw) or Task())
    gen.basharena_task(_run())  # train: odd ids
    assert [s.id for s in seen["dataset"]] == ["11", "13"]
    assert seen["eval_config"].max_steps == 60
    assert seen["eval_config"].token_limit == 2_000_000


def test_logs_record_the_action_view_and_generator(tmp_path):
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import Generate, TaskState, solver

    @solver
    def act():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.messages.append(
                ChatMessageAssistant(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", function="bash", arguments={"cmd": "ls"})
                    ],
                )
            )
            state.output = ModelOutput.from_content("mockllm/model", "")
            return state

        return solve

    task = Task(
        dataset=[
            Sample(id="12", input="t", metadata={"main_task_description": "Do it."})
        ],
        solver=act(),
    )
    log = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]
    run = gen.Run(
        "honeypots",
        "attack",
        "test",
        "BASHARENA_ATTACK_SYSTEM_PROMPT",
        "tinker/openai/gpt-oss-120b",
        None,
        1,
    )
    [sample] = gen.add_trajectory_views(log.location, run).samples
    assert sample.store["base_solution"] == (
        "<action_1>\n<tool>bash</tool>\n<cmd>ls</cmd>\n</action_1>"
    )
    assert sample.store["generator_info"] == {
        "model_info": {
            "model": "openai/gpt-oss-120b",  # without the provider, as APPS logs
            "system_prompt": gen.ATTACK_SYSTEM_PROMPT,
            "base_prompt": "Do it.",
        },
        "eval_mode": "attack",
        "secret_input": "",
    }
    assert isinstance(sample.messages[0], ChatMessageUser)
    assert read_eval_log(log.location).samples[0].store.get("base_solution") is None

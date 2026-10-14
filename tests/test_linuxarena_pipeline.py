import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall

from lasr_labs_2025_control_project.scripts import run_linuxarena as driver
from lasr_labs_2025_control_project.utils import linuxarena_pipeline as pipeline
from lasr_labs_2025_control_project.utils.linuxarena_tinker import (
    ToolDialect,
    bounded_compose,
    resolve_renderer,
)


@pytest.fixture
def configs() -> Any:
    path = pipeline.REPO_ROOT / "experiments/open_models/make_configs.py"
    spec = importlib.util.spec_from_file_location("linuxarena_test_configs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(configs: Any) -> dict[str, Any]:
    configs.ROLE_MODELS["open"] = {
        "U": "tinker/invented/author",
        "T": "tinker/invented/trusted",
        "H": "tinker/invented/honeypot",
    }
    return configs.build_linuxarena(configs.ModelResolver("open", {}))


def test_generated_setting_uses_arbitrary_triple(
    configs: Any, config: dict[str, Any], tmp_path: Path
) -> None:
    configs.CONFIGS = tmp_path
    written, pruned = configs.generate_setting(
        "open", "linuxarena", configs.ModelResolver("open", {}), 8192
    )
    assert not pruned
    assert written == [tmp_path / "linuxarena_open/pipeline.yaml"]
    generated = yaml.safe_load(written[0].read_text())
    assert generated == config
    pipeline.validate_config(generated)
    assert config["models"] == configs.ROLE_MODELS["open"]
    assert "/home/" not in written[0].read_text()
    with pytest.raises(ValueError, match="only --model-class open"):
        configs.build_linuxarena(configs.ModelResolver("closed", {}))


def test_commands_share_tasks_and_side_task_and_use_named_monitors(
    config: dict[str, Any], tmp_path: Path
) -> None:
    for population, role in pipeline.POPULATION_ROLES.items():
        args = pipeline.generation_arguments(config, "task-A", population, tmp_path)
        assert args[args.index("-t") + 1] == "task-A"
        model = json.loads(
            next(arg.split("=", 1)[1] for arg in args if arg.startswith("untrusted="))
        )
        assert model["model_name"].endswith(
            "/" + pipeline.raw_model(config["models"][role])
        )
        if population == "honest":
            assert "-s" not in args
        else:
            assert args[args.index("-s") + 1] == config["generation"]["side_task"]
        assert args[args.index("--max-samples") + 1] == "1"
    args = pipeline.monitoring_arguments(
        config, tmp_path / "raw.eval", tmp_path / "out"
    )
    monitors = json.loads(args[args.index("-m") + 1])
    assert {m["name"] for m in monitors} == {"trusted", "untrusted_suspiciousness"}
    assert all(m["prompt"] == "simple" and m["cot_level"] == "none" for m in monitors)
    for role in ("U", "T"):
        assert pipeline.raw_model(config["models"][role]) in " ".join(args)


@pytest.mark.parametrize("value", ["openai/author", "tinker/tinker://checkpoint"])
def test_only_base_tinker_models_supported(value: str) -> None:
    with pytest.raises(ValueError, match="base tinker"):
        pipeline.raw_model(value)


def test_select_cases_is_deterministic_and_checks_explicit_ids(
    config: dict[str, Any],
) -> None:
    generation = {**config["generation"], "task_count": 2}
    tasks = {key: SimpleNamespace(task_category="fix_bug") for key in ("a", "b", "c")}
    assert pipeline.select_cases(tasks, generation) == pipeline.select_cases(
        dict(reversed(list(tasks.items()))), generation
    )
    generation["main_tasks"] = ["c", "a"]
    assert pipeline.select_cases(tasks, generation) == ["c", "a"]
    generation["main_tasks"] = ["missing"]
    with pytest.raises(ValueError, match="unique task IDs"):
        pipeline.select_cases(tasks, generation)


def test_plan_freezes_triple_and_task_selection(
    config: dict[str, Any], tmp_path: Path
) -> None:
    plan = pipeline.ensure_plan(tmp_path, config, ["a", "b"])
    assert pipeline.ensure_plan(tmp_path, config, ["a", "b"]) == plan
    config["models"]["U"] = "tinker/other/author"
    with pytest.raises(ValueError, match="Run plan differs"):
        pipeline.ensure_plan(tmp_path, config, ["a", "b"])


def test_journal_keeps_failures_and_retries_are_explicit(tmp_path: Path) -> None:
    path = tmp_path / "status.jsonl"
    failed = {
        "job_id": "one",
        "state": "failed",
        "eval_files": ["partial.eval"],
        "actor_id": "actor",
        "population": "attacks",
    }
    complete = {**failed, "state": "complete", "eval_files": ["retry.eval"]}
    for record in (failed, complete):
        pipeline.append_status(path, record)
    records = pipeline.read_jsonl(path)
    assert records == [failed, complete]
    assert pipeline.latest_status(records)["one"]["state"] == "complete"
    assert not pipeline.pending_job(failed, False)
    assert pipeline.pending_job(failed, True)
    assert pipeline.pending_job({"state": "starting"}, False)
    assert not pipeline.pending_job(complete, True)
    sources = pipeline.generation_manifest(records)
    assert {s["path"] for s in sources} == {"partial.eval", "retry.eval"}
    monitored = [{"source": "retry.eval", "eval_files": ["monitored.eval"]}]
    assert {s["path"] for s in pipeline.export_manifest(sources, monitored)} == {
        "partial.eval",
        "monitored.eval",
    }


def test_recover_partial_logs_from_interrupted_attempt(tmp_path: Path) -> None:
    (tmp_path / "partial.eval").touch()
    path = tmp_path / "status.jsonl"
    pipeline.append_status(
        path, {"job_id": "one", "state": "starting", "destination": str(tmp_path)}
    )
    recovered = driver.recover_status(path)
    assert recovered[0]["eval_files"] == [str(tmp_path / "partial.eval")]
    assert pipeline.read_jsonl(path)[0]["state"] == "starting"


@pytest.mark.parametrize(
    "argv,text",
    [
        (["bash", "-lc", "printf '%s' a"], "printf '%s' a"),
        (["sh", "-c", "echo ok"], "echo ok"),
        (["printf", "%s", "a b"], "printf %s 'a b'"),
    ],
)
def test_auto_call_translation_and_native_history(argv: list[str], text: str) -> None:
    dialect = ToolDialect()
    call = ToolCall(
        id="unique", function="bash.exec", arguments={"cmd": argv, "timeout": 120000}
    )
    original_args = copy.deepcopy(call.arguments)
    translation = dialect.translate(call, {"bash"})
    assert translation is not None
    assert call.function == "bash" and call.arguments["cmd"] == text
    assert call.arguments["timeout"] == 120000
    message = ChatMessageAssistant(content="", tool_calls=[call])
    result = ChatMessageTool(content="ok", tool_call_id=call.id, function="bash")
    shown = dialect.native_history([message, result])
    native = shown[0].tool_calls[0]
    assert native.function == "bash.exec" and native.arguments == original_args
    assert shown[1] is result
    assert (
        message.tool_calls is not None
        and message.tool_calls[0].arguments["cmd"] == text
    )
    assert native.id == result.tool_call_id


def test_native_schema_calls_are_unchanged_and_invalid_dialects_fail() -> None:
    dialect = ToolDialect()
    call = ToolCall(id="one", function="bash", arguments={"cmd": "ls"})
    assert dialect.translate(call, {"bash"}) is None
    with pytest.raises(ValueError, match="Unsupported bash"):
        dialect.translate(
            ToolCall(id="two", function="bash", arguments={"cmd": [1]}), {"bash"}
        )
    with pytest.raises(ValueError, match="Unsupported tool name"):
        dialect.translate(
            ToolCall(id="three", function="mystery.exec", arguments={}), {"bash"}
        )
    with pytest.raises(ValueError, match="Unsupported tool dialect"):
        ToolDialect("unknown")
    with pytest.raises(ValueError, match="Unsupported bash"):
        ToolDialect("schema").translate(
            ToolCall(id="four", function="bash", arguments={"cmd": ["ls"]}), {"bash"}
        )


@dataclass
class ModelInfo:
    recommendations: list[str]

    def get_recommended_renderer_names(self, model: str) -> list[str]:
        if model == "unknown/model":
            raise KeyError(model)
        return self.recommendations


def test_renderer_selection_is_metadata_driven_and_unsupported_fails() -> None:
    info = ModelInfo(["hybrid", "hybrid_disable_thinking"])
    assert resolve_renderer("invented/model", {}, False, info) == "hybrid"
    assert (
        resolve_renderer("invented/model", {}, True, info) == "hybrid_disable_thinking"
    )
    assert (
        resolve_renderer("invented/model", {}, True, ModelInfo(["instruct"]))
        == "instruct"
    )
    with pytest.raises(ValueError, match="Unsupported model.*explicit"):
        resolve_renderer("unknown/model", {}, True, info)
    with pytest.raises(ValueError, match="must disable thinking"):
        resolve_renderer("invented/model", {"monitor_renderer": "hybrid"}, True, info)
    assert (
        resolve_renderer("unknown/model", {"renderer": "custom"}, False, info)
        == "custom"
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"privileged": True},
        {"network_mode": "host"},
        {"pid": "host"},
        {"volumes": ["/host:/container"]},
        {"volumes": ["./local:/container"]},
        {"volumes": [{"type": "bind", "source": "/host", "target": "/container"}]},
        {"devices": ["/dev/sda"]},
    ],
)
def test_sandbox_rejects_host_access(
    config: dict[str, Any], patch: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="Refusing"):
        bounded_compose(
            {"services": {"default": {"image": "prebuilt", **patch}}},
            config["resources"],
        )


def test_sandbox_caps_every_service_and_removes_builds(config: dict[str, Any]) -> None:
    original = {
        "services": {
            "default": {"image": "agent", "build": {"context": "."}},
            "sidecar": {"image": "sidecar"},
        }
    }
    bounded = bounded_compose(original, config["resources"])
    assert "build" in original["services"]["default"]
    assert "build" not in bounded["services"]["default"]
    assert bounded["services"]["default"]["cpus"] == 0.5
    assert bounded["services"]["sidecar"]["cpus"] == 0.1
    for service in bounded["services"].values():
        assert service["mem_limit"] == service["memswap_limit"]
        assert (
            service["deploy"]["resources"]["limits"]["memory"] == service["mem_limit"]
        )
    with pytest.raises(ValueError, match="prebuilt image"):
        bounded_compose({"services": {"default": {"build": "."}}}, config["resources"])


def action(population: str, index: int = 0) -> dict[str, Any]:
    return {
        "meta": {
            "action_id": f"{population}-{index}",
            "trajectory_id": f"trajectory-{population}",
            "population": population,
            "split": "test",
            "main_task_ids": ["a"],
            "sample_error": {"message": "timeout"} if index else None,
            "native_monitor_scores": [
                {"monitor": "trusted", "draw": "1", "value": index},
                {
                    "monitor": "untrusted_suspiciousness",
                    "draw": "1",
                    "value": 9 if population != "honest" else 1,
                },
            ],
        }
    }


def test_flat_tables_preserve_every_row_and_supply_all_three_scores() -> None:
    actions = [action(pop, i) for pop in pipeline.POPULATION_ROLES for i in range(2)]
    scores = [
        {"action_id": a["meta"]["action_id"], "margin": i}
        for i, a in enumerate(actions)
        if a["meta"]["population"] != "honest"
    ]
    rows = pipeline.assemble_flat_rows(actions, scores)
    assert len(rows) == 6
    attacks, honeypots, benign = pipeline.statistical_tables(rows)
    assert len(attacks) == len(honeypots) == len(benign) == 2
    for frame in (attacks, honeypots, benign):
        assert "untrusted_suspiciousness_score" in frame.columns
        assert frame.untrusted_suspiciousness_score.notna().all()
    assert attacks.untrusted_collusion_score.notna().all()
    assert rows[1]["sample_error"] == {"message": "timeout"}


def test_flat_tables_keep_missing_and_failed_scores_and_refuse_fitting() -> None:
    actions = [action(pop) for pop in pipeline.POPULATION_ROLES]
    actions[1]["meta"]["native_monitor_scores"][0].update(
        value=None, unscored_reason="no score tool"
    )
    scores = [
        {"action_id": "attacks-0", "margin": None, "error": "request failed"},
        {"action_id": "honeypots-0", "margin": 1},
    ]
    rows = pipeline.assemble_flat_rows(actions, scores)
    assert len(rows) == 3
    assert rows[1]["trusted_errors"] == ["no score tool"]
    assert rows[1]["recognition_error"] == "request failed"
    assert pipeline.missing_scores(rows)["missing"]["untrusted_collusion_score"] == 1
    with pytest.raises(ValueError, match="never drop rows"):
        pipeline.statistical_tables(rows)
    scores.append({"action_id": "attacks-0", "margin": 2})
    assert (
        pipeline.assemble_flat_rows(actions, scores)[1]["untrusted_collusion_score"]
        == 2
    )


def test_sft_and_score_commands_get_models_from_config(
    config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def capture(command: list[str], env_file: Path, dry_run: bool) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(driver, "run_command", capture)
    args = argparse_namespace(tmp_path)
    assert driver.finetune(config, tmp_path, args) == 0
    (tmp_path / "sft/checkpoint.txt").write_text("tinker://run/sampler_weights/final")
    assert driver.score(config, tmp_path, args) == 0
    assert commands[0][commands[0].index("--base-model") + 1] == "invented/author"
    assert commands[0][commands[0].index("--val-frac") + 1] == "0"
    assert commands[1][commands[1].index("--model") + 1] == config["models"]["U"]
    assert (
        commands[2][commands[2].index("--model") + 1]
        == "tinker/tinker://run/sampler_weights/final"
    )


def argparse_namespace(root: Path) -> Any:
    return SimpleNamespace(
        env_file=root / ".env",
        dry_run=True,
        retry_failed=False,
        rescore_complete=False,
        limit=None,
    )


def test_generate_and_monitor_journals_with_mocked_process(
    config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "check_ct_environment", lambda config, root: None)
    monkeypatch.setattr(
        driver, "selected_cases", lambda config, root, dry_run: ["task-a"]
    )
    monkeypatch.setattr(
        driver, "resources", lambda root: {"available_ram_gib": 20, "free_disk_gib": 40}
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(command)
        flag = "-o" if "monitor" in command else "--log-dir"
        destination = Path(command[command.index(flag) + 1])
        (destination / "run.eval").touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    args = argparse_namespace(tmp_path)
    args.dry_run = False
    assert driver.drive(config, tmp_path, args, monitoring=False) == 0
    assert len(commands) == 3
    assert driver.drive(config, tmp_path, args, monitoring=False) == 0
    assert len(commands) == 3
    assert driver.drive(config, tmp_path, args, monitoring=True) == 0
    assert len(commands) == 6
    manifest = json.loads((tmp_path / "export_manifest.json").read_text())
    assert len(manifest) == 3 and all("monitored" in item["path"] for item in manifest)
    args.rescore_complete = True
    assert driver.drive(config, tmp_path, args, monitoring=True) == 0
    assert len(commands) == 9
    assert len(pipeline.read_jsonl(tmp_path / "monitor_status.jsonl")) == 12


def test_plot_delegates_complete_flat_tables_to_existing_pipeline(
    config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lasr_labs_2025_control_project.utils import numerical_pipeline

    actions = [action(pop, i) for pop in pipeline.POPULATION_ROLES for i in range(2)]
    (tmp_path / "exports").mkdir()
    (tmp_path / "sft").mkdir()
    (tmp_path / "exports/actions.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in actions)
    )
    scores = [
        {
            "action_id": a["meta"]["action_id"],
            "margin": 2 if a["meta"]["population"] == "attacks" else -2,
            "own_source": a["meta"]["population"] == "attacks",
        }
        for a in actions
        if a["meta"]["population"] != "honest"
    ]
    for arm in ("base", "sft"):
        (tmp_path / f"sft/scores_test_{arm}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in scores)
        )
    calls: list[dict[str, Any]] = []
    plotted: list[Any] = []

    def build(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return kwargs

    def plots(results: Any, destination_folder: Path) -> None:
        plotted.append(results)
        assert destination_folder == tmp_path / "sft/plot_all"

    monkeypatch.setattr(numerical_pipeline, "build_optimization_config", build)
    monkeypatch.setattr(
        numerical_pipeline, "optimize_parameters", lambda **kwargs: kwargs
    )
    monkeypatch.setattr(
        numerical_pipeline, "build_analysis_payload", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(numerical_pipeline, "save_analysis_payload", lambda *args: None)
    monkeypatch.setattr(numerical_pipeline, "plot_data", plots)
    driver.plot(config, tmp_path)
    assert len(calls) == 2 and len(plotted) == 1
    for kwargs in calls:
        for key in ("attacks_df", "honeypots_df", "benign_df"):
            assert kwargs[key].untrusted_suspiciousness_score.notna().all()
        assert kwargs["attacks_df"].untrusted_collusion_score.notna().all()
    for arm in ("base", "sft"):
        assert (
            len(pipeline.read_jsonl(tmp_path / f"sft/plot_all/flat_{arm}.jsonl")) == 6
        )
    coverage = json.loads((tmp_path / "sft/plot_all/coverage.json").read_text())
    assert coverage["base"]["recognition"]["roc_auc"] == 1


def test_provider_assigns_unique_ids_restores_history_and_softens_forced_monitor_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    from lasr_labs_2025_control_project.utils import linuxarena_tinker as bridge

    registered: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []

    class FakeBase:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name

        async def generate(
            self, input: list[Any], tools: list[Any], tool_choice: Any, config: Any
        ) -> Any:
            requests.append({"input": input, "tools": tools, "choice": tool_choice})
            calls = (
                [
                    ToolCall(
                        id="cookbook-repeated",
                        function="bash.exec",
                        arguments={
                            "cmd": ["bash", "-lc", "echo hello"],
                            "timeout": 120000,
                        },
                    )
                    for _ in range(2)
                ]
                if self.model_name == "fake/policy"
                else []
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))]
            )

    @dataclass
    class ForcedTool:
        name: str

    class FakeService:
        def create_sampling_client(self, base_model: str) -> None:
            return None

    def register(cls: Any, route: str) -> None:
        registered[route] = cls

    modules = {
        "tinker_cookbook.eval.inspect_utils": SimpleNamespace(
            InspectAPIFromTinkerSampling=FakeBase
        ),
        "tinker_cookbook.model_info": ModelInfo(["instruct"]),
        "tinker": SimpleNamespace(ServiceClient=FakeService),
        "inspect_ai.model._registry": SimpleNamespace(modelapi_register=register),
        "inspect_ai.tool": SimpleNamespace(ToolFunction=ForcedTool),
        "inspect_ai.model": SimpleNamespace(GenerateConfig=dict),
    }
    monkeypatch.setattr(bridge, "import_module", lambda name: modules[name])
    monkeypatch.setattr(bridge, "version", lambda name: "0.3.268")
    settings = {"renderer": None, "monitor_renderer": None, "tool_dialect": "auto"}
    options = {
        "models": {
            "False": {"fake/policy": settings},
            "True": {"fake/monitor": settings},
        },
        "translation_log": str(tmp_path / "translations.jsonl"),
    }
    bridge.register_provider(options)
    api = registered["tinker-linuxarena"]("fake/policy")
    output = asyncio.run(api.generate([], [SimpleNamespace(name="bash")], "auto", None))
    calls = output.choices[0].message.tool_calls
    assert len({call.id for call in calls}) == 2
    assert all(
        call.function == "bash" and call.arguments["cmd"] == "echo hello"
        for call in calls
    )
    history = [ChatMessageAssistant(content="", tool_calls=calls)]
    asyncio.run(api.generate(history, [SimpleNamespace(name="bash")], "auto", None))
    shown = requests[-1]["input"][0].tool_calls
    assert all(
        call.function == "bash.exec" and isinstance(call.arguments["cmd"], list)
        for call in shown
    )
    monitor = registered["tinker-linuxarena-monitor"]("fake/monitor")
    output = asyncio.run(
        monitor.generate(
            [],
            [SimpleNamespace(name="score"), SimpleNamespace(name="other")],
            ForcedTool("score"),
            None,
        )
    )
    assert requests[-1]["choice"] == "auto"
    assert [tool.name for tool in requests[-1]["tools"]] == ["score"]
    assert output.choices[0].message.tool_calls == []


def test_missing_previously_recorded_log_stays_in_manifest(tmp_path: Path) -> None:
    path = tmp_path / "status.jsonl"
    missing = str(tmp_path / "deleted.eval")
    pipeline.append_status(
        path,
        {
            "job_id": "one",
            "state": "complete",
            "destination": str(tmp_path),
            "eval_files": [missing],
            "actor_id": "actor",
            "population": "attacks",
        },
    )
    records = driver.recover_status(path)
    assert pipeline.generation_manifest(records)[0]["path"] == missing


def test_dry_run_does_not_freeze_the_plan(
    config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environments = SimpleNamespace(
        get_environment=lambda name: name,
        get_main_tasks_for_env=lambda environment: {},
        get_side_tasks_for_env=lambda environment: {
            config["generation"]["side_task"]: object()
        },
    )
    monkeypatch.setattr(driver, "import_module", lambda name: environments)
    monkeypatch.setattr(driver, "select_cases", lambda tasks, generation: ["task-a"])
    assert driver.selected_cases(config, tmp_path, True) == ["task-a"]
    assert not (tmp_path / "plan.json").exists()
    assert driver.selected_cases(config, tmp_path, False) == ["task-a"]
    assert json.loads((tmp_path / "plan.json").read_text())["cases"] == ["task-a"]

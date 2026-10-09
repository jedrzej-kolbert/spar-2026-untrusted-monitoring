"""make_configs.py emits BashArena configs that the config loader accepts."""

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from lasr_labs_2025_control_project.prompts.basharena_monitor_prompts import (
    BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT,
)
from lasr_labs_2025_control_project.utils.config_loader import (
    DatasetType,
    GeneratorRunConfig,
    MonitorRunConfig,
    load_and_normalize_config,
)

CHECKPOINT = "tinker://run/sampler_weights/basharena_self_rec_action"
GENERATOR = "self_rec/generator"
MONITORS = "self_rec/monitors"


@pytest.fixture(scope="module")
def make_configs() -> Iterator[ModuleType]:
    path = Path(__file__).parents[1] / "experiments/open_models/make_configs.py"
    spec = importlib.util.spec_from_file_location("make_configs", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


@pytest.fixture
def tree(
    make_configs: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """The generated `configs/basharena_open` tree, with one hand-written config."""
    monkeypatch.setattr(make_configs, "CONFIGS", tmp_path)
    root = tmp_path / "basharena_open"
    (root / GENERATOR).mkdir(parents=True)
    (root / GENERATOR / "my_run_config.yaml").write_text("hand-written\n")
    models = make_configs.ModelResolver(
        "open", {"basharena_self_rec_action": CHECKPOINT}
    )
    written, pruned = make_configs.generate_setting("open", "basharena", models, None)
    assert len(written) == 4 and pruned == []
    return root


def test_generators(make_configs: ModuleType, tree: Path) -> None:
    roles = make_configs.ROLE_MODELS["open"]
    # The APPS file names, so sft-steps.md runs the same two commands.
    expected = {
        ("train", "train_generator_config.yaml"): {
            "attacks": ("U", "generator/train/attacks.eval"),
            "honeypots": ("H", "generator/train/honeypots.eval"),
        },
        ("test", "generator_config.yaml"): {
            "attacks": ("U", "generator/attacks.eval"),
            "honest": ("U", "generator/honest.eval"),
            "honeypots": ("H", "generator/honeypots.eval"),
        },
    }
    for (split, name), runs in expected.items():
        config = load_and_normalize_config(
            tree / GENERATOR / name, config_type="generator"
        )
        assert [run.label for run in config.runs] == list(runs)
        for run in config.runs:
            role, save_path = runs[run.label]
            assert isinstance(run, GeneratorRunConfig)
            assert run.generator.dataset_type == DatasetType.BASHARENA
            assert run.generator.data_split == split
            assert run.generator.model_info.model == roles[role]
            assert run.generator.max_tokens is None  # the agent policy's own cap
            assert run.limit == make_configs.BASHARENA_LIMITS[split]
            assert run.save_path == config.experiment_folder / save_path


def test_monitors_use_the_action_view(
    make_configs: ModuleType,
    tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = make_configs.ROLE_MODELS["open"]
    # The loader requires the test logs to exist, relative to the working directory.
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "experiments/open-source-model/basharena/self_rec/generator"
    logs.mkdir(parents=True)
    for label in ("attacks", "honest", "honeypots"):
        (logs / f"{label}.eval").touch()
    for arm, collusion in (("base", roles["U"]), ("sft", f"tinker/{CHECKPOINT}")):
        config = load_and_normalize_config(
            tree / MONITORS / arm / "action_monitor_config.yaml"
        )
        assert {run.label for run in config.runs} == {"attacks", "honest", "honeypots"}
        for run in config.runs:
            assert isinstance(run, MonitorRunConfig)
            assert run.view == "action"
            assert run.action_options is not None
            assert run.action_options.max_history_chars == 120000
            monitor = run.monitors.untrusted_collusion_monitor
            assert monitor is not None
            assert monitor.model_info.model == collusion
            assert (
                monitor.model_info.system_prompt
                == BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT
            )
            trusted = run.monitors.trusted_backdoor_monitor
            assert trusted is not None and trusted.model_info.model == roles["T"]
            assert run.eval_path == config.experiment_folder / (
                f"generator/{run.label}.eval"
            )


def test_role_models_edit_reaches_basharena(
    make_configs: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing `ROLE_MODELS`, as sft-steps.md says to, changes the BashArena configs."""
    edited = {"U": "tinker/org/new-u", "T": "tinker/org/new-t", "H": "tinker/org/new-h"}
    monkeypatch.setattr(make_configs, "CONFIGS", tmp_path)
    monkeypatch.setitem(make_configs.ROLE_MODELS, "open", edited)
    models = make_configs.ModelResolver(
        "open", {"basharena_self_rec_action": CHECKPOINT}
    )
    written, _ = make_configs.generate_setting("open", "basharena", models, None)
    used = {
        line.split("model:", 1)[1].strip()
        for path in written
        for line in path.read_text().splitlines()
        if line.strip().startswith("model:")
    }
    assert used == {*edited.values(), f"tinker/{CHECKPOINT}"}


def test_hand_written_configs_are_kept(tree: Path) -> None:
    assert (tree / GENERATOR / "my_run_config.yaml").read_text() == "hand-written\n"


def test_max_tokens_flag_reaches_the_generators(
    make_configs: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(make_configs, "CONFIGS", tmp_path)
    models = make_configs.ModelResolver("open", {})
    make_configs.generate_setting("open", "basharena", models, 4096)
    config = load_and_normalize_config(
        tmp_path / "basharena_open" / GENERATOR / "generator_config.yaml",
        config_type="generator",
    )
    for run in config.runs:
        assert isinstance(run, GeneratorRunConfig)
        assert run.generator.max_tokens == 4096
    assert models.pending == {"basharena_self_rec_action"}

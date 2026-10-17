"""Offline BashArena config generation, and legacy configs stay byte-identical."""

import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/open_models/make_configs.py"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden in config-generation tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def builder(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("basharena_config_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "CONFIGS", tmp_path / "configs")
    monkeypatch.setattr(module, "DEFAULT_CHECKPOINTS_FILE", tmp_path / "ckpts.json")
    return module


def _monitors(root: Path, suffix: str) -> tuple[dict, dict]:
    path = root / f"self_rec/monitors/{suffix}/base_monitor_config.yaml"
    document = yaml.safe_load(path.read_text())
    entries = document["defaults"]["monitors"]
    return document, {name: cfg for entry in entries for name, cfg in entry.items()}


def test_bash_configs_are_bounded_generators_and_per_action_monitors(builder):
    models = builder.ModelResolver("open", {})
    written, pruned = builder.generate_setting("open", "basharena", models, 4096, 3)
    root = builder.target_dir("open", "basharena")
    assert len(written) == 4 and not pruned
    assert models.pending == {"basharena_self_rec_base"}
    roles = builder.ROLE_MODELS["open"]
    for split, filename, population in (
        ("train", "train_generator_config.yaml", {"attacks", "honeypots"}),
        ("test", "base_generator_config.yaml", {"attacks", "honest", "honeypots"}),
    ):
        text = (root / "self_rec/generator" / filename).read_text()
        assert "--model-class open --settings basharena" in text
        document = yaml.safe_load(text)
        assert document["defaults"] == {
            "data_split": split,
            "dataset_type": "basharena",
            "max_tokens": 4096,
            "limit": 3,
        }
        assert set(document["runs"]) == population
        for label, run in document["runs"].items():
            assert run["model"] == roles["H" if label == "honeypots" else "U"]
            assert run["system_prompt"] == (
                "BASHARENA_HONEST_SYSTEM_PROMPT"
                if label == "honest"
                else "BASHARENA_ATTACK_SYSTEM_PROMPT"
            )
            assert run["eval_mode"] == ("honest" if label == "honest" else "attack")
            assert run["save_path"] == f"generator/{split}/{label}.eval"
    # Base and SFT monitor configs differ only in the collusion monitor's model.
    for suffix, collusion in (
        ("base", roles["U"]),
        ("sft", "tinker/tinker://PENDING__basharena_self_rec_base"),
    ):
        document, monitors = _monitors(root, suffix)
        assert document["defaults"]["view"] == "action"
        assert monitors["untrusted_collusion_monitor"]["model_info"] == {
            "model": collusion,
            "system_prompt": "BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT",
            "base_prompt": "BASHARENA_ACTION_USER_PROMPT",
        }
        for name, role in (
            ("trusted_backdoor_monitor", "T"),
            ("untrusted_backdoor_monitor", "U"),
        ):
            assert monitors[name]["model_info"] == {
                "model": roles[role],
                "system_prompt": "BASHARENA_ACTION_SUSPICIOUSNESS_SYSTEM_PROMPT",
                "base_prompt": "BASHARENA_ACTION_SUSPICIOUSNESS_USER_PROMPT",
            }
        # Action views are not Python and are never paraphrased or minified.
        assert not any(
            m["transformed"] or m["remove_comments"] for m in monitors.values()
        )
        # The runtime writes <output>/<label>.eval; the test logs go to generator/test.
        assert {label: run["eval_path"] for label, run in document["runs"].items()} == {
            label: f"generator/test/{label}.eval"
            for label in ("attacks", "honest", "honeypots")
        }
    assert not list(root.rglob("*transform*"))


def test_generated_monitor_config_loads_as_per_action(builder, monkeypatch, tmp_path):
    from lasr_labs_2025_control_project.utils.config_loader import (
        MonitorRunConfig,
        load_and_normalize_config,
    )

    models = builder.ModelResolver("open", {})
    builder.generate_setting("open", "basharena", models, 4096, 3)
    root = builder.target_dir("open", "basharena")
    document, _ = _monitors(root, "base")
    logs = tmp_path / document["experiment_folder"] / "generator/test"
    logs.mkdir(parents=True)
    for label in document["runs"]:
        (logs / f"{label}.eval").touch()
    monkeypatch.chdir(tmp_path)
    runs = load_and_normalize_config(
        root / "self_rec/monitors/base/base_monitor_config.yaml"
    ).runs
    assert len(runs) == 3
    assert all(isinstance(r, MonitorRunConfig) and r.view == "action" for r in runs)


def test_exact_cli_selection_is_offline_and_repeatable(builder, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--model-class", "open", "--settings", "basharena"]
    )
    builder.main()
    root = builder.target_dir("open", "basharena")
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert len(before) == 4
    assert (
        yaml.safe_load(
            (root / "self_rec/generator/train_generator_config.yaml").read_text()
        )["defaults"]["limit"]
        == 5
    )
    assert not (builder.CONFIGS / "apps_open").exists()
    assert not (builder.CONFIGS / "bigcodebench_open").exists()
    assert not (builder.REPO_ROOT / "experiments").exists()
    builder.main()
    assert before == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    output = capsys.readouterr().out
    assert "basharena_runtime" in output and "basharena_self_rec_base" in output
    assert "--model-class open --settings basharena --checkpoints" in output
    assert 'ckpts.json: {"basharena_self_rec_base"' in output
    assert builder.DEFAULT_SETTINGS == ("apps", "bigcodebench")


def test_checkpoint_is_bash_specific_and_unowned_json_is_preserved(builder, tmp_path):
    apps_checkpoint = "tinker://apps/sampler_weights/recognition"
    bash_checkpoint = "tinker://bash/sampler_weights/recognition"
    models = builder.ModelResolver("open", {"self_rec_base": apps_checkpoint})
    assert "PENDING__" in models.resolve(builder.BASHARENA_SFT_ARM)
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps({builder.BASHARENA_SFT_ARM: bash_checkpoint}))
    checkpoints = builder.load_checkpoints(path)
    models = builder.ModelResolver("open", checkpoints)
    builder.generate_setting("open", "basharena", models, 16384)
    root = builder.target_dir("open", "basharena")
    extra = root / "personal-models.json"
    extra.write_text('{"note":"preserve"}\n')
    builder.generate_setting("open", "basharena", models, 16384)
    assert extra.read_text() == '{"note":"preserve"}\n'
    base = _monitors(root, "base")[1]["untrusted_collusion_monitor"]["model_info"]
    sft = _monitors(root, "sft")[1]["untrusted_collusion_monitor"]["model_info"]
    assert sft["model"] == "tinker/" + bash_checkpoint
    assert base["model"] == builder.ROLE_MODELS["open"]["U"]
    assert not models.pending


@pytest.mark.parametrize(
    "args",
    [
        ["--model-class", "closed"],
        [],
        ["--model-class", "open", "--basharena-limit", "0"],
        ["--model-class", "open", "--max-tokens", "0"],
    ],
)
def test_unsupported_or_unbounded_bash_selection_fails_before_writes(
    builder, monkeypatch, args
):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--settings", "basharena", *args])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2
    assert not builder.CONFIGS.exists()


@pytest.mark.parametrize(
    "model_class,setting,expected",
    [
        (
            "closed",
            "apps",
            "e4e411404220bdef19d3aa48d488947f7ac9afb47f7c539cbd409b50138c99d0",
        ),
        (
            "closed",
            "bigcodebench",
            "f9449d26a52e453d917f07e096bb281a3b31d230518c3976f90ba97d6dfd9dd9",
        ),
        (
            "open",
            "apps",
            "32b0103d8576d5f74580f2de965ca9370365583d89d6b4909a929e58feaf7cd2",
        ),
        (
            "open",
            "bigcodebench",
            "a9342ac3ed089a3eea19183ec1b534f83514dd9f9ee67044b875e1561b449049",
        ),
    ],
)
def test_legacy_configs_remain_byte_identical(builder, model_class, setting, expected):
    models = builder.ModelResolver(model_class, {})
    spec = builder.SPEC[setting]
    documents = {
        gen.path: builder.build_generator(
            gen, models, setting, 16384 if model_class == "open" else None
        )
        for gen in spec.generators
    }
    documents.update(
        {
            tr.path: builder.build_transform(tr, models, setting)
            for tr in spec.transforms
        }
    )
    documents.update(
        {mon.path: builder.build_monitor(mon, models, setting) for mon in spec.monitors}
    )
    digest = hashlib.sha256()
    for key, value in sorted(documents.items()):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(builder.render(value, setting).encode())
        digest.update(b"\0")
    assert digest.hexdigest() == expected

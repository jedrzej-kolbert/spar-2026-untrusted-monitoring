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


def test_bash_configs_are_bounded_native_yaml_and_scorer_json(builder):
    models = builder.ModelResolver("open", {})
    written, pruned = builder.generate_setting("open", "basharena", models, 4096, 3)
    root = builder.target_dir("open", "basharena")
    assert len(written) == 5 and not pruned
    assert models.pending == {"basharena_self_rec_base"}
    workflow = json.loads((root / "self_rec/workflow.json").read_text())
    assert workflow["conditions"] == ["baseline"]
    assert workflow["target_actor"] == builder.ROLE_MODELS["open"]["U"]
    assert workflow["generation_limit_per_population"] == 3
    for split, population in (
        ("train", {"attacks", "honeypots"}),
        ("test", {"attacks", "honest", "honeypots"}),
    ):
        entry = workflow["generations"][split]
        source = root / entry["config"]
        text = source.read_text()
        assert "--model-class open --settings basharena" in text
        document = yaml.safe_load(text)
        assert document["defaults"] == {
            "data_split": split,
            "dataset_type": "basharena",
            "max_tokens": 4096,
            "limit": 3,
        }
        assert set(document["runs"]) == population
        assert entry["output"].endswith(f"/generator/{split}")
        for label, run in document["runs"].items():
            expected_model = builder.ROLE_MODELS["open"][
                "H" if label == "honeypots" else "U"
            ]
            assert run["model"] == entry["actors"][label] == expected_model
            assert run["system_prompt"] == (
                "BASHARENA_HONEST_SYSTEM_PROMPT"
                if label == "honest"
                else "BASHARENA_ATTACK_SYSTEM_PROMPT"
            )
            assert run["eval_mode"] == ("honest" if label == "honest" else "attack")
            assert run["save_path"] == f"generator/{split}/{label}.eval"
    base = json.loads((root / workflow["scoring_models"]["base"]).read_text())
    sft = json.loads((root / workflow["scoring_models"]["sft"]).read_text())
    assert set(base) == {
        "recognition",
        "trusted_suspiciousness",
        "untrusted_suspiciousness",
    }
    assert (
        base["recognition"]
        == base["untrusted_suspiciousness"]
        == workflow["target_actor"]
    )
    assert base["trusted_suspiciousness"] == builder.ROLE_MODELS["open"]["T"]
    # Only the changed role: rescoring shared suspiciousness requests would draw
    # a second, conflicting sample for the same inputs.
    assert sft == {"recognition": "tinker/tinker://PENDING__basharena_self_rec_base"}
    assert not list(root.rglob("*transform*"))
    assert not list(root.rglob("*monitor*.yaml"))


def test_exact_cli_selection_is_offline_and_repeatable(builder, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--model-class", "open", "--settings", "basharena"]
    )
    builder.main()
    root = builder.target_dir("open", "basharena")
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert len(before) == 5
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
    assert "baseline only" in output and "basharena_self_rec_base" in output
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
    base = json.loads((root / "self_rec/scoring/base_models.json").read_text())
    sft = json.loads((root / "self_rec/scoring/sft_models.json").read_text())
    assert sft == {"recognition": "tinker/" + bash_checkpoint}
    assert base["recognition"] != sft["recognition"]
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
            "a6e2fd286a7714016745444472b1bc4bba5ed22f9c29c79c9c05f14777c2d97b",
        ),
        (
            "closed",
            "bigcodebench",
            "8178d3f82bbca167718926a0ce7c55a90300b3b185b18fa5d061db2495ef5e37",
        ),
        (
            "open",
            "apps",
            "a0695d9f7c5ba27be5016af8196c15dd0feff05da6e431efb3e156c12c465e14",
        ),
        (
            "open",
            "bigcodebench",
            "555624dcd9688134220ed9e85ae58232a00b972ed170b21433f9571ce25e5206",
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

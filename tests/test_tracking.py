"""Offline tests for the W&B tracking layer (no network, W&B disabled by conftest)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lasr_labs_2025_control_project.tracking import core, names


@pytest.mark.parametrize(
    "model, alias",
    [
        ("tinker/Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_30b"),
        ("tinker/Qwen/Qwen3.5-4B", "qwen3_5_4b"),
        ("tinker/openai/gpt-oss-120b", "gpt_oss_120b"),
        ("openai/gpt-4.1-2025-04-14", "gpt_4_1"),
        ("openai/gpt-4o-mini", "gpt_4o_mini"),
        ("google/gemini-2.5-flash", "gemini_2_5_flash"),
        ("tinker/tinker://abc-123/sampler_weights/self_rec_u", "ft_self_rec_u"),
        (
            "openai/ft:gpt-4.1-2025-04-14:lasr-labs:causal-base:XyZ",
            "gpt_4_1_ft_causal_base",
        ),
        (
            "openai/ft:gpt-4.1-2025-04-14:lasr-labs:causal-base:XyZ:ckpt-step-40",
            "gpt_4_1_ft_causal_base_ckpt_step_40",
        ),
    ],
)
def test_short_model(model, alias):
    assert names.short_model(model) == alias


@pytest.mark.parametrize(
    "prompt, alias",
    [
        ("causal_attack_prompt", "causal_attack"),
        ("self_rec_subtle_attack_policy_prompt", "self_rec_subtle_attack"),
        ("honest_policy_prompt", "honest"),
        ("CAUSAL_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT", "causal_collusion"),
        ("DEFAULT_HONEST_MONITOR_SYSTEM_PROMPT", "default_honest"),
        ("LLM_SUMMARIZE_SYSTEM_PROMPT", "llm_summarize"),
    ],
)
def test_short_prompt(prompt, alias):
    assert names.short_prompt(prompt) == alias


def test_alias_overrides(monkeypatch):
    monkeypatch.setitem(names.MODEL_ALIASES, "openai/gpt-4o-mini", "t_closed")
    monkeypatch.setitem(names.PROMPT_ALIASES, "honest_policy_prompt", "benign")
    assert names.short_model("openai/gpt-4o-mini") == "t_closed"
    assert names.short_prompt("honest_policy_prompt") == "benign"


def test_chain_names():
    gen = names.generation_name(
        "apps",
        "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "test",
        "attacks",
        "causal_attack_prompt",
    )
    assert gen == "apps-qwen3_30b-test-attacks-causal_attack"
    minified = names.extend(gen, names.stage_segment("min"))
    monitored = names.extend(
        minified,
        names.stage_segment(
            "mon",
            "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507",
            "CAUSAL_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT",
        ),
    )
    assert monitored == (
        "apps-qwen3_30b-test-attacks-causal_attack.min.mon-qwen3_30b-causal_collusion"
    )
    assert names.generation_name(
        "bigcodebench", "m", "train", "honest", "p"
    ).startswith("bcb-")
    assert names.condition_name("apps", "causal", "minified", "base") == (
        "apps.causal.minified.base"
    )
    assert names.run_name(gen, "abcd1234") == f"{gen} · abcd"


def test_fit_truncates_uniquely():
    long_a = "a" * 200
    long_b = "a" * 199 + "b"
    fitted = {names.fit(long_a), names.fit(long_b)}
    assert len(fitted) == 2
    assert all(len(n) <= names.MAX_NAME_LENGTH for n in fitted)
    assert names.fit("short") == "short"


def test_model_ref_name_is_derived_from_id_alone():
    model = "tinker/tinker://abc-123/sampler_weights/self_rec_u"
    assert names.model_ref_name(model) == names.model_ref_name(model)
    assert names.model_ref_name(model).startswith("ft-ft_self_rec_u-")
    assert names.model_ref_name(model) != names.model_ref_name(model + "2")


def _score(main, side=None):
    value = {"main_task_success": main}
    if side is not None:
        value["side_task_success"] = side
    return {"apps_scorer": SimpleNamespace(value=value)}


def _log(status, n_samples, dataset_samples, errors=0):
    samples = [
        SimpleNamespace(
            scores=_score("C", "C"),
            error="boom" if i < errors else None,
            metadata={},
            store={},
        )
        for i in range(n_samples)
    ]
    return SimpleNamespace(
        status=status,
        samples=samples,
        eval=SimpleNamespace(dataset=SimpleNamespace(samples=dataset_samples)),
    )


@pytest.mark.parametrize(
    "status, n, dataset, limit, complete",
    [
        ("success", 10, 10, None, True),
        ("success", 5, 10, 5, True),  # reached its limit
        ("success", 3, 10, 5, False),
        ("error", 10, 10, None, False),
        ("cancelled", 0, 10, None, False),
    ],
)
def test_eval_counts_completeness(status, n, dataset, limit, complete):
    counts = core.eval_counts(_log(status, n, dataset), limit)
    assert counts["complete"] is complete
    assert counts["expected_samples"] == (dataset if limit is None else limit)
    assert counts["samples"] == n


def test_eval_counts_basharena_uses_eval_mode():
    scores = {
        "main_task_success": SimpleNamespace(value=0.5),
        "side_task_success": SimpleNamespace(value="C"),
    }

    def sample(base_solution, error=None):
        return SimpleNamespace(
            scores=scores,
            error=error,
            metadata={"eval_mode": "attack"},
            store={"base_solution": base_solution},
        )

    # kept, then a zero-action trajectory and an errored one, which don't count
    samples = [sample("ls"), sample(" "), sample("ls", error="boom")]
    log = SimpleNamespace(
        status="success",
        samples=samples,
        eval=SimpleNamespace(dataset=SimpleNamespace(samples=3)),
    )
    assert core.eval_counts(log, None)["valid_samples"] == 1


def test_score_rates():
    samples = [
        SimpleNamespace(scores={**_score("C", "I"), "susp": SimpleNamespace(value=4)}),
        SimpleNamespace(scores={**_score("C", "C"), "susp": SimpleNamespace(value=6)}),
    ]
    rates = core._score_rates(samples)
    assert rates["apps_scorer/main_task_success"] == 1.0
    assert rates["apps_scorer/side_task_success"] == 0.5
    assert rates["susp"] == 5.0


def test_disabled_run_still_names_its_output():
    with core.StageRun("generate", "apps-x-test-attacks-p", config={}) as tracker:
        metadata = tracker.eval_metadata()
        tracker.log({"a": 1})
        tracker.summary({"b": 2})
    assert metadata["tracking"]["name"] == "apps-x-test-attacks-p"
    assert metadata["tracking"]["run_id"] is None


@pytest.fixture
def eval_file(tmp_path):
    path = tmp_path / "attacks.eval"
    path.write_bytes(b"not really a zip")
    return path


def test_untracked_input_is_recorded_by_hash(monkeypatch, eval_file):
    monkeypatch.setattr(core, "read_header", lambda path: {})
    resolved = core.resolve_input(eval_file)
    assert resolved.local_path == eval_file
    assert not resolved.tracked and not resolved.unlinked
    assert resolved.md5


def test_tracked_input_without_wandb_is_unlinked(monkeypatch, eval_file):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(core, "_api", lambda: None)
    monkeypatch.setattr(
        core,
        "read_header",
        lambda path: {"name": "gen", "run_id": "abc", "project": "p", "entity": "e"},
    )
    resolved = core.resolve_input(eval_file)
    assert resolved.name == "gen"
    assert resolved.unlinked and "unavailable" in resolved.note


def test_refs_must_be_exact_versions():
    with pytest.raises(ValueError, match="exact version"):
        core.resolve_input("wandb:me/proj/apps-x:latest")
    assert core.is_ref("wandb:me/proj/apps-x:v3")


def _hf_artifact(md5: str) -> SimpleNamespace:
    url = (
        "https://huggingface.co/datasets/org/repo/resolve/"
        + "a" * 40
        + "/untrusted-runs/x/v0/attacks.eval"
    )
    entry = SimpleNamespace(path="attacks.eval", ref=url)
    return SimpleNamespace(
        name="x:v1",
        qualified_name="e/p/x:v1",
        metadata={"md5": md5},
        manifest=SimpleNamespace(entries={"attacks.eval": entry}),
    )


def test_hf_references_download_through_huggingface_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, eval_file: Path
):
    import huggingface_hub

    calls: list[tuple[str, str, str, str]] = []

    def fake_download(repo_id: str, filename: str, repo_type: str, revision: str):
        calls.append((repo_id, filename, repo_type, revision))
        return str(eval_file)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.chdir(tmp_path)
    folder = core.download_artifact(_hf_artifact(core._md5_b64(eval_file)))
    assert (folder / "attacks.eval").read_bytes() == eval_file.read_bytes()
    assert calls == [
        ("org/repo", "untrusted-runs/x/v0/attacks.eval", "dataset", "a" * 40)
    ]
    with pytest.raises(ValueError, match="MD5"):
        core.download_artifact(_hf_artifact("not-the-md5"))


def test_stored_artifacts_download_through_wandb(tmp_path: Path):
    entry = SimpleNamespace(path="attacks.eval", ref=None)
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(entries={"attacks.eval": entry}),
        download=lambda: str(tmp_path),
    )
    assert core.download_artifact(artifact) == tmp_path


def test_unlinked_input_blocks_unless_allowed(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    unlinked = core.ResolvedInput(
        source="a.eval", local_path=Path("a.eval"), name="x", md5="m", tracked=True
    )
    unlinked.note = "producer run failed"
    with pytest.raises(core.UnlinkedInputError, match="producer run failed"):
        with core.StageRun("monitor", "x", config={}, inputs=[unlinked]):
            pass


def test_offline_fallback_refuses_to_drop_links(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(core.StageRun, "_init", lambda self, mode: None)
    linked = core.ResolvedInput(
        source="a.eval",
        local_path=Path("a.eval"),
        name="x",
        md5="m",
        tracked=True,
        artifact=object(),
    )
    with pytest.raises(core.UnlinkedInputError, match="offline"):
        with core.StageRun("monitor", "x", config={}, inputs=[linked]):
            pass

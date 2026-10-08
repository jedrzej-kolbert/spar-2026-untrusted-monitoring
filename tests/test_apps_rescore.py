"""Stored APPS submissions must keep the scores they were given.

Replays 29 stored generations (tests/data/apps_rescore_fixture.json), which
were generated and scored under control-arena 2.1.3, through AppsSetting's
scorer in the Docker sandbox.

- scoring="legacy" (the default) must reproduce the stored 2.1.3 scores
  exactly: a failure means bumping control-arena has changed APPS scoring.
- scoring="upstream" must give the fixture's `expected_scores`; the samples
  where those differ from 2.1.3 carry a `note` explaining why.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from control_arena import EvalMode

from lasr_labs_2025_control_project.scripts.helpers.rescore_apps_eval import rescore

pytestmark = pytest.mark.slow

FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "apps_rescore_fixture.json").read_text()
)
EXPECTED_KEY = {"legacy": "stored_scores_2_1_3", "upstream": "expected_scores"}


def _docker_available() -> bool:
    return bool(shutil.which("docker")) and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _docker_available(), reason="needs a running Docker daemon")
@pytest.mark.parametrize("scoring", ["legacy", "upstream"])
@pytest.mark.parametrize("mode", [EvalMode.HONEST, EvalMode.ATTACK])
def test_stored_submissions_rescore_as_expected(mode: EvalMode, scoring: str):
    samples = [s for s in FIXTURE["samples"] if s["eval_mode"] == mode.value]
    assert samples
    rescored = rescore(
        {s["id"]: s["completion"] for s in samples},
        mode,
        scoring=scoring,  # type: ignore[arg-type]
    )
    expected = EXPECTED_KEY[scoring]
    mismatches = {
        s["id"]: (s[expected], rescored[s["id"]])
        for s in samples
        if rescored[s["id"]] != s[expected]
    }
    assert not mismatches, f"expected vs rescored: {mismatches}"

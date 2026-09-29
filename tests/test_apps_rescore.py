"""Stored APPS submissions must score the same under the installed APPS scorer.

Replays 29 stored generations (tests/data/apps_rescore_fixture.json) through
apps_scorer in the Docker sandbox and compares against expected scores. The
fixture also keeps the original control-arena 2.1.3 scores; the samples whose
expected score differs from 2.1.3 carry a `note` explaining why. A failure here
means the scorer (or the sample/metadata plumbing feeding it) changed behaviour.
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


def _docker_available() -> bool:
    return bool(shutil.which("docker")) and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _docker_available(), reason="needs a running Docker daemon")
@pytest.mark.parametrize("mode", [EvalMode.HONEST, EvalMode.ATTACK])
def test_stored_submissions_rescore_as_expected(mode: EvalMode):
    samples = [s for s in FIXTURE["samples"] if s["eval_mode"] == mode.value]
    assert samples
    rescored = rescore({s["id"]: s["completion"] for s in samples}, mode)
    mismatches = {
        s["id"]: (s["expected_scores"], rescored[s["id"]])
        for s in samples
        if rescored[s["id"]] != s["expected_scores"]
    }
    assert not mismatches, f"expected vs rescored: {mismatches}"

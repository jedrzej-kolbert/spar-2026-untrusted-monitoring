"""BashArena generation, run in its own pinned environment.

BashArena needs control-arena 17, which the main project cannot install next to
its control-arena 2 pin. This package only generates: it writes ordinary `.eval`
logs, which the main project's SFT and monitor steps read as they read APPS logs.
"""

import sys
from pathlib import Path

RUNTIME = {"control-arena": "17.1.2", "inspect-ai": "0.3.201"}

# The main package is not installed here (it pins control-arena 2). The two
# modules this package uses from it, utils/tinker_provider.py and
# utils/basharena_views.py, need neither Control Arena nor a particular Inspect
# version, so they are imported from the checkout this file sits in.
PROJECT_SOURCE = Path(__file__).resolve().parents[3] / "src"
if not (PROJECT_SOURCE / "lasr_labs_2025_control_project").is_dir():
    raise RuntimeError(f"Main project source not found at {PROJECT_SOURCE}")
sys.path.append(str(PROJECT_SOURCE))


def check_runtime() -> dict[str, str]:
    from importlib.metadata import version

    installed = {name: version(name) for name in RUNTIME}
    if installed != RUNTIME:
        raise RuntimeError(f"BashArena runtime needs {RUNTIME}, found {installed}")
    return installed

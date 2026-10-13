"""BashArena generation, run in its own pinned environment.

BashArena needs control-arena 17, which the main project cannot install next to
its control-arena 2 pin. This package only generates: it writes ordinary `.eval`
logs, which the main project's SFT and monitor steps read as they read APPS logs.
"""

import sys
from importlib.metadata import version
from pathlib import Path

# The main package is not installed here (it pins control-arena 2). The two
# modules this package uses from it, utils/tinker_provider.py and
# utils/basharena_views.py, need neither Control Arena nor a particular Inspect
# version, so they are imported from the checkout this file sits in.
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))


def runtime_versions() -> dict[str, str]:
    """Record installed versions; pyproject.toml and uv.lock pin the runtime."""
    return {name: version(name) for name in ("control-arena", "inspect-ai")}

"""BashArena generation and export, run in their own pinned environment.

BashArena needs control-arena 17, which the main project cannot install next to
its control-arena 2 pin. This package holds everything that must run against the
newer version; the main project only reads the JSONL it writes.
"""

RUNTIME = {"control-arena": "17.1.2", "inspect-ai": "0.3.201"}


def check_runtime() -> dict[str, str]:
    from importlib.metadata import version

    installed = {name: version(name) for name in RUNTIME}
    if installed != RUNTIME:
        raise RuntimeError(f"BashArena runtime needs {RUNTIME}, found {installed}")
    return installed

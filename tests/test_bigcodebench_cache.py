import importlib

from inspect_ai.util import ExecResult

# The package re-exports the task function under the module's name.
bcb = importlib.import_module(
    "lasr_labs_2025_control_project.bigcodebench.bigcodebench"
)


class FakeSandbox:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def exec(self, cmd, timeout):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


async def test_creates_missing_cache_dir_and_reuses_result(tmp_path, monkeypatch):
    monkeypatch.setattr(bcb, "CACHE_DIR", tmp_path / "missing" / "cache")
    box = FakeSandbox(ExecResult(True, 0, "out", ""))
    monkeypatch.setattr(bcb, "sandbox", lambda: box)

    first = await bcb.cached_sandbox_exec(["python", "-c", "1"], timeout=10)
    second = await bcb.cached_sandbox_exec(["python", "-c", "1"], timeout=10)

    assert first.success and second.success and second.stdout == "out"
    assert box.calls == 1


async def test_timeout_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(bcb, "CACHE_DIR", tmp_path)
    box = FakeSandbox(TimeoutError())
    monkeypatch.setattr(bcb, "sandbox", lambda: box)

    result = await bcb.cached_sandbox_exec(["python", "-c", "1"], timeout=10)

    assert not result.success
    assert list(tmp_path.iterdir()) == []

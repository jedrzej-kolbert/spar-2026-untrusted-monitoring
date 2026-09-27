"""Offline regression check; run as a module. No API calls or generated-code execution."""

import asyncio
from unittest.mock import AsyncMock

from .contracts import EvaluatorResources, MonitorView
from .contracts import RepresentationKind as Kind
from .jammers import IdentityJammer, MinifierJammer, TCodeRewriteJammer, TSummaryJammer


def rejects(error, operation):
    try:
        operation()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


async def async_rejects(error, operation):
    try:
        await operation()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


async def check():
    view = MonitorView("Print the sum", "print(sum([1, 2]))", Kind.PYTHON_CODE, "p")
    benign = AsyncMock(return_value=view.content)
    call_t = AsyncMock(return_value="Sums the inputs")
    collusion, um, trusted = (AsyncMock(return_value=v) for v in (2.5, 8.0, 3.0))
    resources = EvaluatorResources(benign, call_t, collusion, um, trusted, "fake/T")
    summary = TSummaryJammer(resources.call_t, "fake/T")
    transformed, provenance = await summary(view)
    assert transformed.content == "Sums the inputs"
    assert view.content == "print(sum([1, 2]))"
    assert provenance.model_used == "fake/T"
    assert (await IdentityJammer()(view))[0] == view
    assert (await MinifierJammer()(view))[0].task_description == view.task_description
    await async_rejects(
        ValueError, lambda: TSummaryJammer(AsyncMock(return_value=""), "T")(view)
    )
    await async_rejects(
        SyntaxError,
        lambda: TCodeRewriteJammer(AsyncMock(return_value="not code!"), "T")(view),
    )
    assert (
        await TCodeRewriteJammer(
            AsyncMock(return_value="```python\nprint(3)\n```"), "T"
        )(view)
    )[0].content == "print(3)"
    print("Blue team regression checks passed (offline, no sandbox execution).")


if __name__ == "__main__":
    asyncio.run(check())

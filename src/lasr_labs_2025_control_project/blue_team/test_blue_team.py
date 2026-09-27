"""Offline regression check; run as a module. No API calls or generated-code execution."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from .contracts import EvaluatorResources, MonitorView
from .contracts import RepresentationKind as Kind
from .jammers import IdentityJammer, MinifierJammer, TCodeRewriteJammer, TSummaryJammer
from .policy import FixedMonitoringPolicy, make_resources, parse_score


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
    summary = TSummaryJammer(call_t, "fake/T")
    scores = await FixedMonitoringPolicy(um_jammer=summary)(view, resources)
    assert (
        scores.untrusted_collusion_score,
        scores.untrusted_suspiciousness_score,
        scores.trusted_suspiciousness_score,
    ) == (2.5, 8, 3)
    assert um.await_args.args[0].content == "Sums the inputs"
    assert collusion.await_args.args[0] == um.await_args.args[0]
    assert trusted.await_args.args[0] == view and view.content == "print(sum([1, 2]))"
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
    for text in (
        "oops",
        "<score>nan</score>",
        "<score>11</score>",
        "<score>1</score><score>2</score>",
    ):
        rejects(ValueError, lambda text=text: parse_score(text))
    await async_rejects(
        ValueError,
        lambda: FixedMonitoringPolicy()(
            view,
            replace(resources, um_suspiciousness=AsyncMock(return_value=float("nan"))),
        ),
    )

    from inspect_ai.model import ModelOutput

    seen = []

    class Model:
        def __init__(self, name):
            self.name = name

        async def generate(self, messages, config):
            seen.append((self.name, messages))
            content = {
                "mock/U": "print(3)",
                "mock/UM": "<score>8</score>",
                "mock/T": "<score>3</score>",
                "mock/C": "YES",
            }[self.name]
            return ModelOutput.from_content(self.name, content)

    with (
        patch("inspect_ai.model.get_model", side_effect=Model),
        patch(
            "lasr_labs_2025_control_project.scripts.monitor_solutions_inspect.get_answer_logprobs_from_openai_completion",
            return_value={"YES": -1.0, "NO": -3.0},
        ),
    ):
        real = make_resources(
            {"p": view.task_description},
            generator_model="mock/U",
            um_model="mock/UM",
            trusted_model="mock/T",
            collusion_model="mock/C",
        )
        await async_rejects(KeyError, lambda: real.sample_benign("unknown"))
        assert not seen
        assert await real.sample_benign("p") == "print(3)"
        actual = await FixedMonitoringPolicy()(view, real)
        assert (
            actual.untrusted_collusion_score,
            actual.untrusted_suspiciousness_score,
            actual.trusted_suspiciousness_score,
        ) == (2, 8, 3)
        assert [name for name, _ in seen] == ["mock/U", "mock/C", "mock/UM", "mock/T"]

    with patch(
        "lasr_labs_2025_control_project.scripts.monitor_solutions_inspect.get_answer_logprobs_from_openai_completion",
        return_value={"YES": -1.0, "NO": None},
    ):
        await async_rejects(TypeError, lambda: real.um_collusion(view))
    with patch(
        "lasr_labs_2025_control_project.utils.tinker_provider.collusion_yes_no_logprobs",
        new=AsyncMock(return_value={"YES": -1.0, "NO": -4.0}),
    ) as forced:
        tinker_resources = make_resources(
            {"p": "task"},
            generator_model="mock/U",
            um_model="mock/UM",
            trusted_model="mock/T",
            collusion_model="tinker/fake",
        )
        assert await tinker_resources.um_collusion(view) == 3
        assert forced.await_args.kwargs["model_name"] == "tinker/fake"

    print("Blue team regression checks passed (offline, no sandbox execution).")


if __name__ == "__main__":
    asyncio.run(check())

"""What the monitors see: the code after comment stripping and optional signal jamming.

Mirrors the evaluation pipeline so training rewards match reported metrics:
- ``none``: ``monitor_solutions_inspect`` strips comments (``remove_comments: true`` in
  the monitor configs) and wraps the code in a Python fence.
- ``minify``: ``transform_solutions_inspect.minify_solution``; the monitor reads the
  minified text as is.
- ``paraphrase``: ``transform_solutions_inspect.summarize_solution`` (the LLM
  summarizer); comments are stripped before the summarizer sees the code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from lasr_labs_2025_control_project.utils.utils import (
    apply_python_code_format,
    minify_code,
    remove_comments,
)

from .core import AppsProblem

logger = logging.getLogger(__name__)

ChannelKind = Literal["none", "minify", "paraphrase"]


def _strip_comments(code: str) -> str:
    try:
        return remove_comments(code)
    except Exception:  # malformed code: tokenize can fail; show it unchanged
        return code


@dataclass
class MonitorViewChannel:
    kind: ChannelKind = "none"
    remove_comments: bool = True
    paraphrase_model: str | None = None
    paraphrase_system_prompt: str | None = None
    paraphrase_base_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("none", "minify", "paraphrase"):
            raise ValueError(f"unknown channel: {self.kind}")
        if self.kind == "paraphrase" and not (
            self.paraphrase_model
            and self.paraphrase_system_prompt
            and self.paraphrase_base_prompt
        ):
            raise ValueError("paraphrase channel requires a model and both prompts")

    async def transform(self, problem: AppsProblem, code: str) -> str:
        if self.kind == "none":
            view = _strip_comments(code) if self.remove_comments else code
            return apply_python_code_format(view)
        if self.kind == "minify":
            try:
                return minify_code(code)
            except SyntaxError:
                # Unparseable code: the receiver sees nothing rather than raw source.
                return ""
        return await self._paraphrase(code)

    async def _paraphrase(self, code: str) -> str:
        from inspect_ai.model import (
            ChatMessageSystem,
            ChatMessageUser,
            GenerateConfig,
            get_model,
        )

        from lasr_labs_2025_control_project.utils import tinker_provider  # noqa: F401

        assert self.paraphrase_system_prompt and self.paraphrase_base_prompt
        model = get_model(self.paraphrase_model)
        solution = _strip_comments(code) if self.remove_comments else code
        output = await model.generate(
            [
                ChatMessageSystem(content=self.paraphrase_system_prompt),
                ChatMessageUser(
                    content=self.paraphrase_base_prompt.format(solution=solution)
                ),
            ],
            config=GenerateConfig(reasoning_effort="minimal"),
        )
        if not output.completion.strip():
            raise ValueError("paraphraser returned an empty completion")
        return output.completion

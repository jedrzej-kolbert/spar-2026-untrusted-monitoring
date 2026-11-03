"""Judge prompts learned by centralized ICRL runs, so a pipeline monitor arm can use them.

``scripts/rewrite_solutions_inspect.py`` adds ``JUDGE_<LABEL>`` when it rewrites the test
attacks with a centralized winner: the search's own judge prompt filled with the run's
learned instructions. Test scoring then sets it as the base arm's collusion prompt.
"""

from lasr_labs_2025_control_project.prompts.rl_prompts import (
    RL_CENTRAL_JUDGE_SYSTEM_PROMPT,
)


def _judge(instructions: str) -> str:
    return RL_CENTRAL_JUDGE_SYSTEM_PROMPT.format(decoder_instructions=instructions)

"""Judge prompts learned by centralized ICRL runs, so a pipeline monitor arm can use them.

``experiments/collusion-rl-v2/test.sh score`` adds ``JUDGE_<LABEL>`` for each scored
run: the search's own judge prompt filled with the run's learned instructions.
"""

from lasr_labs_2025_control_project.prompts.rl_prompts import (
    RL_CENTRAL_JUDGE_SYSTEM_PROMPT,
)


def _judge(instructions: str) -> str:
    return RL_CENTRAL_JUDGE_SYSTEM_PROMPT.format(decoder_instructions=instructions)

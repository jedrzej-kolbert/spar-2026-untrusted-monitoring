"""BashArena agent prompts, selectable by name in generator configs.

The policies fill {side_task_description} / {setting_description} themselves, so these
are BashArena's own defaults, unchanged.
"""

from control_arena.settings.bash_arena.policies.attack_policy import (
    DEFAULT_SYSTEM_PROMPT as BASHARENA_ATTACK_SYSTEM_PROMPT,
)
from control_arena.settings.bash_arena.policies.honest_policy import (
    DEFAULT_SYSTEM_PROMPT as BASHARENA_HONEST_SYSTEM_PROMPT,
)

__all__ = ["BASHARENA_ATTACK_SYSTEM_PROMPT", "BASHARENA_HONEST_SYSTEM_PROMPT"]

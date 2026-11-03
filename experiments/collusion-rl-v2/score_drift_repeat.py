"""Score the exact monitor prompt 10 times per mode for a few attacks; is the 28 Oct
value inside today's spread?"""

import asyncio
import math

from dotenv import load_dotenv

load_dotenv("/home/jkk/spar-2026-untrusted-monitoring/.env")

from inspect_ai.log import read_eval_log  # noqa: E402

from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name  # noqa: E402
from lasr_labs_2025_control_project.utils.tinker_provider import (  # noqa: E402
    collusion_yes_no_logprobs,
)
from lasr_labs_2025_control_project.utils.utils import format_solution, remove_comments  # noqa: E402

OLD = (
    "/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/v41-pilot/experiments/apps/monitor/"
    "apps-262144_ft_a1611f-test-attacks-self_rec_subtle_attack.mon-262144_ft_a1611f-self_rec_collusion-c0d2a67a6489/attacks.eval"
)
NEW = (
    "experiments/apps/monitor/apps-262144_ft_a1611f-test-attacks-self_rec_subtle_attack"
    ".mon-262144_ft_a1611f-self_rec_collusion-eaa2a61ef359/attacks.eval"
)
MODEL = "tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"
SYSTEM = get_prompt_by_name("SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT")
USER = get_prompt_by_name("MONITOR_USER_PROMPT")
N = 10
import lasr_labs_2025_control_project.utils.tinker_provider as _tp  # noqa: E402

print("scoring code from", _tp.__file__)


def stored(path: str) -> dict:
    return {
        s.id: (s, s.scores["untrusted_collusion_score"].value)
        for s in read_eval_log(path).samples or []
    }


async def main() -> None:
    old, new = stored(OLD), stored(NEW)
    for sample_id in list(new)[:5]:
        sample = old[sample_id][0]
        meta = sample.metadata
        code = format_solution(
            remove_comments(meta["base_solution"]),
            meta.get("setting_name"),
            meta.get("solution_view"),
        )
        user = USER.format(
            problem_statement=meta["main_task_description"], solution=code
        )
        print(
            f"sample {sample_id}: 28 Oct {old[sample_id][1]:.3f}, 3 Nov pipeline {new[sample_id][1]:.3f}"
        )
        for single in (True, False):
            gaps, sums = [], []
            for _ in range(N):
                out = await collusion_yes_no_logprobs(
                    MODEL, SYSTEM, user, single_prefill=single
                )
                gaps.append(out["YES"] - out["NO"])
                sums.append(math.exp(out["YES"]) + math.exp(out["NO"]))
            inside = min(gaps) <= old[sample_id][1] <= max(gaps)
            print(
                f"  {'one request ' if single else 'two requests'}: YES-NO min {min(gaps):.3f} "
                f"max {max(gaps):.3f} mean {sum(gaps) / N:.3f} distinct {len(set(round(g, 4) for g in gaps))}; "
                f"P(YES)+P(NO) {min(sums):.3f}-{max(sums):.3f}; 28 Oct value inside: {inside}"
            )


asyncio.run(main())

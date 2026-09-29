import math

from lasr_labs_2025_control_project.scripts.monitor_solutions_inspect import (
    collusion_score,
)

NAN = float("nan")


def test_finite_logprobs_give_difference():
    assert collusion_score(-0.5, -2.0) == 1.5


def test_missing_logprob_gives_nan():
    assert math.isnan(collusion_score(NAN, -1.0))
    assert math.isnan(collusion_score(-1.0, NAN))
    assert math.isnan(collusion_score(NAN, NAN))


def test_both_neg_inf_gives_nan():
    assert math.isnan(collusion_score(-math.inf, -math.inf))

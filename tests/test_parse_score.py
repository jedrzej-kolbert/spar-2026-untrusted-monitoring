from lasr_labs_2025_control_project.scripts.monitor_solutions_inspect import (
    parse_score,
)


def test_integer_score():
    assert parse_score("<reasoning>fine</reasoning><score>7</score>") == 7


def test_decimal_score_is_kept():
    assert parse_score("<score>8.9</score>") == 8.9


def test_unparseable_score_is_missing():
    assert parse_score("<score>high</score>") is None
    assert parse_score("no score tag") is None
    assert parse_score(None) is None

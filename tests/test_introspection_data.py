import json

import numpy as np
import pytest

from lasr_labs_2025_control_project.finetuning.introspection_data import (
    extract_property,
    load_rows,
    score_accuracy,
    training_examples,
)
from lasr_labs_2025_control_project.scripts.compare_self_rec_auc import (
    paired_bootstrap_delta,
)


@pytest.mark.parametrize(
    "prop, response, expected",
    [
        ("first_character", " crimson teal", "c"),
        ("second_character", "crimson", "r"),
        ("third_character", "ab", None),
        ("first_word", "crimson teal navy", "crimson"),
        ("second_word", "crimson teal navy", "teal"),
        ("third_word", "crimson teal", None),
        ("starts_with_vowel", "Orange", "true"),
        ("starts_with_vowel", "teal", "false"),
        ("starts_with_vowel", "", None),
        ("among_a_or_c", "C", "true"),
        ("among_a_or_c", "B", "false"),
        ("among_b_or_d", "d", "true"),
        ("identity", " B\n", "B"),
        ("identity", "  ", None),
    ],
)
def test_extract_property(prop, response, expected):
    assert extract_property(prop, response) == expected


def test_extract_ethical_stance_matches_target():
    assert extract_property("ethical_stance", " b", target="B") == "true"
    assert extract_property("ethical_stance", "A", target="B") == "false"
    assert extract_property("ethical_stance", "A", target=None) is None


def test_extract_unknown_property_raises():
    with pytest.raises(ValueError):
        extract_property("last_word", "x")


def _row(task, prop="first_word"):
    return {
        "task": task,
        "behavioral_property": prop,
        "object_level_prompt": [{"role": "user", "content": "obj"}],
        "hypothetical_prompt": [{"role": "user", "content": "meta"}],
    }


def test_training_examples_drops_undefined_and_appends_label():
    rows = [_row("t"), _row("t")]
    out = training_examples(rows, ["crimson", None])
    assert out == [
        {
            "messages": [
                {"role": "user", "content": "meta"},
                {"role": "assistant", "content": "crimson"},
            ],
            "task": "t",
        }
    ]
    # the trainer's prompt list must not be mutated by appending the label
    assert rows[0]["hypothetical_prompt"] == [{"role": "user", "content": "meta"}]


def test_score_accuracy_against_own_answers_and_mode_baseline():
    rows = [_row("a")] * 4 + [_row("b")] * 2
    obj = ["x", "x", "x", "y", "p", None]
    meta = ["X ", "x", "y", "y", "q", "p"]
    result = score_accuracy(rows, obj, meta)
    assert result["n"] == 5
    assert result["per_task"]["a"] == {"n": 4, "accuracy": 0.75, "mode_baseline": 0.75}
    assert result["per_task"]["b"] == {"n": 1, "accuracy": 0.0, "mode_baseline": 1.0}
    assert result["accuracy"] == pytest.approx(3 / 5)
    assert result["mode_baseline"] == pytest.approx(4 / 5)


def test_load_rows_caps_each_file_and_tags_task(tmp_path):
    for name, n in (("colors_long_first_word", 5), ("animals_long_first_word", 2)):
        with (tmp_path / f"{name}.jsonl").open("w") as f:
            for i in range(n):
                f.write(json.dumps({"i": i}) + "\n")
    rows = load_rows(tmp_path, per_file=3, seed=0)
    tasks = [r["task"] for r in rows]
    assert tasks.count("colors_long_first_word") == 3
    assert tasks.count("animals_long_first_word") == 2
    assert load_rows(tmp_path, per_file=3, seed=0) == rows


def test_paired_bootstrap_delta_sign_and_ci():
    rng = np.random.default_rng(1)
    labels = np.array([1] * 100 + [0] * 100)
    base = rng.normal(size=200) + 0.2 * labels
    better = base + 1.5 * labels
    delta, low, high = paired_bootstrap_delta(labels, base, better, n_boot=300)
    assert delta > 0
    assert 0 < low <= delta <= high
    same, low, high = paired_bootstrap_delta(labels, base, base, n_boot=50)
    assert same == low == high == 0

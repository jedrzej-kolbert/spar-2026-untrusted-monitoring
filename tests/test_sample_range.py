"""YAML sample ranges reuse Inspect's native slicing, without regenerating a prefix."""

import pytest
from inspect_ai._eval.task.util import slice_dataset
from inspect_ai.dataset import MemoryDataset, Sample

from lasr_labs_2025_control_project.utils.config_loader import (
    RunLabel,
    _build_base_run_config,
)


def test_native_sample_range(tmp_path):
    config = _build_base_run_config(
        RunLabel.ATTACKS, {"save_path": "new.eval"}, tmp_path, {"limit": [2, 5]}
    )
    dataset = MemoryDataset([Sample(id=i, input=str(i)) for i in range(6)])
    selected = slice_dataset(dataset, limit=config.limit, sample_id=None)
    assert [s.id for s in selected] == [2, 3, 4]
    assert config.limit == (2, 5)


@pytest.mark.parametrize("limit", [[-1, 2], [2, 2], [3, 2], [0], [0, "2"], [False, 2]])
def test_bad_range(limit, tmp_path):
    with pytest.raises(ValueError, match="limit range"):
        _build_base_run_config(
            RunLabel.ATTACKS, {"save_path": "new.eval"}, tmp_path, {"limit": limit}
        )

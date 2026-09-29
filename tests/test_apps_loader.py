"""AppsSetting's JSONL loader must match codeparrot/apps' own loading script.

datasets>=4 can no longer run apps.py, so AppsSetting reads the raw JSONL the
script wrapped. This runs the script's row generator (the code datasets<4
executed) on the same file and checks both paths yield identical samples.
Downloads the APPS files on first run; marked slow.
"""

import importlib.util
from types import SimpleNamespace

import pytest
from control_arena import EvalMode
from huggingface_hub import hf_hub_download

from lasr_labs_2025_control_project.settings.setting import AppsSetting

pytestmark = pytest.mark.slow


def _script_rows(split: str) -> list[dict]:
    script = hf_hub_download("codeparrot/apps", "apps.py", repo_type="dataset")
    spec = importlib.util.spec_from_file_location("codeparrot_apps", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The default "all" config, without building a DatasetBuilder (which would
    # need the removed script-loading machinery).
    builder = SimpleNamespace(config=module.AppsCode.BUILDER_CONFIGS[0])
    assert builder.config.name == module.AppsCode.DEFAULT_CONFIG_NAME
    path = hf_hub_download("codeparrot/apps", f"{split}.jsonl", repo_type="dataset")
    return [row for _, row in module.AppsCode._generate_examples(builder, path)]


@pytest.mark.parametrize("split", ["train", "test"])
def test_jsonl_loader_matches_apps_script(split: str):
    rows = _script_rows(split)
    setting = AppsSetting(original_dataset_split=split, shuffle=False)
    for mode in (EvalMode.HONEST, EvalMode.ATTACK):
        to_sample = setting.record_to_sample_factory(mode)
        # record_to_sample returns [] for skipped problems, else one Sample.
        expected = [s for row in rows if not isinstance(s := to_sample(row), list)]
        actual = list(setting.load_apps_split(split, mode))
        assert len(actual) == len(expected) > 0
        for old, new in zip(expected, actual, strict=True):
            assert (old.id, old.input, old.metadata) == (
                new.id,
                new.input,
                new.metadata,
            )

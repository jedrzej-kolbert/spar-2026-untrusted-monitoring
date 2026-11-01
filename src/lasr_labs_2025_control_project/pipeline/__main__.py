"""Entrypoint: ``uv run python -m lasr_labs_2025_control_project.pipeline experiment=...``"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from lasr_labs_2025_control_project.pipeline.run import run_pipeline
from lasr_labs_2025_control_project.pipeline.schema import Config, register
from lasr_labs_2025_control_project.tracking import PipelineContext

CONF_DIR = str(Path(__file__).resolve().parents[3] / "conf")

register()


@hydra.main(version_base="1.3", config_path=CONF_DIR, config_name="config")
def main(raw: DictConfig) -> None:
    load_dotenv()
    cfg: Config = OmegaConf.to_object(raw)  # type: ignore[assignment]
    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.runtime.output_dir)
    # One W&B group per invocation (a multirun's jobs share it).
    batch = run_dir.parent.name if hydra_cfg.mode == RunMode.MULTIRUN else run_dir.name
    os.environ.setdefault("WANDB_RUN_GROUP", f"{cfg.name}-{batch}")
    hydra_dir = (
        run_dir / hydra_cfg.output_subdir if hydra_cfg.output_subdir else run_dir
    )
    pipeline = PipelineContext(
        name=cfg.name,
        config=OmegaConf.to_container(raw, resolve=True),  # type: ignore[arg-type]
        config_files=[
            path
            for path in (hydra_dir / "config.yaml", hydra_dir / "overrides.yaml")
            if path.exists()
        ],
        tags=[cfg.name, *cfg.wandb_tags],
    )
    run_pipeline(cfg, pipeline)


if __name__ == "__main__":
    main()

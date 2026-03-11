from __future__ import annotations

import argparse
import json
from pathlib import Path

from alice import PipelineConfig, run_pipeline
from steganalysis.lclsmr import LCLSMRConfig
from steganalysis.srnet import SRNetConfig


def _load_config(path: Path) -> PipelineConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("data_root", "feature_root", "feature_model_root", "run_root"):
        if key in raw and raw[key] is not None:
            raw[key] = Path(raw[key])

    if "lclsmr" in raw:
        for key in ("data_root", "feature_root", "output_dir"):
            if key in raw["lclsmr"] and raw["lclsmr"][key] is not None:
                raw["lclsmr"][key] = Path(raw["lclsmr"][key])
    lclsmr = LCLSMRConfig(**raw.pop("lclsmr")) if "lclsmr" in raw else LCLSMRConfig()

    if "srnet" in raw:
        for key in ("data_root", "output_dir", "pretrained_checkpoint"):
            if key in raw["srnet"] and raw["srnet"][key] is not None:
                raw["srnet"][key] = Path(raw["srnet"][key])
    srnet = SRNetConfig(**raw.pop("srnet")) if "srnet" in raw else SRNetConfig()
    return PipelineConfig(lclsmr=lclsmr, srnet=srnet, **raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Alice detector pipeline from a JSON config.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_pipeline(_load_config(args.config))


if __name__ == "__main__":
    main()

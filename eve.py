from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
import os
from pathlib import Path
import subprocess
import sys
import torch
from steganalysis.lclsmr import LCLSMRConfig, detect as detect_lclsmr
from steganalysis.srnet import SRNetConfig
from steganalysis.srnet import SRNET_CONFIG_JSON_ENV
from steganalysis.srnet import detect as detect_srnet
from utils.files import save_json


@dataclass(slots=True)
class PipelineConfig:
    data_root: Path = Path("data/BOSSbase_1.01")
    cover_dir_name: str = "cover"
    stego_dir_name: str = "changeable-sweltering-draft"
    detectors: list[str] = field(default_factory=lambda: ["lclsmr", "srnet"])
    feature_root: Path = Path("data/features")
    feature_model_root: Path = Path("data/features/models")
    run_root: Path = Path("runs")
    image_extension: str = ".pgm"
    validation_suffix: str | None = "9"
    max_train_files: int | None = 1000
    max_val_files: int | None = 100
    lclsmr: LCLSMRConfig = field(default_factory=LCLSMRConfig)
    srnet: SRNetConfig = field(default_factory=SRNetConfig)


CONFIG = PipelineConfig()


def resolve_pipeline_paths(config: PipelineConfig) -> dict[str, Path]:
    cover_dir_path = config.data_root / config.cover_dir_name
    stego_dir_path = config.data_root / config.stego_dir_name
    feature_root = config.feature_root / config.data_root.name
    linear_output_dir = config.feature_model_root / f"{stego_dir_path.name}_linear"
    srnet_output_dir = config.run_root / f"srnet_{stego_dir_path.name}"
    return {
        "cover_dir_path": cover_dir_path,
        "stego_dir_path": stego_dir_path,
        "cover_feature_dir": feature_root / config.cover_dir_name,
        "stego_feature_dir": feature_root / config.stego_dir_name,
        "linear_output_dir": linear_output_dir,
        "srnet_output_dir": srnet_output_dir,
    }


def _distributed_launch_active() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _cuda_visible_devices(gpu_devices: tuple[str, ...]) -> str:
    indices = [str(torch.device(device).index or 0) for device in gpu_devices]
    return ",".join(indices)


def _all_cuda_devices() -> tuple[str, ...] | None:
    if not torch.cuda.is_available():
        return None
    return tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))


def run_srnet_training(config: SRNetConfig) -> None:
    gpu_devices = _all_cuda_devices()

    if gpu_devices is None or len(gpu_devices) <= 1 or _distributed_launch_active():
        detect_srnet(config)
        return

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpu_devices)}",
        "-m",
        "steganalysis.srnet",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices(gpu_devices)
    env[SRNET_CONFIG_JSON_ENV] = json.dumps(asdict(config), default=str)
    subprocess.run(command, check=True, env=env)


def run_lclsmr_detector(config: PipelineConfig, paths: dict[str, Path]) -> Path:
    lclsmr_config = replace(
        config.lclsmr,
        data_root=config.data_root,
        cover_dir_name=config.cover_dir_name,
        stego_dir_name=config.stego_dir_name,
        feature_root=config.feature_root / config.data_root.name,
        output_dir=paths["linear_output_dir"],
        image_suffix=config.image_extension,
        fixed_val_suffix=config.validation_suffix,
        max_train_pairs=config.max_train_files,
        max_val_pairs=config.max_val_files,
    )
    detect_lclsmr(lclsmr_config)
    return paths["linear_output_dir"]


def run_srnet_detector(config: PipelineConfig, paths: dict[str, Path]) -> Path:
    srnet_config = replace(
        config.srnet,
        data_root=config.data_root,
        cover_dir_name=config.cover_dir_name,
        stego_dir_name=paths["stego_dir_path"].name,
        output_dir=paths["srnet_output_dir"],
        extensions=(config.image_extension,),
        fixed_val_suffix=config.validation_suffix,
        max_train_pairs=config.max_train_files,
        max_val_pairs=config.max_val_files,
    )
    run_srnet_training(srnet_config)
    return paths["srnet_output_dir"]


def run_pipeline(config: PipelineConfig = CONFIG) -> dict[str, Path]:
    paths = resolve_pipeline_paths(config)
    save_json(config.run_root / f"pipeline_{paths['stego_dir_path'].name}.json", asdict(config))

    detector_runners = {
        "lclsmr": lambda: run_lclsmr_detector(config, paths),
        "srnet": lambda: run_srnet_detector(config, paths),
    }

    outputs = dict(paths)
    for detector_name in config.detectors:
        runner = detector_runners.get(detector_name)
        if runner is None:
            available = ", ".join(sorted(detector_runners))
            raise ValueError(f"Unknown detector '{detector_name}'. Expected one of: {available}")
        outputs[f"{detector_name}_output"] = runner()

    if "lclsmr" in config.detectors:
        print(f"Linear report: {paths['linear_output_dir'] / 'classification_report.txt'}")
    if "srnet" in config.detectors:
        print(f"SRNet best report: {paths['srnet_output_dir'] / 'classification_report_best.txt'}")
    return outputs


def main() -> None:
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()

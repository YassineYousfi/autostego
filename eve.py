from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, UTC
import os
from pathlib import Path
import subprocess
import sys
import torch
from steganalysis.fusion import FusionConfig, detect as detect_fusion
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
    run_tag: str | None = None
    lclsmr: LCLSMRConfig = field(default_factory=LCLSMRConfig)
    srnet: SRNetConfig = field(default_factory=SRNetConfig)


CONFIG = PipelineConfig()


def resolve_pipeline_paths(config: PipelineConfig, run_tag: str) -> dict[str, Path]:
    cover_dir_path = config.data_root / config.cover_dir_name
    stego_dir_path = config.data_root / config.stego_dir_name
    feature_root = config.feature_root / config.data_root.name
    linear_output_dir = config.feature_model_root / f"{stego_dir_path.name}_linear_{run_tag}"
    srnet_output_dir = config.run_root / f"srnet_{stego_dir_path.name}_{run_tag}"
    fusion_output_dir = config.run_root / f"fusion_{stego_dir_path.name}_{run_tag}"
    return {
        "cover_dir_path": cover_dir_path,
        "stego_dir_path": stego_dir_path,
        "cover_feature_dir": feature_root / config.cover_dir_name,
        "stego_feature_dir": feature_root / config.stego_dir_name,
        "linear_output_dir": linear_output_dir,
        "srnet_output_dir": srnet_output_dir,
        "fusion_output_dir": fusion_output_dir,
    }


def resolve_run_tag(config: PipelineConfig) -> str:
    if config.run_tag:
        return config.run_tag
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


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


def run_fusion_detector(config: PipelineConfig, paths: dict[str, Path]) -> Path:
    linear_output = paths.get("lclsmr_output")
    if linear_output is None:
        linear_output = run_lclsmr_detector(config, paths)
        paths["lclsmr_output"] = linear_output

    srnet_output = paths.get("srnet_output")
    if srnet_output is None:
        srnet_output = run_srnet_detector(config, paths)
        paths["srnet_output"] = srnet_output

    fusion_config = FusionConfig(
        data_root=config.data_root,
        cover_dir_name=config.cover_dir_name,
        stego_dir_name=config.stego_dir_name,
        feature_root=config.feature_root / config.data_root.name,
        linear_model_dir=linear_output,
        srnet_output_dir=srnet_output,
        output_dir=paths["fusion_output_dir"],
        image_suffix=config.image_extension,
        fixed_val_suffix=config.validation_suffix,
        max_train_pairs=config.max_train_files,
        max_val_pairs=config.max_val_files,
    )
    detect_fusion(fusion_config)
    return paths["fusion_output_dir"]


def save_run_summary(config: PipelineConfig, outputs: dict[str, Path], run_tag: str) -> None:
    stego_name = outputs["stego_dir_path"].name
    summary = {
        "stego_dir": stego_name,
        "run_tag": run_tag,
        "detectors": config.detectors,
        "outputs": {key: str(value) for key, value in outputs.items() if isinstance(value, Path)},
    }
    save_json(config.run_root / f"result_{stego_name}_{run_tag}.json", summary)
    summary_lines = [f"run_tag: {run_tag}"]
    for detector_name in config.detectors:
        output_path = outputs.get(f"{detector_name}_output")
        if output_path is not None:
            summary_lines.append(f"{detector_name}: {output_path}")
    (config.run_root / f"result_{stego_name}_{run_tag}.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def run_pipeline(config: PipelineConfig = CONFIG) -> dict[str, Path]:
    run_tag = resolve_run_tag(config)
    paths = resolve_pipeline_paths(config, run_tag)
    save_json(config.run_root / f"pipeline_{paths['stego_dir_path'].name}_{run_tag}.json", asdict(config))

    detector_runners = {
        "lclsmr": lambda: run_lclsmr_detector(config, paths),
        "srnet": lambda: run_srnet_detector(config, paths),
        "fusion": lambda: run_fusion_detector(config, paths),
    }

    outputs = dict(paths)
    for detector_name in config.detectors:
        runner = detector_runners.get(detector_name)
        if runner is None:
            available = ", ".join(sorted(detector_runners))
            raise ValueError(f"Unknown detector '{detector_name}'. Expected one of: {available}")
        outputs[f"{detector_name}_output"] = runner()

    save_run_summary(config, outputs, run_tag)
    if "lclsmr" in config.detectors:
        print(f"Linear report: {paths['linear_output_dir'] / 'classification_report.txt'}")
    if "srnet" in config.detectors:
        print(f"SRNet best report: {paths['srnet_output_dir'] / 'classification_report_best.txt'}")
    if "fusion" in config.detectors:
        print(f"Fusion report: {paths['fusion_output_dir'] / 'classification_report.txt'}")
    return outputs


def main() -> None:
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from steganalysis.lclsmr import LCLSMRConfig, train_classifier
from steganalysis.srnet import SRNetConfig
from steganalysis.srnet import train as train_srnet
from steganalysis.srm import SRMDirConfig, extract_dir
from steganography.embed_dir import EmbedDirConfig, embed_dir, generate_stego_dir
from utils.files import save_json


@dataclass(slots=True)
class PipelineConfig:
    data_root: Path = Path("data/BOSSbase_1.01")
    cover_dir_name: str = "cover"
    algorithm: str = "hill"
    payload: float = 0.4
    stego_dir: Path | None = None
    feature_root: Path = Path("data/features")
    feature_model_root: Path = Path("data/features/models")
    run_root: Path = Path("runs")
    image_extension: str = ".pgm"
    validation_suffix: str | None = "9"
    max_train_files: int | None = 1000
    max_val_files: int | None = 100
    embed_workers: int | None = None
    srm_gpu_devices: tuple[str, ...] | None = None
    lclsmr: LCLSMRConfig = field(default_factory=LCLSMRConfig)
    srnet: SRNetConfig = field(default_factory=SRNetConfig)


CONFIG = PipelineConfig()


def resolve_paths(config: PipelineConfig) -> dict[str, Path]:
    cover_dir = config.data_root / config.cover_dir_name
    stego_dir = config.stego_dir or generate_stego_dir(config.data_root)
    feature_root = config.feature_root / config.data_root.name
    linear_output = config.feature_model_root / f"{stego_dir.name}_linear"
    srnet_output = config.run_root / f"srnet_{stego_dir.name}"
    return {
        "cover_dir": cover_dir,
        "stego_dir": stego_dir,
        "cover_features": feature_root / config.cover_dir_name,
        "stego_features": feature_root / stego_dir.name,
        "linear_output": linear_output,
        "srnet_output": srnet_output,
    }


def run_pipeline(config: PipelineConfig = CONFIG) -> dict[str, Path]:
    paths = resolve_paths(config)
    save_json(config.run_root / f"pipeline_{paths['stego_dir'].name}.json", asdict(config))

    lclsmr_config = replace(
        config.lclsmr,
        cover_feature_dir=paths["cover_features"],
        stego_feature_dir=paths["stego_features"],
        output_dir=paths["linear_output"],
        fixed_val_suffix=config.validation_suffix,
        max_train_pairs=config.max_train_files,
        max_val_pairs=config.max_val_files,
    )
    srnet_config = replace(
        config.srnet,
        data_root=config.data_root,
        cover_dir=config.cover_dir_name,
        stego_dir=paths["stego_dir"].name,
        output_dir=paths["srnet_output"],
        extensions=(config.image_extension,),
        fixed_val_suffix=config.validation_suffix,
        max_train_pairs=config.max_train_files,
        max_val_pairs=config.max_val_files,
    )

    embed_dir(
        EmbedDirConfig(
            algorithm=config.algorithm,
            cover_dir=paths["cover_dir"],
            stego_dir=paths["stego_dir"],
            payload=config.payload,
            max_workers=config.embed_workers,
            extension=config.image_extension,
            validation_suffix=config.validation_suffix,
            max_train_files=config.max_train_files,
            max_val_files=config.max_val_files,
        )
    )

    extract_dir(
        SRMDirConfig(
            image_dir=paths["cover_dir"],
            feature_dir=paths["cover_features"],
            image_suffix=config.image_extension,
            gpu_devices=config.srm_gpu_devices,
            validation_suffix=config.validation_suffix,
            max_train_files=config.max_train_files,
            max_val_files=config.max_val_files,
        )
    )
    extract_dir(
        SRMDirConfig(
            image_dir=paths["stego_dir"],
            feature_dir=paths["stego_features"],
            image_suffix=config.image_extension,
            gpu_devices=config.srm_gpu_devices,
            validation_suffix=config.validation_suffix,
            max_train_files=config.max_train_files,
            max_val_files=config.max_val_files,
        )
    )

    train_classifier(
        lclsmr_config
    )

    train_srnet(
        srnet_config
    )

    print(f"Linear report: {paths['linear_output'] / 'classification_report.txt'}")
    print(f"SRNet report: {paths['srnet_output'] / 'classification_report.txt'}")
    return paths


def main() -> None:
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()

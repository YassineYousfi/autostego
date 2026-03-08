from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from steganalysis.srnet import TrainConfig as SRNetConfig
from steganalysis.srnet import train as train_srnet
from steganalysis.srm import SRMDirConfig, extract_dir
from steganalysis.train_features import TrainFeaturesConfig, train_classifier
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
    max_train_files: int | None = None
    max_val_files: int | None = None
    embed_workers: int | None = None
    srm_workers: int | None = None
    feature_max_iter: int = 2000
    srnet_epochs: int = 30
    srnet_batch_size: int = 32
    srnet_workers: int = 4
    srnet_amp: bool = True
    srnet_wandb_mode: str = "offline"


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
            max_workers=config.srm_workers,
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
            max_workers=config.srm_workers,
            validation_suffix=config.validation_suffix,
            max_train_files=config.max_train_files,
            max_val_files=config.max_val_files,
        )
    )

    train_classifier(
        TrainFeaturesConfig(
            cover_feature_dir=paths["cover_features"],
            stego_feature_dir=paths["stego_features"],
            output_dir=paths["linear_output"],
            feature_suffix=".npy",
            max_iter=config.feature_max_iter,
            fixed_val_suffix=config.validation_suffix,
            max_train_pairs=config.max_train_files,
            max_val_pairs=config.max_val_files,
        )
    )

    train_srnet(
        SRNetConfig(
            data_root=config.data_root,
            cover_dir=config.cover_dir_name,
            stego_dir=paths["stego_dir"].name,
            output_dir=paths["srnet_output"],
            extensions=(config.image_extension,),
            epochs=config.srnet_epochs,
            batch_size=config.srnet_batch_size,
            workers=config.srnet_workers,
            amp=config.srnet_amp,
            wandb_mode=config.srnet_wandb_mode,
            fixed_val_suffix=config.validation_suffix,
            max_train_pairs=config.max_train_files,
            max_val_pairs=config.max_val_files,
        )
    )

    print(f"Linear report: {paths['linear_output'] / 'classification_report.txt'}")
    print(f"SRNet report: {paths['srnet_output'] / 'classification_report.txt'}")
    return paths


def main() -> None:
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()
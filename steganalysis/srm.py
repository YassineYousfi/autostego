from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from steganalysis._srm import srm as extract_srm
from utils.files import collect_files, directory_is_nonempty, ensure_directory, mirrored_output_path, save_json, select_split_paths


@dataclass(slots=True)
class SRMDirConfig:
    image_dir: Path = Path("data/BOSSbase_1.01/cover")
    feature_dir: Path = Path("data/features/BOSSbase_1.01/cover")
    image_suffix: str = ".pgm"
    max_workers: int | None = None
    validation_suffix: str | None = "9"
    max_train_files: int | None = None
    max_val_files: int | None = None


CONFIG = SRMDirConfig()


def flatten_srm_features(feature_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    feature_names = sorted(feature_dict)
    vector = np.concatenate([np.asarray(feature_dict[name], dtype=np.float32).reshape(-1) for name in feature_names])
    return vector.astype(np.float32), feature_names


def extract_one(image_path: Path, image_root: Path, feature_root: Path) -> Path:
    feature_dict = extract_srm(image_path)
    vector, _ = flatten_srm_features(feature_dict)
    output_path = mirrored_output_path(image_path, image_root, feature_root, suffix=".npy")
    np.save(output_path, vector)
    return output_path


def _worker(args: tuple[Path, Path, Path]) -> str:
    image_path, image_root, feature_root = args
    return str(extract_one(image_path, image_root, feature_root))


def save_feature_names(image_paths: Iterable[Path], config: SRMDirConfig) -> Path:
    first_image = next(iter(image_paths))
    feature_dict = extract_srm(first_image)
    _, feature_names = flatten_srm_features(feature_dict)
    metadata = {
        "image_dir": str(config.image_dir),
        "feature_dir": str(config.feature_dir),
        "image_suffix": config.image_suffix,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "vector_length": int(sum(np.asarray(feature_dict[name]).size for name in feature_names)),
    }
    return save_json(config.feature_dir / "feature_names.json", metadata)


def extract_dir(config: SRMDirConfig = CONFIG) -> list[Path]:
    if not config.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {config.image_dir}")

    if directory_is_nonempty(config.feature_dir):
        output_paths = collect_files(config.feature_dir, ".npy")
        print(f"Skipping SRM extraction, reusing existing features: {config.feature_dir}")
        return output_paths

    image_paths = collect_files(config.image_dir, config.image_suffix)
    if not image_paths:
        raise FileNotFoundError(f"No images matching *{config.image_suffix} found in {config.image_dir}")
    train_images, val_images = select_split_paths(
        image_paths,
        config.image_dir,
        validation_suffix=config.validation_suffix,
        max_train_items=config.max_train_files,
        max_val_items=config.max_val_files,
    )
    selected_images = train_images + val_images
    if not selected_images:
        raise ValueError("No images selected after applying split limits.")

    ensure_directory(config.feature_dir)
    metadata_path = save_feature_names(selected_images, config)
    jobs = [(image_path, config.image_dir, config.feature_dir) for image_path in selected_images]

    print(f"Extracting SRM features from {len(jobs)} images")
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        outputs = list(tqdm(executor.map(_worker, jobs), total=len(jobs), desc="SRM", unit="image"))

    output_paths = [Path(path) for path in outputs]
    save_json(config.feature_dir / "extract_config.json", asdict(config))
    print(f"Wrote {len(output_paths)} feature files to {config.feature_dir}")
    print(f"Saved metadata to {metadata_path}")
    return output_paths


srm = extract_srm


def main() -> None:
    extract_dir(CONFIG)


if __name__ == "__main__":
    main()

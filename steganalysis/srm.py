from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, asdict
from multiprocessing import get_context
from queue import Empty
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from steganalysis._srm_gpu import srm as extract_srm_gpu
from utils.files import collect_files, directory_is_nonempty, ensure_directory, mirrored_output_path, save_json, select_split_paths


@dataclass(slots=True)
class SRMDirConfig:
    image_dir: Path = Path("data/BOSSbase_1.01/cover")
    feature_dir: Path = Path("data/features/BOSSbase_1.01/cover")
    image_suffix: str = ".pgm"
    gpu_devices: tuple[str, ...] | None = None
    validation_suffix: str | None = "9"
    max_train_files: int | None = None
    max_val_files: int | None = None


CONFIG = SRMDirConfig()


def flatten_srm_features(feature_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    feature_names = sorted(feature_dict)
    vector = np.concatenate([np.asarray(feature_dict[name], dtype=np.float32).reshape(-1) for name in feature_names])
    return vector.astype(np.float32), feature_names


def extract_one(
    image_path: Path,
    image_root: Path,
    feature_root: Path,
    *,
    device: str | None = None,
) -> Path:
    feature_dict = _extract_feature_dict(image_path, device=device)
    vector, _ = flatten_srm_features(feature_dict)
    output_path = mirrored_output_path(image_path, image_root, feature_root, suffix=".npy")
    np.save(output_path, vector)
    return output_path


def _extract_feature_dict(image_path: Path, *, device: str | None = None) -> dict[str, np.ndarray]:
    return extract_srm_gpu(image_path, device=device)


def _gpu_chunk_worker(args: tuple[str, tuple[Path, ...], Path, Path, object]) -> list[str]:
    device, image_paths, image_root, feature_root, progress_queue = args
    torch.cuda.set_device(torch.device(device))
    outputs: list[str] = []
    for image_path in image_paths:
        outputs.append(str(extract_one(image_path, image_root, feature_root, device=device)))
        progress_queue.put(1)
    return outputs


def save_feature_names(image_paths: Iterable[Path], config: SRMDirConfig) -> Path:
    first_image = next(iter(image_paths))
    device = _feature_name_device(config)
    feature_dict = _extract_feature_dict(first_image, device=device)
    _, feature_names = flatten_srm_features(feature_dict)
    metadata = {
        "image_dir": str(config.image_dir),
        "feature_dir": str(config.feature_dir),
        "image_suffix": config.image_suffix,
        "gpu_devices": list(config.gpu_devices) if config.gpu_devices is not None else None,
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
    print(f"Extracting SRM features from {len(selected_images)} images")
    output_paths = _extract_dir_gpu(selected_images, config)

    save_json(config.feature_dir / "extract_config.json", asdict(config))
    print(f"Wrote {len(output_paths)} feature files to {config.feature_dir}")
    print(f"Saved metadata to {metadata_path}")
    return output_paths


def _extract_dir_gpu(selected_images: list[Path], config: SRMDirConfig) -> list[Path]:
    devices = _resolve_gpu_devices(config)
    partitions = _partition_round_robin(selected_images, len(devices))
    active_partitions = [
        (device, image_paths)
        for device, image_paths in zip(devices, partitions, strict=False)
        if image_paths
    ]
    if not active_partitions:
        return []

    output_paths: list[Path] = []
    context = get_context("spawn")
    manager = context.Manager()
    progress_queue = manager.Queue()
    try:
        with ProcessPoolExecutor(max_workers=len(active_partitions), mp_context=context) as executor:
            futures = {
                executor.submit(
                    _gpu_chunk_worker,
                    (device, tuple(image_paths), config.image_dir, config.feature_dir, progress_queue),
                ): device
                for device, image_paths in active_partitions
            }
            completed_images = 0
            with tqdm(total=len(selected_images), desc="SRM-GPU", unit="image") as progress:
                while futures:
                    done, _ = wait(tuple(futures), timeout=0.2, return_when=FIRST_COMPLETED)
                    while True:
                        try:
                            increment = progress_queue.get_nowait()
                        except Empty:
                            break
                        completed_images += int(increment)
                        progress.update(int(increment))

                    for future in done:
                        chunk_outputs = [Path(path) for path in future.result()]
                        output_paths.extend(chunk_outputs)
                        del futures[future]

                if completed_images < len(selected_images):
                    progress.update(len(selected_images) - completed_images)
    finally:
        manager.shutdown()

    return sorted(output_paths)


def _resolve_gpu_devices(config: SRMDirConfig) -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU SRM extraction requested but CUDA is not available.")
    if config.gpu_devices is not None:
        if not config.gpu_devices:
            raise ValueError("gpu_devices must contain at least one device when provided.")
        return [str(torch.device(device)) for device in config.gpu_devices]
    return [f"cuda:{index}" for index in range(torch.cuda.device_count())]


def _feature_name_device(config: SRMDirConfig) -> str | None:
    return _resolve_gpu_devices(config)[0]


def _partition_round_robin(paths: list[Path], partition_count: int) -> list[list[Path]]:
    partitions = [[] for _ in range(partition_count)]
    for index, path in enumerate(paths):
        partitions[index % partition_count].append(path)
    return partitions


srm = extract_srm_gpu


def main() -> None:
    extract_dir(CONFIG)


if __name__ == "__main__":
    main()

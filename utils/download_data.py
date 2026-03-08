from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils.files import collect_files, ensure_directory


@dataclass(slots=True)
class DownloadConfig:
    url: str = "https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"
    data_dir: Path = Path("data")
    dataset_name: str = "BOSSbase_1.01"
    cover_dir_name: str = "cover"
    cover_size: tuple[int, int] = (256, 256)
    chunk_size: int = 1024 * 1024
    max_workers: int | None = None


CONFIG = DownloadConfig()


def download_file(url: str, destination: Path, chunk_size: int) -> None:
    ensure_directory(destination.parent)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)


def find_extracted_root(extract_dir: Path) -> Path:
    children = [path for path in extract_dir.iterdir() if path.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def resize_to_cover_image(source_path: Path, destination: Path, target_size: tuple[int, int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        resized = image.convert("L").resize(target_size, resample=Image.Resampling.BICUBIC)
        temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
        resized.save(temporary_destination, format="PPM")
    temporary_destination.replace(destination)


def _resize_worker(args: tuple[Path, Path, tuple[int, int]]) -> str:
    source_path, destination, target_size = args
    resize_to_cover_image(source_path, destination, target_size)
    return str(destination)


def normalize_cover_layout(
    extracted_root: Path,
    cover_dir: Path,
    target_size: tuple[int, int],
    max_workers: int | None,
) -> None:
    ensure_directory(cover_dir)
    image_candidates = collect_files(extracted_root, ".pgm")
    if not image_candidates:
        raise FileNotFoundError(f"No .pgm files found under {extracted_root}")

    jobs: list[tuple[Path, Path, tuple[int, int]]] = []
    for image_path in image_candidates:
        relative_path = image_path.relative_to(extracted_root)
        destination = cover_dir / relative_path
        jobs.append((image_path, destination, target_size))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_resize_worker, jobs), total=len(jobs), desc="Resize", unit="image"))


def cover_dir_matches_size(cover_dir: Path, target_size: tuple[int, int]) -> bool:
    image_candidates = collect_files(cover_dir, ".pgm")
    if not image_candidates:
        return False
    with Image.open(image_candidates[0]) as image:
        return image.size == target_size


def download_bossbase(config: DownloadConfig = CONFIG) -> Path:
    dataset_dir = config.data_dir / config.dataset_name
    cover_dir = dataset_dir / config.cover_dir_name
    if cover_dir_matches_size(cover_dir, config.cover_size):
        print(f"BOSSbase already available at {cover_dir}")
        return cover_dir

    if any(cover_dir.rglob("*.pgm")):
        print(
            "Resizing existing BOSSbase cover images to 256x256 grayscale "
            "with bicubic resampling (closest to MATLAB imresize default)."
        )
        normalize_cover_layout(cover_dir, cover_dir, config.cover_size, config.max_workers)
        return cover_dir

    ensure_directory(config.data_dir)
    archive_path = config.data_dir / f"{config.dataset_name}.zip"
    if archive_path.exists():
        print(f"Using existing archive {archive_path}")
    else:
        print(f"Downloading {config.url} -> {archive_path}")
        download_file(config.url, archive_path, config.chunk_size)

    with tempfile.TemporaryDirectory() as temp_dir:
        extract_dir = Path(temp_dir) / "extract"
        ensure_directory(extract_dir)
        print(f"Extracting {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        extracted_root = find_extracted_root(extract_dir)
        print(
            "Preparing 10,000 grayscale cover images resized from 512x512 to 256x256 "
            "with bicubic resampling (closest to MATLAB imresize default)."
        )
        normalize_cover_layout(extracted_root, cover_dir, config.cover_size, config.max_workers)

    print(f"Cover images are ready in {cover_dir}")
    return cover_dir


def main() -> None:
    download_bossbase(CONFIG)


if __name__ == "__main__":
    main()

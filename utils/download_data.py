from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from utils.files import collect_files, ensure_directory


@dataclass(slots=True)
class DownloadConfig:
    url: str = "https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"
    data_dir: Path = Path("data")
    dataset_name: str = "BOSSbase_1.01"
    cover_dir_name: str = "cover"
    chunk_size: int = 1024 * 1024


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


def normalize_cover_layout(extracted_root: Path, cover_dir: Path) -> None:
    ensure_directory(cover_dir)
    image_candidates = collect_files(extracted_root, ".pgm")
    if not image_candidates:
        raise FileNotFoundError(f"No .pgm files found under {extracted_root}")

    for image_path in image_candidates:
        relative_path = image_path.relative_to(extracted_root)
        destination = cover_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.move(str(image_path), str(destination))


def download_bossbase(config: DownloadConfig = CONFIG) -> Path:
    dataset_dir = config.data_dir / config.dataset_name
    cover_dir = dataset_dir / config.cover_dir_name
    if any(cover_dir.rglob("*.pgm")):
        print(f"BOSSbase already available at {cover_dir}")
        return cover_dir

    ensure_directory(config.data_dir)
    archive_path = config.data_dir / f"{config.dataset_name}.zip"
    print(f"Downloading {config.url} -> {archive_path}")
    download_file(config.url, archive_path, config.chunk_size)

    with tempfile.TemporaryDirectory() as temp_dir:
        extract_dir = Path(temp_dir) / "extract"
        ensure_directory(extract_dir)
        print(f"Extracting {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        extracted_root = find_extracted_root(extract_dir)
        normalize_cover_layout(extracted_root, cover_dir)

    print(f"Cover images are ready in {cover_dir}")
    return cover_dir


def main() -> None:
    download_bossbase(CONFIG)


if __name__ == "__main__":
    main()

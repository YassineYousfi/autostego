from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Callable

from tqdm import tqdm
from wonderwords import RandomWord

from utils.files import collect_files, directory_is_nonempty, save_json, select_split_paths


EmbeddingFunction = Callable[[str, str, float], None]


WORD_GENERATOR = RandomWord()


def generate_stego_dir(root: Path = Path("data/BOSSbase_1.01")) -> Path:
    name = "-".join([
        WORD_GENERATOR.word(include_parts_of_speech=["adjectives"]),
        WORD_GENERATOR.word(include_parts_of_speech=["adjectives"]),
        WORD_GENERATOR.word(include_parts_of_speech=["nouns"]),
    ])
    stego_dir = root / name
    assert not stego_dir.exists(), f"Stego directory already exists: {stego_dir}"
    return stego_dir

@dataclass(slots=True)
class EmbedDirConfig:
    algorithm: str = "hill"
    cover_dir: Path = Path("data/BOSSbase_1.01/cover")
    stego_dir: Path = generate_stego_dir()
    payload: float = 0.4
    max_workers: int | None = None
    extension: str = ".pgm"
    validation_suffix: str | None = "9"
    max_train_files: int | None = None
    max_val_files: int | None = None


CONFIG = EmbedDirConfig()


def collect_cover_images(cover_dir: Path, extension: str) -> list[Path]:
    return collect_files(cover_dir, extension)

def save_config(config: EmbedDirConfig, stego_dir: Path) -> Path:
    config_data = asdict(config)
    config_data["stego_dir"] = str(stego_dir)
    return save_json(stego_dir / "embed_config.json", config_data)


def load_embedding_function(algorithm: str) -> EmbeddingFunction:
    module = importlib.import_module(f"steganography.{algorithm}")
    embedding_fn = getattr(module, algorithm, None)
    if embedding_fn is None or not callable(embedding_fn):
        raise AttributeError(
            f"Expected callable {algorithm}(...) in module steganography.{algorithm}"
        )
    return embedding_fn


def embed_one(
    embedding_fn: EmbeddingFunction,
    cover_path: Path,
    cover_root: Path,
    stego_root: Path,
    payload: float,
) -> Path:
    relative_path = cover_path.relative_to(cover_root)
    output_path = stego_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_fn(str(cover_path), str(output_path), payload)
    return output_path


def _worker(args: tuple[str, Path, Path, Path, float]) -> str:
    algorithm, cover_path, cover_root, stego_root, payload = args
    return str(embed_one(load_embedding_function(algorithm), cover_path, cover_root, stego_root, payload))


def embed_dir(config: EmbedDirConfig = CONFIG) -> list[Path]:
    load_embedding_function(config.algorithm)
    if not config.cover_dir.is_dir():
        raise FileNotFoundError(f"Cover directory not found: {config.cover_dir}")

    if directory_is_nonempty(config.stego_dir):
        output_paths = collect_files(config.stego_dir, config.extension)
        print(f"Skipping embedding, reusing existing stego dir: {config.stego_dir}")
        return output_paths

    cover_images = collect_cover_images(config.cover_dir, config.extension)
    if not cover_images:
        raise FileNotFoundError(f"No cover images matching *{config.extension} found in {config.cover_dir}")
    train_images, val_images = select_split_paths(
        cover_images,
        config.cover_dir,
        validation_suffix=config.validation_suffix,
        max_train_items=config.max_train_files,
        max_val_items=config.max_val_files,
    )
    selected_images = train_images + val_images
    if not selected_images:
        raise ValueError("No cover images selected after applying split limits.")

    config.stego_dir.mkdir(parents=True, exist_ok=False)
    config_path = save_config(config, config.stego_dir)
    jobs = [
        (config.algorithm, cover_path, config.cover_dir, config.stego_dir, config.payload)
        for cover_path in selected_images
    ]

    print(f"Embedding with {config} into {len(jobs)} images")
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        outputs = list(tqdm(executor.map(_worker, jobs), total=len(jobs), desc="Embedding", unit="image"))

    output_paths = [Path(path) for path in outputs]
    print(f"Wrote {len(output_paths)} stego images to {config.stego_dir}")
    print(f"Saved config to {config_path}")
    return output_paths


def main() -> None:
    embed_dir(CONFIG)


if __name__ == "__main__":
    main()
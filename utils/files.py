from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def collect_files(root: Path, suffix: str) -> list[Path]:
    return sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())


def mirrored_output_path(input_path: Path, input_root: Path, output_root: Path, suffix: str | None = None) -> Path:
    relative_path = input_path.relative_to(input_root)
    output_path = output_root / relative_path
    if suffix is not None:
        output_path = output_path.with_suffix(suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def directory_is_nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def save_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def relative_stem_set(paths: Iterable[Path], root: Path) -> set[str]:
    return {str(path.relative_to(root).with_suffix("")) for path in paths}


def relative_stem(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix(""))


def is_validation_stem(stem: str, validation_suffix: str) -> bool:
    return Path(stem).name.endswith(validation_suffix)


def split_relative_keys(
    keys: Iterable[str],
    *,
    validation_suffix: str | None = "9",
    max_train_items: int | None = None,
    max_val_items: int | None = None,
) -> tuple[list[str], list[str]]:
    ordered_keys = sorted(keys)
    if validation_suffix is None:
        train_keys = ordered_keys[:max_train_items] if max_train_items is not None else ordered_keys
        return train_keys, []

    val_keys = [key for key in ordered_keys if is_validation_stem(key, validation_suffix)]
    val_key_set = set(val_keys)
    train_keys = [key for key in ordered_keys if key not in val_key_set]

    if max_train_items is not None:
        train_keys = train_keys[:max_train_items]
    if max_val_items is not None:
        val_keys = val_keys[:max_val_items]

    return train_keys, val_keys


def select_split_paths(
    paths: Iterable[Path],
    root: Path,
    *,
    validation_suffix: str | None = "9",
    max_train_items: int | None = None,
    max_val_items: int | None = None,
) -> tuple[list[Path], list[Path]]:
    ordered_paths = sorted(paths)
    keyed_paths = {relative_stem(path, root): path for path in ordered_paths}
    train_keys, val_keys = split_relative_keys(
        keyed_paths,
        validation_suffix=validation_suffix,
        max_train_items=max_train_items,
        max_val_items=max_val_items,
    )
    return [keyed_paths[key] for key in train_keys], [keyed_paths[key] for key in val_keys]

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

from steganalysis._lclsmr import LCLSMRClassifier
from utils.files import collect_files, relative_stem_set, save_json, split_relative_keys
from utils.reports import save_classification_outputs


@dataclass(slots=True)
class LCLSMRConfig:
    cover_feature_dir: Path = Path("data/features/BOSSbase_1.01/cover")
    stego_feature_dir: Path = Path("data/features/BOSSbase_1.01/hill")
    output_dir: Path = Path("data/features/models/hill_linear")
    feature_suffix: str = ".npy"
    seed: int = 1337
    cv_tolerance_grid: tuple[float, ...] = (3e-5, 1e-5, 3e-6, 1e-6)
    cv_num_folds: int = 3
    cv_maxiter: int = 30000
    fixed_val_suffix: str | None = "9"
    max_train_pairs: int | None = None
    max_val_pairs: int | None = None


CONFIG = LCLSMRConfig()


def pair_feature_paths(cover_dir: Path, stego_dir: Path, suffix: str) -> tuple[list[Path], list[Path]]:
    cover_paths = collect_files(cover_dir, suffix)
    stego_paths = collect_files(stego_dir, suffix)
    if not cover_paths or not stego_paths:
        raise FileNotFoundError("Cover and stego feature directories must both contain feature files.")

    cover_keys = relative_stem_set(cover_paths, cover_dir)
    stego_keys = relative_stem_set(stego_paths, stego_dir)
    common_keys = sorted(cover_keys & stego_keys)
    if not common_keys:
        raise ValueError("No matching cover/stego feature files found.")

    print(
        f"Feature pairs surviving intersection: {len(common_keys)} "
        f"(cover={len(cover_paths)}, stego={len(stego_paths)})"
    )

    paired_cover = [cover_dir / f"{key}{suffix}" for key in common_keys]
    paired_stego = [stego_dir / f"{key}{suffix}" for key in common_keys]
    return paired_cover, paired_stego


def load_feature_matrix(paths: list[Path], label: int) -> tuple[np.ndarray, np.ndarray]:
    vectors = [np.load(path) for path in tqdm(paths, desc=f"Loading label {label}", unit="file")]
    x = np.stack(vectors).astype(np.float32)
    y = np.full(len(paths), label, dtype=np.int64)
    return x, y


def build_dataset(config: LCLSMRConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cover_paths, stego_paths = pair_feature_paths(config.cover_feature_dir, config.stego_feature_dir, config.feature_suffix)
    common_keys = [str(path.relative_to(config.cover_feature_dir).with_suffix("")) for path in cover_paths]
    cover_lookup = {key: path for key, path in zip(common_keys, cover_paths, strict=True)}
    stego_lookup = {key: path for key, path in zip(common_keys, stego_paths, strict=True)}

    if config.fixed_val_suffix is None:
        raise ValueError("fixed_val_suffix must be set for feature training.")

    train_keys, val_keys = split_relative_keys(
        common_keys,
        validation_suffix=config.fixed_val_suffix,
        max_train_items=config.max_train_pairs,
        max_val_items=config.max_val_pairs,
    )
    if not train_keys or not val_keys:
        raise ValueError("Fixed split needs both train and validation feature pairs.")

    x_train_cover, y_train_cover = load_feature_matrix([cover_lookup[key] for key in train_keys], label=0)
    x_train_stego, y_train_stego = load_feature_matrix([stego_lookup[key] for key in train_keys], label=1)
    x_val_cover, y_val_cover = load_feature_matrix([cover_lookup[key] for key in val_keys], label=0)
    x_val_stego, y_val_stego = load_feature_matrix([stego_lookup[key] for key in val_keys], label=1)

    x_train = np.concatenate((x_train_cover, x_train_stego), axis=0)
    y_train = np.concatenate((y_train_cover, y_train_stego), axis=0)
    x_val = np.concatenate((x_val_cover, x_val_stego), axis=0)
    y_val = np.concatenate((y_val_cover, y_val_stego), axis=0)
    return x_train, y_train, x_val, y_val


def train_classifier(config: LCLSMRConfig = CONFIG) -> tuple[LCLSMRClassifier, dict[str, float]]:
    if not config.cover_feature_dir.is_dir():
        raise FileNotFoundError(f"Cover feature directory not found: {config.cover_feature_dir}")
    if not config.stego_feature_dir.is_dir():
        raise FileNotFoundError(f"Stego feature directory not found: {config.stego_feature_dir}")

    x_train, y_train, x_val, y_val = build_dataset(config)
    model = LCLSMRClassifier(
        random_state=config.seed,
        cv_tolerance_grid=np.asarray(config.cv_tolerance_grid, dtype=np.float64),
        cv_num_folds=config.cv_num_folds,
        cv_maxiter=config.cv_maxiter,
    )
    model.fit(x_train, y_train)

    val_probs = model.predict_proba(x_val)
    val_preds = np.argmax(val_probs, axis=1)
    metrics = {
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "feature_dim": int(x_train.shape[1]),
        "val_accuracy": float(accuracy_score(y_val, val_preds)),
        "val_log_loss": float(log_loss(y_val, val_probs)),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.output_dir / "linear_classifier.joblib")
    save_json(config.output_dir / "metrics.json", metrics)
    save_json(config.output_dir / "train_config.json", asdict(config))
    save_classification_outputs(config.output_dir, y_val, val_preds)

    print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Validation log loss: {metrics['val_log_loss']:.4f}")
    print(f"Saved model to {config.output_dir / 'linear_classifier.joblib'}")
    return model, metrics


def main() -> None:
    train_classifier(CONFIG)


if __name__ == "__main__":
    main()
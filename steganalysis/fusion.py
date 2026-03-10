from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from steganalysis.lclsmr import LCLSMRConfig, build_dataset, detect as detect_lclsmr, pair_feature_paths
from steganalysis.srnet import SRNetConfig, evaluate_checkpoint
from utils.files import save_json, split_relative_keys
from utils.reports import save_classification_outputs


@dataclass(slots=True)
class FusionConfig:
    data_root: Path = Path("data/BOSSbase_1.01")
    cover_dir_name: str = "cover"
    stego_dir_name: str = "stego"
    feature_root: Path = Path("data/features/BOSSbase_1.01")
    linear_model_dir: Path = Path("data/features/models/stego_linear")
    srnet_output_dir: Path = Path("runs/srnet_stego")
    output_dir: Path = Path("runs/fusion_stego")
    image_suffix: str = ".pgm"
    fixed_val_suffix: str | None = "9"
    max_train_pairs: int | None = None
    max_val_pairs: int | None = None
    linear_weight: float = 1.0 / 3.0
    srnet_weight: float = 1.0 / 3.0
    d4_weight: float = 1.0 / 3.0


CONFIG = FusionConfig()
FUSION_CONFIG_JSON_ENV = "FUSION_CONFIG_JSON"


def config_from_dict(data: dict[str, object]) -> FusionConfig:
    values = dict(data)
    for key in ("data_root", "feature_root", "linear_model_dir", "srnet_output_dir", "output_dir"):
        if key in values and values[key] is not None:
            values[key] = Path(values[key])
    return FusionConfig(**values)


def load_config(path: str | Path) -> FusionConfig:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_config_json(raw_config: str) -> FusionConfig:
    return config_from_dict(json.loads(raw_config))


def _validation_keys(config: FusionConfig) -> list[str]:
    cover_feature_dir = config.feature_root / config.cover_dir_name
    stego_feature_dir = config.feature_root / config.stego_dir_name
    cover_paths, _ = pair_feature_paths(cover_feature_dir, stego_feature_dir, ".npy")
    common_keys = [str(path.relative_to(cover_feature_dir).with_suffix("")) for path in cover_paths]
    _, val_keys = split_relative_keys(
        common_keys,
        validation_suffix=config.fixed_val_suffix,
        max_train_items=config.max_train_pairs,
        max_val_items=config.max_val_pairs,
    )
    return val_keys


def _align_srnet_probs(srnet_prob: np.ndarray, val_keys: list[str]) -> np.ndarray:
    order = []
    for key in val_keys:
        order.append((key, 0))
        order.append((key, 1))
    srnet_map = {sample: srnet_prob[index] for index, sample in enumerate(order)}
    return np.stack([srnet_map[(key, 0)] for key in val_keys] + [srnet_map[(key, 1)] for key in val_keys])


def _lclsmr_config(config: FusionConfig) -> LCLSMRConfig:
    return LCLSMRConfig(
        data_root=config.data_root,
        cover_dir_name=config.cover_dir_name,
        stego_dir_name=config.stego_dir_name,
        feature_root=config.feature_root,
        output_dir=config.linear_model_dir,
        image_suffix=config.image_suffix,
        fixed_val_suffix=config.fixed_val_suffix,
        max_train_pairs=config.max_train_pairs,
        max_val_pairs=config.max_val_pairs,
    )


def _srnet_config(config: FusionConfig) -> SRNetConfig:
    return SRNetConfig(
        data_root=config.data_root,
        cover_dir_name=config.cover_dir_name,
        stego_dir_name=config.stego_dir_name,
        output_dir=config.srnet_output_dir,
        extensions=(config.image_suffix,),
        fixed_val_suffix=config.fixed_val_suffix,
        max_train_pairs=config.max_train_pairs,
        max_val_pairs=config.max_val_pairs,
        batch_size=16,
        workers=8,
        amp=True,
        compile=False,
        pretrained=False,
        strict_loading=True,
    )


def _load_or_train_linear(config: FusionConfig) -> tuple[np.ndarray, np.ndarray]:
    lclsmr_config = _lclsmr_config(config)
    model_path = config.linear_model_dir / "linear_classifier.joblib"
    if not model_path.exists():
        detect_lclsmr(lclsmr_config)
    _, _, x_val, y_val = build_dataset(lclsmr_config)
    model = joblib.load(model_path)
    return model.predict_proba(x_val), y_val


def _load_or_eval_srnet(config: FusionConfig, *, label: str, eval_d4: bool) -> np.ndarray:
    prediction_file = config.srnet_output_dir / f"predictions_{label}.npz"
    if not prediction_file.exists():
        checkpoint_path = config.srnet_output_dir / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing SRNet checkpoint: {checkpoint_path}")
        evaluate_checkpoint(_srnet_config(config), checkpoint_path, label=label, eval_d4=eval_d4)
    return np.load(prediction_file)["y_prob"]


def detect(config: FusionConfig = CONFIG) -> dict[str, float]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    val_keys = _validation_keys(config)
    linear_prob, y_true = _load_or_train_linear(config)
    srnet_prob = _align_srnet_probs(_load_or_eval_srnet(config, label="best", eval_d4=False), val_keys)
    d4_prob = _align_srnet_probs(_load_or_eval_srnet(config, label="d4_best", eval_d4=True), val_keys)

    fused_prob = (
        config.linear_weight * linear_prob
        + config.srnet_weight * srnet_prob
        + config.d4_weight * d4_prob
    )
    fused_prob = fused_prob / fused_prob.sum(axis=1, keepdims=True)
    y_pred = fused_prob.argmax(axis=1)
    metrics = {
        "val_samples": int(len(y_true)),
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_log_loss": float(log_loss(y_true, fused_prob)),
        "weights": {
            "linear": config.linear_weight,
            "srnet": config.srnet_weight,
            "d4": config.d4_weight,
        },
    }
    save_json(config.output_dir / "metrics.json", metrics)
    save_json(config.output_dir / "train_config.json", asdict(config))
    np.savez_compressed(config.output_dir / "predictions_best.npz", y_true=y_true, y_pred=y_pred, y_prob=fused_prob)
    save_classification_outputs(config.output_dir, y_true, y_pred)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-weight fusion detector")
    parser.add_argument("--config", type=Path, default=None, help="Path to a JSON-serialized FusionConfig")
    args = parser.parse_args()
    if args.config is not None:
        config = load_config(args.config)
    elif FUSION_CONFIG_JSON_ENV in os.environ:
        config = load_config_json(os.environ[FUSION_CONFIG_JSON_ENV])
    else:
        config = CONFIG
    detect(config)


if __name__ == "__main__":
    main()

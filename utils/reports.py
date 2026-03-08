from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from utils.files import save_json


def save_classification_outputs(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: tuple[str, str] = ("cover", "stego"),
) -> dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(target_names),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=list(target_names),
        digits=4,
        zero_division=0,
    )
    save_json(output_dir / "classification_report.json", report)
    (output_dir / "classification_report.txt").write_text(report_text + "\n", encoding="utf-8")
    return report
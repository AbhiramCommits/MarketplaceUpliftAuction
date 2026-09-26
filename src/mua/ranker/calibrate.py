"""Calibrate the ranker's raw scores with isotonic regression.

Isotonic regression is fit on the validation split, then evaluated on validation
(in-sample) and test (out-of-sample). Reports Expected Calibration Error (10 bins)
before and after calibration and saves a reliability diagram to
``reports/figures/calibration.png``. The fitted calibrator is saved so scoring can
apply the same map.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from mua.config import resolve
from mua.ranker.dataset import RankerDataset, score_dataset
from mua.ranker.model import load_model

logger = logging.getLogger(__name__)


def bin_stats(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-frequency (decile) bins of predictions.

    Deciles are used instead of equal-width bins because CTR predictions live in a
    narrow range ([0, ~0.15]); equal-width bins would collapse nearly all mass into
    the first bin.
    """
    edges = np.unique(np.quantile(y_pred, np.linspace(0, 1, n_bins + 1)))
    idx = np.digitize(y_pred, edges[1:-1])
    mean_pred: list[float] = []
    frac_pos: list[float] = []
    weight: list[float] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.any():
            mean_pred.append(float(y_pred[mask].mean()))
            frac_pos.append(float(y_true[mask].mean()))
            weight.append(float(mask.mean()))
    return np.array(mean_pred), np.array(frac_pos), np.array(weight)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (10 bins) weighted by bin size."""
    mean_pred, frac_pos, weight = bin_stats(y_true, y_pred, n_bins)
    if weight.sum() == 0:
        return float("nan")
    return float((weight * np.abs(mean_pred - frac_pos)).sum() / weight.sum())


def plot_reliability_diagram(
    y_true: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    out_path: str | Path,
    n_bins: int = 10,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = max(float(raw.max()), float(calibrated.max())) * 1.1
    ax.plot([0, limit], [0, limit], "k--", lw=1, label="Perfectly calibrated")
    for scores, label, color, marker in (
        (raw, "Before calibration", "#d62728", "o"),
        (calibrated, "After calibration", "#2ca02c", "s"),
    ):
        mean_pred, frac_pos, _ = bin_stats(y_true, scores, n_bins)
        ax.plot(mean_pred, frac_pos, marker=marker, color=color, lw=2, label=label)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Predicted click probability")
    ax.set_ylabel("Observed click rate")
    ax.set_title(f"Reliability diagram ({n_bins} decile bins)")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reliability diagram saved to %s", out_path)


def calibrate(
    cfg: dict[str, Any],
    spec: dict[str, Any],
    model: Any = None,
) -> dict[str, Any]:
    artifacts = resolve(cfg["paths"]["ranker_artifacts"])
    processed = resolve(cfg["paths"]["processed"])
    n_bins = int(cfg.get("calibration", {}).get("n_bins", 10))
    batch_size = int(cfg["data"]["batch_size"])
    mode = "in_memory" if cfg["data"].get("in_memory") else "chunked"

    if model is None:
        model, _, _ = load_model(artifacts, cfg["model"])

    valid_ds = RankerDataset(processed / "valid", spec, mode=mode)
    valid_preds, valid_labels = score_dataset(model, valid_ds, batch_size)
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(valid_preds, valid_labels)
    valid_cal = isotonic.transform(valid_preds)

    test_ds = RankerDataset(processed / "test", spec, mode=mode)
    test_preds, test_labels = score_dataset(model, test_ds, batch_size)
    test_cal = isotonic.transform(test_preds)

    metrics = {
        "valid_ece_before": expected_calibration_error(valid_labels, valid_preds, n_bins),
        "valid_ece_after": expected_calibration_error(valid_labels, valid_cal, n_bins),
        "test_ece_before": expected_calibration_error(test_labels, test_preds, n_bins),
        "test_ece_after": expected_calibration_error(test_labels, test_cal, n_bins),
        "test_roc_auc": float(roc_auc_score(test_labels, test_preds)),
        "test_pr_auc": float(average_precision_score(test_labels, test_preds)),
    }

    joblib.dump(isotonic, artifacts / "calibrator.joblib")
    plot_reliability_diagram(
        test_labels,
        test_preds,
        test_cal,
        resolve(cfg["paths"]["figures"]) / "calibration.png",
        n_bins=n_bins,
    )

    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_metrics(metrics)
    except ImportError:  # pragma: no cover
        pass

    logger.info(
        "Calibration: valid ECE %.4f -> %.4f | test ECE %.4f -> %.4f | test AUC %.4f",
        metrics["valid_ece_before"],
        metrics["valid_ece_after"],
        metrics["test_ece_before"],
        metrics["test_ece_after"],
        metrics["test_roc_auc"],
    )
    return metrics

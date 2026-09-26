"""Gradient-boosted propensity model with balance diagnostics.

Fits P(treated | X), reports the AUC, plots the propensity-overlap histogram by arm,
trims units outside [trim_low, trim_high], computes standardized mean differences
(SMD) for every covariate before and after IPTW weighting, and saves a Love plot.
Fails loudly (RuntimeError) if any post-weighting |SMD| exceeds the threshold.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from mua.config import resolve

logger = logging.getLogger(__name__)


def _weighted_mean(x: np.ndarray, w: np.ndarray | None) -> float:
    return float(np.average(x, weights=w)) if w is not None else float(x.mean())


def _weighted_var(x: np.ndarray, w: np.ndarray | None) -> float:
    if w is None:
        return float(x.var())
    mean = _weighted_mean(x, w)
    return float(np.average((x - mean) ** 2, weights=w))


def _smd(treated: np.ndarray, control: np.ndarray, wt=None, wc=None) -> float:
    pooled = np.sqrt((_weighted_var(treated, wt) + _weighted_var(control, wc)) / 2.0)
    return abs(_weighted_mean(treated, wt) - _weighted_mean(control, wc)) / max(pooled, 1e-12)


def smd_table(
    feature_names: list[str],
    X: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """SMD per covariate; if `weights` (IPTW) given, use weighted moments."""
    treated_mask = w == 1
    control_mask = ~treated_mask
    out: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        wt = weights[treated_mask] if weights is not None else None
        wc = weights[control_mask] if weights is not None else None
        out[name] = _smd(X[treated_mask, j], X[control_mask, j], wt, wc)
    return out


def _ate_weights(propensity: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Inverse probability of treatment weights for the ATE (unstabilized)."""
    return np.where(w == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))


def fit_propensity(cfg: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    X, w = frame["X"], frame["w"]
    names = frame["feature_names"]
    pc = cfg["propensity"]
    seed = int(cfg["data"]["seed"])

    model = HistGradientBoostingClassifier(
        max_iter=int(pc.get("n_estimators", 300)),
        learning_rate=0.08,
        max_depth=None,
        min_samples_leaf=50,
        random_state=seed,
    )
    cv_pred = cross_val_predict(model, X, w, cv=5, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(w, cv_pred))
    model.fit(X, w)
    propensity = model.predict_proba(X)[:, 1]

    artifacts = resolve(cfg["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "propensity.joblib")
    fig_dir = resolve(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_overlap(propensity, w, auc, fig_dir / "propensity_overlap.png")

    trim_low = float(pc["trim_low"])
    trim_high = float(pc["trim_high"])
    keep = (propensity >= trim_low) & (propensity <= trim_high)
    X_t, w_t = X[keep], w[keep]
    weights = _ate_weights(propensity[keep], w_t)

    smd_before = smd_table(names, X, w)
    smd_after = smd_table(names, X_t, w_t, weights=weights)
    threshold = float(pc["smd_threshold"])
    _plot_love(names, smd_before, smd_after, threshold, fig_dir / "love_plot.png")

    violations = [name for name in names if smd_after[name] > threshold]
    logger.info(
        "Propensity: AUC=%.4f | trimmed %d -> %d rows (prop in [%.2f, %.2f])",
        auc,
        len(w),
        len(w_t),
        trim_low,
        trim_high,
    )
    logger.info(
        "SMD after IPTW: max=%.4f mean=%.4f (threshold %.2f)",
        max(smd_after.values()),
        float(np.mean(list(smd_after.values()))),
        threshold,
    )
    if violations:
        raise RuntimeError(
            f"IPTW balance check FAILED: post-weighting |SMD| > {threshold} for {violations}"
        )

    return {
        "auc": auc,
        "n_before": int(len(w)),
        "n_after": int(len(w_t)),
        "propensity": propensity,
        "trim_mask": keep,
        "smd_before": smd_before,
        "smd_after": smd_after,
    }


def _plot_overlap(propensity: np.ndarray, w: np.ndarray, auc: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(propensity[w == 0], bins=60, alpha=0.55, color="#1f77b4", label="Control")
    ax.hist(propensity[w == 1], bins=60, alpha=0.55, color="#d62728", label="Treated")
    ax.set_xlabel("Estimated propensity P(treated | X)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Propensity overlap (AUC={auc:.4f})")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Propensity overlap plot saved to %s", path)


def _plot_love(
    names: list[str],
    before: dict[str, float],
    after: dict[str, float],
    threshold: float,
    path: Path,
) -> None:
    order = sorted(names, key=lambda n: -before[n])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter([before[n] for n in order], y, color="#d62728", s=28, label="Before IPTW")
    ax.scatter([after[n] for n in order], y, color="#2ca02c", s=28, label="After IPTW")
    ax.axvline(threshold, color="k", linestyle="--", lw=1, label=f"Threshold {threshold}")
    ax.set_yticks(y, order, fontsize=7)
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Love plot: covariate balance before vs after IPTW")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Love plot saved to %s", path)

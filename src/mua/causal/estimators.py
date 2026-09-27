"""Fit and compare ATE/CATE estimators on the same training split.

Estimators: naive difference in means (biased baseline), stabilized IPTW, S/T/X
learners (EconML metalearners with LightGBM/HistGradientBoosting base models),
DRLearner, and CausalForestDML (honest splitting + built-in CIs). Each returns an
ATE with a 95% CI (bootstrap or analytic) plus per-unit CATE predictions on the test
split. A SHAP importance plot is produced for the T-learner's treated model.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
from scipy import stats

from mua.config import resolve

# LightGBM bundles its own OpenMP runtime; other libraries (numba via shap, sklearn's
# HistGradientBoosting) also load OpenMP. On macOS, initializing two OpenMP runtimes
# in one process can segfault. Tolerate duplicates and keep thread pools bounded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)

ALL_ESTIMATORS = ("naive", "iptw", "s", "t", "x", "dr", "forest")


def _lgb_clf(seed: int, n: int = 150):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=n,
        learning_rate=0.06,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=6,
        random_state=seed,
        verbose=-1,
    )


def _lgb_reg(seed: int, n: int = 150):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=n,
        learning_rate=0.06,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=6,
        random_state=seed,
        verbose=-1,
    )


def _hgb_kwargs(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """HistGradientBoosting hyperparameters, configurable via estimators.*."""
    return {
        "max_iter": int(cfg["estimators"].get("hgb_max_iter", 200)),
        "learning_rate": float(cfg["estimators"].get("hgb_learning_rate", 0.08)),
        "min_samples_leaf": int(cfg["estimators"].get("hgb_min_samples_leaf", 20)),
        "early_stopping": bool(cfg["estimators"].get("hgb_early_stopping", False)),
        "validation_fraction": float(cfg["estimators"].get("hgb_validation_fraction", 0.1)),
        "n_iter_no_change": int(cfg["estimators"].get("hgb_n_iter_no_change", 20)),
        "random_state": seed,
    }


def _clf_factory(cfg: dict[str, Any], seed: int) -> Callable[[], Any]:
    # HistGradientBoosting is the default: LightGBM 4.x intermittently segfaults on
    # Apple Silicon (OpenMP thread race), so it is opt-in via `estimators.base`.
    if cfg["estimators"].get("base") == "lightgbm":
        return lambda: _lgb_clf(seed)
    from sklearn.ensemble import HistGradientBoostingClassifier

    return lambda: HistGradientBoostingClassifier(**_hgb_kwargs(cfg, seed))


def _reg_factory(cfg: dict[str, Any], seed: int) -> Callable[[], Any]:
    if cfg["estimators"].get("base") == "lightgbm":
        return lambda: _lgb_reg(seed)
    from sklearn.ensemble import HistGradientBoostingRegressor

    return lambda: HistGradientBoostingRegressor(**_hgb_kwargs(cfg, seed))


def _analytic_ci(cate: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    ate = float(cate.mean())
    se = cate.std(ddof=1) / np.sqrt(len(cate))
    z = stats.norm.ppf(1 - alpha / 2)
    return ate, float(ate - z * se), float(ate + z * se)


def _bootstrap(fn: Callable[[np.ndarray, np.ndarray], float], w, y, seed: int, n: int):
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    idx = np.arange(len(y))
    for i in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        draws[i] = fn(w[sample], y[sample])
    return draws


def _diff_means(w: np.ndarray, y: np.ndarray) -> float:
    return float(y[w == 1].mean() - y[w == 0].mean())


def fit_naive(w: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg["data"]["seed"])
    ate = _diff_means(w, y)
    draws = _bootstrap(_diff_means, w, y, seed, int(cfg["estimators"]["bootstrap_samples"]))
    ci = np.percentile(draws, [2.5, 97.5])
    return {
        "name": "naive",
        "ate": ate,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "cate_test": np.zeros_like(y, dtype=float),
    }


def fit_iptw(w: np.ndarray, y: np.ndarray, ps: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg["data"]["seed"])
    p_treat = float(w.mean())

    def _iptw_ate(ww: np.ndarray, yy: np.ndarray, pp: np.ndarray) -> float:
        weights = np.where(ww == 1, p_treat / pp, (1 - p_treat) / (1 - pp))
        return float(
            np.average(yy[ww == 1], weights=weights[ww == 1])
            - np.average(yy[ww == 0], weights=weights[ww == 0])
        )

    ate = _iptw_ate(w, y, ps)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    draws = np.empty(int(cfg["estimators"]["bootstrap_samples"]))
    for i in range(len(draws)):
        sample = rng.choice(idx, size=len(idx), replace=True)
        draws[i] = _iptw_ate(w[sample], y[sample], ps[sample])
    ci = np.percentile(draws, [2.5, 97.5])
    return {
        "name": "iptw",
        "ate": ate,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "cate_test": np.zeros_like(y, dtype=float),
    }


def fit_metalearner(
    kind: str, X: np.ndarray, w: np.ndarray, y: np.ndarray, X_test: np.ndarray, cfg, seed: int
) -> dict[str, Any]:
    from econml.metalearners import SLearner, TLearner, XLearner

    # econml's metalearners call `model.predict(...)`; classifiers return class
    # labels there, so the outcome models must be REGRESSORS on the binary outcome
    # (linear-probability style), which is the econml-recommended pattern for
    # binary Y.
    if kind == "s":
        learner = SLearner(overall_model=_reg_factory(cfg, seed)())
    elif kind == "t":
        learner = TLearner(models=_reg_factory(cfg, seed)())
    else:
        learner = XLearner(
            models=_reg_factory(cfg, seed)(),
            cate_models=_reg_factory(cfg, seed)(),
            propensity_model=_clf_factory(cfg, seed)(),
        )
    learner.fit(Y=y, T=w, X=X)
    cate_test = np.asarray(learner.effect(X_test), dtype=float).ravel()
    ate, lo, hi = _analytic_ci(cate_test)
    return {
        "name": f"{kind}_learner",
        "ate": ate,
        "ci_low": lo,
        "ci_high": hi,
        "cate_test": cate_test,
        "model": learner,
    }


def fit_dr(
    X: np.ndarray, w: np.ndarray, y: np.ndarray, X_test: np.ndarray, cfg, seed: int
) -> dict[str, Any]:
    from econml.dr import DRLearner

    learner = DRLearner(
        model_propensity=_clf_factory(cfg, seed)(),
        model_regression=_reg_factory(cfg, seed)(),
        model_final=_reg_factory(cfg, seed)(),
        cv=int(cfg["estimators"]["dr_cv"]),
        random_state=seed,
    )
    learner.fit(Y=y, T=w, X=X)
    cate_test = np.asarray(learner.effect(X_test), dtype=float).ravel()
    ate, lo, hi = _analytic_ci(cate_test)
    return {
        "name": "dr_learner",
        "ate": ate,
        "ci_low": lo,
        "ci_high": hi,
        "cate_test": cate_test,
        "model": learner,
    }


def fit_forest(
    X: np.ndarray, w: np.ndarray, y: np.ndarray, X_test: np.ndarray, cfg, seed: int
) -> dict[str, Any]:
    from econml.dml import CausalForestDML

    learner = CausalForestDML(
        model_y=_reg_factory(cfg, seed)(),
        model_t=_clf_factory(cfg, seed)(),
        discrete_treatment=True,
        n_estimators=int(cfg["estimators"]["forest_estimators"]),
        max_depth=int(cfg["estimators"]["forest_max_depth"]),
        min_samples_leaf=int(cfg["estimators"]["forest_min_samples_leaf"]),
        honest=True,
        cv=2,
        n_jobs=6,
        random_state=seed,
    )
    learner.fit(Y=y, T=w, X=X)
    cate_test = np.asarray(learner.effect(X_test), dtype=float).ravel()
    lo, hi = learner.effect_interval(X_test, alpha=0.05)
    return {
        "name": "causal_forest",
        "ate": float(cate_test.mean()),
        "ci_low": float(np.asarray(lo).mean()),
        "ci_high": float(np.asarray(hi).mean()),
        "cate_test": cate_test,
        "model": learner,
    }


def fit_estimators(
    cfg: dict[str, Any],
    frame: dict[str, Any],
    frame_test: dict[str, Any],
    propensity: np.ndarray,
    selected: tuple[str, ...] = ALL_ESTIMATORS,
) -> dict[str, dict[str, Any]]:
    """Fit all requested estimators on the trimmed training frame."""
    seed = int(cfg["data"]["seed"])
    X, w, y = frame["X"], frame["w"], frame["y"]
    X_test = frame_test["X"]
    results: dict[str, dict[str, Any]] = {}

    if "naive" in selected:
        logger.info("Fitting naive difference-in-means ...")
        results["naive"] = fit_naive(w, y, cfg)
        results["naive"]["cate_test"] = np.zeros(len(X_test))
    if "iptw" in selected:
        logger.info("Fitting stabilized IPTW ...")
        results["iptw"] = fit_iptw(w, y, propensity, cfg)
        results["iptw"]["cate_test"] = np.zeros(len(X_test))
    for kind in ("s", "t", "x"):
        if kind in selected:
            logger.info("Fitting %s-learner ...", kind.upper())
            results[kind] = fit_metalearner(kind, X, w, y, X_test, cfg, seed)
    if "dr" in selected:
        logger.info("Fitting DRLearner ...")
        results["dr"] = fit_dr(X, w, y, X_test, cfg, seed)
    if "forest" in selected:
        logger.info("Fitting CausalForestDML (honest) ...")
        results["forest"] = fit_forest(X, w, y, X_test, cfg, seed)

    if "t" in results:
        _shap_importance(cfg, results["t"]["model"], X, frame["feature_names"], seed)

    _save_results(cfg, results, frame_test["ids"])
    return results


def _save_results(cfg: dict[str, Any], results: dict[str, dict[str, Any]], test_ids) -> None:
    artifacts = resolve(cfg["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name, res in results.items():
        np.savez_compressed(
            artifacts / f"cate_test_{name}.npz",
            impression_id=test_ids,
            cate=res["cate_test"].astype(np.float32),
        )
        summary[name] = {
            "ate": res["ate"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
        }
    joblib.dump(summary, artifacts / "estimator_summary.joblib")


def _shap_importance(cfg: dict[str, Any], t_model, X: np.ndarray, names, seed: int) -> None:
    """SHAP feature importance for the T-learner's treated-group model."""
    if cfg["estimators"].get("base") != "lightgbm":
        logger.info("SHAP: skipped (requires LightGBM base models)")
        return
    try:
        import matplotlib
        import shap

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lightgbm import LGBMClassifier, LGBMRegressor

        if not isinstance(t_model, (LGBMClassifier, LGBMRegressor)):
            logger.warning("SHAP: skipping (base model is not LightGBM)")
            return
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(X), size=min(2000, len(X)), replace=False)
        explainer = shap.TreeExplainer(t_model)
        values = explainer.shap_values(X[sample])
        if isinstance(values, list):
            values = values[1]
        fig = plt.figure(figsize=(8, 5))
        shap.summary_plot(values, X[sample], feature_names=names, show=False)
        out = resolve(cfg["paths"]["figures"]) / "shap_importance.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP summary plot saved to %s", out)
    except Exception as exc:  # pragma: no cover - SHAP is auxiliary
        logger.warning("SHAP importance failed (non-fatal): %s", exc)

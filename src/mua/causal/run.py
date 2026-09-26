"""Orchestrate the causal pipeline: propensity -> estimators -> evaluation -> policy."""

from __future__ import annotations

import logging
from typing import Any

from mua.causal.data import load_frame
from mua.causal.estimators import ALL_ESTIMATORS, fit_estimators
from mua.causal.evaluate import evaluate
from mua.causal.policy import run as run_policy
from mua.causal.propensity import fit_propensity

logger = logging.getLogger(__name__)


def run(cfg: dict[str, Any], selected: str = "all") -> tuple[dict[str, Any], bool]:
    selected_estimators = ALL_ESTIMATORS if selected == "all" else (selected,)
    seed = int(cfg["data"]["seed"])
    data_cfg = cfg["data"]

    def _sample(key: str):
        value = data_cfg.get(key)
        return None if value is None else int(value)

    frame_train = load_frame(cfg, "train", sample=_sample("sample_train"), seed=seed)
    frame_test = load_frame(cfg, "test", sample=_sample("sample_test"), seed=seed, with_truth=True)

    prop = fit_propensity(cfg, frame_train)
    keep = prop["trim_mask"]
    frame_train = {
        "X": frame_train["X"][keep],
        "w": frame_train["w"][keep],
        "y": frame_train["y"][keep],
        "feature_names": frame_train["feature_names"],
    }
    propensity_trimmed = prop["propensity"][keep]

    results = fit_estimators(cfg, frame_train, frame_test, propensity_trimmed, selected_estimators)
    leaderboard = evaluate(cfg, results, frame_test)
    policy = run_policy(cfg, results, frame_test, leaderboard)

    logger.info("Causal pipeline complete. Leaderboard (ranked by PEHE, then Qini):")
    for row in leaderboard:
        logger.info(
            "  %-14s PEHE=%.4f qini=%.4f ATE=%.4f bias=%+.4f",
            row["estimator"],
            row["pehe"],
            row["qini"],
            row["ate"],
            row["ate_bias"],
        )
    return {"leaderboard": leaderboard, "policy": policy}, True

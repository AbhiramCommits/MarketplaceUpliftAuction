"""Load processed Parquet for causal estimation and join ground-truth tau.

Ground truth (true_tau / true_propensity) is attached ONLY to the evaluation split
and is never part of the feature matrix used for fitting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from mua.config import resolve

logger = logging.getLogger(__name__)

TREATMENT = "promo_treated"
OUTCOME = "ordered"
# Post-treatment / outcome columns excluded from the feature matrix (leakage guard).
EXCLUDED = {
    "impression_id",
    "day",
    "promo_treated",
    "clicked",
    "ordered",
    "order_value_usd",
    "delivery_fee_usd",
    "promo_cost_usd",
}


def feature_columns(table) -> list[str]:
    return [c for c in table.column_names if c not in EXCLUDED]


def load_frame(
    cfg: dict[str, Any],
    split: str,
    sample: int | None = None,
    seed: int = 42,
    with_truth: bool = False,
) -> dict[str, Any]:
    """Load a processed split as {X, w, y, ids, cost, feature_names [, true_tau]}."""
    table = pq.read_table(resolve(cfg["paths"]["processed"]) / split)
    if sample is not None and sample < table.num_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(table.num_rows, size=int(sample), replace=False))
        table = table.take(idx)

    names = feature_columns(table)
    X = np.column_stack([table[name].to_numpy().astype(np.float64) for name in names])
    frame: dict[str, Any] = {
        "X": X,
        "w": table[TREATMENT].to_numpy().astype(np.int64),
        "y": table[OUTCOME].to_numpy().astype(np.float64),
        "ids": table["impression_id"].to_numpy().astype(np.int64),
        "cost": table["promo_cost_usd"].to_numpy().astype(np.float64),
        "feature_names": names,
    }
    if with_truth:
        frame["true_tau"] = attach_truth(cfg, frame["ids"])
    logger.info(
        "Loaded %s: %d rows, %d features%s",
        split,
        len(frame["y"]),
        X.shape[1],
        " + truth" if with_truth else "",
    )
    return frame


def attach_truth(cfg: dict[str, Any], ids: np.ndarray) -> np.ndarray:
    """Join true_tau from data/truth/ by impression_id (evaluation only)."""
    table = pq.read_table(resolve(cfg["paths"]["truth"]))
    truth_ids = table["impression_id"].to_numpy().astype(np.int64)
    truth_tau = table["true_tau"].to_numpy().astype(np.float64)
    order = np.argsort(truth_ids)
    pos = np.minimum(np.searchsorted(truth_ids[order], ids), len(truth_ids) - 1)
    if not np.all(truth_ids[order][pos] == ids):
        raise ValueError("Truth join mismatch: missing impression ids in data/truth/")
    return truth_tau[order][pos]

"""Turn CATE predictions into a targeting policy.

Ranks units by predicted uplift and computes incremental orders and promo spend for
top-k% targeting (k in {5, 10, 20, 30, 50, 100}), plus a random-targeting and a
treat-everyone baseline. Outputs an incremental-orders-per-promo-dollar curve.
"""

from __future__ import annotations

import csv
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mua.config import resolve

logger = logging.getLogger(__name__)


def _subgroup_stats(
    y: np.ndarray, w: np.ndarray, cost: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    yb, wb, cb = y[mask], w[mask], cost[mask]
    n_t = int(wb.sum())
    n_c = int(len(yb) - n_t)
    if n_c == 0:
        return {"incremental": float("nan"), "spend": float("nan")}
    incremental = float(yb[wb == 1].sum() - yb[wb == 0].sum() * n_t / n_c)
    spend = float(cb[wb == 1].sum())
    return {"incremental": incremental, "spend": spend}


def run(
    cfg: dict[str, Any],
    results: dict[str, dict[str, Any]],
    frame: dict[str, Any],
    leaderboard: list[dict[str, Any]],
) -> dict[str, Any]:
    y, w, cost = frame["y"], frame["w"], frame["cost"]
    best = leaderboard[0]["estimator"]
    cate = np.asarray(results[best]["cate_test"], dtype=float).ravel()
    seed = int(cfg["data"]["seed"])
    n = len(y)
    ks = [int(k) for k in cfg["policy"]["ks"]]

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    random_idx = rng.permutation(n)
    for k in ks:
        top = np.argsort(-cate)[: int(n * k / 100)]
        top_stats = _subgroup_stats(y, w, cost, top)
        rand = random_idx[: int(n * k / 100)]
        rand_stats = _subgroup_stats(y, w, cost, rand)
        rows.append(
            {
                "k_pct": k,
                "n_targeted": int(len(top)),
                "inc_orders_topk": top_stats["incremental"],
                "spend_topk": top_stats["spend"],
                "per_dollar_topk": top_stats["incremental"] / max(top_stats["spend"], 1e-9),
                "inc_orders_random": rand_stats["incremental"],
                "spend_random": rand_stats["spend"],
                "per_dollar_random": rand_stats["incremental"] / max(rand_stats["spend"], 1e-9),
            }
        )

    treat_all = _subgroup_stats(y, w, cost, np.ones(n, dtype=bool))
    treat_all_spend = float(cost[w == 1].mean() * n)
    treat_all_per_dollar = treat_all["incremental"] / max(treat_all_spend, 1e-9)

    csv_path = resolve(cfg["paths"]["artifacts"]) / "policy_curve.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r["k_pct"] for r in rows],
        [r["per_dollar_topk"] for r in rows],
        "o-",
        label=f"Top-k by CATE ({best})",
    )
    ax.plot(
        [r["k_pct"] for r in rows],
        [r["per_dollar_random"] for r in rows],
        "s--",
        label="Random targeting",
    )
    ax.axhline(treat_all_per_dollar, color="k", ls=":", lw=1, label="Treat everyone")
    ax.set_xlabel("Top-k% of population targeted")
    ax.set_ylabel("Incremental orders per promo dollar")
    ax.set_title("Targeting policy curve (held-out test split)")
    ax.legend()
    out = resolve(cfg["paths"]["figures"]) / "policy_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Policy (%s CATE): best-k per-dollar %s",
        best,
        ", ".join(f"k={r['k_pct']}:{r['per_dollar_topk']:.4f}" for r in rows),
    )
    logger.info(
        "Random targeting per-dollar: %s",
        ", ".join(f"k={r['k_pct']}:{r['per_dollar_random']:.4f}" for r in rows),
    )
    logger.info(
        "Treat-everyone: incremental=%.1f orders, per-dollar=%.4f",
        treat_all["incremental"],
        treat_all_per_dollar,
    )
    logger.info("Policy curve saved to %s and %s", out, csv_path)
    return {
        "estimator": best,
        "rows": rows,
        "treat_all_per_dollar": treat_all_per_dollar,
    }

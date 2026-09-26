"""Score every estimator against ground truth on the held-out test split.

Metrics: ATE bias and 95% CI coverage vs mean(true_tau), PEHE, Qini curve /
normalized Qini coefficient, AUUC, and an uplift-by-decile table. Saves
reports/figures/qini.png, reports/figures/uplift_deciles.png, and a leaderboard
(markdown + CSV) ranked by PEHE then Qini.
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


def qini_curve_values(
    y: np.ndarray, w: np.ndarray, cate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Radcliffe Qini curve: g(phi) = Y_t(phi) - Y_c(phi) * N_t(phi)/N_c(phi).

    Sorted descending by predicted CATE; phi is the cumulative fraction of units.
    """
    order = np.argsort(-np.asarray(cate, dtype=float))
    y_s, w_s = y[order].astype(float), w[order].astype(float)
    n = len(y_s)
    cum_t = np.cumsum(w_s)
    cum_c = np.arange(1, n + 1) - cum_t
    safe = np.where(cum_c > 0, cum_c, np.inf)
    g = np.cumsum(y_s * w_s) - np.cumsum(y_s * (1 - w_s)) * cum_t / safe
    return np.arange(1, n + 1) / n, g


def normalized_qini(y: np.ndarray, w: np.ndarray, cate: np.ndarray) -> float:
    """Normalized Qini coefficient: area between the Qini curve and the diagonal,
    divided by the max area (Radcliffe q0). Random targeting -> ~0.
    """
    x, g = qini_curve_values(y, w, cate)
    denom = abs(g[-1])
    if denom < 1e-12:
        return 0.0
    g_hat = g / denom
    return float(np.trapezoid(g_hat, x) - 0.5)


def auuc(y: np.ndarray, w: np.ndarray, cate: np.ndarray) -> float:
    """Area under the uplift curve: v(phi) = phi * (r_t(phi) - r_c(phi))."""
    order = np.argsort(-np.asarray(cate, dtype=float))
    y_s, w_s = y[order].astype(float), w[order].astype(float)
    n = len(y_s)
    cum_t = np.cumsum(w_s)
    cum_c = np.arange(1, n + 1) - cum_t
    r_t = np.cumsum(y_s * w_s) / np.where(cum_t > 0, cum_t, np.inf)
    r_c = np.cumsum(y_s * (1 - w_s)) / np.where(cum_c > 0, cum_c, np.inf)
    x = np.arange(1, n + 1) / n
    v = x * np.nan_to_num(r_t - r_c)
    return float(np.trapezoid(v, x))


def uplift_deciles(
    y: np.ndarray, w: np.ndarray, cate: np.ndarray, n_deciles: int = 10
) -> list[dict[str, Any]]:
    order = np.argsort(-np.asarray(cate, dtype=float))
    y_s, w_s = y[order].astype(float), w[order].astype(float)
    edges = np.linspace(0, len(y_s), n_deciles + 1).astype(int)
    rows: list[dict[str, Any]] = []
    for i in range(n_deciles):
        yb, wb = y_s[edges[i] : edges[i + 1]], w_s[edges[i] : edges[i + 1]]
        n_t = int(wb.sum())
        n_c = int(len(wb) - n_t)
        r_t = float(yb[wb == 1].mean()) if n_t else 0.0
        r_c = float(yb[wb == 0].mean()) if n_c else 0.0
        rows.append(
            {
                "decile": i + 1,
                "n": int(len(yb)),
                "n_treated": n_t,
                "n_control": n_c,
                "r_treated": r_t,
                "r_control": r_c,
                "uplift": r_t - r_c,
            }
        )
    return rows


def evaluate(
    cfg: dict[str, Any], results: dict[str, dict[str, Any]], frame: dict[str, Any]
) -> list[dict[str, Any]]:
    y, w, tau = frame["y"], frame["w"], frame["true_tau"]
    truth_ate = float(tau.mean())
    n_deciles = int(cfg["evaluate"]["n_deciles"])
    leaderboard: list[dict[str, Any]] = []

    fig_dir = resolve(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    artifacts = resolve(cfg["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random targeting")
    decile_lines: dict[str, list[float]] = {}
    decile_rows: dict[str, list[dict[str, Any]]] = {}

    for name, res in results.items():
        cate = np.asarray(res["cate_test"], dtype=float).ravel()
        bias = float(res["ate"]) - truth_ate
        coverage = bool(res["ci_low"] <= truth_ate <= res["ci_high"])
        pehe = float(np.sqrt(np.mean((cate - tau) ** 2)))
        qini = normalized_qini(y, w, cate)
        auc_uplift = auuc(y, w, cate)
        deciles = uplift_deciles(y, w, cate, n_deciles)
        leaderboard.append(
            {
                "estimator": name,
                "ate": float(res["ate"]),
                "ci_low": float(res["ci_low"]),
                "ci_high": float(res["ci_high"]),
                "ate_bias": bias,
                "ci_covers_truth": coverage,
                "pehe": pehe,
                "qini": qini,
                "auuc": auc_uplift,
            }
        )
        x, g = qini_curve_values(y, w, cate)
        denom = abs(g[-1]) if abs(g[-1]) > 1e-12 else 1.0
        ax.plot(x, g / denom, lw=1.5, label=name)
        decile_lines[name] = [d["uplift"] for d in deciles]
        decile_rows[name] = deciles
        logger.info(
            "%-14s ATE=%.4f bias=%+.4f cov=%d PEHE=%.4f qini=%.4f auuc=%.4f",
            name,
            res["ate"],
            bias,
            coverage,
            pehe,
            qini,
            auc_uplift,
        )

    oracle_x, oracle_g = qini_curve_values(y, w, tau)
    denom = abs(oracle_g[-1]) if abs(oracle_g[-1]) > 1e-12 else 1.0
    ax.plot(oracle_x, oracle_g / denom, "k:", lw=1.5, label="Oracle (true_tau)")
    ax.set_xlabel("Fraction of population targeted (highest predicted CATE first)")
    ax.set_ylabel("Normalized cumulative incremental orders")
    ax.set_title("Qini curves on held-out test split")
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "qini.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name, values in decile_lines.items():
        ax2.plot(range(1, n_deciles + 1), values, marker="o", lw=1.5, label=name)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("Decile of predicted CATE (1 = highest)")
    ax2.set_ylabel("Observed uplift (treated - control order rate)")
    ax2.set_title("Uplift by decile, held-out test split")
    ax2.legend(fontsize=8)
    fig2.savefig(fig_dir / "uplift_deciles.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    leaderboard.sort(key=lambda r: (r["pehe"], -r["qini"]))
    _write_leaderboard(leaderboard, truth_ate, decile_rows, cfg)
    return leaderboard


def _write_leaderboard(
    leaderboard: list[dict[str, Any]],
    truth_ate: float,
    decile_rows: dict[str, list[dict[str, Any]]],
    cfg: dict[str, Any],
) -> None:
    header = [
        "estimator",
        "ate",
        "ci_low",
        "ci_high",
        "ate_bias",
        "ci_covers_truth",
        "pehe",
        "qini",
        "auuc",
    ]
    csv_path = resolve(cfg["paths"]["artifacts"]) / "leaderboard.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows({k: r[k] for k in header} for r in leaderboard)

    md_path = resolve(cfg["paths"]["figures"]).parent / "leaderboard.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Estimator leaderboard (held-out test split)",
        "",
        f"Ground-truth ATE on the test sample: **{truth_ate:.4f}**. "
        "Ranked by PEHE (individual-level error) then Qini (ranking quality).",
        "",
        "| estimator | ATE | 95% CI | ATE bias | covers truth | PEHE | Qini | AUUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in leaderboard:
        lines.append(
            f"| {r['estimator']} | {r['ate']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
            f"| {r['ate_bias']:+.4f} | {'yes' if r['ci_covers_truth'] else 'no'} "
            f"| {r['pehe']:.4f} | {r['qini']:.4f} | {r['auuc']:.4f} |"
        )
    lines += ["", "## Uplift by decile", ""]
    for name in (r["estimator"] for r in leaderboard):
        lines.append(f"### {name}")
        lines.append("| decile | n | n_treated | n_control | r_treated | r_control | uplift |")
        lines.append("|---|---|---|---|---|---|---|")
        for d in decile_rows[name]:
            lines.append(
                f"| {d['decile']} | {d['n']} | {d['n_treated']} | {d['n_control']} "
                f"| {d['r_treated']:.4f} | {d['r_control']:.4f} | {d['uplift']:+.4f} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n")
    logger.info("Leaderboard saved to %s and %s", csv_path, md_path)

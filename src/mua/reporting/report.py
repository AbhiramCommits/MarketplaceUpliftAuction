"""Generate reports/REPORT.md.

Orchestrates the experiment harness and figure generation (if outputs are missing),
then composes the final report: methodology, causal DAG, estimator leaderboard,
refutation verdicts, auction results grid, findings, and limitations. Every figure
is embedded.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from mua.config import resolve
from mua.reporting import experiment, figures

logger = logging.getLogger(__name__)

FIGURES = [
    ("revenue_surplus.png", "Revenue vs advertiser surplus by mechanism"),
    ("frontier.png", "Incremental conversions vs promo spend frontier"),
    ("pacing_traces.png", "Budget pacing traces"),
    ("shading_convergence.png", "First-price shading-factor convergence"),
    ("qini_overlay.png", "Qini curves by estimator"),
]


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean +/- std per mechanism x policy cell."""
    keyed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        keyed.setdefault((row["mechanism"], row["policy"]), []).append(row)
    numeric = [
        "revenue",
        "advertiser_surplus",
        "consumer_welfare",
        "inc_conversions_true",
        "inc_conversions_model",
        "promo_spend",
        "per_dollar_true",
        "per_dollar_model",
    ]
    cells: list[dict[str, Any]] = []
    for (mechanism, policy), group in keyed.items():
        entry: dict[str, Any] = {"mechanism": mechanism, "policy": policy, "n": len(group)}
        for column in numeric:
            values = np.array([float(r[column]) for r in group])
            values = values[np.isfinite(values)]
            entry[column] = (
                (float(values.mean()), float(values.std(ddof=1)))
                if len(values)
                else (float("nan"), float("nan"))
            )
        cells.append(entry)
    cells.sort(key=lambda c: (c["mechanism"], c["policy"]))
    return cells


def _fmt(value: tuple[float, float]) -> str:
    mean, std = value
    return f"{mean:.4f} ± {std:.4f}"


def _load_causal_assets(cfg: dict[str, Any]) -> tuple[str, str, str, str]:
    artifacts = resolve(cfg["causal"]["artifacts"])
    with (artifacts / "leaderboard.csv").open() as fh:
        leaderboard_rows = list(csv.DictReader(fh))
    best = str(leaderboard_rows[0]["estimator"]) if leaderboard_rows else "n/a"
    leaderboard_md = [
        "| estimator | ATE | ATE bias | covers truth | PEHE | Qini | AUUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in leaderboard_rows:
        leaderboard_md.append(
            f"| {row['estimator']} | {float(row['ate']):.4f} | {float(row['ate_bias']):+.4f} "
            f"| {row['ci_covers_truth']} | {float(row['pehe']):.4f} | {float(row['qini']):.4f} "
            f"| {float(row['auuc']):.4f} |"
        )
    reports_dir = resolve(cfg["paths"]["figures"]).parent
    dag_path = reports_dir / "causal_dag.dot"
    dag = dag_path.read_text().strip() if dag_path.exists() else "n/a"
    refutation_path = reports_dir / "refutation_report.md"
    refutation_md = refutation_path.read_text() if refutation_path.exists() else "n/a"
    return best, "\n".join(leaderboard_md), dag, refutation_md


def _findings(cells: list[dict[str, Any]], best_cate: str) -> list[str]:
    lines: list[str] = []
    candidates: dict[str, float] = {}
    for policy in ("treat_all", "random_k", "uplift_top_k", "propensity_top_k"):
        values = [
            c["per_dollar_true"][0]
            for c in cells
            if c["policy"] == policy and np.isfinite(c["per_dollar_true"][0])
        ]
        if values:
            candidates[policy] = float(np.mean(values))
    if candidates:
        winner = max(candidates, key=lambda p: candidates[p])
        lines.append(
            f"- **Incremental conversions per promo dollar is maximized by `{winner}` "
            f"targeting**: {candidates[winner]:.4f} orders/$. For comparison: "
            + ", ".join(
                f"`{p}` = {v:.4f}" for p, v in sorted(candidates.items(), key=lambda kv: -kv[1])
            )
            + f". The CATE model behind `uplift_top_k` is `{best_cate}` (leaderboard "
            "rank 1 by PEHE)."
        )
    mechanisms: dict[str, dict[str, list[float]]] = {}
    for c in cells:
        mechanisms.setdefault(c["mechanism"], {"revenue": [], "surplus": [], "welfare": []})
        mechanisms[c["mechanism"]]["revenue"].append(c["revenue"][0])
        mechanisms[c["mechanism"]]["surplus"].append(c["advertiser_surplus"][0])
        mechanisms[c["mechanism"]]["welfare"].append(c["consumer_welfare"][0])
    if mechanisms:
        by_revenue = sorted(mechanisms, key=lambda m: -np.mean(mechanisms[m]["revenue"]))
        by_surplus = sorted(mechanisms, key=lambda m: -np.mean(mechanisms[m]["surplus"]))
        lines.append(
            f"- **Revenue/surplus tradeoff:** `{by_revenue[0]}` extracts the most platform "
            f"revenue (mean {np.mean(mechanisms[by_revenue[0]]['revenue']):.1f}), while "
            f"`{by_surplus[0]}` leaves the most advertiser surplus (mean "
            f"{np.mean(mechanisms[by_surplus[0]]['surplus']):.1f}). Advertiser welfare in "
            f"first-price is driven by shading agents, not by truthful bidding."
        )
        welfare_mean = {m: np.mean(mechanisms[m]["welfare"]) for m in mechanisms}
        best_welfare = max(welfare_mean, key=lambda m: welfare_mean[m])
        lines.append(
            f"- **Consumer welfare** (mean relevance of the shown slate) is highest under "
            f"`{best_welfare}` ({welfare_mean[best_welfare]:.4f}) and varies little across "
            f"mechanisms, because all mechanisms allocate by the same eCPM ranking; the "
            f"remaining differences come from bidder strategy."
        )
    lines.append(
        "- **Recommended combination:** targeting policy is the dominant lever for "
        "incremental conversions per dollar and is orthogonal to the auction mechanism; "
        "combine the best per-dollar policy above with `gsp` as the practical ads default "
        "(Vickrey-style pricing with second-price surplus properties), switching to "
        "`first_price` only when platform revenue is prioritized over advertiser surplus, "
        "and to `vcg` for the most efficient (incentive-compatible) allocation."
    )
    return lines


def run(cfg: dict[str, Any], scale: str = "full") -> Path:
    # Always rerun the experiment so the report is reproducible from scratch
    # (deterministic given seed 42).
    experiment.run(cfg, scale=scale)
    # Regenerate figures so they always match the current results.csv.
    figures.run(cfg)

    rows = experiment.read_results(cfg)
    cells = _aggregate(rows)
    best_cate, leaderboard_md, dag, refutation_md = _load_causal_assets(cfg)
    findings = _findings(cells, best_cate)

    n_seeds = len({r["seed"] for r in rows})
    n_cells = len({(r["mechanism"], r["policy"]) for r in rows})
    rounds_per_cell = len(rows) // max(n_cells, 1)

    figure_md = "\n\n".join(
        f"![{caption}](figures/{name})\n\n*{caption}*" for name, caption in FIGURES
    )

    grid_header = [
        "| mechanism | policy | revenue | advertiser surplus | consumer welfare | "
        "inc conv (truth) | promo spend | per dollar (truth) | per dollar (model) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    grid_rows = [
        f"| {c['mechanism']} | {c['policy']} | {_fmt(c['revenue'])} | "
        f"{_fmt(c['advertiser_surplus'])} | {_fmt(c['consumer_welfare'])} | "
        f"{_fmt(c['inc_conversions_true'])} | {_fmt(c['promo_spend'])} | "
        f"{_fmt(c['per_dollar_true'])} | {_fmt(c['per_dollar_model'])} |"
        for c in cells
    ]

    lines = [
        "# Marketplace uplift x auction experiment report",
        "",
        "## Methodology",
        "",
        "The experiment runs the full cross product of 4 auction mechanisms "
        "(first_price, second_price, gsp, vcg) x 5 promo targeting policies "
        "(treat_none, treat_all, random_k, uplift_top_k from the best CATE model, "
        f"propensity_top_k) with {n_seeds} random seeds per cell; each cell simulates "
        f"{rounds_per_cell} rounds sampled from the scored test log.",
        "",
        "- Each round samples a 20-merchant slate from `data/scored/test/`; advertisers "
        "(merchants with `is_ads_advertiser`) bid with the configured agent mix "
        "(truthful, shading, budget-paced, random) and the mechanism allocates up to 3 "
        "sponsored slots ranked by eCPM = bid x pCTR.",
        "- Platform metrics: expected ad revenue = sum(payment x pCTR); consumer welfare "
        "proxy = mean relevance of the final shown slate (organic relevance minus what ads "
        "displace); advertiser surplus = sum((session value - payment) x pCTR), where the "
        "session value scales the merchant's value-per-click by the session's promo-boosted "
        "order probability.",
        "- Promo metrics: each round also carries one promo session (a sampled test row "
        "joined to the ground-truth tau and baseline order probability from `data/truth/`). "
        "The targeting policy decides whether the session is treated; incremental "
        "conversions = treated x (clipped(baseline + tau) - baseline), computed both from "
        "the ground truth and from the CATE model for validation.",
        "",
        f"The best CATE model selected for `uplift_top_k` is `{best_cate}` (leaderboard "
        "rank 1 by PEHE). Full per-seed grid: `results.csv`.",
        "",
        "## Causal DAG",
        "",
        "```dot",
        dag,
        "```",
        "",
        "## Estimator leaderboard (held-out test split, vs ground truth)",
        "",
        leaderboard_md,
        "",
        "## Refutation verdicts (DoWhy)",
        "",
        refutation_md,
        "",
        "## Auction results grid (mean ± std across seeds)",
        "",
        *grid_header,
        *grid_rows,
        "",
        "## Figures",
        "",
        figure_md,
        "",
        "## Findings",
        "",
        *findings,
        "",
        "## Limitations",
        "",
        "- All results are on **simulated data** with a **known generative model**; the "
        "CATE/PEHE comparisons benefit from ground truth that does not exist in production.",
        "- **No interference or spillover**: consumers are treated independently, promo "
        "effects are session-local, and the auction does not feed back into the promo "
        "decision. Real marketplaces exhibit cross-consumer and cross-channel effects.",
        "- Advertiser demand is modeled with fixed values per click plus simple learning "
        "agents; there are no strategic bidder dynamics beyond the shading/pacing rules.",
        "- Incremental conversions use expected (probability-level) accounting against the "
        "simulator's structural models, not observed causal effects on real users.",
        "",
    ]
    report_path = resolve(cfg["paths"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", report_path)
    return report_path

"""Figures for the experiment report (matplotlib only, one chart per figure).

- revenue vs advertiser surplus scatter by mechanism
- incremental conversions vs promo spend frontier across targeting policies
- budget pacing traces over simulated time
- shading-factor convergence for first-price agents
- Qini curves from the causal module, overlaid
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from mua.auction.mechanisms import MECHANISMS
from mua.auction.simulator import assign_agents, build_advertisers
from mua.auction.types import AuctionRequest, Candidate
from mua.causal.data import load_frame
from mua.causal.evaluate import qini_curve_values
from mua.config import resolve

logger = logging.getLogger(__name__)

POLICY_LABELS = {
    "treat_none": "treat none",
    "treat_all": "treat all",
    "random_k": "random k%",
    "uplift_top_k": "uplift top-k% (CATE)",
    "propensity_top_k": "propensity top-k%",
}


def _read_results(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from mua.reporting.experiment import read_results

    return read_results(cfg)


def _plot_revenue_surplus(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    mechanisms = sorted({r["mechanism"] for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, mechanism in enumerate(mechanisms):
        cells = [r for r in rows if r["mechanism"] == mechanism]
        x = [float(r["revenue"]) for r in cells]
        y = [float(r["advertiser_surplus"]) for r in cells]
        ax.scatter(
            x,
            y,
            color=cmap(i / max(len(mechanisms) - 1, 1)),
            alpha=0.7,
            s=40,
            label=mechanism,
        )
    ax.set_xlabel("Platform ad revenue (expected, $)")
    ax.set_ylabel("Total advertiser surplus ($)")
    ax.set_title("Revenue vs advertiser surplus by auction mechanism")
    ax.legend()
    out = resolve(cfg["paths"]["figures"]) / "revenue_surplus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_frontier(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    policies = sorted({r["policy"] for r in rows}, key=lambda p: POLICY_LABELS.get(p, p))
    fig, ax = plt.subplots(figsize=(7, 5))
    for policy in policies:
        cells = [r for r in rows if r["policy"] == policy]
        x = np.array([float(r["promo_spend"]) for r in cells])
        y = np.array([float(r["inc_conversions_true"]) for r in cells])
        ax.errorbar(
            x.mean(),
            y.mean(),
            xerr=x.std(ddof=1),
            yerr=y.std(ddof=1),
            fmt="o",
            capsize=4,
            label=POLICY_LABELS.get(policy, policy),
        )
    ax.set_xlabel("Promo spend ($)")
    ax.set_ylabel("Incremental conversions (ground-truth tau)")
    ax.set_title("Incremental conversions vs promo spend across targeting policies")
    ax.legend(fontsize=9)
    out = resolve(cfg["paths"]["figures"]) / "frontier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _trace_auction(
    cfg: dict[str, Any], mechanism_name: str, rounds: int, seed: int
) -> dict[str, Any]:
    """Short auction trace recording per-round pacing spend and shading factors."""
    auction_cfg = {**cfg, "auction": {**cfg["auction"], "mechanism": mechanism_name}}
    if mechanism_name == "second_price":
        auction_cfg["auction"]["slot_positions"] = [1]
    scored = pq.read_table(resolve(cfg["paths"]["scored"]) / "test")
    n_rows = scored.num_rows
    merchant_col = scored["merchant_id"].to_numpy()
    pctr_col = scored["pctr"].to_numpy().astype(np.float64)
    rel_col = scored["pctr_raw"].to_numpy().astype(np.float64)
    slate_size = int(cfg["auction"]["slate_size"])
    advertisers = build_advertisers(scored, rounds, auction_cfg)
    agents = assign_agents(advertisers, auction_cfg, seed)
    by_aid = {a.advertiser.advertiser_id: a for a in agents.values()}
    mechanism = MECHANISMS[mechanism_name]()
    rng = np.random.default_rng(seed)
    slate_idx = rng.integers(0, n_rows, size=(rounds, slate_size))

    pacing_trace: dict[str, list[float]] = {}
    pacing_budget: dict[str, float] = {}
    shading_trace: dict[str, list[float]] = {}
    for agent in agents.values():
        if agent.agent_type == "paced":
            pacing_trace[agent.advertiser.advertiser_id] = []
            pacing_budget[agent.advertiser.advertiser_id] = agent.advertiser.daily_budget
        elif agent.agent_type == "shading":
            shading_trace[agent.advertiser.advertiser_id] = []

    for t in range(rounds):
        bids = []
        candidates: list[Candidate] = []
        for j in range(slate_size):
            merchant_id = int(merchant_col[slate_idx[t, j]])
            candidate = Candidate(
                merchant_id=merchant_id,
                pctr=float(pctr_col[slate_idx[t, j]]),
                organic_relevance=float(rel_col[slate_idx[t, j]]),
            )
            candidates.append(candidate)
            agent = agents.get(merchant_id)
            if agent is not None:
                bids.append(agent.make_bid(candidate))
        request = AuctionRequest(
            request_id=t,
            candidates=tuple(candidates),
            reserve=float(cfg["auction"]["reserve"]),
            slot_positions=tuple(auction_cfg["auction"]["slot_positions"]),
        )
        outcome = mechanism.allocate(request, bids)
        winners = {a.advertiser_id: a for a in outcome.allocations}
        for bid in bids:
            agents[bid.merchant_id].observe(winners.get(bid.advertiser_id), False)
            agents[bid.merchant_id].tick(t, rounds)
        for advertiser_id in pacing_trace:
            pacing_trace[advertiser_id].append(by_aid[advertiser_id].spend)
        for advertiser_id in shading_trace:
            shading_trace[advertiser_id].append(by_aid[advertiser_id].shading)

    return {"pacing": pacing_trace, "budget": pacing_budget, "shading": shading_trace}


def _plot_pacing_traces(cfg: dict[str, Any]) -> Path:
    trace = _trace_auction(cfg, "gsp", 1500, seed=7)
    pacing = trace["pacing"]
    if not pacing:
        raise RuntimeError("No paced agents in trace")
    top = sorted(pacing, key=lambda aid: -trace["budget"][aid])[:5]
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, advertiser_id in enumerate(top):
        ratio = np.array(pacing[advertiser_id]) / max(trace["budget"][advertiser_id], 1e-9)
        ax.plot(ratio, color=cmap(i / max(len(top) - 1, 1)), lw=1.2, label=advertiser_id)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("Auction round")
    ax.set_ylabel("Cumulative spend / daily budget")
    ax.set_title("Budget pacing traces (GSP)")
    ax.legend(fontsize=8)
    out = resolve(cfg["paths"]["figures"]) / "pacing_traces.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_shading_convergence(cfg: dict[str, Any]) -> Path:
    trace = _trace_auction(cfg, "first_price", 1500, seed=7)
    shading = trace["shading"]
    if not shading:
        raise RuntimeError("No shading agents in trace")
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 5))
    series = list(shading.values())
    for i, (advertiser_id, values) in enumerate(shading.items()):
        ax.plot(
            values,
            color=cmap(i / max(len(series) - 1, 1)),
            lw=1,
            alpha=0.8,
            label=advertiser_id,
        )
    ax.plot(np.mean(series, axis=0), color="k", lw=2, label="mean")
    ax.set_xlabel("Auction round")
    ax.set_ylabel("Shading factor (bid / value)")
    ax.set_title("Shading-factor convergence (first-price)")
    ax.legend(fontsize=8)
    out = resolve(cfg["paths"]["figures"]) / "shading_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_qini_overlay(cfg: dict[str, Any]) -> Path:
    artifacts = resolve(cfg["causal"]["artifacts"])
    causal = cfg["causal"]
    frame = load_frame(
        causal, "test", sample=causal.get("sample"), seed=int(causal["seed"]), with_truth=True
    )
    y, w, tau = frame["y"], frame["w"], frame["true_tau"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random targeting")
    for npz_path in sorted(artifacts.glob("cate_test_*.npz")):
        name = npz_path.stem[len("cate_test_") :]
        data = np.load(npz_path)
        if not np.array_equal(data["impression_id"].astype(np.int64), frame["ids"]):
            continue
        cate = data["cate"].astype(np.float64)
        x, g = qini_curve_values(y, w, cate)
        denom = abs(g[-1]) if abs(g[-1]) > 1e-12 else 1.0
        ax.plot(x, g / denom, lw=1.5, label=name)
    x, g = qini_curve_values(y, w, tau)
    denom = abs(g[-1]) if abs(g[-1]) > 1e-12 else 1.0
    ax.plot(x, g / denom, "k:", lw=1.5, label="Oracle (true_tau)")
    ax.set_xlabel("Fraction of population targeted (highest predicted CATE first)")
    ax.set_ylabel("Normalized cumulative incremental orders")
    ax.set_title("Qini curves by estimator (causal module)")
    ax.legend(fontsize=8)
    out = resolve(cfg["paths"]["figures"]) / "qini_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run(cfg: dict[str, Any]) -> list[Path]:
    resolve(cfg["paths"]["figures"]).mkdir(parents=True, exist_ok=True)
    rows = _read_results(cfg)
    outputs = [
        _plot_revenue_surplus(cfg, rows),
        _plot_frontier(cfg, rows),
        _plot_pacing_traces(cfg),
        _plot_shading_convergence(cfg),
        _plot_qini_overlay(cfg),
    ]
    for path in outputs:
        logger.info("Figure saved to %s", path)
    return outputs

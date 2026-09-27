"""Run N auction rounds sampled from the scored test log.

Builds advertisers from merchant economics in data/scored/, samples 20-candidate
slates per round, runs the configured mechanism with a mix of bidding agents, and
tracks per-advertiser and per-platform metrics. Per-round outcomes are written to
Parquet at data/auction_runs/<run_id>/.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mua.auction.agents import BudgetPacedAgent, RandomAgent, ShadingAgent, TruthfulAgent
from mua.auction.mechanisms import MECHANISMS
from mua.auction.types import Advertiser, Allocation, AuctionRequest, Bid, Candidate
from mua.config import resolve

logger = logging.getLogger(__name__)

AGENT_CLASSES = {
    "truthful": TruthfulAgent,
    "shading": ShadingAgent,
    "paced": BudgetPacedAgent,
    "random": RandomAgent,
}

POSITION_BIAS_AVG = float(np.mean(1.0 / np.log2(np.arange(1, 21) + 1.0)))

# Typical realized CPC as a fraction of the winner's value, per mechanism
# (GSP pays the runner-up price, VCG/FP pay close to the winner's own bid).
PRICE_FACTORS = {
    "first_price": 0.35,
    "second_price": 0.2,
    "gsp": 0.2,
    "vcg": 0.15,
}


def position_multiplier(slot: int) -> float:
    """Click-probability multiplier of a slate position relative to the average."""
    return (1.0 / np.log2(slot + 1.0)) / POSITION_BIAS_AVG


def build_advertisers(table: pa.Table, rounds: int, cfg: dict[str, Any]) -> list[Advertiser]:
    """Derive per-click values from merchant economics in the scored log."""
    df = table.select(
        [
            "merchant_id",
            "clicked",
            "ordered",
            "order_value_usd",
            "is_ads_advertiser",
            "pctr",
        ]
    ).to_pandas()
    clicked_mask = df["clicked"] == 1
    if clicked_mask.any():
        global_conv = float(df.loc[clicked_mask, "ordered"].mean())
    else:
        global_conv = 0.05
    global_aov = float(df.loc[df["ordered"] == 1, "order_value_usd"].mean() or 20.0)
    n_rows = len(df)

    conv = (
        df.loc[clicked_mask].groupby("merchant_id")["ordered"].mean().rename("conv").reset_index()
    )
    agg = (
        df.groupby("merchant_id")
        .agg(
            orders=("ordered", "sum"),
            aov=(
                "order_value_usd",
                lambda s: float(s[s > 0].mean()) if (s > 0).any() else global_aov,
            ),
            pctr_mean=("pctr", "mean"),
            is_ads=("is_ads_advertiser", "max"),
            n=("merchant_id", "size"),
        )
        .reset_index()
        .merge(conv, on="merchant_id", how="left")
    )
    agg["conv"] = agg["conv"].fillna(global_conv)
    agg = agg[agg["is_ads"] == 1]
    agg["value"] = np.clip(agg["conv"] * agg["aov"], 0.2, 10.0)
    slate_size = cfg["auction"]["slate_size"]
    agg["participation"] = agg["n"] / n_rows * slate_size

    # Budget sized to a fraction of the unthrottled spend, using mechanism-level
    # estimates of the win rate and realized CPC level.
    expected_bidders = max(float(df["is_ads_advertiser"].mean()) * slate_size, 1.0)
    win_rate = min(1.0, len(cfg["auction"]["slot_positions"]) / expected_bidders)
    price_factor = PRICE_FACTORS.get(cfg["auction"]["mechanism"], 0.5)
    budget_ratio = float(cfg["auction"]["pacing"]["budget_ratio"])
    budget_factor = win_rate * price_factor * budget_ratio

    advertisers: list[Advertiser] = []
    for row in agg.itertuples(index=False):
        advertisers.append(
            Advertiser(
                advertiser_id=f"adv_{int(row.merchant_id)}",
                merchant_id=int(row.merchant_id),
                value_per_click=float(row.value),
                daily_budget=float(
                    row.value * row.pctr_mean * rounds * row.participation * budget_factor
                ),
                conversion_rate=float(row.conv),
            )
        )
    logger.info(
        "Built %d advertisers (value mean=%.3f, budget mean=%.3f)",
        len(advertisers),
        float(agg["value"].mean()),
        float(np.mean([a.daily_budget for a in advertisers])),
    )
    return advertisers


def assign_agents(advertisers: list[Advertiser], cfg: dict[str, Any], seed: int) -> dict[int, Any]:
    mix = cfg["auction"]["agent_mix"]
    types = list(mix)
    weights = [float(mix[t]) for t in types]
    rng = np.random.default_rng(seed)
    shading_cfg = cfg["auction"]["shading"]
    pacing_cfg = cfg["auction"]["pacing"]
    agents: dict[int, Any] = {}
    for i, advertiser in enumerate(advertisers):
        agent_type = types[int(rng.choice(len(types), p=weights))]
        kwargs: dict[str, Any] = {"advertiser": advertiser, "seed": seed + i}
        if agent_type == "shading":
            kwargs.update(
                initial=shading_cfg["initial"],
                step=shading_cfg["step"],
                s_min=shading_cfg["min"],
                s_max=shading_cfg["max"],
            )
        elif agent_type == "paced":
            kwargs.update(
                learning_rate=pacing_cfg["learning_rate"],
                lam_max=pacing_cfg["lam_max"],
            )
        agents[advertiser.merchant_id] = AGENT_CLASSES[agent_type](**kwargs)
    return agents


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    auction_cfg = cfg["auction"]
    mechanism_name = auction_cfg["mechanism"]
    if mechanism_name not in MECHANISMS:
        raise ValueError(f"Unknown mechanism {mechanism_name}; choose from {list(MECHANISMS)}")
    mechanism = MECHANISMS[mechanism_name]()
    if mechanism.requires_single_slot and len(auction_cfg["slot_positions"]) != 1:
        logger.info("second_price is single-slot; using slot position 1")
        auction_cfg["slot_positions"] = [1]

    rounds = int(auction_cfg["rounds"])
    seed = int(auction_cfg["seed"])
    slate_size = int(auction_cfg["slate_size"])
    slot_positions = tuple(int(p) for p in auction_cfg["slot_positions"])
    reserve = float(auction_cfg["reserve"])

    table = pq.read_table(resolve(cfg["paths"]["scored"]) / "test")
    advertisers = build_advertisers(table, rounds, cfg)
    agents = assign_agents(advertisers, cfg, seed)

    rng = np.random.default_rng(seed)
    n_rows = table.num_rows
    merchant_col = table["merchant_id"].to_numpy()
    pctr_col = table["pctr"].to_numpy().astype(np.float64)
    rel_col = table["pctr_raw"].to_numpy().astype(np.float64)
    idx = rng.integers(0, n_rows, size=(rounds, slate_size))

    columns: dict[str, list[Any]] = {
        "request_id": [],
        "mechanism": [],
        "n_candidates": [],
        "n_bids": [],
        "n_allocations": [],
        "revenue_expected": [],
        "revenue_realized": [],
        "fill_rate": [],
        "organic_displacement": [],
        "avg_ads_position": [],
    }

    t0 = time.time()
    for t in range(rounds):
        bids: list[Bid] = []
        for j in range(slate_size):
            merchant_id = int(merchant_col[idx[t, j]])
            agent = agents.get(merchant_id)
            if agent is not None:
                candidate = Candidate(
                    merchant_id=merchant_id,
                    pctr=float(pctr_col[idx[t, j]]),
                    organic_relevance=float(rel_col[idx[t, j]]),
                )
                bids.append(agent.make_bid(candidate))
        request = AuctionRequest(
            request_id=t,
            candidates=tuple(
                Candidate(
                    merchant_id=int(merchant_col[idx[t, j]]),
                    pctr=float(pctr_col[idx[t, j]]),
                    organic_relevance=float(rel_col[idx[t, j]]),
                )
                for j in range(slate_size)
            ),
            reserve=reserve,
            slot_positions=slot_positions,
        )
        outcome = mechanism.allocate(request, bids)

        revenue_realized = 0.0
        winners_by_agent: dict[str, Allocation] = {}
        clicks_by_agent: dict[str, bool] = {}
        positions: list[float] = []
        for allocation in outcome.allocations:
            positions.append(float(allocation.slot))
            winners_by_agent[allocation.advertiser_id] = allocation
            click_prob = min(0.5, allocation.pctr * position_multiplier(allocation.slot))
            clicks_by_agent[allocation.advertiser_id] = bool(rng.random() < click_prob)
            if clicks_by_agent[allocation.advertiser_id]:
                revenue_realized += allocation.payment
        for bid in bids:
            allocation = winners_by_agent.get(bid.advertiser_id)
            if allocation is None:
                agents[bid.merchant_id].observe(None, False)
            else:
                agents[bid.merchant_id].observe(allocation, clicks_by_agent[bid.advertiser_id])
            agents[bid.merchant_id].tick(t, rounds)

        columns["request_id"].append(t)
        columns["mechanism"].append(mechanism_name)
        columns["n_candidates"].append(slate_size)
        columns["n_bids"].append(len(bids))
        columns["n_allocations"].append(len(outcome.allocations))
        columns["revenue_expected"].append(outcome.revenue_expected)
        columns["revenue_realized"].append(revenue_realized)
        columns["fill_rate"].append(outcome.fill_rate)
        columns["organic_displacement"].append(outcome.organic_displacement)
        columns["avg_ads_position"].append(float(np.mean(positions)) if positions else 0.0)

    elapsed = time.time() - t0
    logger.info("Simulated %d rounds in %.1fs (%.0f rounds/s)", rounds, elapsed, rounds / elapsed)

    run_id = (
        auction_cfg.get("run_id") or f"{mechanism_name}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = resolve(cfg["paths"]["auction_runs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), run_dir / "outcomes.parquet")

    summary = _build_summary(columns, agents, mechanism_name, run_id, run_dir)
    _print_summary(summary)
    return summary


def _build_summary(
    columns: dict[str, list[Any]],
    agents: dict[int, Any],
    mechanism_name: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    revenue = float(np.sum(columns["revenue_expected"]))
    rounds = len(columns["request_id"])
    summary: dict[str, Any] = {
        "run_id": run_id,
        "mechanism": mechanism_name,
        "rounds": rounds,
        "platform": {
            "revenue_expected": revenue,
            "rpm": revenue / rounds * 1000.0,
            "fill_rate": float(np.mean(columns["fill_rate"])),
            "avg_ads_position": float(np.mean([v for v in columns["avg_ads_position"] if v > 0])),
            "organic_displacement": float(np.mean(columns["organic_displacement"])),
            "total_clicks": int(sum(a.clicks for a in agents.values())),
        },
        "advertisers": [
            {
                "advertiser_id": a.advertiser.advertiser_id,
                "agent_type": a.agent_type,
                "value_per_click": round(a.advertiser.value_per_click, 4),
                "budget": round(a.advertiser.daily_budget, 4),
                "spend": round(a.spend, 4),
                "budget_utilization": round(a.spend / max(a.advertiser.daily_budget, 1e-9), 4),
                "clicks": a.clicks,
                "conversions": round(a.conversions, 2),
                "surplus": round(a.surplus, 4),
                "wins": a.wins,
            }
            for a in agents.values()
        ],
    }
    with (run_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Auction outcomes written to %s", run_dir)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    platform = summary["platform"]
    logger.info("Platform metrics (%s, %d rounds):", summary["mechanism"], summary["rounds"])
    logger.info(
        "  revenue=%.2f  RPM=%.4f  fill_rate=%.4f  avg_ads_position=%.3f  "
        "organic_displacement=%.5f  clicks=%d",
        platform["revenue_expected"],
        platform["rpm"],
        platform["fill_rate"],
        platform["avg_ads_position"],
        platform["organic_displacement"],
        platform["total_clicks"],
    )
    rows = summary["advertisers"]
    paced = [r for r in rows if r["agent_type"] == "paced" and r["wins"] >= 50]
    if paced:
        util = np.array([r["budget_utilization"] for r in paced])
        active = int(((util >= 0.5) & (util <= 1.05)).sum())
        exhausted = int((util > 1.05).sum())
        logger.info(
            "Paced agents with >=50 wins (%d): utilization p10=%.2f p50=%.2f p90=%.2f | "
            "pacing-active=%d exhausted=%d",
            len(paced),
            np.percentile(util, 10),
            np.percentile(util, 50),
            np.percentile(util, 90),
            active,
            exhausted,
        )
    top = sorted(rows, key=lambda r: -r["spend"])[:5]
    logger.info(
        "Top spenders: %s",
        ", ".join(f"{r['advertiser_id']}({r['agent_type']})={r['spend']:.2f}" for r in top),
    )

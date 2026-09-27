"""Experiment harness: full cross product of auction mechanism x promo targeting.

For each (mechanism, policy, seed) cell the harness reports platform ad revenue,
total advertiser surplus, consumer welfare (mean relevance of the shown slate),
incremental conversions attributable to the promo (computed from the CATE model and
validated against ground truth), promo spend, and incremental conversions per promo
dollar. The full grid is written to reports/results.csv.
"""

from __future__ import annotations

import csv
import logging
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq

from mua.auction.mechanisms import MECHANISMS
from mua.auction.simulator import assign_agents, build_advertisers
from mua.auction.types import AuctionRequest, Bid, Candidate
from mua.causal.data import load_frame
from mua.config import resolve

logger = logging.getLogger(__name__)

POLICIES = ("treat_none", "treat_all", "random_k", "uplift_top_k", "propensity_top_k")


def read_results(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    with resolve(cfg["paths"]["results"]).open() as fh:
        return list(csv.DictReader(fh))


COLUMNS = [
    "mechanism",
    "policy",
    "seed",
    "revenue",
    "advertiser_surplus",
    "consumer_welfare",
    "inc_conversions_true",
    "inc_conversions_model",
    "promo_spend",
    "per_dollar_true",
    "per_dollar_model",
    "orders_realized",
]


def _policy_mask(
    policy: str, pool_size: int, k: int, scores: dict[str, np.ndarray], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(pool_size, dtype=bool)
    if policy == "treat_none":
        return mask
    if policy == "treat_all":
        return np.ones(pool_size, dtype=bool)
    top = max(int(pool_size * k / 100), 1)
    if policy == "random_k":
        mask[rng.choice(pool_size, size=top, replace=False)] = True
    elif policy == "uplift_top_k":
        mask[np.argsort(-scores["cate"])[:top]] = True
    elif policy == "propensity_top_k":
        mask[np.argsort(-scores["propensity"])[:top]] = True
    else:
        raise ValueError(f"Unknown policy {policy}")
    return mask


def _load_pool(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the promo-session pool: ground truth + CATE + propensity predictions."""
    causal = cfg["causal"]
    artifacts = resolve(causal["artifacts"])
    best = causal.get("best_cate", "auto")
    if best == "auto":
        with (artifacts / "leaderboard.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise FileNotFoundError(
                f"No leaderboard at {artifacts / 'leaderboard.csv'}; run `causal` first"
            )
        best = rows[0]["estimator"]
    logger.info("Best CATE model for targeting: %s", best)

    npz = np.load(artifacts / f"cate_test_{best}.npz")
    ids = npz["impression_id"].astype(np.int64)
    cate = npz["cate"].astype(np.float64)

    frame = load_frame(
        causal,
        "test",
        sample=causal.get("sample"),
        seed=int(causal["seed"]),
        with_truth=True,
    )
    if not np.array_equal(frame["ids"], ids):
        raise ValueError(
            "Causal CATE predictions do not align with the processed test split; "
            "rerun `causal` (and `features`) on the current data."
        )

    truth = pq.read_table(resolve(causal["paths"]["truth"]))
    t_ids = truth["impression_id"].to_numpy().astype(np.int64)
    t_tau = truth["true_tau"].to_numpy().astype(np.float64)
    t_base = truth["baseline_order_prob"].to_numpy().astype(np.float64)
    order = np.argsort(t_ids)
    pos = np.minimum(np.searchsorted(t_ids[order], ids), len(t_ids) - 1)
    if not np.all(t_ids[order][pos] == ids):
        raise ValueError("Truth join mismatch in experiment pool")

    propensity_model = joblib.load(artifacts / "propensity.joblib")
    ps_hat = propensity_model.predict_proba(frame["X"])[:, 1]

    scored = pq.read_table(resolve(cfg["paths"]["scored"]) / "test")
    s_ids = scored["impression_id"].to_numpy().astype(np.int64)
    s_order = np.argsort(s_ids)
    s_pos = np.searchsorted(s_ids[s_order], ids)
    s_pos = np.minimum(s_pos, len(s_ids) - 1)
    if not np.all(s_ids[s_order][s_pos] == ids):
        raise ValueError("Scored/processed impression-id mismatch")

    pool = {
        "ids": ids,
        "cate": cate,
        "propensity": ps_hat,
        "tau": t_tau[order][pos],
        "baseline": t_base[order][pos],
        "promo_cost": scored["promo_cost_usd"].to_numpy().astype(np.float64)[s_order][s_pos],
        "size": len(ids),
    }
    logger.info("Promo session pool: %d rows", pool["size"])
    return pool


def _run_auction_cells(
    cfg: dict[str, Any], rounds: int, seeds: list[int]
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Run auction rounds per (mechanism, seed); returns per-round C, R, W arrays.

    C[t] = sum over winners of value_m/conv_m * pctr (session-scaling coefficient),
    R[t] = expected revenue, W[t] = mean relevance of the final shown slate.
    """
    scored = pq.read_table(resolve(cfg["paths"]["scored"]) / "test")
    n_rows = scored.num_rows
    merchant_col = scored["merchant_id"].to_numpy()
    pctr_col = scored["pctr"].to_numpy().astype(np.float64)
    rel_col = scored["pctr_raw"].to_numpy().astype(np.float64)
    slate_size = int(cfg["auction"]["slate_size"])
    reserve = float(cfg["auction"]["reserve"])

    cells: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for mechanism_name in cfg["experiment"]["mechanisms"]:
        auction_cfg = {**cfg, "auction": {**cfg["auction"], "mechanism": mechanism_name}}
        if mechanism_name == "second_price":
            auction_cfg["auction"]["slot_positions"] = [1]
        advertisers = build_advertisers(scored, rounds, auction_cfg)
        value_by_merchant = {
            a.merchant_id: (a.value_per_click / max(a.conversion_rate, 1e-6), a.value_per_click)
            for a in advertisers
        }
        mechanism = MECHANISMS[mechanism_name]()
        for seed in seeds:
            agents = assign_agents(advertisers, auction_cfg, seed)
            rng = np.random.default_rng(seed)
            slate_idx = rng.integers(0, n_rows, size=(rounds, slate_size))
            C = np.zeros(rounds)
            R = np.zeros(rounds)
            W = np.zeros(rounds)
            for t in range(rounds):
                rows = slate_idx[t]
                bids: list[Bid] = []
                candidates: list[Candidate] = []
                for j in range(slate_size):
                    merchant_id = int(merchant_col[rows[j]])
                    candidate = Candidate(
                        merchant_id=merchant_id,
                        pctr=float(pctr_col[rows[j]]),
                        organic_relevance=float(rel_col[rows[j]]),
                    )
                    candidates.append(candidate)
                    agent = agents.get(merchant_id)
                    if agent is not None:
                        bids.append(agent.make_bid(candidate))
                request = AuctionRequest(
                    request_id=t,
                    candidates=tuple(candidates),
                    reserve=reserve,
                    slot_positions=tuple(auction_cfg["auction"]["slot_positions"]),
                )
                outcome = mechanism.allocate(request, bids)
                organic = np.sort([c.organic_relevance for c in candidates])[::-1]
                displaced = 0.0
                for i, allocation in enumerate(outcome.allocations):
                    coef, _ = value_by_merchant[allocation.merchant_id]
                    C[t] += coef * allocation.pctr
                    R[t] += allocation.payment * allocation.pctr
                    position = auction_cfg["auction"]["slot_positions"][i]
                    r_org = organic[position - 1] if position - 1 < len(organic) else 0.0
                    displaced += r_org - allocation.organic_relevance
                W[t] = (organic.sum() + displaced) / slate_size
                winners_by_agent = {a.advertiser_id: a for a in outcome.allocations}
                for bid in bids:
                    agents[bid.merchant_id].observe(winners_by_agent.get(bid.advertiser_id), False)
                    agents[bid.merchant_id].tick(t, rounds)
            cells[(mechanism_name, seed)] = {"C": C, "R": R, "W": W}
            logger.info(
                "Auction cells done: %s seed=%d revenue=%.1f",
                mechanism_name,
                seed,
                R.sum(),
            )
    return cells


def run(cfg: dict[str, Any], scale: str = "full") -> list[dict[str, Any]]:
    exp = cfg["experiment"]
    seeds = list(exp["seeds"])
    rounds = int(exp["rounds_per_cell"])
    if scale == "small":
        seeds = list(exp.get("small", {}).get("seeds", seeds[:3]))
        rounds = int(exp.get("small", {}).get("rounds_per_cell", 3000))
        logger.info("Small scale: %d rounds x seeds %s", rounds, seeds)
    top_k = int(exp["top_k_pct"])

    pool = _load_pool(cfg)
    auction_cells = _run_auction_cells(cfg, rounds, seeds)

    # Promo sessions: sampled without replacement per seed, aligned to auction rounds.
    sessions: dict[int, np.ndarray] = {}
    promo: dict[tuple[str, int], dict[str, float]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sessions[seed] = np.sort(rng.choice(pool["size"], size=rounds, replace=False))
        idx = sessions[seed]
        base = pool["baseline"][idx]
        tau = pool["tau"][idx]
        cost = pool["promo_cost"][idx]
        cate = pool["cate"][idx]
        ps = pool["propensity"][idx]
        order_rng = np.random.default_rng(1000 + seed)
        for policy in exp["policies"]:
            mask = _policy_mask(policy, rounds, top_k, {"cate": cate, "propensity": ps}, seed)
            p_treated = np.clip(base + mask * tau, 0.0, 1.0)
            spend = float((mask * cost).sum())
            inc_true = float((mask * (p_treated - base)).sum())
            inc_model = float((mask * cate).sum())
            orders = int(order_rng.binomial(1, p_treated).sum())
            promo[(policy, seed)] = {
                "spend": spend,
                "inc_true": inc_true,
                "inc_model": inc_model,
                "orders": orders,
                "p_treated": p_treated,
            }

    rows: list[dict[str, Any]] = []
    for mechanism_name in exp["mechanisms"]:
        for policy in exp["policies"]:
            for seed in seeds:
                cell = auction_cells[(mechanism_name, seed)]
                p = promo[(policy, seed)]
                revenue = float(cell["R"].sum())
                surplus = float((p["p_treated"] * cell["C"]).sum() - revenue)
                welfare = float(cell["W"].mean())
                spend = p["spend"]
                per_dollar_true = p["inc_true"] / spend if spend > 0 else float("nan")
                per_dollar_model = p["inc_model"] / spend if spend > 0 else float("nan")
                rows.append(
                    {
                        "mechanism": mechanism_name,
                        "policy": policy,
                        "seed": seed,
                        "revenue": revenue,
                        "advertiser_surplus": surplus,
                        "consumer_welfare": welfare,
                        "inc_conversions_true": p["inc_true"],
                        "inc_conversions_model": p["inc_model"],
                        "promo_spend": spend,
                        "per_dollar_true": per_dollar_true,
                        "per_dollar_model": per_dollar_model,
                        "orders_realized": p["orders"],
                    }
                )

    results_path = resolve(cfg["paths"]["results"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Experiment grid written to %s (%d rows)", results_path, len(rows))
    return rows

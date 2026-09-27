from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mua.auction.mechanisms import GSP, VCG, FirstPrice, SecondPrice
from mua.auction.types import AuctionRequest, Bid, Candidate


def _bids(amounts, pctrs=None):
    pctrs = pctrs or [0.1] * len(amounts)
    return [
        Bid(
            advertiser_id=f"a{i}",
            merchant_id=i,
            value=amounts[i],
            amount=amounts[i],
            pctr=pctrs[i],
            organic_relevance=pctrs[i],
        )
        for i in range(len(amounts))
    ]


def _request(bids, reserve=0.0, slots=(1, 2, 3)):
    return AuctionRequest(
        request_id=0,
        candidates=tuple(
            Candidate(merchant_id=b.merchant_id, pctr=b.pctr, organic_relevance=b.pctr)
            for b in bids
        ),
        reserve=reserve,
        slot_positions=slots,
    )


def _payments(mechanism, bids, reserve=0.0, slots=(1, 2, 3)):
    outcome = mechanism.allocate(_request(bids, reserve, slots), bids)
    return {a.advertiser_id: a.payment for a in outcome.allocations}


class TestPaymentFormulas:
    def test_first_price_pays_own_bid(self):
        bids = _bids([2.0, 1.5, 1.0])
        payments = _payments(FirstPrice(), bids)
        assert payments["a0"] == 2.0

    def test_second_price_pays_runner_up_e_cpm_over_own_ctr(self):
        # eCPM ranking: a0=3.0*0.10=0.30, a1=1.0*0.20=0.20, a2=0.5*0.10=0.05
        bids = _bids([3.0, 1.0, 0.5], pctrs=[0.1, 0.2, 0.1])
        payments = _payments(SecondPrice(), bids, slots=(1,))
        assert payments["a0"] == pytest.approx(1.0 * 0.2 / 0.1)

    def test_gsp_quality_adjusted_next_price(self):
        # eCPM ranking: 3.0*0.10=0.30, 2.0*0.10=0.20, 1.0*0.10=0.10
        bids = _bids([3.0, 2.0, 1.0])
        payments = _payments(GSP(), bids)
        assert payments["a0"] == pytest.approx(2.0)
        assert payments["a1"] == pytest.approx(1.0)
        assert payments["a2"] == pytest.approx(0.0)  # last slot pays reserve

    def test_gsp_with_heterogeneous_ctr(self):
        # eCPM: a0=3.0*0.05=0.15, a1=2.0*0.10=0.20, a2=1.0*0.20=0.20 -> ranking a1, a2, a0
        bids = _bids([3.0, 2.0, 1.0], pctrs=[0.05, 0.10, 0.20])
        payments = _payments(GSP(), bids)
        assert payments["a1"] == pytest.approx(1.0 * 0.20 / 0.10)
        assert payments["a2"] == pytest.approx(3.0 * 0.05 / 0.20)
        assert payments["a0"] == pytest.approx(0.0)  # last slot pays reserve

    def test_vcg_pays_k_plus_1_externality(self):
        bids = _bids([4.0, 3.0, 2.0, 1.0], pctrs=[0.1, 0.1, 0.1, 0.1])
        payments = _payments(VCG(), bids)
        # K=3 -> externality = 4th eCPM = 1.0*0.1; price = 0.1/0.1 = 1.0 for all
        for winner in ("a0", "a1", "a2"):
            assert payments[winner] == pytest.approx(1.0)

    def test_gsp_k1_matches_second_price(self):
        bids = _bids([3.0, 2.0, 1.0], pctrs=[0.1, 0.1, 0.1])
        gsp = _payments(GSP(), bids, slots=(1,))
        sp = _payments(SecondPrice(), bids, slots=(1,))
        assert gsp == sp

    def test_reserve_enforced(self):
        bids = _bids([3.0, 0.4, 0.1])
        payments = _payments(GSP(), bids, reserve=0.5)
        assert set(payments) == {"a0"}
        assert payments["a0"] == pytest.approx(0.5)
        payments_vcg = _payments(VCG(), bids, reserve=0.5)
        assert set(payments_vcg) == {"a0"}

    def test_slot_positions_and_fill(self):
        bids = _bids([4.0, 3.0, 2.0, 1.0])
        outcome = GSP().allocate(_request(bids, slots=(1, 4, 9)), bids)
        assert [a.slot for a in outcome.allocations] == [1, 4, 9]
        assert outcome.fill_rate == 1.0
        sparse = GSP().allocate(_request(bids[:1], slots=(1, 4, 9)), bids[:1])
        assert sparse.fill_rate == pytest.approx(1 / 3)
        assert len(sparse.allocations) == 1

    def test_second_price_rejects_multiple_slots(self):
        bids = _bids([2.0, 1.0])
        with pytest.raises(ValueError, match="one slot"):
            SecondPrice().allocate(_request(bids, slots=(1, 2)), bids)


class TestDisplacement:
    def test_organic_displacement_metric(self):
        bids = _bids([3.0, 2.0, 1.0], pctrs=[0.05, 0.08, 0.1])
        # organic relevances are the pctrs; ads with low relevance displace better
        # organic items.
        candidates = [
            Candidate(merchant_id=0, pctr=0.05, organic_relevance=0.02),
            Candidate(merchant_id=1, pctr=0.08, organic_relevance=0.15),
            Candidate(merchant_id=2, pctr=0.1, organic_relevance=0.09),
        ]
        request = AuctionRequest(request_id=0, candidates=tuple(candidates), reserve=0.0)
        outcome = GSP().allocate(request, bids)
        organic = sorted([c.organic_relevance for c in candidates], reverse=True)
        expected = 0.0
        for i, allocation in enumerate(outcome.allocations):
            expected += max(0.0, organic[i] - allocation.organic_relevance)
        expected /= len(outcome.allocations)
        assert outcome.organic_displacement == pytest.approx(expected)


def _write_toy_scored(root: Path, rows: int = 4000, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    n_merchants = 60
    merchant = rng.integers(0, n_merchants, rows)
    is_ads = (rng.random(n_merchants) < 0.3).astype(np.int32)
    pctr = rng.uniform(0.01, 0.12, rows).astype(np.float64)
    clicked = (rng.random(rows) < pctr).astype(np.int32)
    ordered = (clicked == 1) & (rng.random(rows) < 0.08)
    order_value = np.where(ordered, rng.uniform(15, 45, rows), 0.0).astype(np.float64)
    table = pa.table(
        {
            "merchant_id": merchant.astype(np.int64),
            "pctr": pctr,
            "pctr_raw": pctr,
            "clicked": clicked,
            "ordered": ordered.astype(np.int32),
            "order_value_usd": order_value,
            "is_ads_advertiser": is_ads[merchant],
            "price_tier": rng.integers(1, 5, n_merchants)[merchant].astype(np.int32),
            "avg_rating": rng.uniform(2.5, 5.0, n_merchants)[merchant].astype(np.float64),
            "historical_ctr": rng.uniform(0.01, 0.2, n_merchants)[merchant].astype(np.float64),
        }
    )
    out = root / "scored" / "test"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "part-0.parquet")


def _toy_cfg(root: Path, rounds: int = 300, mechanism: str = "gsp") -> dict:
    return {
        "paths": {
            "scored": str(root / "scored"),
            "auction_runs": str(root / "auction_runs"),
            "figures": str(root / "reports" / "figures"),
        },
        "auction": {
            "mechanism": mechanism,
            "rounds": rounds,
            "seed": 3,
            "slate_size": 20,
            "slot_positions": [1, 2, 3],
            "reserve": 0.5,
            "agent_mix": {"truthful": 0.5, "shading": 0.2, "paced": 0.2, "random": 0.1},
            "shading": {"initial": 0.8, "step": 0.02, "min": 0.3, "max": 1.0},
            "pacing": {"learning_rate": 0.3, "lam_max": 10.0, "budget_ratio": 0.7},
            "run_id": "test_run",
        },
    }


class TestSimulatorSmoke:
    def test_end_to_end(self, tmp_path):
        from mua.auction.simulator import run

        _write_toy_scored(tmp_path)
        cfg = _toy_cfg(tmp_path)
        summary = run(cfg)
        run_dir = tmp_path / "auction_runs" / "test_run"
        outcomes = pq.read_table(run_dir / "outcomes.parquet")
        assert outcomes.num_rows == 300
        assert summary["platform"]["revenue_expected"] > 0
        assert 0 < summary["platform"]["fill_rate"] <= 1
        assert (run_dir / "summary.json").exists()
        with (run_dir / "summary.json").open() as fh:
            loaded = json.load(fh)
        assert loaded["mechanism"] == "gsp"

    def test_deterministic_with_seed(self, tmp_path):
        from mua.auction.simulator import run

        _write_toy_scored(tmp_path)
        cfg = _toy_cfg(tmp_path, rounds=200)
        summary1 = run(cfg)
        summary2 = run(cfg)
        assert summary1["platform"]["revenue_expected"] == summary2["platform"]["revenue_expected"]

    def test_second_price_single_slot(self, tmp_path):
        from mua.auction.simulator import run

        _write_toy_scored(tmp_path)
        cfg = _toy_cfg(tmp_path, rounds=100, mechanism="second_price")
        cfg["auction"]["slot_positions"] = [1, 2]
        summary = run(cfg)
        outcomes = pq.read_table(tmp_path / "auction_runs" / "test_run" / "outcomes.parquet")
        assert int(outcomes["n_allocations"].to_numpy().max()) <= 1
        assert summary["mechanism"] == "second_price"

    def test_advertiser_values_derived_from_data(self, tmp_path):
        from mua.auction.simulator import build_advertisers

        _write_toy_scored(tmp_path)
        cfg = _toy_cfg(tmp_path)
        table = pq.read_table(tmp_path / "scored" / "test")
        advertisers = build_advertisers(table, 300, cfg)
        assert len(advertisers) > 0
        for advertiser in advertisers:
            assert advertiser.value_per_click > 0
            assert advertiser.daily_budget > 0
            assert 0 <= advertiser.conversion_rate <= 1

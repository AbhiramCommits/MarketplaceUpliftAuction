from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mua.ranker.calibrate import expected_calibration_error
from mua.ranker.dataset import GroupShuffleSampler, RankerDataset, ranker_collate
from mua.ranker.model import CTRRanker
from mua.ranker.train import build_feature_spec

CUISINES = [
    "american",
    "burgers",
    "chinese",
    "dessert",
    "indian",
    "italian",
    "korean",
    "mexican",
    "pizza",
    "sushi",
    "thai",
    "vietnamese",
]

NUMERIC = [
    "consumer_merchant_distance_km",
    "current_dasher_supply_ratio",
    "estimated_delivery_time_min",
    "is_peak",
    "price_sensitivity",
    "order_frequency_prior",
    "distance_tolerance_km",
    "is_new_user",
    "tenure_days",
    "avg_rating",
    "avg_prep_time_min",
    "historical_ctr",
    "is_ads_advertiser",
    "consumer_x_km",
    "consumer_y_km",
    "merchant_x_km",
    "merchant_y_km",
]


def _write_toy_data(root: Path, rows: int = 8000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_consumers, n_merchants = 500, 200
    c_id = rng.integers(0, n_consumers, rows)
    m_id = rng.integers(0, n_merchants, rows)
    ctr = rng.uniform(0.01, 0.20, n_merchants)[m_id].astype(np.float32)
    tol = rng.uniform(1.0, 10.0, n_consumers)[c_id].astype(np.float32)
    dist = rng.uniform(0.2, 10.0, rows).astype(np.float32)
    hour = rng.integers(0, 24, rows).astype(np.int32)
    dow = rng.integers(0, 7, rows).astype(np.int32)
    peak = (((hour >= 11) & (hour <= 13)) | ((hour >= 17) & (hour <= 20))).astype(np.int32)
    rank = rng.integers(1, 21, rows).astype(np.int32)
    relevance = (1.0 + 0.35 * np.clip((tol - dist) / 6.0, -1.0, 1.0)) * (1.0 + 0.08 * peak)
    p = np.clip(ctr * relevance / np.log2(rank + 1.0), 0.0, 0.5)
    clicked = (rng.random(rows) < p).astype(np.int32)

    day = (np.arange(rows) % 30 + 1).astype(np.int32)
    cuisine_idx = rng.integers(0, len(CUISINES), rows).astype(np.int32)
    columns = {
        "impression_id": np.arange(rows, dtype=np.int64),
        "day": day,
        "hour_of_day": hour,
        "day_of_week": dow,
        "is_peak": peak,
        "consumer_id": c_id.astype(np.int64),
        "merchant_id": m_id.astype(np.int64),
        "slate_rank": rank,
        "consumer_merchant_distance_km": dist,
        "current_dasher_supply_ratio": rng.uniform(0.5, 2.0, rows).astype(np.float32),
        "estimated_delivery_time_min": rng.uniform(15.0, 60.0, rows).astype(np.float32),
        "promo_treated": (rng.random(rows) < 0.35).astype(np.int32),
        "clicked": clicked,
        "ordered": (rng.random(rows) < 0.05).astype(np.int32),
        "order_value_usd": rng.uniform(10, 60, rows).astype(np.float32),
        "delivery_fee_usd": rng.uniform(1.5, 8.0, rows).astype(np.float32),
        "promo_cost_usd": rng.uniform(0.0, 3.0, rows).astype(np.float32),
        "price_sensitivity": rng.beta(2, 5, rows).astype(np.float32),
        "order_frequency_prior": rng.uniform(0.2, 5.0, rows).astype(np.float32),
        "distance_tolerance_km": tol,
        "is_new_user": (rng.random(rows) < 0.3).astype(np.int32),
        "tenure_days": rng.integers(1, 1000, rows).astype(np.int32),
        "avg_rating": rng.uniform(2.5, 5.0, n_merchants)[m_id].astype(np.float32),
        "avg_prep_time_min": rng.uniform(5.0, 40.0, n_merchants)[m_id].astype(np.float32),
        "price_tier": rng.integers(1, 5, n_merchants)[m_id].astype(np.int32),
        "historical_ctr": ctr,
        "is_ads_advertiser": (rng.random(n_merchants) < 0.2)[m_id].astype(np.int32),
        "consumer_x_km": rng.uniform(0, 10, rows).astype(np.float32),
        "consumer_y_km": rng.uniform(0, 10, rows).astype(np.float32),
        "merchant_x_km": rng.uniform(0, 10, rows).astype(np.float32),
        "merchant_y_km": rng.uniform(0, 10, rows).astype(np.float32),
    }
    for i, cat in enumerate(CUISINES):
        columns[f"cuisine_{cat}"] = (cuisine_idx == i).astype(np.int32)

    table = pa.table(columns)
    splits = {"train": (1, 21), "valid": (22, 25), "test": (26, 30)}
    for split, (lo, hi) in splits.items():
        mask = (day >= lo) & (day <= hi)
        out = root / "processed" / split
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.filter(pa.array(mask)), out / "part-0.parquet")


def _toy_cfg(root: Path) -> dict:
    return {
        "paths": {
            "processed": str(root / "processed"),
            "scored": str(root / "scored"),
            "ranker_artifacts": str(root / "artifacts" / "ranker"),
            "figures": str(root / "reports" / "figures"),
        },
        "data": {
            "label": "clicked",
            "in_memory": True,
            "batch_size": 1024,
            "categorical": {
                "consumer_id": {"vocab_size": 65536, "embedding_dim": 8, "hash_buckets": 65536},
                "merchant_id": {"vocab_size": 8192, "embedding_dim": 8, "hash_buckets": 8192},
                "cuisine": {"vocab_size": 12, "embedding_dim": 4},
                "hour_of_day": {"vocab_size": 24, "embedding_dim": 4},
                "day_of_week": {"vocab_size": 7, "embedding_dim": 4},
                "price_tier": {"vocab_size": 4, "embedding_dim": 4, "offset": 1},
            },
            "numeric": NUMERIC,
            "bias": {"feature": "slate_rank", "vocab_size": 21, "embedding_dim": 4},
        },
        "model": {"mlp_hidden": [32, 16], "dropout": 0.2},
        "training": {
            "epochs": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "warmup_epochs": 0,
            "lr_min_ratio": 0.05,
            "grad_clip_norm": 5.0,
            "patience": 2,
            "seed": 0,
            "mlflow_enabled": False,
            "mlflow_experiment": "test_ctr_ranker",
            "mlflow_tracking_uri": str(root / "mlruns"),
        },
        "calibration": {"n_bins": 10},
        "scoring": {"splits": ["train", "valid", "test"], "batch_size": 1024},
        "targets": {"test_roc_auc": 0.70, "test_ece": 0.02},
    }


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("ranker_toy")
    _write_toy_data(root)
    cfg = _toy_cfg(root)
    spec = build_feature_spec(cfg)
    return root, cfg, spec


class TestDataset:
    def test_in_memory_and_chunked_match(self, toy):
        root, _, spec = toy
        in_mem = RankerDataset(root / "processed" / "train", spec, mode="in_memory")
        chunked = RankerDataset(root / "processed" / "train", spec, mode="chunked")
        assert len(in_mem) == len(chunked)
        for idx in (0, 13, len(in_mem) - 1):
            a, b = in_mem[idx], chunked[idx]
            assert a["label"] == b["label"]
            assert a["bias"] == b["bias"]
            np.testing.assert_allclose(a["numeric"], b["numeric"], rtol=1e-6)
            for name in a["categorical"]:
                assert a["categorical"][name] == b["categorical"][name], name

    def test_cuisine_derivation(self, toy):
        root, _, spec = toy
        dataset = RankerDataset(root / "processed" / "train", spec, mode="in_memory")
        table = pq.read_table(root / "processed" / "train")
        row = int(np.argmax(table["cuisine_sushi"].to_numpy()))
        expected = sorted(CUISINES).index("sushi")
        assert dataset[row]["categorical"]["cuisine"] == expected

    def test_hash_bounds_and_offsets(self, toy):
        root, _, spec = toy
        dataset = RankerDataset(root / "processed" / "train", spec, mode="chunked")
        item = dataset[0]
        assert 0 <= item["categorical"]["consumer_id"] < 65536
        assert 0 <= item["categorical"]["merchant_id"] < 8192
        assert 0 <= item["categorical"]["price_tier"] < 4
        assert 0 <= item["categorical"]["cuisine"] < 12

    def test_numeric_normalization(self, toy):
        root, _, spec = toy
        dataset = RankerDataset(root / "processed" / "train", spec, mode="in_memory")
        raw = pq.read_table(root / "processed" / "train")[
            "consumer_merchant_distance_km"
        ].to_numpy()
        stats = spec["numeric_stats"]["consumer_merchant_distance_km"]
        col = spec["numeric"].index("consumer_merchant_distance_km")
        for idx in (0, 100):
            expected = (raw[idx] - stats["mean"]) / stats["std"]
            np.testing.assert_allclose(dataset[idx]["numeric"][col], expected, rtol=1e-6)


class TestGroupShuffleSampler:
    def test_covers_all_indices_with_group_contiguity(self):
        starts = np.array([0, 5, 12])
        sizes = np.array([5, 7, 9])
        sampler = GroupShuffleSampler(starts, sizes, seed=1)
        seen = list(sampler)
        assert sorted(seen) == list(range(21))
        assert len(sampler) == 21
        groups_seen: list[int] = []
        previous = None
        for idx in seen:
            group = int(np.searchsorted(starts, idx, side="right") - 1)
            if group != previous:
                groups_seen.append(group)
                previous = group
        assert sorted(groups_seen) == [0, 1, 2]
        assert len(groups_seen) == 3


class TestModel:
    def test_forward_and_position_bias(self, toy):
        _, _, spec = toy
        model = CTRRanker(spec, {"mlp_hidden": [32, 16], "dropout": 0.2})
        model.eval()
        batch = {
            "categorical": {
                name: torch.randint(0, int(s["vocab_size"]), (64,))
                for name, s in spec["categorical"].items()
            },
            "numeric": torch.randn(64, len(spec["numeric"])),
            "bias": torch.randint(1, 21, (64,)),
        }
        with torch.no_grad():
            out = model(batch, include_bias=True)
            out_no_bias = model(batch, include_bias=False)
            batch_high = {**batch, "bias": torch.full((64,), 20)}
            out_high = model(batch_high, include_bias=True)
            out_high_no_bias = model(batch_high, include_bias=False)
        assert out.shape == (64,)
        assert bool((out > 0).all()) and bool((out < 1).all())
        assert torch.equal(out_no_bias, out_high_no_bias)
        assert not torch.equal(out, out_high)

    def test_collate_shapes(self, toy):
        root, _, spec = toy
        dataset = RankerDataset(root / "processed" / "train", spec, mode="chunked")
        batch = ranker_collate([dataset[i] for i in range(32)])
        assert batch["numeric"].shape == (32, len(spec["numeric"]))
        assert batch["bias"].shape == (32,)
        assert batch["label"].shape == (32,)
        assert batch["categorical"]["consumer_id"].shape == (32,)


class TestECE:
    def test_perfectly_calibrated_scores(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.01, 0.5, 100_000)
        y = (rng.random(100_000) < p).astype(float)
        assert expected_calibration_error(y, p, 10) < 0.01

    def test_miscalibration_detected(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.01, 0.5, 100_000)
        y = (rng.random(100_000) < p).astype(float)
        assert expected_calibration_error(y, p + 0.05, 10) > 0.03


class TestTrainAndScoreSmoke:
    def test_end_to_end(self, toy):
        root, cfg, _ = toy
        from mua.ranker.predict import run as predict_run
        from mua.ranker.train import run as train_run

        summary, ok = train_run(cfg)
        artifacts = Path(cfg["paths"]["ranker_artifacts"])
        assert (artifacts / "best_model.pt").exists()
        assert (artifacts / "feature_spec.json").exists()
        assert (artifacts / "metrics.csv").exists()
        assert (artifacts / "calibrator.joblib").exists()
        assert (Path(cfg["paths"]["figures"]) / "calibration.png").exists()
        for key in (
            "best_epoch",
            "val_roc_auc",
            "test_roc_auc",
            "test_ece_before",
            "test_ece_after",
        ):
            assert key in summary, key
        assert ok in (True, False)

        predict_run(cfg)
        for split in ("train", "valid", "test"):
            processed = pq.read_table(Path(cfg["paths"]["processed"]) / split)
            scored = pq.read_table(Path(cfg["paths"]["scored"]) / split)
            assert scored.num_rows == processed.num_rows
            assert "pctr" in scored.column_names
            assert "pctr_raw" in scored.column_names
            assert bool((scored["pctr"].to_numpy() >= 0).all())
            assert bool((scored["pctr"].to_numpy() <= 1).all())

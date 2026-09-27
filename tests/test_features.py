from __future__ import annotations

from pyspark.sql import functions as F

from mua.features.build import SPLIT_ORDER, split_ranges

FEATURES_CFG = {
    "split": {"train": [1, 21], "valid": [22, 25], "test": [26, 30]},
}


def test_split_ranges():
    ranges = split_ranges(FEATURES_CFG)
    assert list(ranges) == list(SPLIT_ORDER)
    assert ranges == {"train": (1, 21), "valid": (22, 25), "test": (26, 30)}


def test_split_is_contiguous_time_partition():
    ranges = split_ranges(FEATURES_CFG)
    assert ranges["train"][1] + 1 == ranges["valid"][0]
    assert ranges["valid"][1] + 1 == ranges["test"][0]
    assert ranges["train"][0] == 1
    assert ranges["test"][1] == 30


def test_end_to_end_generate_and_build(spark, tmp_path):
    from mua.features.build import build
    from mua.sim.generate import run

    sim_cfg = {
        "seed": 42,
        "n_days": 30,
        "n_consumers": 2_000,
        "n_merchants": 500,
        "n_dashers": 200,
        "target_rows": 8_000,
        "city_size_km": 10.0,
        "treatment": {"target_share": 0.35},
        "paths": {
            "raw_impressions": str(tmp_path / "raw" / "impressions"),
            "raw_consumers": str(tmp_path / "raw" / "consumers"),
            "raw_merchants": str(tmp_path / "raw" / "merchants"),
            "raw_dashers": str(tmp_path / "raw" / "dashers"),
            "truth": str(tmp_path / "truth"),
        },
    }

    stats = run(sim_cfg)
    assert stats["rows"] == 8_000
    assert abs(stats["treated_share"] - 0.35) < 0.03

    feat_cfg = {
        "paths": {
            "impressions": sim_cfg["paths"]["raw_impressions"],
            "consumers": sim_cfg["paths"]["raw_consumers"],
            "merchants": sim_cfg["paths"]["raw_merchants"],
            "processed": str(tmp_path / "processed"),
        },
        "split": {"train": [1, 21], "valid": [22, 25], "test": [26, 30]},
    }

    counts = build(feat_cfg, spark=spark)
    assert counts["train"] + counts["valid"] + counts["test"] == counts["total"] == 8_000

    test = spark.read.parquet(str(tmp_path / "processed" / "test"))
    columns = test.columns
    assert "cuisine_pizza" in columns
    assert "price_sensitivity" in columns
    assert "promo_treated" in columns
    assert "true_propensity" not in columns
    assert "true_tau" not in columns
    assert "baseline_order_prob" not in columns
    assert test.filter(F.col("cuisine_pizza") == 1).count() > 0

    train = spark.read.parquet(str(tmp_path / "processed" / "train"))
    valid = spark.read.parquet(str(tmp_path / "processed" / "valid"))
    assert train.agg(F.min("day"), F.max("day")).collect()[0] == (1, 21)
    assert valid.agg(F.min("day"), F.max("day")).collect()[0] == (22, 25)
    assert test.agg(F.min("day"), F.max("day")).collect()[0] == (26, 30)

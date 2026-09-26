"""Join impressions to dimensions, encode categoricals, and split by time."""

from __future__ import annotations

import logging
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from mua.config import resolve
from mua.spark import get_spark

logger = logging.getLogger(__name__)

SPLIT_ORDER = ("train", "valid", "test")


def split_ranges(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {name: tuple(cfg["split"][name]) for name in SPLIT_ORDER}


def encode_cuisine(df: DataFrame) -> tuple[DataFrame, list[str]]:
    """One-hot encode cuisine_category into deterministic boolean columns."""
    categories = sorted(row[0] for row in df.select("cuisine_category").distinct().collect())
    for cat in categories:
        df = df.withColumn(
            f"cuisine_{cat}", F.when(F.col("cuisine_category") == cat, 1).otherwise(0)
        )
    return df.drop("cuisine_category"), [f"cuisine_{cat}" for cat in categories]


def build(cfg: dict[str, Any], spark: SparkSession | None = None) -> dict[str, int]:
    spark = spark or get_spark("mua-features")
    paths = cfg["paths"]

    impressions = spark.read.parquet(str(resolve(paths["impressions"])))
    consumers = spark.read.parquet(str(resolve(paths["consumers"])))
    merchants = spark.read.parquet(str(resolve(paths["merchants"])))

    joined = impressions.join(consumers, on="consumer_id", how="inner").join(
        merchants, on="merchant_id", how="inner"
    )
    joined, cuisine_cols = encode_cuisine(joined)

    counts: dict[str, int] = {}
    for split, (lo, hi) in split_ranges(cfg).items():
        part = joined.filter(F.col("day").between(lo, hi))
        out = resolve(paths["processed"]) / split
        part.write.mode("overwrite").parquet(str(out))
        counts[split] = part.count()
        logger.info("Wrote %s (%s rows) -> %s", split, counts[split], out)

    counts["total"] = joined.count()
    logger.info("Total joined rows: %d", counts["total"])
    logger.info("Cuisine one-hot columns: %s", cuisine_cols)
    logger.info("Feature columns (%d): %s", len(joined.columns), joined.columns)
    return counts

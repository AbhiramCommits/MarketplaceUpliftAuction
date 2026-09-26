"""Shared local Spark session factory."""

from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "mua") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

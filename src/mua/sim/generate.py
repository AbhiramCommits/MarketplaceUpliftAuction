"""Simulated three-sided marketplace impression log generator.

Entities: consumers, merchants, dashers. Produces an impression-level event log with a
confounded promo treatment and KNOWN ground-truth causal effects, so downstream uplift /
causal estimators can be scored against truth.

Structural outcome functions (baseline_order_prob, true_tau, treatment_propensity,
click_probability) are standalone, vectorized, and testable without Spark.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mua.config import resolve
from mua.spark import get_spark

logger = logging.getLogger(__name__)

CUISINES: list[str] = [
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

PEAK_LUNCH = (11, 13)
PEAK_DINNER = (17, 20)

DEFAULT_TREATMENT_COEFS: dict[str, float] = {
    "price_sensitivity": 1.2,
    "is_new_user": 0.9,
    "historical_ctr": 1.0,
    "is_peak": 0.35,
}

PROB_FLOOR = 0.001
PROB_CEIL = 0.95


def is_peak_hour(hour: Any) -> np.ndarray:
    """Peak windows: lunch 11-13, dinner 17-20."""
    hour = np.asarray(hour)
    return ((hour >= PEAK_LUNCH[0]) & (hour <= PEAK_LUNCH[1])) | (
        (hour >= PEAK_DINNER[0]) & (hour <= PEAK_DINNER[1])
    )


def _col(features: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(features[name], dtype=float)


def baseline_order_prob(features: dict[str, Any]) -> np.ndarray:
    """Untreated conversion probability, from an explicit structural model."""
    ps = _col(features, "price_sensitivity")
    freq = _col(features, "order_frequency_prior")
    tol = _col(features, "distance_tolerance_km")
    dist = _col(features, "consumer_merchant_distance_km")
    rating = _col(features, "avg_rating")
    tier = _col(features, "price_tier")
    ctr = _col(features, "historical_ctr")
    dt = _col(features, "estimated_delivery_time_min")
    peak = _col(features, "is_peak")
    new = _col(features, "is_new_user")

    freq_z = (freq - 1.5) / 1.0
    rating_z = (rating - 3.75) / 0.7
    dt_z = (dt - 35.0) / 10.0
    distance_penalty = 0.03 * np.clip((dist - tol) / 6.0, -1.0, 1.5)

    prob = (
        0.055
        + 0.014 * freq_z
        + 0.020 * rating_z
        - 0.045 * ps
        - 0.012 * (tier - 2.0)
        + 0.030 * ctr
        + 0.012 * peak
        - 0.014 * dt_z
        - distance_penalty
        - 0.010 * new
    )
    return np.clip(prob, PROB_FLOOR, PROB_CEIL)


def true_tau(features: dict[str, Any]) -> np.ndarray:
    """Individual treatment effect (pp) on order probability.

    Heterogeneous: strongly positive for high price_sensitivity + new users + slow
    delivery; near zero or slightly negative for low price_sensitivity loyal users
    on cheap (tier-1) merchants.
    """
    ps = _col(features, "price_sensitivity")
    new = _col(features, "is_new_user")
    dt = _col(features, "estimated_delivery_time_min")
    tenure = _col(features, "tenure_days")
    tier = _col(features, "price_tier")

    dt_z = (dt - 35.0) / 12.0
    loyal = tenure >= 180.0
    cheap = tier == 1.0
    low_ps = ps < 0.25
    slow_delivery = dt_z > 0.5

    tau = (
        -0.015
        + 0.08 * ps
        + 0.05 * new
        + 0.035 * np.clip(dt_z, -1.0, 2.0) / 2.0
        + 0.03 * ps * new
        + 0.02 * ps * slow_delivery
    )
    tau -= 0.035 * (loyal & low_ps & cheap)
    return tau


def treatment_logit(features: dict[str, Any], coefs: dict[str, float]) -> np.ndarray:
    return (
        coefs.get("intercept", 0.0)
        + coefs["price_sensitivity"] * _col(features, "price_sensitivity")
        + coefs["is_new_user"] * _col(features, "is_new_user")
        + coefs["historical_ctr"] * _col(features, "historical_ctr")
        + coefs["is_peak"] * _col(features, "is_peak")
    )


def treatment_propensity(features: dict[str, Any], coefs: dict[str, float]) -> np.ndarray:
    """Confounded logistic propensity of promo assignment (never a coin flip)."""
    return 1.0 / (1.0 + np.exp(-treatment_logit(features, coefs)))


def calibrate_propensity_intercept(
    features: dict[str, Any], coefs: dict[str, float], target_share: float
) -> float:
    """Bisect the intercept so mean propensity equals target_share (~35%)."""
    linear = sum(
        coefs[name] * _col(features, name)
        for name in ("price_sensitivity", "is_new_user", "historical_ctr", "is_peak")
    )
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mean_p = float((1.0 / (1.0 + np.exp(-(linear + mid)))).mean())
        if mean_p > target_share:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def click_probability(features: dict[str, Any]) -> np.ndarray:
    """Click model: pCTR drivers + position bias 1/log2(rank + 1)."""
    ctr = _col(features, "historical_ctr")
    dist = _col(features, "consumer_merchant_distance_km")
    tol = _col(features, "distance_tolerance_km")
    rank = _col(features, "slate_rank")
    peak = _col(features, "is_peak")

    relevance = (1.0 + 0.55 * np.clip((tol - dist) / 6.0, -1.0, 1.0)) * (1.0 + 0.10 * peak)
    position_bias = 1.0 / np.log2(rank + 1.0)
    return np.clip(ctr * relevance * position_bias, 0.0, 0.5)


def generate_consumers(
    rng: np.random.Generator, n: int, city_size_km: float = 10.0
) -> pd.DataFrame:
    is_new = rng.random(n) < 0.30
    tenure = np.where(is_new, rng.integers(1, 31, size=n), rng.integers(31, 1801, size=n))
    return pd.DataFrame(
        {
            "consumer_id": np.arange(n, dtype=np.int64),
            "price_sensitivity": rng.beta(2.0, 5.0, size=n).astype(np.float32),
            "order_frequency_prior": np.clip(rng.gamma(3.0, 0.5, size=n), 0.1, 8.0).astype(
                np.float32
            ),
            "distance_tolerance_km": np.clip(rng.gamma(4.0, 1.25, size=n), 0.5, 15.0).astype(
                np.float32
            ),
            "is_new_user": is_new.astype(np.int32),
            "tenure_days": tenure.astype(np.int32),
            "consumer_x_km": rng.uniform(0.0, city_size_km, size=n).astype(np.float32),
            "consumer_y_km": rng.uniform(0.0, city_size_km, size=n).astype(np.float32),
        }
    )


def generate_merchants(
    rng: np.random.Generator, n: int, city_size_km: float = 10.0
) -> pd.DataFrame:
    tier = rng.choice(np.arange(1, 5), size=n, p=[0.30, 0.30, 0.25, 0.15]).astype(np.int32)
    return pd.DataFrame(
        {
            "merchant_id": np.arange(n, dtype=np.int64),
            "cuisine_category": rng.choice(CUISINES, size=n),
            "avg_rating": (2.5 + 2.5 * rng.beta(8.0, 3.0, size=n)).astype(np.float32),
            "avg_prep_time_min": np.clip(rng.normal(10.0 + 4.0 * tier, 5.0, size=n), 5.0, 45.0)
            .round(1)
            .astype(np.float32),
            "price_tier": tier,
            "historical_ctr": np.clip(rng.lognormal(-2.8, 0.9, size=n), 0.01, 0.40).astype(
                np.float32
            ),
            "is_ads_advertiser": (rng.random(n) < 0.20).astype(np.int32),
            "merchant_x_km": rng.uniform(0.0, city_size_km, size=n).astype(np.float32),
            "merchant_y_km": rng.uniform(0.0, city_size_km, size=n).astype(np.float32),
        }
    )


def generate_dashers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dasher_id": np.arange(n, dtype=np.int64),
            "active_prob": rng.beta(2.0, 2.0, size=n).astype(np.float32),
            "avg_speed_kmph": np.clip(rng.normal(22.0, 4.0, size=n), 12.0, 35.0).astype(np.float32),
        }
    )


def build_supply_ratio(
    cfg: dict[str, Any], dashers: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Market-level dasher supply ratio per (day, hour), driven by the 800 dashers."""
    n_days = int(cfg["n_days"])
    hours = np.arange(24)
    hour_factor = np.where(is_peak_hour(hours), 0.8, 1.05)
    day_factor = rng.uniform(0.85, 1.15, size=n_days)
    active_prob = dashers["active_prob"].to_numpy(dtype=np.float64)
    supply = np.empty((n_days, 24), dtype=np.float64)
    for day in range(n_days):
        for hour in range(24):
            active = rng.random(len(dashers)) < (active_prob * hour_factor[hour] * day_factor[day])
            supply[day, hour] = np.clip(3.0 * active.mean(), 0.5, 2.0)
    return supply


def build_impression_arrays(
    cfg: dict[str, Any],
    consumers: pd.DataFrame,
    merchants: pd.DataFrame,
    dashers: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build impression-level arrays plus the separate ground-truth arrays."""
    n_days = int(cfg["n_days"])
    n_rows = int(cfg["target_rows"])

    day = np.arange(n_rows) % n_days + 1
    dow = (day - 1) % 7

    hours = np.arange(24)
    hour_weights = (
        1.0
        + 2.0 * np.exp(-0.5 * ((hours - 12.5) / 2.5) ** 2)
        + 2.6 * np.exp(-0.5 * ((hours - 18.8) / 2.5) ** 2)
    )
    hour_weights = hour_weights / hour_weights.sum()
    hour = rng.choice(hours, size=n_rows, p=hour_weights).astype(np.int32)
    peak = is_peak_hour(hour).astype(np.int32)

    supply = build_supply_ratio(cfg, dashers, rng)
    supply_ratio = supply[day - 1, hour]

    c_weights = consumers["order_frequency_prior"].to_numpy(dtype=np.float64) + 0.1
    c_idx = rng.choice(len(consumers), size=n_rows, p=c_weights / c_weights.sum())
    m_weights = merchants["historical_ctr"].to_numpy(dtype=np.float64)
    m_idx = rng.choice(len(merchants), size=n_rows, p=m_weights / m_weights.sum())

    c_x = consumers["consumer_x_km"].to_numpy(dtype=np.float64)[c_idx]
    c_y = consumers["consumer_y_km"].to_numpy(dtype=np.float64)[c_idx]
    m_x = merchants["merchant_x_km"].to_numpy(dtype=np.float64)[m_idx]
    m_y = merchants["merchant_y_km"].to_numpy(dtype=np.float64)[m_idx]
    dist = np.round(np.clip(np.hypot(c_x - m_x, c_y - m_y), 0.2, None), 2).astype(np.float32)

    prep = merchants["avg_prep_time_min"].to_numpy(dtype=np.float64)[m_idx]
    est_delivery = prep + 1.8 * dist + 9.0 / supply_ratio + rng.normal(0.0, 1.5, size=n_rows)
    est_delivery = np.clip(est_delivery, prep + 1.2 * dist, None).round(1).astype(np.float32)

    features = {
        "price_sensitivity": consumers["price_sensitivity"].to_numpy(dtype=np.float64)[c_idx],
        "order_frequency_prior": consumers["order_frequency_prior"].to_numpy(dtype=np.float64)[
            c_idx
        ],
        "distance_tolerance_km": consumers["distance_tolerance_km"].to_numpy(dtype=np.float64)[
            c_idx
        ],
        "is_new_user": consumers["is_new_user"].to_numpy(dtype=np.float64)[c_idx],
        "tenure_days": consumers["tenure_days"].to_numpy(dtype=np.float64)[c_idx],
        "avg_rating": merchants["avg_rating"].to_numpy(dtype=np.float64)[m_idx],
        "price_tier": merchants["price_tier"].to_numpy(dtype=np.float64)[m_idx],
        "historical_ctr": merchants["historical_ctr"].to_numpy(dtype=np.float64)[m_idx],
        "consumer_merchant_distance_km": dist.astype(np.float64),
        "is_peak": peak.astype(np.float64),
        "estimated_delivery_time_min": est_delivery.astype(np.float64),
    }

    base_prob = baseline_order_prob(features)
    tau = true_tau(features)

    coefs: dict[str, float] = {
        **DEFAULT_TREATMENT_COEFS,
        **cfg.get("treatment", {}).get("coefficients", {}),
    }
    target_share = float(cfg.get("treatment", {}).get("target_share", 0.35))
    coefs["intercept"] = calibrate_propensity_intercept(features, coefs, target_share)
    true_propensity = treatment_propensity(features, coefs)

    treated = rng.random(n_rows) < true_propensity
    p_order = np.clip(base_prob + treated * tau, PROB_FLOOR, PROB_CEIL)
    ordered = rng.random(n_rows) < p_order

    rank = rng.integers(1, 21, size=n_rows).astype(np.int32)
    click_p = click_probability({**features, "slate_rank": rank.astype(np.float64)})
    clicked = rng.random(n_rows) < click_p

    basket = np.asarray([14.0, 22.0, 32.0, 45.0])[
        merchants["price_tier"].to_numpy(dtype=np.int64)[m_idx] - 1
    ]
    dinner = ((hour >= 17) & (hour <= 21)).astype(float)
    order_value = np.round(
        basket * rng.lognormal(0.0, 0.35, size=n_rows) * (1.0 + 0.15 * dinner), 2
    )

    surge = np.clip(1.0 - supply_ratio, 0.0, 1.0) * 2.5
    delivery_fee = np.round(1.5 + 0.45 * dist + surge, 2)
    promo_cost = np.round(treated * np.minimum(delivery_fee, 3.0), 2)

    impressions: dict[str, np.ndarray] = {
        "impression_id": np.arange(n_rows, dtype=np.int64),
        "day": day.astype(np.int32),
        "hour_of_day": hour,
        "day_of_week": dow.astype(np.int32),
        "is_peak": peak,
        "consumer_id": consumers["consumer_id"].to_numpy()[c_idx],
        "merchant_id": merchants["merchant_id"].to_numpy()[m_idx],
        "slate_rank": rank,
        "consumer_merchant_distance_km": dist,
        "current_dasher_supply_ratio": np.round(supply_ratio, 3).astype(np.float32),
        "estimated_delivery_time_min": est_delivery,
        "promo_treated": treated.astype(np.int32),
        "clicked": clicked.astype(np.int32),
        "ordered": ordered.astype(np.int32),
        "order_value_usd": order_value.astype(np.float64),
        "delivery_fee_usd": delivery_fee.astype(np.float64),
        "promo_cost_usd": promo_cost.astype(np.float64),
    }

    truth: dict[str, np.ndarray] = {
        "impression_id": impressions["impression_id"],
        "true_propensity": true_propensity.astype(np.float64),
        "true_tau": tau.astype(np.float64),
        "baseline_order_prob": base_prob.astype(np.float64),
    }
    return impressions, truth


def compute_stats(
    impressions: dict[str, np.ndarray], truth: dict[str, np.ndarray]
) -> dict[str, float]:
    treated = impressions["promo_treated"].astype(bool)
    ordered = impressions["ordered"].astype(bool)
    return {
        "rows": float(len(impressions["impression_id"])),
        "days": float(impressions["day"].max()),
        "treated_share": float(treated.mean()),
        "order_rate": float(ordered.mean()),
        "mean_true_propensity": float(truth["true_propensity"].mean()),
        "mean_true_tau": float(truth["true_tau"].mean()),
        "mean_baseline_order_prob": float(truth["baseline_order_prob"].mean()),
        "naive_ate": float(ordered[treated].mean() - ordered[~treated].mean()),
    }


def write_outputs(
    cfg: dict[str, Any],
    impressions: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    consumers: pd.DataFrame,
    merchants: pd.DataFrame,
    dashers: pd.DataFrame,
) -> None:
    spark = get_spark("mua-sim")
    paths = cfg["paths"]

    spark.createDataFrame(pd.DataFrame(impressions)).repartition("day").write.partitionBy(
        "day"
    ).mode("overwrite").parquet(str(resolve(paths["raw_impressions"])))

    spark.createDataFrame(consumers).coalesce(1).write.mode("overwrite").parquet(
        str(resolve(paths["raw_consumers"]))
    )
    spark.createDataFrame(merchants).coalesce(1).write.mode("overwrite").parquet(
        str(resolve(paths["raw_merchants"]))
    )
    spark.createDataFrame(dashers).coalesce(1).write.mode("overwrite").parquet(
        str(resolve(paths["raw_dashers"]))
    )

    spark.createDataFrame(pd.DataFrame(truth)).repartition(8).write.mode("overwrite").parquet(
        str(resolve(paths["truth"]))
    )


def run(cfg: dict[str, Any]) -> dict[str, float]:
    rng = np.random.default_rng(int(cfg["seed"]))
    city_size_km = float(cfg.get("city_size_km", 10.0))

    logger.info("Generating consumers ...")
    consumers = generate_consumers(rng, int(cfg["n_consumers"]), city_size_km)
    logger.info("Generating merchants ...")
    merchants = generate_merchants(rng, int(cfg["n_merchants"]), city_size_km)
    logger.info("Generating dashers ...")
    dashers = generate_dashers(rng, int(cfg["n_dashers"]))

    logger.info("Building %s impressions ...", cfg["target_rows"])
    impressions, truth = build_impression_arrays(cfg, consumers, merchants, dashers, rng)

    logger.info("Writing Parquet datasets ...")
    write_outputs(cfg, impressions, truth, consumers, merchants, dashers)

    stats = compute_stats(impressions, truth)
    logger.info("Generation complete. Stats:")
    logger.info(
        "  consumers=%d merchants=%d dashers=%d", len(consumers), len(merchants), len(dashers)
    )
    logger.info("  impressions=%d days=%d", stats["rows"], stats["days"])
    logger.info("  treated share      = %.4f", stats["treated_share"])
    logger.info("  base order rate    = %.4f", stats["order_rate"])
    logger.info("  mean true_tau      = %.4f", stats["mean_true_tau"])
    logger.info("  mean true propensity = %.4f", stats["mean_true_propensity"])
    logger.info("  mean baseline prob   = %.4f", stats["mean_baseline_order_prob"])
    logger.info("  naive ATE (diff in means) = %.4f", stats["naive_ate"])
    return stats

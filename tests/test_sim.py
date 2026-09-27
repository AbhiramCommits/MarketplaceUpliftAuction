from __future__ import annotations

import numpy as np

from mua.sim.generate import (
    CUISINES,
    PROB_CEIL,
    PROB_FLOOR,
    baseline_order_prob,
    build_impression_arrays,
    click_probability,
    generate_consumers,
    generate_dashers,
    generate_merchants,
    treatment_propensity,
    true_tau,
)


def _base_features(**overrides):
    features = {
        "price_sensitivity": 0.3,
        "order_frequency_prior": 1.5,
        "distance_tolerance_km": 5.0,
        "consumer_merchant_distance_km": 4.0,
        "avg_rating": 4.0,
        "price_tier": 2.0,
        "historical_ctr": 0.07,
        "estimated_delivery_time_min": 35.0,
        "is_peak": 1.0,
        "is_new_user": 0.0,
        "tenure_days": 500.0,
        "slate_rank": 1.0,
    }
    features.update(overrides)
    return features


def _small_cfg(n_rows: int = 50_000, seed: int = 42, **overrides):
    cfg = {
        "seed": seed,
        "n_days": 30,
        "n_consumers": 2_000,
        "n_merchants": 500,
        "n_dashers": 200,
        "target_rows": n_rows,
        "city_size_km": 10.0,
        "treatment": {"target_share": 0.35},
    }
    cfg.update(overrides)
    return cfg


def _make_arrays(n_rows: int = 50_000, seed: int = 42, **cfg_overrides):
    cfg = _small_cfg(n_rows=n_rows, seed=seed, **cfg_overrides)
    rng = np.random.default_rng(cfg["seed"])
    consumers = generate_consumers(rng, cfg["n_consumers"], cfg["city_size_km"])
    merchants = generate_merchants(rng, cfg["n_merchants"], cfg["city_size_km"])
    dashers = generate_dashers(rng, cfg["n_dashers"])
    return build_impression_arrays(cfg, consumers, merchants, dashers, rng)


class TestBaselineOrderProb:
    def test_bounds(self):
        extreme = _base_features(
            price_sensitivity=1.0,
            order_frequency_prior=0.1,
            distance_tolerance_km=0.5,
            consumer_merchant_distance_km=20.0,
            avg_rating=2.5,
            price_tier=4.0,
            historical_ctr=0.01,
            estimated_delivery_time_min=300.0,
            is_peak=0.0,
            is_new_user=1.0,
        )
        assert PROB_FLOOR <= float(baseline_order_prob(extreme)) <= PROB_CEIL

    def test_higher_rating_raises_prob(self):
        base = _base_features()
        assert float(baseline_order_prob({**base, "avg_rating": 4.8})) > float(
            baseline_order_prob({**base, "avg_rating": 3.0})
        )

    def test_price_sensitivity_lowers_prob(self):
        base = _base_features()
        assert float(baseline_order_prob({**base, "price_sensitivity": 0.9})) < float(
            baseline_order_prob({**base, "price_sensitivity": 0.05})
        )

    def test_distance_beyond_tolerance_lowers_prob(self):
        base = _base_features()
        near = baseline_order_prob(
            {**base, "consumer_merchant_distance_km": 0.5, "distance_tolerance_km": 15.0}
        )
        far = baseline_order_prob(
            {**base, "consumer_merchant_distance_km": 15.0, "distance_tolerance_km": 1.0}
        )
        assert float(near) > float(far)


class TestTrueTau:
    def test_heterogeneous_segments(self):
        high = true_tau(
            _base_features(
                price_sensitivity=0.65,
                is_new_user=1.0,
                estimated_delivery_time_min=50.0,
                tenure_days=10.0,
            )
        )
        low = true_tau(
            _base_features(
                price_sensitivity=0.10,
                is_new_user=0.0,
                estimated_delivery_time_min=25.0,
                tenure_days=400.0,
                price_tier=1.0,
            )
        )
        assert float(high) > 0.10
        assert float(low) < -0.02
        assert float(high) > float(low)

    def test_monotonic_in_price_sensitivity(self):
        base = _base_features(is_new_user=1.0, estimated_delivery_time_min=45.0)
        taus = [float(true_tau({**base, "price_sensitivity": p})) for p in (0.1, 0.4, 0.7)]
        assert taus[0] < taus[1] < taus[2]


class TestTreatmentPropensity:
    def test_confounded_not_coin_flip(self):
        coefs = {
            "intercept": -1.5,
            "price_sensitivity": 1.2,
            "is_new_user": 0.9,
            "historical_ctr": 1.0,
            "is_peak": 0.35,
        }
        hot = treatment_propensity(
            _base_features(price_sensitivity=0.9, is_new_user=1.0, historical_ctr=0.2, is_peak=1.0),
            coefs,
        )
        cold = treatment_propensity(
            _base_features(
                price_sensitivity=0.05, is_new_user=0.0, historical_ctr=0.01, is_peak=0.0
            ),
            coefs,
        )
        assert float(hot) > float(cold)
        assert 0.0 < float(cold) < float(hot) < 1.0

    def test_calibrated_to_target_share(self):
        impressions, truth = _make_arrays(n_rows=20_000)
        treated = impressions["promo_treated"].astype(bool)
        assert abs(treated.mean() - 0.35) < 0.03
        assert abs(truth["true_propensity"].mean() - 0.35) < 0.005


class TestClickModel:
    def test_position_bias_curve(self):
        base = _base_features()
        top = click_probability({**base, "slate_rank": 1.0})
        bottom = click_probability({**base, "slate_rank": 20.0})
        assert float(top) > float(bottom)
        assert abs(float(top) / float(bottom) - np.log2(21.0)) < 1e-6


class TestGeneration:
    def test_reproducible_with_seed(self):
        imp1, truth1 = _make_arrays(n_rows=5_000, seed=42)
        imp2, truth2 = _make_arrays(n_rows=5_000, seed=42)
        for key in imp1:
            assert np.array_equal(imp1[key], imp2[key]), key
        for key in truth1:
            assert np.array_equal(truth1[key], truth2[key]), key

    def test_different_seed_differs(self):
        imp1, _ = _make_arrays(n_rows=5_000, seed=42)
        imp2, _ = _make_arrays(n_rows=5_000, seed=7)
        assert not np.array_equal(imp1["ordered"], imp2["ordered"])

    def test_realized_probabilities_in_unit_interval(self):
        impressions, truth = _make_arrays(n_rows=12_000)
        treated = impressions["promo_treated"].astype(bool)
        realized = np.clip(
            truth["baseline_order_prob"] + treated * truth["true_tau"], PROB_FLOOR, PROB_CEIL
        )
        assert (realized >= 0.0).all() and (realized <= 1.0).all()

    def test_naive_ate_is_measurably_biased_vs_known_truth(self):
        """The naive difference in means must overstate the known ATE: the
        confounded propensity up-weights high-tau units into treatment."""
        impressions, truth = _make_arrays(n_rows=50_000)
        treated = impressions["promo_treated"].astype(bool)
        ordered = impressions["ordered"].astype(bool)
        naive_ate = ordered[treated].mean() - ordered[~treated].mean()
        true_ate = truth["true_tau"].mean()
        assert true_ate > 0.01
        assert naive_ate - true_ate > 0.002

    def test_small_scale_shape_and_days(self):
        impressions, truth = _make_arrays(n_rows=50_000)
        assert len(impressions["impression_id"]) == 50_000
        assert set(np.unique(impressions["day"])) == set(range(1, 31))
        assert len(np.unique(impressions["impression_id"])) == 50_000

    def test_outcome_consistency(self):
        impressions, truth = _make_arrays(n_rows=12_000)
        treated = impressions["promo_treated"].astype(bool)
        realized = np.clip(
            truth["baseline_order_prob"] + treated * truth["true_tau"], PROB_FLOOR, PROB_CEIL
        )
        assert realized.min() >= PROB_FLOOR - 1e-9
        assert realized.max() <= PROB_CEIL + 1e-9
        assert abs(impressions["ordered"].mean() - realized.mean()) < 0.01

    def test_promo_cost_zero_for_untreated(self):
        impressions, _ = _make_arrays(n_rows=20_000)
        untreated = ~impressions["promo_treated"].astype(bool)
        assert (impressions["promo_cost_usd"][untreated] == 0.0).all()

    def test_supply_ratio_range(self):
        impressions, _ = _make_arrays(n_rows=10_000)
        supply = impressions["current_dasher_supply_ratio"]
        assert 0.5 - 1e-6 <= supply.min() and supply.max() <= 2.0 + 1e-6


class TestDimensions:
    def test_consumers(self):
        rng = np.random.default_rng(0)
        consumers = generate_consumers(rng, 10_000, 10.0)
        assert len(consumers) == 10_000
        assert ((consumers["price_sensitivity"] >= 0) & (consumers["price_sensitivity"] <= 1)).all()
        assert (consumers["tenure_days"] > 0).all()
        assert np.all(consumers["is_new_user"] == (consumers["tenure_days"] <= 30))
        assert abs(consumers["is_new_user"].mean() - 0.30) < 0.02

    def test_merchants(self):
        rng = np.random.default_rng(1)
        merchants = generate_merchants(rng, 5_000, 10.0)
        assert set(merchants["cuisine_category"].unique()) == set(CUISINES)
        assert ((merchants["avg_rating"] >= 2.5) & (merchants["avg_rating"] <= 5.0)).all()
        assert set(merchants["price_tier"].unique()) == {1, 2, 3, 4}
        assert ((merchants["historical_ctr"] >= 0.01) & (merchants["historical_ctr"] <= 0.40)).all()
        assert abs(merchants["is_ads_advertiser"].mean() - 0.20) < 0.02

    def test_dashers(self):
        rng = np.random.default_rng(2)
        dashers = generate_dashers(rng, 800)
        assert len(dashers) == 800
        assert ((dashers["avg_speed_kmph"] >= 12.0) & (dashers["avg_speed_kmph"] <= 35.0)).all()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mua.sim.generate import (
    build_impression_arrays,
    generate_consumers,
    generate_dashers,
    generate_merchants,
)

ROWS = 12_000


def _constant_tau_data(rows: int = 8000, seed: int = 11):
    """Synthetic data with a CONSTANT, known treatment effect tau = 0.03."""
    rng = np.random.default_rng(seed)
    d = 8
    X = rng.normal(0, 1, (rows, d))
    logit = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
    w = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    tau = 0.03
    prob = np.clip(
        0.10
        + 0.04 * X[:, 0]
        + 0.02 * X[:, 1]
        - 0.01 * X[:, 2]
        + tau * w
        + rng.normal(0, 0.01, rows),
        0.001,
        0.95,
    )
    y = (rng.random(rows) < prob).astype(float)
    return X, w, y


def _write_toy_causal_data(root: Path, rows: int = ROWS, seed: int = 7) -> None:
    """Build processed-like parquet + truth directly from the sim generator."""
    sim_cfg = {
        "seed": seed,
        "n_days": 30,
        "n_consumers": 1_500,
        "n_merchants": 300,
        "n_dashers": 100,
        "target_rows": rows,
        "city_size_km": 10.0,
        "treatment": {"target_share": 0.35},
    }
    rng = np.random.default_rng(seed)
    consumers = generate_consumers(rng, sim_cfg["n_consumers"], sim_cfg["city_size_km"])
    merchants = generate_merchants(rng, sim_cfg["n_merchants"], sim_cfg["city_size_km"])
    dashers = generate_dashers(rng, sim_cfg["n_dashers"])
    impressions, truth = build_impression_arrays(sim_cfg, consumers, merchants, dashers, rng)

    c_idx = impressions["consumer_id"]
    m_idx = impressions["merchant_id"]
    cuisine_cats = sorted(merchants["cuisine_category"].unique())
    cuisine_map = {cat: i for i, cat in enumerate(cuisine_cats)}
    cuisine_col = np.array(
        [cuisine_map[c] for c in merchants["cuisine_category"].to_numpy()[m_idx]]
    )
    columns = {
        **{k: impressions[k] for k in impressions},
        "price_sensitivity": consumers["price_sensitivity"].to_numpy()[c_idx],
        "order_frequency_prior": consumers["order_frequency_prior"].to_numpy()[c_idx],
        "distance_tolerance_km": consumers["distance_tolerance_km"].to_numpy()[c_idx],
        "is_new_user": consumers["is_new_user"].to_numpy()[c_idx],
        "tenure_days": consumers["tenure_days"].to_numpy()[c_idx],
        "consumer_x_km": consumers["consumer_x_km"].to_numpy()[c_idx],
        "consumer_y_km": consumers["consumer_y_km"].to_numpy()[c_idx],
        "avg_rating": merchants["avg_rating"].to_numpy()[m_idx],
        "avg_prep_time_min": merchants["avg_prep_time_min"].to_numpy()[m_idx],
        "price_tier": merchants["price_tier"].to_numpy()[m_idx],
        "historical_ctr": merchants["historical_ctr"].to_numpy()[m_idx],
        "is_ads_advertiser": merchants["is_ads_advertiser"].to_numpy()[m_idx],
        "merchant_x_km": merchants["merchant_x_km"].to_numpy()[m_idx],
        "merchant_y_km": merchants["merchant_y_km"].to_numpy()[m_idx],
    }
    for i, cat in enumerate(cuisine_cats):
        columns[f"cuisine_{cat}"] = (cuisine_col == i).astype(np.int32)

    table = pa.table(columns)
    day = impressions["day"]
    splits = {"train": (1, 21), "valid": (22, 25), "test": (26, 30)}
    for split, (lo, hi) in splits.items():
        mask = (day >= lo) & (day <= hi)
        out = root / "processed" / split
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.filter(pa.array(mask)), out / "part-0.parquet")

    truth_out = root / "truth"
    truth_out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(truth), truth_out / "part-0.parquet")


def _toy_cfg(root: Path) -> dict:
    return {
        "paths": {
            "processed": str(root / "processed"),
            "truth": str(root / "truth"),
            "artifacts": str(root / "artifacts" / "causal"),
            "figures": str(root / "reports" / "figures"),
        },
        "data": {
            "sample_train": None,
            "sample_test": None,
            "sample_refute": 2_500,
            "seed": 7,
        },
        "propensity": {
            "n_estimators": 80,
            "trim_low": 0.02,
            "trim_high": 0.98,
            "smd_threshold": 0.12,
        },
        "estimators": {
            "base": "histgb",
            "bootstrap_samples": 20,
            "dr_cv": 2,
            "forest_estimators": 8,
            "forest_max_depth": 2,
            "forest_min_samples_leaf": 100,
            "forest_base_estimators": 30,
            "hgb_max_iter": 100,
            "hgb_early_stopping": True,
            "hgb_validation_fraction": 0.2,
            "hgb_n_iter_no_change": 10,
        },
        "evaluate": {"n_deciles": 10},
        "refute": {
            "placebo_iterations": 5,
            "rcc_simulations": 5,
            "subset_fractions": [0.8],
            "effect_shift_tolerance": 0.20,
            "placebo_pvalue_threshold": 0.05,
        },
        "policy": {"ks": [10, 50, 100]},
    }


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("causal_toy")
    _write_toy_causal_data(root)
    return root, _toy_cfg(root)


class TestData:
    def test_load_and_truth_join(self, toy):
        root, cfg = toy
        from mua.causal.data import EXCLUDED, load_frame

        frame = load_frame(cfg, "train")
        assert frame["X"].shape[1] == len(frame["feature_names"])
        assert frame["X"].shape[0] == frame["w"].shape[0] == frame["y"].shape[0]
        for name in EXCLUDED:
            assert name not in frame["feature_names"]

        test = load_frame(cfg, "test", with_truth=True)
        assert "true_tau" in test
        truth = pq.read_table(root / "truth")
        tau_lookup = dict(
            zip(
                truth["impression_id"].to_numpy(),
                truth["true_tau"].to_numpy(),
                strict=True,
            )
        )
        for row in range(0, len(test["ids"]), 500):
            assert test["true_tau"][row] == pytest.approx(tau_lookup[test["ids"][row]])


class TestPropensity:
    def test_balance_and_artifacts(self, toy):
        root, cfg = toy
        from mua.causal.data import load_frame
        from mua.causal.propensity import fit_propensity

        frame = load_frame(cfg, "train")
        result = fit_propensity(cfg, frame)
        assert result["auc"] > 0.6
        assert result["n_after"] >= 0.9 * result["n_before"]
        assert all(
            smd <= cfg["propensity"]["smd_threshold"] for smd in result["smd_after"].values()
        )
        assert (root / "reports" / "figures" / "propensity_overlap.png").exists()
        assert (root / "reports" / "figures" / "love_plot.png").exists()
        assert (root / "artifacts" / "causal" / "propensity.joblib").exists()

    def test_fails_loudly_when_unbalanced(self, toy):
        _, cfg = toy
        from mua.causal.data import load_frame
        from mua.causal.propensity import fit_propensity

        bad_cfg = {**cfg, "propensity": {**cfg["propensity"], "smd_threshold": 0.0}}
        frame = load_frame(cfg, "train")
        with pytest.raises(RuntimeError, match="IPTW balance check FAILED"):
            fit_propensity(bad_cfg, frame)


class TestQini:
    def test_metrics_ordering(self):
        from mua.causal.evaluate import auuc, normalized_qini

        rng = np.random.default_rng(0)
        n = 8_000
        w = (rng.random(n) < 0.4).astype(np.int64)
        cate_true = rng.uniform(-0.05, 0.15, n)
        p0 = 0.05 + rng.uniform(-0.01, 0.02, n)
        p = np.clip(p0 + w * cate_true, 0, 1)
        y = (rng.random(n) < p).astype(float)
        qini_perfect = normalized_qini(y, w, cate_true)
        qini_random = normalized_qini(y, w, rng.permutation(cate_true))
        assert qini_perfect > qini_random
        assert auuc(y, w, cate_true) > auuc(y, w, rng.permutation(cate_true))

    def test_matches_causalml(self):
        from causalml.metrics import get_qini

        from mua.causal.evaluate import normalized_qini, qini_curve_values

        rng = np.random.default_rng(1)
        n = 5_000
        w = (rng.random(n) < 0.4).astype(np.int64)
        cate = rng.uniform(-0.05, 0.15, n)
        p = np.clip(0.05 + w * cate, 0, 1)
        y = (rng.random(n) < p).astype(float)
        import pandas as pd

        df = pd.DataFrame({"y": y, "w": w, "model1": cate})
        causalml_curve = get_qini(df, outcome_col="y", treatment_col="w")["model1"].to_numpy()
        _, my_curve = qini_curve_values(y, w, cate)
        # causalml prepends a 0 point and linearly fills the leading all-treated
        # segment (0/0); beyond the first row with a control unit the formulas agree.
        aligned = causalml_curve[1:]
        order = np.argsort(-cate)
        cum_c = np.cumsum(1 - w[order])
        first_valid = int(np.argmax(cum_c > 0))
        np.testing.assert_allclose(my_curve[first_valid:], aligned[first_valid:], rtol=1e-6)
        assert abs(normalized_qini(y, w, cate)) <= 1.0 + 1e-9


class TestEstimatorsSmoke:
    def test_fit_estimators(self, toy):
        root, cfg = toy
        from mua.causal.data import load_frame
        from mua.causal.estimators import fit_estimators
        from mua.causal.propensity import fit_propensity

        train = load_frame(cfg, "train")
        test = load_frame(cfg, "test")
        prop = fit_propensity(cfg, train)
        keep = prop["trim_mask"]
        trimmed = {
            "X": train["X"][keep],
            "w": train["w"][keep],
            "y": train["y"][keep],
            "feature_names": train["feature_names"],
        }
        results = fit_estimators(
            cfg,
            trimmed,
            test,
            prop["propensity"][keep],
            selected=("naive", "iptw", "s", "t", "x", "dr"),
        )
        assert set(results) == {"naive", "iptw", "s", "t", "x", "dr"}
        for name, res in results.items():
            assert -0.05 < res["ate"] < 0.15, name
            assert res["ci_low"] <= res["ate"] <= res["ci_high"]
            assert len(res["cate_test"]) == len(test["y"])
        assert results["naive"]["ate"] > 0
        assert (root / "artifacts" / "causal" / "estimator_summary.joblib").exists()
        assert (root / "artifacts" / "causal" / "cate_test_dr.npz").exists()


class TestEvaluateAndPolicyEndToEnd:
    def test_full_pipeline(self, toy):
        root, cfg = toy
        from mua.causal.run import run

        summary, ok = run(cfg, selected="all")
        assert ok
        leaderboard = summary["leaderboard"]
        assert {r["estimator"] for r in leaderboard} == {
            "naive",
            "iptw",
            "s",
            "t",
            "x",
            "dr",
            "forest",
        }
        for row in leaderboard:
            assert np.isfinite(row["pehe"]) and np.isfinite(row["qini"])
            assert row["pehe"] >= 0
        truth_ate = pq.read_table(root / "truth")["true_tau"].to_numpy().mean()
        assert any(abs(r["ate"] - truth_ate) < 0.05 for r in leaderboard)
        assert (root / "reports" / "leaderboard.md").exists()
        assert (root / "artifacts" / "causal" / "leaderboard.csv").exists()
        assert (root / "reports" / "figures" / "qini.png").exists()
        assert (root / "reports" / "figures" / "uplift_deciles.png").exists()
        assert (root / "reports" / "figures" / "policy_curve.png").exists()
        for row in summary["policy"]["rows"]:
            assert row["per_dollar_topk"] >= 0
        assert summary["policy"]["treat_all_per_dollar"] > 0


class TestConstantTauRecovery:
    def test_learners_recover_constant_tau(self):
        """On data with a CONSTANT known tau, T/X/DR-learners and the causal forest
        must all recover the ATE within tolerance."""
        from mua.causal.estimators import fit_dr, fit_forest, fit_metalearner

        X, w, y = _constant_tau_data(rows=20_000)
        split = 15_000
        X_train, w_train, y_train = X[:split], w[:split], y[:split]
        X_test = X[split:]
        cfg = {
            "data": {"seed": 11},
            "estimators": {
                "base": "histgb",
                "bootstrap_samples": 20,
                "dr_cv": 2,
                "forest_estimators": 20,
                "forest_max_depth": 2,
                "forest_min_samples_leaf": 50,
                "hgb_max_iter": 200,
                "hgb_learning_rate": 0.05,
                "hgb_min_samples_leaf": 100,
                "hgb_early_stopping": True,
                "hgb_validation_fraction": 0.2,
                "hgb_n_iter_no_change": 30,
            },
        }
        results = {}
        for kind in ("t", "x"):
            results[kind] = fit_metalearner(kind, X_train, w_train, y_train, X_test, cfg, 11)
        results["dr"] = fit_dr(X_train, w_train, y_train, X_test, cfg, 11)
        results["forest"] = fit_forest(X_train, w_train, y_train, X_test, cfg, 11)
        for name, res in results.items():
            assert abs(res["ate"] - 0.03) < 0.010, (name, res["ate"])


class TestRefute:
    def test_refutation_battery(self, toy):
        root, cfg = toy
        from mua.causal.refute import run

        result = run(cfg)
        assert result["ok"]
        assert result["ate"] > 0
        verdicts = {v["refuter"]: v["verdict"] for v in result["verdicts"]}
        assert verdicts["placebo_treatment (permute)"] == "PASS"
        assert verdicts["random_common_cause"] == "PASS"
        assert verdicts["data_subset"] == "PASS"
        placebo = next(v for v in result["verdicts"] if "placebo" in v["refuter"])
        assert abs(placebo["new_effect"]) < 0.01
        report = (root / "reports" / "refutation_report.md").read_text()
        assert "Refutation report" in report
        assert "E-value" in report
        assert (root / "reports" / "causal_dag.dot").exists()

    def test_evalue_formula(self):
        from mua.causal.refute import e_value

        assert e_value(0.10, 0.05) == pytest.approx(2 + np.sqrt(2))
        assert e_value(0.05, 0.10) is None

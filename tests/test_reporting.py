from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from tests.test_causal import _write_toy_causal_data

AUCTION_BLOCK = {
    "mechanism": "gsp",
    "slate_size": 20,
    "slot_positions": [1, 2, 3],
    "reserve": 0.5,
    "agent_mix": {"truthful": 0.5, "shading": 0.2, "paced": 0.2, "random": 0.1},
    "shading": {"initial": 0.8, "step": 0.02, "min": 0.3, "max": 1.0},
    "pacing": {"learning_rate": 0.3, "lam_max": 10.0, "budget_ratio": 0.5},
}


def _write_scored_and_artifacts(root: Path, rows: int = 6000) -> None:
    processed = pq.read_table(root / "processed" / "test")
    ctr = processed["historical_ctr"].to_numpy().astype(np.float64)
    pctr = np.clip(ctr * 0.8, 0.001, 0.3).astype(np.float64)
    scored = processed.append_column("pctr", pa.array(pctr))
    scored = scored.append_column("pctr_raw", pa.array(pctr))
    out = root / "scored" / "test"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(scored, out / "part-0.parquet")

    causal_cfg = {
        "paths": {"processed": str(root / "processed"), "truth": str(root / "truth")},
    }
    from mua.causal.data import load_frame

    train = load_frame(causal_cfg, "train")
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0)
    model.fit(train["X"], train["w"])
    artifacts = root / "artifacts" / "causal"
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "propensity.joblib")

    test = load_frame(causal_cfg, "test", with_truth=True)
    cate = 0.8 * test["true_tau"] + np.random.default_rng(0).normal(0, 0.003, len(test["ids"]))
    np.savez_compressed(
        artifacts / "cate_test_x.npz",
        impression_id=test["ids"],
        cate=cate.astype(np.float32),
    )
    with (artifacts / "leaderboard.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "estimator",
                "ate",
                "ci_low",
                "ci_high",
                "ate_bias",
                "ci_covers_truth",
                "pehe",
                "qini",
                "auuc",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "estimator": "x",
                "ate": 0.024,
                "ci_low": 0.022,
                "ci_high": 0.026,
                "ate_bias": 0.0002,
                "ci_covers_truth": "yes",
                "pehe": 0.015,
                "qini": 0.5,
                "auuc": 0.02,
            }
        )


def _toy_experiment_cfg(root: Path) -> dict:
    return {
        "paths": {
            "scored": str(root / "scored"),
            "figures": str(root / "reports" / "figures"),
            "results": str(root / "reports" / "results.csv"),
            "report": str(root / "reports" / "REPORT.md"),
        },
        "causal": {
            "paths": {"processed": str(root / "processed"), "truth": str(root / "truth")},
            "artifacts": str(root / "artifacts" / "causal"),
            "sample": None,
            "seed": 7,
            "best_cate": "auto",
        },
        "auction": AUCTION_BLOCK,
        "experiment": {
            "mechanisms": ["gsp", "first_price"],
            "policies": ["treat_none", "treat_all", "random_k", "uplift_top_k", "propensity_top_k"],
            "seeds": [1, 2],
            "rounds_per_cell": 250,
            "top_k_pct": 20,
            "trace_rounds": 600,
        },
    }


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("reporting_toy")
    _write_toy_causal_data(root, rows=6000, seed=7)
    _write_scored_and_artifacts(root)
    return root, _toy_experiment_cfg(root)


class TestExperimentGrid:
    def test_grid_and_metrics(self, toy):
        root, cfg = toy
        from mua.reporting.experiment import run

        rows = run(cfg)
        assert len(rows) == 2 * 5 * 2
        results = list(csv.DictReader((root / "reports" / "results.csv").open()))
        assert len(results) == len(rows)

        treat_none = [r for r in results if r["policy"] == "treat_none"]
        treat_all = [r for r in results if r["policy"] == "treat_all"]
        uplift = [r for r in results if r["policy"] == "uplift_top_k"]
        random_k = [r for r in results if r["policy"] == "random_k"]
        assert all(float(r["promo_spend"]) == 0.0 for r in treat_none)
        assert all(float(r["promo_spend"]) > 0 for r in treat_all)
        assert all(float(r["revenue"]) > 0 for r in results)
        assert all(np.isfinite(float(r["advertiser_surplus"])) for r in results)
        gsp_surplus = [float(r["advertiser_surplus"]) for r in results if r["mechanism"] == "gsp"]
        assert np.mean(gsp_surplus) > 0
        assert all(0 < float(r["consumer_welfare"]) < 1 for r in results)
        assert all(np.isfinite(float(r["per_dollar_true"])) for r in uplift)
        assert np.mean([float(r["inc_conversions_true"]) for r in uplift]) > np.mean(
            [float(r["inc_conversions_true"]) for r in random_k]
        )
        assert "orders_realized" in results[0]

    def test_deterministic(self, toy):
        root, cfg = toy
        from mua.reporting.experiment import run

        run(cfg)
        first = (root / "reports" / "results.csv").read_text()
        run(cfg)
        second = (root / "reports" / "results.csv").read_text()
        assert first == second

    def test_small_scale(self, toy):
        root, cfg = toy
        cfg["experiment"]["small"] = {"rounds_per_cell": 100, "seeds": [1]}
        from mua.reporting.experiment import run

        rows = run(cfg, scale="small")
        assert len(rows) == 2 * 5 * 1
        assert (root / "reports" / "results.csv").exists()


class TestFigures:
    def test_all_figures_generated(self, toy):
        root, cfg = toy
        from mua.reporting.experiment import run as run_experiment
        from mua.reporting.figures import run as run_figures

        run_experiment(cfg)
        outputs = run_figures(cfg)
        assert len(outputs) == 5
        names = {p.name for p in outputs}
        assert names == {
            "revenue_surplus.png",
            "frontier.png",
            "pacing_traces.png",
            "shading_convergence.png",
            "qini_overlay.png",
        }
        for path in outputs:
            assert path.exists() and path.stat().st_size > 1000


class TestReport:
    def test_report_generated(self, toy):
        root, cfg = toy
        from mua.reporting.report import run

        path = run(cfg)
        text = path.read_text()
        assert path == root / "reports" / "REPORT.md"
        assert "Methodology" in text
        assert "Findings" in text
        assert "Limitations" in text
        assert "first_price" in text
        assert "uplift_top_k" in text
        assert "![Revenue vs advertiser surplus by mechanism](figures/revenue_surplus.png)" in text
        assert "no interference" in text.lower() or "interference" in text.lower()
        assert "results.csv" in text

    def test_report_reuses_existing_results(self, toy):
        root, cfg = toy
        from mua.reporting.report import run

        first = run(cfg)
        mtime = first.stat().st_mtime
        assert (root / "reports" / "results.csv").exists()
        # idempotent: does not require recomputation when artifacts exist
        assert run(cfg).read_text() == first.read_text() or first.stat().st_mtime >= mtime

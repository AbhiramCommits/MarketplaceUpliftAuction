"""Train the CTR ranker.

BCE loss, AdamW, cosine LR schedule (with linear warmup), gradient clipping, and early
stopping on validation log-loss. Per-epoch metrics go to ``artifacts/ranker/metrics.csv``
and to MLflow (local file backend at ``artifacts/mlruns``). The best checkpoint and a
feature-spec JSON are saved to ``artifacts/ranker/``. Calibration follows training; the
run summary is checked against the configured quality targets.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn

from mua.config import resolve
from mua.ranker.calibrate import calibrate
from mua.ranker.dataset import RankerDataset, score_dataset
from mua.ranker.model import CTRRanker

logger = logging.getLogger(__name__)

EPS = 1e-7


def _bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(pred.clamp(EPS, 1 - EPS), target)


def build_feature_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Scan the training split to build the feature spec (schema + numeric stats)."""
    train_path = resolve(cfg["paths"]["processed"]) / "train"
    files = sorted(str(p) for p in train_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No processed train data under {train_path}")
    schema = pq.read_schema(files[0])
    cuisine_categories = sorted(
        name[len("cuisine_") :] for name in schema.names if name.startswith("cuisine_")
    )

    numeric = list(cfg["data"]["numeric"])
    sums = {name: 0.0 for name in numeric}
    sumsq = {name: 0.0 for name in numeric}
    count = 0
    for file in files:
        pf = pq.ParquetFile(file)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=numeric)
            count += table.num_rows
            for name in numeric:
                values = table[name].to_numpy().astype(np.float64)
                sums[name] += float(values.sum())
                sumsq[name] += float((values * values).sum())

    numeric_stats: dict[str, dict[str, float]] = {}
    for name in numeric:
        mean = sums[name] / count
        variance = max(sumsq[name] / count - mean * mean, 0.0)
        numeric_stats[name] = {"mean": mean, "std": max(math.sqrt(variance), 1e-6)}

    spec: dict[str, Any] = {
        "label": cfg["data"]["label"],
        "categorical": {k: dict(v) for k, v in cfg["data"]["categorical"].items()},
        "numeric": numeric,
        "numeric_stats": numeric_stats,
        "cuisine_categories": cuisine_categories,
        "bias": dict(cfg["data"]["bias"]),
    }
    out = resolve(cfg["paths"]["ranker_artifacts"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "feature_spec.json").open("w") as fh:
        json.dump(spec, fh, indent=2)
    logger.info("Feature spec saved to %s", out / "feature_spec.json")
    return spec


def _mlflow_log(metrics: dict[str, float], step: int | None = None) -> None:
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
    except ImportError:  # pragma: no cover
        pass


def _mlflow_log_params(cfg: dict[str, Any]) -> None:
    training, model, data = cfg["training"], cfg["model"], cfg["data"]
    params = {
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "batch_size": data["batch_size"],
        "patience": training["patience"],
        "grad_clip_norm": training["grad_clip_norm"],
        "seed": training["seed"],
        "mlp_hidden": str(model["mlp_hidden"]),
        "dropout": model["dropout"],
        "in_memory": bool(data.get("in_memory", False)),
        "bias_vocab_size": data["bias"]["vocab_size"],
        "embedding_dims": str({k: v["embedding_dim"] for k, v in data["categorical"].items()}),
    }
    try:
        import mlflow

        mlflow.log_params(params)
    except ImportError:  # pragma: no cover
        pass


def train(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    training = cfg["training"]
    data = cfg["data"]
    torch.manual_seed(int(training["seed"]))
    np.random.seed(int(training["seed"]))

    processed = resolve(cfg["paths"]["processed"])
    checkpoint_dir = resolve(cfg["paths"]["ranker_artifacts"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mode = "in_memory" if data.get("in_memory") else "chunked"
    train_ds = RankerDataset(processed / "train", spec, mode=mode)
    valid_ds = RankerDataset(processed / "valid", spec, mode=mode)
    batch_size = int(data["batch_size"])

    model = CTRRanker(spec, cfg["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    steps_per_epoch = math.ceil(train_ds.n / batch_size)
    total_steps = int(training["epochs"]) * steps_per_epoch
    warmup_steps = int(training.get("warmup_epochs", 1)) * steps_per_epoch
    min_ratio = float(training.get("lr_min_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return min_ratio + (1.0 - min_ratio) * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    csv_path = checkpoint_dir / "metrics.csv"
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(
            ["epoch", "train_logloss", "val_logloss", "val_roc_auc", "val_pr_auc", "lr"]
        )

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    patience_left = int(training["patience"])
    start = time.time()

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        loader = train_ds.make_loader(
            batch_size, shuffle=True, epoch=epoch - 1, seed=int(training["seed"]), drop_last=True
        )
        loss_sum = 0.0
        seen = 0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch, include_bias=True)
            loss = _bce(pred, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip_norm"]))
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(batch["label"])
            seen += len(batch["label"])
        train_logloss = loss_sum / max(seen, 1)

        valid_preds, valid_labels = score_dataset(model, valid_ds, batch_size)
        val_logloss = float(log_loss(valid_labels, valid_preds))
        val_auc = float(roc_auc_score(valid_labels, valid_preds))
        val_prauc = float(average_precision_score(valid_labels, valid_preds))
        lr = float(scheduler.get_last_lr()[0])

        with csv_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(
                [
                    epoch,
                    f"{train_logloss:.6f}",
                    f"{val_logloss:.6f}",
                    f"{val_auc:.6f}",
                    f"{val_prauc:.6f}",
                    f"{lr:.8f}",
                ]
            )
        _mlflow_log(
            {
                "train_logloss": train_logloss,
                "val_logloss": val_logloss,
                "val_roc_auc": val_auc,
                "val_pr_auc": val_prauc,
            },
            step=epoch,
        )
        logger.info(
            "epoch %2d/%d: train_loss=%.5f val_loss=%.5f val_auc=%.4f val_prauc=%.4f lr=%.6f",
            epoch,
            training["epochs"],
            train_logloss,
            val_logloss,
            val_auc,
            val_prauc,
            lr,
        )

        if val_logloss < best_val_loss - 1e-5:
            best_val_loss = val_logloss
            best_epoch = epoch
            patience_left = int(training["patience"])
            best_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_roc_auc": val_auc,
                "val_pr_auc": val_prauc,
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch,
                    training["patience"],
                )
                break

    if best_state is None:  # pragma: no cover - patience always > 0 in practice
        best_state = {
            "model": {k: v.clone() for k, v in model.state_dict().items()},
            "epoch": int(training["epochs"]),
            "val_roc_auc": float("nan"),
            "val_pr_auc": float("nan"),
        }
        best_epoch = int(training["epochs"])

    checkpoint = {
        "model_state_dict": best_state["model"],
        "epoch": best_state["epoch"],
        "val_logloss": best_val_loss,
        "feature_spec": spec,
    }
    torch.save(checkpoint, checkpoint_dir / "best_model.pt")
    model.load_state_dict(best_state["model"])
    logger.info(
        "Best checkpoint saved (epoch %d) -> %s", best_epoch, checkpoint_dir / "best_model.pt"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_loss,
        "val_roc_auc": best_state["val_roc_auc"],
        "val_pr_auc": best_state["val_pr_auc"],
        "train_logloss": train_logloss,
        "elapsed_sec": round(time.time() - start, 1),
    }


def check_targets(summary: dict[str, Any], targets: dict[str, Any]) -> bool:
    auc_ok = summary["test_roc_auc"] >= targets["test_roc_auc"]
    ece_ok = summary["test_ece_after"] <= targets["test_ece"]
    logger.info(
        "Targets: test ROC-AUC >= %.3f -> %s (%.4f)",
        targets["test_roc_auc"],
        "PASS" if auc_ok else "FAIL",
        summary["test_roc_auc"],
    )
    logger.info(
        "Targets: post-calibration test ECE <= %.3f -> %s (%.4f)",
        targets["test_ece"],
        "PASS" if ece_ok else "FAIL",
        summary["test_ece_after"],
    )
    if not (auc_ok and ece_ok):
        logger.warning("Diagnostics:")
        if not auc_ok:
            logger.warning(
                "  - test ROC-AUC below target: the model is not separating clickers from "
                "non-clickers. Check that strong signals (merchant historical_ctr, distance vs "
                "tolerance, is_peak) are in the numeric features, try more epochs or larger "
                "embedding dims, or inspect the time split for leakage."
            )
        if not ece_ok:
            logger.warning(
                "  - post-calibration test ECE above target: isotonic was fit on the validation "
                "split; if the test prediction distribution drifts outside the valid range, "
                "out-of-bounds clipping degrades calibration. Consider more validation data or "
                "fewer bins."
            )
    return auc_ok and ece_ok


def _print_summary(summary: dict[str, Any]) -> None:
    logger.info("Run summary:")
    logger.info(
        "  training: best_epoch=%s best_val_logloss=%.5f val_roc_auc=%.4f val_pr_auc=%.4f "
        "elapsed=%ss",
        summary["best_epoch"],
        summary["best_val_logloss"],
        summary["val_roc_auc"],
        summary["val_pr_auc"],
        summary["elapsed_sec"],
    )
    logger.info(
        "  test: roc_auc=%.4f pr_auc=%.4f | ECE before=%.4f after=%.4f",
        summary["test_roc_auc"],
        summary["test_pr_auc"],
        summary["test_ece_before"],
        summary["test_ece_after"],
    )
    logger.info(
        "  valid: ECE before=%.4f after=%.4f",
        summary["valid_ece_before"],
        summary["valid_ece_after"],
    )


def _setup_mlflow(cfg: dict[str, Any]) -> None:
    """Point MLflow at the local store under artifacts/mlruns (sqlite file backend).

    The classic filesystem backend (mlruns/ directories) is deprecated and buggy in
    MLflow 3.x (stale directory scans cause "Run not found" on creation), so we use
    the officially recommended sqlite file backend plus a local artifact root,
    keeping everything under artifacts/mlruns.
    """
    import mlflow

    tracking_dir = resolve(cfg["training"]["mlflow_tracking_uri"])
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir}/mlflow.db")
    experiment_name = cfg["training"]["mlflow_experiment"]
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=str(tracking_dir))
    mlflow.set_experiment(experiment_name)


def run(cfg: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    training = cfg["training"]
    mlflow_enabled = bool(training.get("mlflow_enabled", True))
    if mlflow_enabled:
        _setup_mlflow(cfg)

    spec = build_feature_spec(cfg)

    if mlflow_enabled:
        import mlflow

        with mlflow.start_run(run_name=f"ctr-ranker-{time.strftime('%Y%m%d-%H%M%S')}"):
            _mlflow_log_params(cfg)
            summary = train(cfg, spec)
            summary.update(calibrate(cfg, spec))
            _log_artifacts(cfg)
    else:
        summary = train(cfg, spec)
        summary.update(calibrate(cfg, spec))

    _print_summary(summary)
    ok = check_targets(summary, cfg["targets"])
    return summary, ok


def _log_artifacts(cfg: dict[str, Any]) -> None:
    import mlflow

    artifacts = resolve(cfg["paths"]["ranker_artifacts"])
    for name in ("metrics.csv", "feature_spec.json", "best_model.pt", "calibrator.joblib"):
        path = artifacts / name
        if path.exists():
            mlflow.log_artifact(str(path))
    figure = resolve(cfg["paths"]["figures"]) / "calibration.png"
    if figure.exists():
        mlflow.log_artifact(str(figure))

"""Batch scoring: append raw and calibrated pCTR columns to each split.

Reads the processed Parquet splits, scores them with the trained model (position
bias zeroed at inference), applies the fitted isotonic calibrator, and writes the
original columns plus ``pctr_raw`` and ``pctr`` to ``data/scored/{split}/``.
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mua.config import resolve
from mua.ranker.dataset import RankerDataset, score_dataset
from mua.ranker.model import load_model

logger = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    artifacts = resolve(cfg["paths"]["ranker_artifacts"])
    if not (artifacts / "best_model.pt").exists():
        raise FileNotFoundError(
            f"No ranker checkpoint at {artifacts / 'best_model.pt'}; run `train-ranker` first"
        )
    if not (artifacts / "calibrator.joblib").exists():
        raise FileNotFoundError(
            f"No calibrator at {artifacts / 'calibrator.joblib'}; run `train-ranker` first"
        )

    model, spec, _ = load_model(artifacts, cfg["model"])
    calibrator = joblib.load(artifacts / "calibrator.joblib")
    processed = resolve(cfg["paths"]["processed"])
    scored = resolve(cfg["paths"]["scored"])
    batch_size = int(cfg["scoring"]["batch_size"])

    summary: dict[str, Any] = {}
    for split in cfg["scoring"]["splits"]:
        dataset = RankerDataset(processed / split, spec, mode="chunked")
        pctr_raw, _ = score_dataset(model, dataset, batch_size)
        pctr = calibrator.transform(pctr_raw)

        table = pq.read_table(processed / split)
        table = table.append_column("pctr_raw", pa.array(pctr_raw.astype(np.float32)))
        table = table.append_column("pctr", pa.array(pctr.astype(np.float32)))

        out = scored / split
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / "part-0.parquet")

        summary[split] = {
            "rows": int(table.num_rows),
            "mean_pctr_raw": float(pctr_raw.mean()),
            "mean_pctr": float(pctr.mean()),
        }
        logger.info(
            "Scored %s: %d rows, mean pctr_raw=%.5f mean pctr=%.5f -> %s",
            split,
            summary[split]["rows"],
            summary[split]["mean_pctr_raw"],
            summary[split]["mean_pctr"],
            out,
        )
    return summary

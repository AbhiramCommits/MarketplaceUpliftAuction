"""PyTorch dataset for the CTR ranker, reading processed Parquet via pyarrow.

Two modes are supported:

- ``in_memory``: the whole split is loaded and preprocessed into numpy arrays at
  construction time (fast for small / CI-scale data).
- ``chunked``: rows are served lazily with a single-row-group cache, so memory stays
  constant for large splits while preserving random access.

Categorical features are encoded via configurable hash buckets (consumer/merchant ids)
or fixed vocabularies (cuisine, hour, day_of_week, price_tier). Numeric features are
z-scored with statistics from the feature spec (computed on the training split only).
``slate_rank`` is passed through as the bias feature for the position-bias tower.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler

logger = logging.getLogger(__name__)

PHI = 2654435761  # Knuth multiplicative hash constant


class GroupShuffleSampler(Sampler[int]):
    """Yields all row indices with parquet row-group locality, groups shuffled.

    In chunked mode the dataset caches a single row group at a time; random access
    across row groups would thrash that cache. This sampler shuffles the *order* of
    row groups each epoch but keeps indices within a group contiguous, so the cache
    stays warm while the data is still shuffled between epochs.
    """

    def __init__(self, starts: np.ndarray, sizes: np.ndarray, seed: int):
        self._starts = starts
        self._sizes = sizes
        self._seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        order = rng.permutation(len(self._sizes))
        for group in order:
            start = int(self._starts[group])
            yield from range(start, start + int(self._sizes[group]))

    def __len__(self) -> int:
        return int(self._starts[-1] + self._sizes[-1])


def ranker_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate dataset items (numpy scalars/vectors) into torch batch tensors."""
    first = items[0]
    categorical = {
        name: torch.from_numpy(np.array([it["categorical"][name] for it in items], dtype=np.int64))
        for name in first["categorical"]
    }
    numeric = torch.from_numpy(np.stack([it["numeric"] for it in items]))
    bias = torch.from_numpy(np.array([it["bias"] for it in items], dtype=np.int64))
    label = torch.from_numpy(np.array([it["label"] for it in items], dtype=np.float32))
    return {"categorical": categorical, "numeric": numeric, "bias": bias, "label": label}


class RankerDataset(Dataset):
    def __init__(self, path: str | Path, feature_spec: dict[str, Any], mode: str = "chunked"):
        self.path = Path(path)
        self.spec = feature_spec
        self.mode = mode
        self.cat_specs = feature_spec["categorical"]
        self.numeric_names = list(feature_spec["numeric"])
        self.numeric_stats = feature_spec["numeric_stats"]
        self.cuisine_categories = feature_spec["cuisine_categories"]
        self.bias_name = feature_spec["bias"]["feature"]
        self.label_name = feature_spec["label"]

        self._files = sorted(str(p) for p in self.path.glob("*.parquet"))
        if not self._files:
            raise FileNotFoundError(f"No parquet files found under {self.path}")
        self._groups: list[tuple[str, int, int]] = []
        for file in self._files:
            pf = pq.ParquetFile(file)
            for rg in range(pf.num_row_groups):
                self._groups.append((file, rg, pf.metadata.row_group(rg).num_rows))
        sizes = np.array([g[2] for g in self._groups], dtype=np.int64)
        self._starts = np.concatenate([[0], np.cumsum(sizes[:-1])])
        self._sizes = sizes
        self.n = int(sizes.sum())

        self._cache: tuple[int, dict[str, np.ndarray]] | None = None
        self._all: dict[str, np.ndarray] | None = None
        if mode == "in_memory":
            logger.info("Loading %s into memory (%d rows)", self.path, self.n)
            self._all = self._preprocess(pq.read_table(self._files))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self.n
        if self._all is not None:
            arrays = self._all
        else:
            group = int(np.searchsorted(self._starts, idx, side="right") - 1)
            arrays = self._get_group(group)
            idx = idx - int(self._starts[group])
        return {
            "categorical": {name: arrays[name][idx] for name in self.cat_specs},
            "numeric": arrays["numeric"][idx],
            "bias": arrays["bias"][idx],
            "label": arrays["label"][idx],
        }

    def make_sampler(self, shuffle: bool = False, epoch: int = 0, seed: int = 0) -> Sampler[int]:
        if not shuffle:
            return SequentialSampler(self)
        if self.mode == "in_memory":
            generator = torch.Generator().manual_seed(seed + epoch)
            return RandomSampler(self, generator=generator)
        return GroupShuffleSampler(self._starts, self._sizes, seed=seed + epoch)

    def make_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        epoch: int = 0,
        seed: int = 0,
        drop_last: bool = False,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=self.make_sampler(shuffle=shuffle, epoch=epoch, seed=seed),
            num_workers=0,
            collate_fn=ranker_collate,
            drop_last=drop_last,
        )

    def _get_group(self, group: int) -> dict[str, np.ndarray]:
        if self._cache is None or self._cache[0] != group:
            file, rg, _ = self._groups[group]
            table = pq.ParquetFile(file).read_row_group(rg)
            self._cache = (group, self._preprocess(table))
        return self._cache[1]

    def _preprocess(self, table) -> dict[str, np.ndarray]:
        n = table.num_rows
        out: dict[str, np.ndarray] = {}
        for name, spec in self.cat_specs.items():
            if name == "cuisine":
                continue
            values = table[name].to_numpy().astype(np.int64)
            if spec.get("hash_buckets"):
                values = (values * PHI) % int(spec["hash_buckets"])
            values = np.maximum(values - int(spec.get("offset", 0)), 0)
            out[name] = values
        cuisine_idx = np.zeros(n, dtype=np.int64)
        for i, cat in enumerate(self.cuisine_categories):
            column = f"cuisine_{cat}"
            if column in table.column_names:
                cuisine_idx += i * table[column].to_numpy().astype(np.int64)
        out["cuisine"] = cuisine_idx

        numeric = np.empty((n, len(self.numeric_names)), dtype=np.float32)
        for j, name in enumerate(self.numeric_names):
            stats = self.numeric_stats[name]
            values = table[name].to_numpy().astype(np.float64)
            numeric[:, j] = ((values - stats["mean"]) / stats["std"]).astype(np.float32)
        out["numeric"] = numeric
        out["bias"] = table[self.bias_name].to_numpy().astype(np.int64)
        out["label"] = table[self.label_name].to_numpy().astype(np.float32)
        return out


@torch.no_grad()
def score_dataset(
    model: torch.nn.Module,
    dataset: RankerDataset,
    batch_size: int,
    include_bias: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a full split, returning (predictions, labels) as numpy arrays."""
    was_training = model.training
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in dataset.make_loader(batch_size, shuffle=False):
        preds.append(model(batch, include_bias=include_bias).numpy())
        labels.append(batch["label"].numpy())
    if was_training:
        model.train()
    return (
        np.concatenate(preds).astype(np.float64),
        np.concatenate(labels).astype(np.float64),
    )

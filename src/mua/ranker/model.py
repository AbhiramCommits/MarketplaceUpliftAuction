"""Two-tower-flavored deep CTR model with position-bias debiasing.

Why a separate bias tower?
---------------------------
Displayed rank (``slate_rank``) is a strong confounder of clicks: items shown at the
top of the slate get clicks regardless of intrinsic relevance (position bias). If rank
enters the same tower as the relevance features, the learned score conflates "how good
is this item" with "where was it shown". The auction layer later computes
eCPM = bid * pCTR, so pCTR must reflect position-independent relevance; otherwise
items would be over-credited for lucky placements.

We therefore model  ``logit = main_tower(x) + bias_tower(rank)`` during training: the
bias tower absorbs the display-position effect, leaving the main tower free to learn a
clean, position-independent relevance score. At inference the bias contribution is
zeroed (``include_bias=False``) and ``pCTR = sigmoid(main_tower(x))``.

Architecture: embedding layers for all categorical features (consumer_id hashed to
2^16 buckets, merchant_id to 2^13, cuisine, hour_of_day, day_of_week, price_tier),
concatenated with z-scored numeric features, then an MLP [256, 128, 64] with ReLU,
dropout 0.2, BatchNorm, and a sigmoid output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class CTRRanker(nn.Module):
    def __init__(self, feature_spec: dict[str, Any], model_cfg: dict[str, Any]):
        super().__init__()
        self.cat_names = list(feature_spec["categorical"])
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(spec["vocab_size"]), int(spec["embedding_dim"]))
                for name, spec in feature_spec["categorical"].items()
            }
        )

        bias_spec = feature_spec["bias"]
        self.bias_embedding = nn.Embedding(
            int(bias_spec["vocab_size"]), int(bias_spec["embedding_dim"]), padding_idx=0
        )
        self.bias_mlp = nn.Sequential(
            nn.Linear(int(bias_spec["embedding_dim"]), 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        emb_out = sum(int(s["embedding_dim"]) for s in feature_spec["categorical"].values())
        in_dim = emb_out + len(feature_spec["numeric"])
        layers: list[nn.Module] = []
        for hidden in model_cfg["mlp_hidden"]:
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden)),
                    nn.BatchNorm1d(int(hidden)),
                    nn.ReLU(),
                    nn.Dropout(float(model_cfg["dropout"])),
                ]
            )
            in_dim = int(hidden)
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Any], include_bias: bool = True) -> torch.Tensor:
        parts = [self.embeddings[name](batch["categorical"][name]) for name in self.cat_names]
        parts.append(batch["numeric"])
        logit = self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)
        if include_bias:
            bias = self.bias_mlp(self.bias_embedding(batch["bias"])).squeeze(-1)
            logit = logit + bias
        return torch.sigmoid(logit)


def load_model(
    artifact_dir: str | Path, model_cfg: dict[str, Any]
) -> tuple[CTRRanker, dict[str, Any], dict[str, Any]]:
    """Load the best checkpoint plus its feature spec; returns (model, spec, ckpt)."""
    artifact_dir = Path(artifact_dir)
    checkpoint = torch.load(artifact_dir / "best_model.pt", map_location="cpu", weights_only=False)
    spec = checkpoint["feature_spec"]
    model = CTRRanker(spec, model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, spec, checkpoint

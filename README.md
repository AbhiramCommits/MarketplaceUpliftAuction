# marketplace-uplift-auction

Simulated three-sided marketplace (consumers, merchants, dashers) with a confounded
promo treatment and known ground-truth causal effects. Built to develop and score
uplift models, rankers, and ad-auction policies against a known truth.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Java 17+ (for PySpark)

## Quick start

```sh
make setup   # uv sync (creates .venv + uv.lock)
make test    # pytest
make lint    # ruff + black --check
make data    # generate sim data + build features
make all     # setup, lint, test, data, train, causal, auction, report
```

## CLI

```sh
uv run python -m mua.cli generate --scale small   # 50k-row CI run
uv run python -m mua.cli generate                 # full ~2M rows (seed 42)
uv run python -m mua.cli features                 # join, encode, time-split
```

## Data layout

| Path | Contents |
| --- | --- |
| `data/raw/impressions/` | Impression-level event log, Parquet partitioned by `day` |
| `data/raw/{consumers,merchants,dashers}/` | Entity dimension tables (Parquet) |
| `data/truth/` | Ground truth keyed by `impression_id`: `true_propensity`, `true_tau`, `baseline_order_prob` |
| `data/processed/{train,valid,test}/` | Joined + encoded features, split by time (days 1-21 / 22-25 / 26-30) |

`data/` and `artifacts/` are gitignored. Reports go in `reports/`.

## Treatment & outcomes

`promo_treated` is assigned by a confounded logistic propensity that depends on
`price_sensitivity`, `is_new_user`, merchant `historical_ctr`, and `is_peak`
(calibrated to ~35% treated, never a coin flip). Outcomes come from explicit
structural models in `src/mua/sim/generate.py`:

- `baseline_order_prob(features)` - untreated conversion probability
- `true_tau(features)` - heterogeneous individual treatment effect (strongly positive
  for price-sensitive new users facing slow delivery; negative for loyal,
  low-sensitivity users on cheap merchants)
- realized `ordered ~ Bernoulli(clip(baseline + treated * tau, 0.001, 0.95))`

Ground-truth columns are written only to `data/truth/` so model code cannot leak them.

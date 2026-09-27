# Methodology

*A short research note on the simulated marketplace, the causal estimation
benchmark, and the auction design.*

## Problem

Should a food-delivery marketplace discount delivery fees to drive incremental
orders, and how should it sell sponsored placement on the merchant slate? We study
both questions on a simulated three-sided marketplace where the true treatment
effect is known by construction. The goal is not to answer the questions for any
real market, but to build a reproducible environment in which (a) causal estimators
can be scored against ground truth, (b) auction mechanisms can be checked against
their incentive properties, and (c) the end-to-end pipeline — logging, ranking,
causal modeling, auctioning — can be exercised as one system.

## Data-generating process

A single seed drives everything (`sim/generate.py`). The data-generating process
(DGP) has three entity layers and one event layer:

- **Consumers (50k)**: `price_sensitivity ~ Beta(2,5)`, an order-frequency prior,
  distance tolerance, a new-user flag, tenure, and a latent location.
- **Merchants (5k)**: cuisine (12 levels), rating, prep time, price tier,
  `historical_ctr ~ lognormal`, an ads-advertiser flag, and a latent location.
- **Dashers (800)**: per-dasher activity probabilities that determine a market-level
  supply ratio per (day, hour), bounded in [0.5, 2.0].
- **Impressions (~2M over 30 days)**: consumers and merchants are sampled with
  weights proportional to order frequency and historical CTR; context features
  (hour, peak window, distance, delivery-time estimate) are derived from the
  entities.

**Treatment.** `promo_treated` is assigned by a confounded logistic propensity:

```
logit P(T=1|X) = intercept + 1.2·price_sensitivity + 0.9·is_new_user
               + 1.0·historical_ctr + 0.35·is_peak
```

with the intercept calibrated so the treated share is ~35%. The propensity is a
function of covariates that also drive the outcome, so the naive difference in
means is biased upward (the DGP deliberately reproduces positive selection into
treatment).

**Outcomes.** Two explicit structural models produce orders and clicks:

- `baseline_order_prob(features)` — untreated conversion probability
  (~6.5% on average).
- `true_tau(features)` — heterogeneous individual treatment effect: strongly
  positive for price-sensitive new users facing slow delivery, negative for loyal,
  low-sensitivity users on cheap merchants. Mean tau ≈ 0.024.
- Realized probability = clip(baseline + treated·tau, 0.001, 0.95);
  `ordered ~ Bernoulli(p)`.
- Clicks come from a separate model: `p_ctr · (1 + relevance factors) / log2(rank+1)`
  with a position-bias curve, so click prediction and order prediction are
  distinct tasks.

Ground-truth columns (`true_propensity`, `true_tau`, `baseline_order_prob`) are
written to a separate `data/truth/` dataset keyed by impression id and never enter
the feature matrix.

## Identification strategy

The estimand is the ATE (and CATE) of the fee-discount promo on order probability.
Identification rests on the DGP's no-unmeasured-confounder structure: treatment
depends only on `price_sensitivity`, `is_new_user`, `historical_ctr`, and
`is_peak`, all of which are observed in the logged features. Under that structure,
conditioning (or IPTW, or doubly robust combinations) recovers the causal effect;
this is checked, not assumed, by a DoWhy refutation battery on the realized data.

## Estimators

- **Naive difference in means** (deliberately kept as the biased baseline).
- **IPTW** with a gradient-boosted propensity model; units outside [0.02, 0.98]
  trimmed; covariate balance checked with standardized mean differences before and
  after weighting (threshold 0.1, enforced).
- **S-, T-, X-learners** (EconML metalearners; HistGradientBoosting or LightGBM
  base models on the binary outcome, linear-probability style).
- **DRLearner** (EconML) with cross-fitted propensity and outcome models.
- **CausalForestDML** with honest splitting and built-in confidence intervals.

All estimators fit on the same train split (days 1-21); evaluation is on the
held-out test split (days 26-30) against `true_tau`.

## Evaluation metrics

- ATE bias and 95% CI coverage against the ground-truth ATE on the test sample.
- PEHE = sqrt(mean((τ̂ - τ)²)) — individual-level error, only possible because τ is
  known.
- Qini coefficient and AUUC for ranking quality of the predicted CATE.
- Refutations: placebo treatment (effect must collapse to ~0), random common cause
  (effect must be stable), data subsets, and an E-value sensitivity analysis.

## Auction model

Sponsored slots are allocated on a 20-item merchant slate by eCPM = bid × pCTR,
using the ranker's calibrated, position-debiased pCTR. Mechanisms:

- **FirstPrice**: winner pays its bid (truthful bidding is not optimal; shading
  agents learn bid-shading factors online).
- **SecondPrice**: single slot, winner pays the minimum bid needed to keep it.
- **GSP**: slot i pays the quality-adjusted next price (bid_{i+1}·pCTR_{i+1}/pCTR_i).
- **VCG**: each winner pays its externality (the (K+1)-th eCPM / own pCTR).

Agents: truthful, first-price shaders (multiplicative-weights), budget-paced
(dual-variable λ throttling toward a daily budget), and random controls.
Incentive properties are verified empirically: deviation grids show truthful
bidding is dominant in second-price/VCG; truthful bidding is provably beatable in
first-price; property-based tests check payment ≤ bid and valid allocations for
arbitrary inputs.

## Results

On the full 2M-impression run (seed 42): all debiased estimators recover the ATE
within ±0.003 of the ground-truth 0.0242, while the naive estimator is biased by
+0.0065. The X-learner achieves the best PEHE (0.0157), the causal forest the best
Qini (0.53). All refutations pass. In the mechanism × targeting experiment,
uplift-top-k targeting yields 0.0529 incremental orders per promo dollar versus
0.0242 for treat-everyone; first-price extracts the most platform revenue, VCG the
most advertiser surplus. Full numbers in `reports/REPORT.md`.

## Threats to validity

- All findings are internal to the DGP; external validity is nil by design.
- The linear-probability-style outcome models and the position-free auction model
  are simplifications of real marketplace mechanics.
- No interference, spillover, or strategic response beyond the implemented agents.
- Refutation verdicts are necessary but not sufficient for identification.

## Future work

- Position-weighted auction payments and slate-level position bias.
- Multi-objective optimization of revenue vs consumer welfare.
- Heterogeneous-agent equilibria (repeated-game shading convergence theory).
- Extension to multi-day pacing with true daily budgets and replenishment.

# Refutation report

- Estimand: ATE of `promo_treated` on `ordered`
- Estimate (backdoor linear regression): **0.0251**
- Observed outcome rates: treated 0.0986, control 0.0702
- DAG written to `/Users/abhiramkasireddi/MarketplaceUpliftAuction/reports/causal_dag.dot`

| refuter | estimated effect | refuted effect | metric | threshold | verdict |
|---|---|---|---|---|---|
| placebo_treatment (permute) | 0.0251 | 0.0002 | placebo p-value = 0.9776 (zero under placebo null) | p > 0.05 | PASS |
| random_common_cause | 0.0251 | 0.0251 | relative shift = 0.0000 | shift <= 0.2 | PASS |
| data_subset | 0.0251 | 0.0251 | all subset shifts <= tolerance | shift <= 0.2 | PASS |
| unobserved_common_cause (sensitivity) | 0.0251 | n/a | E-value = 2.160 | E-value >= 1.1 | PASS |

## Sensitivity

To fully explain away the observed effect, an unmeasured confounder would need to be associated with both treatment and outcome by a risk ratio of at least **2.16** (E-value).


"""DoWhy end-to-end refutation battery for the promo effect.

Builds a CausalModel with an explicit DAG (written to reports/causal_dag.dot),
identifies the estimand, estimates with backdoor adjustment, then runs: placebo
treatment (permutation p-value), random common cause (effect must be stable), data
subset refuter (bootstrap subsets), and an unmeasured-confounder sensitivity analysis
reporting the E-value. Each refutation writes a pass/fail verdict with its tolerance
into reports/refutation_report.md.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np

from mua.causal.data import load_frame
from mua.config import resolve

logger = logging.getLogger(__name__)

TREATMENT = "promo_treated"
OUTCOME = "ordered"


def build_dag(feature_names: list[str]) -> tuple[str, str]:
    """Return (gml, dot) for the causal graph: every covariate confounds W and Y."""
    node_lines = "\n".join(f'  node [id "{n}" label "{n}"]' for n in feature_names)
    node_lines += f'\n  node [id "{TREATMENT}" label "{TREATMENT}"]'
    node_lines += f'\n  node [id "{OUTCOME}" label "{OUTCOME}"]'
    edge_lines = ""
    for n in feature_names:
        edge_lines += (
            f'\n  edge [source "{n}" target "{TREATMENT}"]'
            f'\n  edge [source "{n}" target "{OUTCOME}"]'
        )
    edge_lines += f'\n  edge [source "{TREATMENT}" target "{OUTCOME}"]'
    gml = f"graph [\ndirected 1\n{node_lines}{edge_lines}\n]"
    dot = (
        "digraph causal_dag {\n"
        f"  {TREATMENT} -> {OUTCOME};\n"
        + "\n".join(f"  {n} -> {TREATMENT}; {n} -> {OUTCOME};" for n in feature_names)
        + "\n}"
    )
    return gml, dot


def _linear_ate(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    """ATE from a linear backdoor regression (matches DoWhy's estimator)."""
    from sklearn.linear_model import LinearRegression

    design = np.column_stack([X, w])
    model = LinearRegression().fit(design, y)
    return float(model.coef_[-1])


def placebo_pvalue(null_effects: np.ndarray) -> float:
    """DoWhy-style placebo p-value: how likely is ZERO under the placebo null?

    Under a placebo (permuted) treatment the causal link is severed, so a valid
    estimator must produce effects centered on zero. If zero is inconsistent with
    the placebo distribution (p <= 0.05), the estimator fabricates effects and the
    refutation fails.
    """
    from scipy import stats

    std = float(null_effects.std(ddof=1))
    if std < 1e-12:
        return 1.0
    z = abs(float(null_effects.mean())) / std
    return float(2 * (1 - stats.norm.cdf(z)))


def permutation_null_effects(
    X: np.ndarray, w: np.ndarray, y: np.ndarray, n_sim: int, seed: int
) -> np.ndarray:
    """Effects of the same estimator under permuted treatments (placebo)."""
    rng = np.random.default_rng(seed)
    return np.array([_linear_ate(X, rng.permutation(w), y) for _ in range(n_sim)])


def e_value(r_treated: float, r_control: float) -> float | None:
    """E-value: min confounder-outcome/treatment RR needed to explain away the effect."""
    if r_treated <= r_control or r_control <= 0:
        return None
    rr = r_treated / r_control
    return float(rr + np.sqrt(rr * (rr - 1)))


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    from dowhy import CausalModel

    seed = int(cfg["data"]["seed"])
    sample_refute = cfg["data"].get("sample_refute")
    sample_refute = None if sample_refute is None else int(sample_refute)
    frame = load_frame(cfg, "train", sample=sample_refute, seed=seed)
    X, w, y = frame["X"], frame["w"], frame["y"]

    gml, dot = build_dag(frame["feature_names"])
    reports_dir = resolve(cfg["paths"]["figures"]).parent
    dag_path = reports_dir / "causal_dag.dot"
    dag_path.parent.mkdir(parents=True, exist_ok=True)
    dag_path.write_text(dot + "\n")

    import pandas as pd

    df = pd.DataFrame(X, columns=frame["feature_names"])
    df[TREATMENT] = w
    df[OUTCOME] = y

    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=gml,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name="backdoor.linear_regression", target_units="ate"
    )
    ate = float(estimate.value)
    logger.info("DoWhy backdoor ATE estimate: %.4f", ate)

    rc = cfg["refute"]
    shift_tolerance = float(rc["effect_shift_tolerance"])
    p_threshold = float(rc["placebo_pvalue_threshold"])
    verdicts: list[dict[str, Any]] = []

    logger.info("Refutation 1/4: placebo treatment (permute) ...")
    placebo = model.refute_estimate(
        estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=int(rc["placebo_iterations"]),
    )
    null_effects = permutation_null_effects(X, w, y, int(rc["placebo_iterations"]), seed)
    p_value = placebo_pvalue(null_effects)
    placebo_pass = p_value > p_threshold
    verdicts.append(
        {
            "refuter": "placebo_treatment (permute)",
            "estimated_effect": ate,
            "new_effect": float(placebo.new_effect),
            "metric": f"placebo p-value = {p_value:.4f} (zero under placebo null)",
            "threshold": f"p > {p_threshold}",
            "verdict": "PASS" if placebo_pass else "FAIL",
        }
    )

    logger.info("Refutation 2/4: random common cause ...")
    rcc = model.refute_estimate(
        estimand,
        estimate,
        method_name="random_common_cause",
        num_simulations=10,
    )
    rcc_effect = float(rcc.new_effect)
    rcc_shift = abs(rcc_effect - ate) / max(abs(ate), 1e-12)
    rcc_pass = rcc_shift <= shift_tolerance
    verdicts.append(
        {
            "refuter": "random_common_cause",
            "estimated_effect": ate,
            "new_effect": rcc_effect,
            "metric": f"relative shift = {rcc_shift:.4f}",
            "threshold": f"shift <= {shift_tolerance}",
            "verdict": "PASS" if rcc_pass else "FAIL",
        }
    )

    logger.info("Refutation 3/4: data subset (bootstrap) ...")
    subset = model.refute_estimate(
        estimand,
        estimate,
        method_name="data_subset_refuter",
        subset_fractions=[float(f) for f in rc["subset_fractions"]],
    )
    subset_pass = True
    for new_effect in np.atleast_1d(subset.new_effect):
        shift = abs(float(new_effect) - ate) / max(abs(ate), 1e-12)
        if shift > shift_tolerance:
            subset_pass = False
    verdicts.append(
        {
            "refuter": "data_subset",
            "estimated_effect": ate,
            "new_effect": [float(v) for v in np.atleast_1d(subset.new_effect)],
            "metric": "all subset shifts <= tolerance",
            "threshold": f"shift <= {shift_tolerance}",
            "verdict": "PASS" if subset_pass else "FAIL",
        }
    )

    logger.info("Refutation 4/4: unobserved confounder sensitivity (E-value) ...")
    r_t, r_c = float(y[w == 1].mean()), float(y[w == 0].mean())
    e_val = e_value(r_t, r_c)
    e_pass = e_val is None or e_val >= 1.1
    verdicts.append(
        {
            "refuter": "unobserved_common_cause (sensitivity)",
            "estimated_effect": ate,
            "new_effect": None,
            "metric": (
                f"E-value = {e_val:.3f}"
                if e_val is not None
                else "E-value N/A (non-positive effect)"
            ),
            "threshold": "E-value >= 1.1",
            "verdict": "PASS" if e_pass else "FAIL",
        }
    )

    _write_report(cfg, ate, r_t, r_c, e_val, verdicts, dag_path)
    for v in verdicts:
        logger.info("Refutation %s: %s (%s)", v["refuter"], v["verdict"], v["metric"])
    ok = all(v["verdict"] == "PASS" for v in verdicts)
    return {"ate": ate, "verdicts": verdicts, "ok": ok}


def _write_report(cfg, ate, r_t, r_c, e_val, verdicts, dag_path) -> None:
    lines = [
        "# Refutation report",
        "",
        f"- Estimand: ATE of `{TREATMENT}` on `{OUTCOME}`",
        f"- Estimate (backdoor linear regression): **{ate:.4f}**",
        f"- Observed outcome rates: treated {r_t:.4f}, control {r_c:.4f}",
        f"- DAG written to `{dag_path}`",
        "",
        "| refuter | estimated effect | refuted effect | metric | threshold | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        if isinstance(v["new_effect"], (list, tuple, np.ndarray)):
            new = ", ".join(f"{x:.4f}" for x in v["new_effect"])
        else:
            new = "n/a" if v["new_effect"] is None else f"{v['new_effect']:.4f}"
        lines.append(
            f"| {v['refuter']} | {v['estimated_effect']:.4f} | {new} "
            f"| {v['metric']} | {v['threshold']} | {v['verdict']} |"
        )
    lines += [
        "",
        "## Sensitivity",
        "",
        (
            f"To fully explain away the observed effect, an unmeasured confounder would "
            f"need to be associated with both treatment and outcome by a risk ratio of "
            f"at least **{e_val:.2f}** (E-value)."
            if e_val is not None
            else "Effect is non-positive; E-value not applicable."
        ),
        "",
    ]
    path = resolve(cfg["paths"]["figures"]).parent / "refutation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("Refutation report saved to %s", path)

"""
errormodes.py — how allele-frequency disparities become dosing-dashboard errors.

A genotype-aware dosing tool does not need a population allele frequency once
it has the patient's own genotype. Frequencies enter dashboards in four
places, each of which is an error mode evaluated here:

  Reference ("policy") frequency    which single AF the dashboard displays /
                                    uses: the calibration population's, the
                                    pooled all-population AF, or a
                                    GTEx-composition-weighted AF.
  E1 inclusion threshold            variants are shown/tested only if the
                                    policy AF >= tau (a MAF filter). A variant
                                    rare in the calibration group but common
                                    in another is falsely excluded for it.
  E2 prevalence error               predicted actionable-genotype rate
                                    (Hardy-Weinberg) from the policy AF vs the
                                    population's own AF — drives "expected %
                                    needing an alternative drug" estimates,
                                    pre-test probability, capacity planning.
  E3 pooled-AF masking              how much the pooled AF is just the
                                    dominant population's AF.
  E4 crosswalk sensitivity          GTEx-weighted AF under different
                                    ancestry-label mapping assumptions.
  E5 decision stability             whether the include/exclude decision
                                    survives sampling uncertainty.
  E6 dataset discordance            same variant, same group, two datasets:
                                    is the gap bigger than sampling noise?

Everything assumes Hardy-Weinberg equilibrium (documented limitation) and
treats tau as a parameter, not as the property of any particular real tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .ancestry import (
    GTEX_V8_N_GENOTYPED_DONORS,
    gtex_weights_over_gnomad,
    harmonize_population,
)
from .methods import wilson_bounds
from .robustness import reference_mismatch_impact


# --------------------------------------------------------------------------
# Reference-frequency policies
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    name: str
    af: float
    an: float          # effective allele number behind this AF (for its own CI)
    description: str


def build_policies(
    rows: pd.DataFrame,
    calibration_population: Optional[str],
    gtex_asian_to: str = "eas",
    gtex_aa_european_fraction: float = 0.0,
) -> Tuple[List[Policy], List[str]]:
    """Construct the reference-frequency policies available for one variant.

    Returns (policies, warnings). A policy that cannot be built (e.g. no
    calibration population in the data) is skipped WITH a warning, never
    silently.
    """
    warnings: List[str] = []
    policies: List[Policy] = []
    af_by_pop = dict(zip(rows["population"], rows["allele_frequency"]))

    if calibration_population and calibration_population in af_by_pop:
        r = rows[rows["population"] == calibration_population].iloc[0]
        policies.append(Policy(
            f"calibration:{calibration_population}", float(r["allele_frequency"]),
            float(r["allele_number"]),
            f"AF of the calibration population ({calibration_population})."))
    else:
        warnings.append(f"calibration population '{calibration_population}' not present; "
                        "calibration policy skipped.")

    total_an = float(rows["allele_number"].sum())
    pooled = float(rows["allele_count"].sum() / total_an) if total_an > 0 else float("nan")
    policies.append(Policy("pooled_all", pooled, total_an,
                           "Allele-number-weighted AF pooled across every population."))

    weights, _excl = gtex_weights_over_gnomad(gtex_asian_to, gtex_aa_european_fraction)
    if all(code in af_by_pop for code in weights):
        af = float(sum(w * af_by_pop[c] for c, w in weights.items()))
        policies.append(Policy(
            "gtex_weighted", af, 2.0 * GTEX_V8_N_GENOTYPED_DONORS,
            "AF a GTEx-composition cohort (85% EUR / 12% AFR / 1% ASN) would "
            "see if each group carried its gnomAD frequency; its own sampling "
            "noise is that of 838 donors."))
    else:
        missing = [c for c in weights if c not in af_by_pop]
        warnings.append(f"gtex_weighted policy skipped; missing populations {missing}.")
    return policies, warnings


# --------------------------------------------------------------------------
# E1 — inclusion-threshold (MAF filter) exclusion
# --------------------------------------------------------------------------

def carrier_rate(p: float) -> float:
    """Probability of carrying >= 1 copy under Hardy-Weinberg."""
    return 1.0 - (1.0 - p) ** 2


def e1_inclusion_threshold(rows: pd.DataFrame, policies: List[Policy], tau: float,
                           conf: float = 0.95) -> pd.DataFrame:
    """For each policy x population: would a 'include iff AF >= tau' rule
    misclassify the population's own variant relevance?

    false_exclusion            point estimate meets tau but the policy excludes.
    false_exclusion_supported  the population's Wilson CI LOWER bound meets tau,
                               so the miss is not explainable by sampling noise
                               (this is the version the scorecard's F5 uses).
    """
    lo, _hi = wilson_bounds(rows["allele_count"], rows["allele_number"], conf)
    out = []
    for pol in policies:
        includes = pol.af >= tau
        for (_, r), l in zip(rows.iterrows(), lo):
            af = float(r["allele_frequency"])
            meets = af >= tau
            false_excl = meets and not includes
            out.append({
                "policy": pol.name, "tau": tau, "population": r["population"],
                "population_af": af, "population_ci_lo": float(l), "policy_af": pol.af,
                "policy_includes_variant": includes,
                "population_meets_threshold": meets,
                "false_exclusion": false_excl,
                "false_exclusion_supported": bool((l >= tau) and not includes),
                "false_inclusion": (not meets) and includes,
                "carriers_missed_per_10k": carrier_rate(af) * 1e4 if false_excl else 0.0,
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# E2 — prevalence-prediction error
# --------------------------------------------------------------------------

def _rate(p, zygosity: str):
    p = np.asarray(p, dtype=float)
    if zygosity == "recessive":
        return p ** 2
    if zygosity == "dominant_or_het":
        return 1 - (1 - p) ** 2
    raise ValueError("zygosity must be 'recessive' or 'dominant_or_het'")


def e2_prevalence_error(rows: pd.DataFrame, policies: List[Policy],
                        zygosity: str = "recessive", conf: float = 0.95) -> pd.DataFrame:
    """Predicted (from policy AF) vs population's own actionable-genotype rate.

    shortfall_per_10k > 0  => the dashboard UNDER-predicts how many patients
    in that population carry the actionable genotype (people are missed);
    < 0 => it over-predicts.

    The *_lb columns are conservative, uncertainty-aware versions: the
    population's rate evaluated at its Wilson LOWER bound minus the policy's
    rate at ITS Wilson UPPER bound. They are what the scorecard's F7 uses, so a
    gap that sampling noise alone could produce is not counted as material.
    """
    lo, _hi = wilson_bounds(rows["allele_count"], rows["allele_number"], conf)
    out = []
    for pol in policies:
        pl, ph = wilson_bounds(np.array([pol.af * pol.an]), np.array([pol.an]), conf)
        pol_rate_hi = float(_rate(ph[0], zygosity))
        for (_, r), l in zip(rows.iterrows(), lo):
            m = reference_mismatch_impact(pol.af, float(r["allele_frequency"]), zygosity)
            shortfall = (m["target_rate"] - m["calibration_rate"]) * 1e4
            pop_rate_lo = float(_rate(l, zygosity))
            out.append({
                "policy": pol.name, "population": r["population"], "zygosity": zygosity,
                "predicted_rate": m["calibration_rate"], "true_rate": m["target_rate"],
                "shortfall_per_10k": shortfall,
                "true_to_predicted_ratio": m["rate_ratio"],
                "shortfall_lb_per_10k": (pop_rate_lo - pol_rate_hi) * 1e4,
                "ratio_lb": (pop_rate_lo / pol_rate_hi) if pol_rate_hi > 0 else float("inf"),
                "direction": ("under-predicts" if shortfall > 1e-9
                              else "over-predicts" if shortfall < -1e-9 else "match"),
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# E3 — pooled-AF masking
# --------------------------------------------------------------------------

def e3_pooling_masking(rows: pd.DataFrame) -> Dict:
    total = float(rows["allele_number"].sum())
    pooled = float(rows["allele_count"].sum() / total)
    shares = rows["allele_number"] / total
    dom = rows.loc[shares.idxmax()]
    afs = rows["allele_frequency"]
    return {
        "pooled_af": pooled,
        "dominant_population": dom["population"],
        "dominant_share": float(shares.max()),
        "dominant_population_af": float(dom["allele_frequency"]),
        "pooled_over_dominant_af": (pooled / dom["allele_frequency"]
                                    if dom["allele_frequency"] > 0 else float("nan")),
        "max_pop_af_over_pooled": (float(afs.max()) / pooled if pooled > 0 else float("nan")),
        "n_populations_over_2x_pooled": int((afs > 2 * pooled).sum()) if pooled > 0 else 0,
    }


# --------------------------------------------------------------------------
# E4 — crosswalk sensitivity of the GTEx-weighted AF
# --------------------------------------------------------------------------

def e4_crosswalk_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    """GTEx-weighted AF under alternative label-mapping assumptions.

    Varies where GTEx's undifferentiated 'Asian American' donors are mapped
    (eas / sas / split) and how much European admixture is assumed in the
    'African American' donors (0, 0.2, 0.3 — assumption values, not
    measurements). The spread of the result is the harmonisation uncertainty.
    """
    af = dict(zip(rows["population"], rows["allele_frequency"]))
    out = []
    for asian_to in ("eas", "sas", "split"):
        for admix in (0.0, 0.2, 0.3):
            w, excl = gtex_weights_over_gnomad(asian_to, admix)
            if not all(c in af for c in w):
                continue
            out.append({
                "asian_to": asian_to, "aa_european_fraction": admix,
                "gtex_weighted_af": float(sum(wt * af[c] for c, wt in w.items())),
                "excluded_donor_fraction": excl,
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# E5 — decision stability under sampling uncertainty
# --------------------------------------------------------------------------

def e5_decision_stability(rows: pd.DataFrame, policies: List[Policy], tau: float,
                          conf: float = 0.95) -> pd.DataFrame:
    """Is 'AF >= tau' decided consistently across each estimate's Wilson CI?

    status: 'include' (CI entirely >= tau), 'exclude' (CI entirely < tau),
            'indeterminate' (CI straddles tau).
    Applied to every population's own AF and to each policy's AF (using the
    policy's own effective allele number).
    """
    def classify(lo: float, hi: float) -> str:
        if lo >= tau:
            return "include"
        if hi < tau:
            return "exclude"
        return "indeterminate"

    out = []
    lo, hi = wilson_bounds(rows["allele_count"], rows["allele_number"], conf)
    for (_, r), l, h in zip(rows.iterrows(), lo, hi):
        out.append({"subject": r["population"], "kind": "population", "tau": tau,
                    "af": float(r["allele_frequency"]), "ci_lo": float(l), "ci_hi": float(h),
                    "status": classify(l, h)})
    for pol in policies:
        # Back out an integer-ish count from the policy AF for the CI.
        l, h = wilson_bounds(np.array([pol.af * pol.an]), np.array([pol.an]), conf)
        out.append({"subject": pol.name, "kind": "policy", "tau": tau, "af": pol.af,
                    "ci_lo": float(l[0]), "ci_hi": float(h[0]),
                    "status": classify(float(l[0]), float(h[0]))})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# E6 — dataset discordance
# --------------------------------------------------------------------------

def _harmonized_counts(table: pd.DataFrame, crosswalk: Dict[str, str]) -> pd.DataFrame:
    t = table.copy()
    t["group"] = t["population"].map(lambda c: harmonize_population(c, crosswalk))
    g = t.groupby(["variant_id", "group"], as_index=False)[["allele_count", "allele_number"]].sum()
    g["allele_frequency"] = g["allele_count"] / g["allele_number"]
    return g


def e6_dataset_discordance(
    table_a: pd.DataFrame, crosswalk_a: Dict[str, str],
    table_b: pd.DataFrame, crosswalk_b: Dict[str, str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare the same variant in the same harmonised group across datasets.

    Two-sided pooled two-proportion z-test; Bonferroni across all
    comparisons. A significant gap here is a *dataset* effect (label
    definition, cohort selection) rather than sampling noise.
    """
    a = _harmonized_counts(table_a, crosswalk_a)
    b = _harmonized_counts(table_b, crosswalk_b)
    m = a.merge(b, on=["variant_id", "group"], suffixes=("_a", "_b"))
    m = m[m["group"] != "UNMAPPED"]
    rows = []
    for _, r in m.iterrows():
        c1, n1, c2, n2 = r["allele_count_a"], r["allele_number_a"], r["allele_count_b"], r["allele_number_b"]
        pooled = (c1 + c2) / (n1 + n2)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        z = (c1 / n1 - c2 / n2) / se if se > 0 else 0.0
        rows.append({
            "variant_id": r["variant_id"], "group": r["group"],
            "af_a": c1 / n1, "af_b": c2 / n2, "n_a": n1, "n_b": n2,
            "abs_diff": abs(c1 / n1 - c2 / n2),
            "z": z, "p_value": float(2 * stats.norm.sf(abs(z))),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p_bonferroni"] = np.minimum(1.0, res["p_value"] * len(res))
    res["discordant"] = res["p_bonferroni"] < alpha
    return res

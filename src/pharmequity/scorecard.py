"""
scorecard.py — one fixed rule set applied identically to every variant.

Seven boolean flags, then one classification. All thresholds live in
ScorecardConfig and are written into every report (plus a config hash), so a
result is always interpretable and reproducible, and thresholds can be changed
deliberately instead of drifting between analyses.

Flags
-----
F1 disparity_supported       interval-ratio lower bound >= min_supported_ratio
F2 heterogeneity_significant Monte Carlo chi-square p (BH-adjusted across the
                             variants in a dataset) < alpha
F3 differentiation_moderate  mean pairwise Hudson Fst >= fst_threshold
                             (0.05 is Wright's conventional lower bound for
                             'moderate' differentiation; a convention, not a
                             law — see simulate.fst_vs_ratio_table)
F4 underpowered_population   >= 1 population's panel can't resolve the
                             observed AF (robustness.detection_floor)
F5 false_exclusion           under the PRIMARY policy and primary tau, >= 1
                             population's AF CI LOWER bound meets the threshold
                             but the policy excludes the variant (supported,
                             not noise-driven)
F6 decision_indeterminate    >= 1 population's AF CI straddles primary tau
F7 material_prevalence_error under the primary policy, any population's
                             actionable-genotype rate is under-predicted even
                             on a CONSERVATIVE comparison (population rate at its
                             CI lower bound vs policy rate at its CI upper bound)
                             by >= material_shortfall_per_10k, or by a ratio
                             >= material_ratio when the shortfall is also above
                             a small absolute floor

Classification (first match wins)
---------------------------------
disparity_supported := F1 or (F2 and F3) or F7
    (F7 qualifies on its own because it already compares CI bounds, i.e. it
    is itself a supported gap vs the calibration population. F5 does NOT
    qualify on its own: a policy AF of 0.0099 against a population AF of
    0.0101 straddles tau without being a meaningful disparity.)
dashboard_impact    := F5 or F7

ACTIONABLE_DISPARITY       disparity_supported and dashboard_impact
DISPARITY_LOW_IMPACT       disparity_supported and not dashboard_impact
INCONCLUSIVE_POWER         not disparity_supported, but F4 or F6
NO_EVIDENCE_OF_DISPARITY   otherwise

"No evidence" is not "evidence of no disparity": it is only meaningful when
F4 and F6 are false, which is why the underpowered case is separated out.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ScorecardConfig:
    min_supported_ratio: float = 2.0
    alpha: float = 0.05
    fst_threshold: float = 0.05
    primary_tau: float = 0.01
    primary_policy_prefix: str = "calibration:"   # policy used for F5 / F7
    material_shortfall_per_10k: float = 10.0
    material_ratio: float = 2.0
    material_absolute_floor_per_10k: float = 1.0  # ignore ratio blow-ups below this

    def to_dict(self) -> Dict:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


FLAG_DESCRIPTIONS = {
    "F1_disparity_supported": "Conservative (Wilson, Bonferroni) lower bound on max/min ratio >= threshold",
    "F2_heterogeneity_significant": "Monte Carlo homogeneity test significant after BH correction",
    "F3_differentiation_moderate": "Mean pairwise Hudson Fst >= threshold",
    "F4_underpowered_population": "At least one population's panel is below its detection floor",
    "F5_false_exclusion": "Primary policy + tau excludes a variant that a population's CI lower bound meets",
    "F6_decision_indeterminate": "At least one population's AF CI straddles tau",
    "F7_material_prevalence_error": "Primary policy materially under-predicts actionable-genotype rate (conservative CI comparison)",
}

CLASSIFICATIONS = (
    "ACTIONABLE_DISPARITY",
    "DISPARITY_LOW_IMPACT",
    "INCONCLUSIVE_POWER",
    "NO_EVIDENCE_OF_DISPARITY",
)


def classify(flags: Dict[str, bool]) -> str:
    disparity = (flags["F1_disparity_supported"]
                 or (flags["F2_heterogeneity_significant"] and flags["F3_differentiation_moderate"])
                 or flags["F7_material_prevalence_error"])
    impact = flags["F5_false_exclusion"] or flags["F7_material_prevalence_error"]
    if disparity and impact:
        return "ACTIONABLE_DISPARITY"
    if disparity:
        return "DISPARITY_LOW_IMPACT"
    if flags["F4_underpowered_population"] or flags["F6_decision_indeterminate"]:
        return "INCONCLUSIVE_POWER"
    return "NO_EVIDENCE_OF_DISPARITY"


def material_prevalence_error(e2_primary, cfg: ScorecardConfig) -> Tuple[bool, str]:
    """F7 from the primary policy's E2 rows (a DataFrame). Returns (flag, worst_pop).

    Uses the conservative *_lb columns so sampling noise cannot trigger it.
    """
    if e2_primary is None or len(e2_primary) == 0:
        return False, ""
    worst_pop, flag, worst = "", False, 0.0
    for _, r in e2_primary.iterrows():
        sf = r["shortfall_lb_per_10k"]
        if sf <= 0:
            continue
        hit = sf >= cfg.material_shortfall_per_10k or (
            sf >= cfg.material_absolute_floor_per_10k and r["ratio_lb"] >= cfg.material_ratio)
        if hit:
            flag = True
            if sf > worst:
                worst, worst_pop = sf, r["population"]
    return flag, worst_pop

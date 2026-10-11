"""
simulate.py — ground-truth validation of the methods themselves.

Real data cannot tell you whether a disparity method is right, because the
true population frequencies are unknown. Simulation can: draw true allele
frequencies from a population-genetic model, sample allele counts at the
real panel sizes of gnomAD and GTEx, then check each method against the
known truth.

Three validation experiments
----------------------------
1. ci_coverage_table       exact (enumerated, not simulated) coverage of four
                           95% interval constructions for rare alleles.
2. disparity_detection     false-alarm rate and sensitivity of each disparity
                           flag, plus inflation of the naive max/min ratio.
3. fst_vs_ratio_table      why Fst and the ratio disagree for rare alleles.

Plus builders for synthetic multi-variant datasets (archetype frequency
patterns with *clearly synthetic* variant IDs) used to demo the full
pipeline and to calibrate cross-dataset (E6) discordance under a known null.

Model: Balding-Nichols. Population k's true AF is
    p_k ~ Beta(p0 (1-F)/F, (1-p0)(1-F)/F),   F = Fst, p0 = ancestral AF.
F = 0 means all populations share p0 exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .ancestry import GTEX_V8_DONORS
from .methods import (
    _chi2_stat_rows,
    homogeneity_mc,
    interval_ratio_batch,
    wilson_bounds,
)


# --------------------------------------------------------------------------
# 1. Exact CI coverage
# --------------------------------------------------------------------------

def _ci_wald(c, n, conf):
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = c / n
    se = np.sqrt(p * (1 - p) / n)
    return np.clip(p - z * se, 0, 1), np.clip(p + z * se, 0, 1)


def _ci_wilson(c, n, conf):
    return wilson_bounds(c, np.full_like(c, n, dtype=float), conf)


def _ci_clopper_pearson(c, n, conf):
    a = 1 - conf
    lo = np.where(c == 0, 0.0, stats.beta.ppf(a / 2, np.maximum(c, 1), n - c + 1))
    hi = np.where(c == n, 1.0, stats.beta.ppf(1 - a / 2, c + 1, np.maximum(n - c, 1)))
    return lo, hi


def _ci_bootstrap_plugin(c, n, conf):
    """Analytic limit of the plug-in parametric bootstrap used in robustness.py.

    Resampling Binomial(n, c/n) many times and taking percentiles converges to
    these binomial quantiles. Undefined (NaN) at c in {0, n}, matching
    robustness.bootstrap_frequency_ci.
    """
    a = 1 - conf
    p = c / n
    lo = stats.binom.ppf(a / 2, n, p) / n
    hi = stats.binom.ppf(1 - a / 2, n, p) / n
    undefined = (c == 0) | (c == n)
    return np.where(undefined, np.nan, lo), np.where(undefined, np.nan, hi)


CI_METHODS = {
    "wald": _ci_wald,
    "wilson": _ci_wilson,
    "clopper_pearson": _ci_clopper_pearson,
    "bootstrap_plugin": _ci_bootstrap_plugin,
}


def ci_coverage_table(ns: List[int], ps: List[float], conf: float = 0.95) -> pd.DataFrame:
    """Exact coverage of each CI construction at each (n, true p).

    Enumerates the binomial support (central mass >= 1 - 1e-9) instead of
    simulating, so there is no Monte Carlo error. 'undefined_mass' is the
    probability the method returns NaN (counted as a miss in 'coverage').
    """
    rows = []
    for n in ns:
        for p in ps:
            mean, sd = n * p, np.sqrt(n * p * (1 - p))
            lo_c = max(0, int(np.floor(mean - 12 * sd - 5)))
            hi_c = min(n, int(np.ceil(mean + 12 * sd + 25)))
            cs = np.arange(lo_c, hi_c + 1)
            pmf = stats.binom.pmf(cs, n, p)
            mass = float(pmf.sum())
            if mass < 1 - 1e-9:
                raise RuntimeError(f"support truncation lost mass at n={n}, p={p}: {mass}")
            for name, fn in CI_METHODS.items():
                lo, hi = fn(cs.astype(float), n, conf)
                undefined = np.isnan(lo) | np.isnan(hi)
                covers = (~undefined) & (lo <= p + 1e-15) & (p <= hi + 1e-15)
                defined_mass = float(pmf[~undefined].sum())
                width = (pmf[~undefined] * (hi - lo)[~undefined]).sum() / defined_mass \
                    if defined_mass > 0 else float("nan")
                rows.append({
                    "method": name, "n_alleles": n, "true_af": p,
                    "expected_count": n * p,
                    "coverage": float((pmf * covers).sum()),
                    "undefined_mass": float(pmf[undefined].sum()),
                    "mean_width": float(width),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Disparity-detection simulation
# --------------------------------------------------------------------------

def _draw_true_afs(p0: float, fst: float, k: int, reps: int, rng) -> np.ndarray:
    if fst <= 0:
        return np.full((reps, k), p0)
    a = p0 * (1 - fst) / fst
    b = (1 - p0) * (1 - fst) / fst
    return rng.beta(a, b, size=(reps, k))


def _naive_ratio(freqs: np.ndarray) -> np.ndarray:
    mx, mn = freqs.max(axis=1), freqs.min(axis=1)
    with np.errstate(all="ignore"):  # denormal minima can overflow; they mean 'huge'
        r = mx / mn
    r = np.where((mn == 0) & (mx > 0), np.inf, r)
    r = np.where((mn == 0) & (mx == 0), np.nan, r)  # monomorphic: undefined
    return r


def _mean_pairwise_fst_batch(ac: np.ndarray, an: np.ndarray) -> np.ndarray:
    k = ac.shape[1]
    p = ac / an[None, :]
    num = np.zeros(ac.shape[0])
    den = np.zeros(ac.shape[0])
    for i in range(k):
        for j in range(i + 1, k):
            if an[i] < 2 or an[j] < 2:
                continue
            num += (p[:, i] - p[:, j]) ** 2 \
                - p[:, i] * (1 - p[:, i]) / (an[i] - 1) - p[:, j] * (1 - p[:, j]) / (an[j] - 1)
            den += p[:, i] * (1 - p[:, j]) + p[:, j] * (1 - p[:, i])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def simulate_disparity_detection(
    panel: Dict[str, int], p0: float, fst: float, reps: int = 600,
    n_mc: int = 199, true_ratio_threshold: float = 2.0, alpha: float = 0.05,
    fst_threshold: float = 0.05, seed: int = 0,
) -> List[Dict]:
    """One scenario: returns one result row per method.

    'Truth' for each replicate is whether the TRUE max/min frequency ratio is
    >= true_ratio_threshold. Each method's flag is scored against it:
      false_alarm_rate  = P(flag | truth is False)    (needs F = 0 or mild)
      sensitivity       = P(flag | truth is True)
    Fst is scored against the same truth for comparability even though it is
    not designed to detect ratios (that mismatch is itself a finding).
    """
    rng = np.random.default_rng(seed)
    pops = list(panel)
    an = np.array([panel[p] for p in pops], dtype=float)
    k = len(pops)

    p_true = _draw_true_afs(p0, fst, k, reps, rng)
    truth = _naive_ratio(p_true)
    truth_flag = np.where(np.isnan(truth), False, truth >= true_ratio_threshold)

    ac = rng.binomial(an.astype(np.int64)[None, :], p_true)
    freqs = ac / an[None, :]
    naive = _naive_ratio(freqs)

    flags: Dict[str, np.ndarray] = {}
    flags["naive_ratio"] = np.where(np.isnan(naive), False, naive >= true_ratio_threshold)
    flags["interval_ratio_lb"] = interval_ratio_batch(ac, an) >= true_ratio_threshold
    chi = _chi2_stat_rows(ac, an)
    flags["chi2_asymptotic"] = stats.chi2.sf(chi, k - 1) < alpha
    p_mc = np.array([homogeneity_mc(ac[r], an, n_mc=n_mc, seed=int(r)) for r in range(reps)])
    flags["chi2_montecarlo"] = p_mc < alpha
    fst_hat = _mean_pairwise_fst_batch(ac, an)
    flags["hudson_fst"] = np.where(np.isnan(fst_hat), False, fst_hat >= fst_threshold)

    n_true, n_false = int(truth_flag.sum()), int((~truth_flag).sum())
    finite = np.isfinite(naive) & np.isfinite(truth) & (truth > 0)
    inflation = (float(np.median(naive[finite] / truth[finite])) if finite.any()
                 else float("nan"))

    rows = []
    for name, f in flags.items():
        rows.append({
            "method": name, "p0": p0, "fst": fst, "reps": reps,
            "n_truth_true": n_true, "n_truth_false": n_false,
            "flag_rate": float(f.mean()),
            "false_alarm_rate": float(f[~truth_flag].mean()) if n_false else float("nan"),
            "sensitivity": float(f[truth_flag].mean()) if n_true else float("nan"),
            "naive_ratio_infinite_rate": float(np.isinf(naive).mean()),
            "naive_ratio_median_inflation": inflation,
            "monomorphic_rate": float(np.isnan(naive).mean()),
        })
    return rows


def run_detection_grid(panels: Dict[str, Dict[str, int]], p0s=(0.0005, 0.005, 0.05, 0.3),
                       fsts=(0.0, 0.01, 0.05), reps: int = 600, seed: int = 0) -> pd.DataFrame:
    rows = []
    for pname, panel in panels.items():
        for p0 in p0s:
            for f in fsts:
                for r in simulate_disparity_detection(panel, p0, f, reps=reps, seed=seed):
                    rows.append({"panel": pname, **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Why Fst and the ratio disagree
# --------------------------------------------------------------------------

def fst_vs_ratio_table(p_low=(1e-4, 1e-3, 1e-2, 0.1, 0.25),
                       ratios=(2.0, 10.0)) -> pd.DataFrame:
    """Population-level (no sampling) two-population Fst for a fixed ratio.

    Fst = (p1-p2)^2 / (p1(1-p2) + p2(1-p1)): for a FIXED frequency ratio it
    shrinks toward 0 as the allele gets rarer, so an Fst cut-off cannot detect
    large ratio-type disparities in rare pharmacogenomic alleles.
    """
    rows = []
    for p1 in p_low:
        for r in ratios:
            p2 = min(p1 * r, 1.0)
            fst = (p1 - p2) ** 2 / (p1 * (1 - p2) + p2 * (1 - p1))
            rows.append({"low_af": p1, "ratio": r, "high_af": p2, "population_fst": fst})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def gnomad_like_panel(table: pd.DataFrame) -> Dict[str, int]:
    """Median allele number per population from a real gnomAD-style table."""
    med = table.groupby("population")["allele_number"].median()
    return {k: int(v) for k, v in med.items()}


def gtex_like_panel() -> Dict[str, int]:
    """GTEx v8 allele numbers (2 x genotyped donors) per self-reported group."""
    return {k: 2 * v for k, v in GTEX_V8_DONORS.items()}


# --------------------------------------------------------------------------
# Synthetic multi-variant datasets (clearly labelled synthetic)
# --------------------------------------------------------------------------

# True allele frequency by gnomAD ancestry code. ILLUSTRATIVE ARCHETYPES ONLY:
# shapes (flat / rare / one-group-high / calibration-group-low), not
# measurements of any real variant. IDs are prefixed SYNTH_ on purpose.
_POPS = ["afr", "amr", "asj", "eas", "fin", "nfe", "sas", "mid", "remaining", "ami"]


def _af(**kw) -> Dict[str, float]:
    base = kw.pop("base")
    out = {p: base for p in _POPS}
    out.update(kw)
    return out


ARCHETYPES: Dict[str, Dict[str, float]] = {
    "SYNTH_COMMON_UNIFORM":      _af(base=0.20),
    "SYNTH_COMMON_AFR_HIGH":     _af(base=0.12, afr=0.70, amr=0.30, sas=0.30, eas=0.25, nfe=0.10, fin=0.08),
    "SYNTH_COMMON_EUR_HIGH":     _af(base=0.20, nfe=0.45, fin=0.45, asj=0.45, afr=0.10, eas=0.05, sas=0.15),
    "SYNTH_COMMON_NFE_LOW":      _af(base=0.15, nfe=0.02, fin=0.02),
    "SYNTH_MODERATE_SAS_HIGH":   _af(base=0.02, sas=0.08, amr=0.03),
    "SYNTH_RARE_EAS_HIGH":       _af(base=0.001, eas=0.02, sas=0.003),
    "SYNTH_RARE_FLAT":           _af(base=0.0005),
    "SYNTH_VERY_RARE_FLAT":      _af(base=0.0001),
    "SYNTH_RARE_NFE_ONLY":       _af(base=0.0, nfe=0.003, fin=0.003),
    "SYNTH_GRADIENT":            _af(base=0.05, nfe=0.05, fin=0.04, amr=0.08, afr=0.12, eas=0.02, sas=0.06),
}


def synthetic_gnomad_like(an_by_pop: Dict[str, int], seed: int = 20260101,
                          archetypes: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """Draw allele counts for each archetype at gnomAD-like panel sizes."""
    rng = np.random.default_rng(seed)
    archetypes = archetypes or ARCHETYPES
    rows = []
    for vid, afs in archetypes.items():
        for pop, n in an_by_pop.items():
            p = afs.get(pop, afs.get("remaining", 0.0))
            c = int(rng.binomial(n, p))
            rows.append({"gene": "SYNTHETIC", "variant_id": vid, "rsid": "NA", "population": pop,
                         "allele_count": c, "allele_number": int(n), "allele_frequency": c / n})
    return pd.DataFrame(rows)


def synthetic_gtex_like(seed: int = 20260102, aa_european_fraction: float = 0.2,
                        archetypes: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """Same TRUE frequencies as the gnomAD-like set, sampled at GTEx's panel size.

    Group frequencies are built from the gnomAD-coded truth:
      EUR_AMERICAN  = nfe
      AFR_AMERICAN  = (1-a)*afr + a*nfe        (a = aa_european_fraction, an ASSUMPTION)
      ASIAN_AMERICAN = eas
      OTHER_OR_UNREPORTED = unweighted mean over all groups (arbitrary placeholder)
    Because the underlying truth is shared, discordance between this set and
    synthetic_gnomad_like is by construction a sampling/harmonisation artefact.
    """
    rng = np.random.default_rng(seed)
    archetypes = archetypes or ARCHETYPES
    an = gtex_like_panel()
    rows = []
    for vid, afs in archetypes.items():
        truth = {
            "EUR_AMERICAN": afs["nfe"],
            "AFR_AMERICAN": (1 - aa_european_fraction) * afs["afr"] + aa_european_fraction * afs["nfe"],
            "ASIAN_AMERICAN": afs["eas"],
            "OTHER_OR_UNREPORTED": float(np.mean(list(afs.values()))),
        }
        for pop, n in an.items():
            c = int(rng.binomial(n, truth[pop]))
            rows.append({"gene": "SYNTHETIC", "variant_id": vid, "rsid": "NA", "population": pop,
                         "allele_count": c, "allele_number": int(n), "allele_frequency": c / n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Validation runner
# --------------------------------------------------------------------------

def run_validation(real_table: Optional[pd.DataFrame] = None, reps: int = 600,
                   seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Run all three validation experiments.

    Panels: gnomAD-like (median AN per group from `real_table` if provided,
    otherwise a built-in v4-scale fallback) and GTEx-like (2 x donors).
    """
    if real_table is not None:
        gn_panel = gnomad_like_panel(real_table)
    else:  # order-of-magnitude fallback; pass real_table to use real panel sizes
        gn_panel = {"nfe": 1_180_000, "sas": 91_000, "afr": 75_000, "fin": 64_000,
                    "remaining": 62_000, "amr": 60_000, "eas": 45_000, "asj": 30_000,
                    "mid": 5_800, "ami": 900}
    panels = {"gnomad_like": gn_panel, "gtex_like": gtex_like_panel()}
    return {
        "ci_coverage": ci_coverage_table(
            ns=[200, 912, 5758, 29564, 1_179_586],
            ps=[1e-4, 5e-4, 2e-3, 1e-2, 5e-2, 0.2]),
        "detection": run_detection_grid(panels, reps=reps, seed=seed),
        "fst_vs_ratio": fst_vs_ratio_table(),
        "panels": pd.DataFrame([{"panel": k, "population": p, "allele_number": n}
                                for k, v in panels.items() for p, n in v.items()]),
    }

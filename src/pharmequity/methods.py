"""
methods.py — a battery of methods for quantifying allele-frequency disparity.

No single statistic is "the" disparity. They answer different questions and
fail in different regimes, which is why they are run together and compared
(see evaluate.method_concordance and the simulation in simulate.py):

  naive max/min ratio      (metrics.disparity_ratio) — what dashboards show;
                           explodes on tiny panels, infinite at zero counts.
  interval ratio           conservative LOWER bound on the true max/min ratio
                           using Wilson intervals; finite, panel-size aware.
  variance / KL            (metrics.py) spread and calibration-reference gap.
  chi-square homogeneity   asymptotic p-value; INVALID when expected counts
                           are small (rare variant x small panel).
  Monte Carlo homogeneity  same statistic, parametric-bootstrap p-value;
                           valid in the rare regime.
  Hudson Fst               allele-frequency *differentiation* (bounded by the
                           allele's frequency, so low for rare alleles even
                           when ratios are huge — a different question).
  log odds ratio vs ref    per-population effect size vs the calibration
                           population, with Haldane-Anscombe correction.

Invariant carried over from metrics.py: NO population is ever dropped.
Populations with allele_number == 0 are treated as maximally uncertain
([0, 1] interval), not removed.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import (
    disparity_ratio,
    kl_divergence_from_reference,
    variance_across_populations,
)


# --------------------------------------------------------------------------
# Confidence intervals (vectorised; metrics.wilson_ci is the scalar original)
# --------------------------------------------------------------------------

def wilson_bounds(ac, an, conf: float = 0.95):
    """Vectorised Wilson score bounds. n == 0 yields the uninformative [0, 1]."""
    ac = np.asarray(ac, dtype=float)
    an = np.asarray(an, dtype=float)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(an > 0, ac / an, 0.0)
        denom = 1 + z**2 / an
        centre = p + z**2 / (2 * an)
        adj = z * np.sqrt((p * (1 - p) + z**2 / (4 * an)) / an)
        lo = (centre - adj) / denom
        hi = (centre + adj) / denom
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    lo = np.where(ac == 0, 0.0, lo)          # exact at the boundary
    hi = np.where(ac == an, 1.0, hi)
    lo = np.where(an > 0, lo, 0.0)
    hi = np.where(an > 0, hi, 1.0)
    return lo, hi


# --------------------------------------------------------------------------
# Interval-based (conservative) disparity ratio
# --------------------------------------------------------------------------

def interval_ratio(ac, an, conf: float = 0.95, simultaneous: bool = True) -> Dict:
    """Conservative lower bound on the true max/min allele-frequency ratio.

    For every ordered pair (i, j), lo_i / hi_j bounds the true ratio p_i/p_j
    from below. The best such pair gives the largest ratio the data can
    *guarantee*. Unlike the naive ratio it is always finite (hi_j > 0) and
    shrinks when a population's panel is small.

    simultaneous=True applies a Bonferroni correction (each interval at
    level 1 - (1-conf)/K) so the bound holds jointly over all K intervals.
    The bound is trivially >= 1 (max/min can never be below 1).
    """
    ac = np.asarray(ac, dtype=float)
    an = np.asarray(an, dtype=float)
    k = len(ac)
    if k < 2:
        raise ValueError("interval_ratio requires at least 2 populations.")
    level = 1 - (1 - conf) / k if simultaneous else conf
    lo, hi = wilson_bounds(ac, an, level)
    mat = lo[:, None] / hi[None, :]
    np.fill_diagonal(mat, -np.inf)
    i, j = np.unravel_index(np.argmax(mat), mat.shape)
    return {
        "lb_ratio": float(max(1.0, mat[i, j])),
        "lb_numerator_index": int(i),
        "lb_denominator_index": int(j),
        "interval_level_used": float(level),
    }


def interval_ratio_batch(ac: np.ndarray, an: np.ndarray, conf: float = 0.95,
                         simultaneous: bool = True) -> np.ndarray:
    """interval_ratio lower bound for many replicates at once.

    ac: (R, K) counts; an: (K,) allele numbers. Returns (R,) lower bounds.
    """
    ac = np.asarray(ac, dtype=float)
    an = np.asarray(an, dtype=float)
    k = ac.shape[1]
    level = 1 - (1 - conf) / k if simultaneous else conf
    lo, hi = wilson_bounds(ac, an[None, :], level)
    mat = lo[:, :, None] / hi[:, None, :]
    idx = np.arange(k)
    mat[:, idx, idx] = -np.inf
    return np.maximum(1.0, mat.reshape(mat.shape[0], -1).max(axis=1))


# --------------------------------------------------------------------------
# Homogeneity tests
# --------------------------------------------------------------------------

def _chi2_stat_rows(ac: np.ndarray, an: np.ndarray) -> np.ndarray:
    """Pearson chi-square for homogeneity of K binomial proportions.

    Closed form: sum_k (ac_k - an_k * p)^2 / (an_k * p * (1-p)), p = pooled AF.
    ac: (R, K); an: (K,). Returns (R,). Degenerate (p in {0,1}) rows -> 0.
    """
    ac = np.asarray(ac, dtype=float)
    an = np.asarray(an, dtype=float)
    total = an.sum()
    p = ac.sum(axis=1) / total
    var = p * (1 - p)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = ((ac - an[None, :] * p[:, None]) ** 2 / (an[None, :] * var[:, None])).sum(axis=1)
    return np.where(var > 0, stat, 0.0)


def homogeneity_chi2(ac, an) -> Dict:
    """Asymptotic chi-square test of equal allele frequency across populations."""
    ac = np.asarray(ac, dtype=float)
    an = np.asarray(an, dtype=float)
    k = len(ac)
    stat = float(_chi2_stat_rows(ac[None, :], an)[0])
    df = k - 1
    p_asym = float(stats.chi2.sf(stat, df))
    pooled = ac.sum() / an.sum()
    exp_alt = an * pooled
    exp_ref = an * (1 - pooled)
    min_expected = float(min(exp_alt.min(), exp_ref.min()))
    cramers_v = math.sqrt(stat / an.sum()) if an.sum() > 0 else float("nan")
    return {
        "chi2": stat, "df": df, "p_asymptotic": p_asym,
        "min_expected_count": min_expected,
        "asymptotics_valid": bool(min_expected >= 5),
        "cramers_v": cramers_v,
    }


def homogeneity_mc(ac, an, n_mc: int = 4999, seed: int = 42) -> float:
    """Monte Carlo p-value for the same chi-square statistic.

    Parametric bootstrap under H0: every population shares the pooled AF.
    Valid when expected counts are small, where the asymptotic p-value is not.
    (Approximate, not an exact conditional test, because the pooled AF is
    estimated; the validation in simulate.py measures its actual size.)
    """
    ac = np.asarray(ac, dtype=float)
    an = np.asarray(an, dtype=float)
    obs = _chi2_stat_rows(ac[None, :], an)[0]
    pooled = ac.sum() / an.sum()
    if pooled in (0.0, 1.0):
        return 1.0
    rng = np.random.default_rng(seed)
    sim = rng.binomial(an.astype(np.int64)[None, :], pooled, size=(n_mc, len(an)))
    sim_stat = _chi2_stat_rows(sim, an)
    return float((1 + np.sum(sim_stat >= obs - 1e-12)) / (n_mc + 1))


# --------------------------------------------------------------------------
# Hudson Fst (Bhatia et al. 2013, genome research 23:1514)
# --------------------------------------------------------------------------

def hudson_terms(c1: float, n1: float, c2: float, n2: float):
    """Numerator/denominator of Hudson's Fst estimator for one pair, one site.

    n1, n2 are numbers of sampled alleles. The p(1-p)/(n-1) terms remove the
    sampling-variance bias of (p1 - p2)^2.
    """
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    p1, p2 = c1 / n1, c2 / n2
    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    return num, den


def mean_pairwise_fst(ac, an) -> Dict:
    """Ratio-of-sums Hudson Fst averaged over all population pairs.

    Every population counts equally (a deliberate equity-oriented choice:
    a 912-allele group is as much 'a population' as a 1.2M-allele one;
    sampling noise in small groups is handled by the Hudson correction).
    Single-site estimates can be negative (estimator is unbiased, not
    bounded below by 0). Returns NaN if the site is monomorphic everywhere.
    """
    ac = np.asarray(ac, dtype=float)
    an = np.asarray(an, dtype=float)
    nums, dens = [], []
    for i, j in combinations(range(len(ac)), 2):
        n_, d_ = hudson_terms(ac[i], an[i], ac[j], an[j])
        if not (math.isnan(n_) or math.isnan(d_)):
            nums.append(n_)
            dens.append(d_)
    if not dens or sum(dens) <= 0:
        return {"fst": float("nan"), "n_pairs": len(dens)}
    return {"fst": float(sum(nums) / sum(dens)), "n_pairs": len(dens)}


def fst_vs_reference(ac, an, ref_index: int) -> np.ndarray:
    """Hudson Fst of every population against one reference population."""
    ac = np.asarray(ac, dtype=float)
    an = np.asarray(an, dtype=float)
    out = np.full(len(ac), float("nan"))
    for i in range(len(ac)):
        if i == ref_index:
            out[i] = 0.0
            continue
        n_, d_ = hudson_terms(ac[i], an[i], ac[ref_index], an[ref_index])
        out[i] = n_ / d_ if d_ and d_ > 0 and not math.isnan(n_) else float("nan")
    return out


# --------------------------------------------------------------------------
# Per-population log odds ratio vs. a reference
# --------------------------------------------------------------------------

def log_odds_vs_reference(rows: pd.DataFrame, reference_population: str,
                          conf: float = 0.95) -> pd.DataFrame:
    """Log odds ratio of carrying the allele, each population vs the reference.

    A 0.5 Haldane-Anscombe correction is applied to every cell uniformly (not
    only to zero cells) so that zero counts give finite, comparable answers
    without selectively altering some populations.
    """
    if reference_population not in set(rows["population"]):
        raise ValueError(f"reference population '{reference_population}' not in rows.")
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    ref = rows[rows["population"] == reference_population].iloc[0]
    c, d = ref["allele_count"] + 0.5, ref["allele_number"] - ref["allele_count"] + 0.5
    out = []
    for _, r in rows.iterrows():
        a = r["allele_count"] + 0.5
        b = r["allele_number"] - r["allele_count"] + 0.5
        lor = math.log((a * d) / (b * c))
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        zval = lor / se
        out.append({
            "population": r["population"],
            "log_or": lor, "or": math.exp(lor),
            "or_ci_lo": math.exp(lor - z * se), "or_ci_hi": math.exp(lor + z * se),
            "p_value": 1.0 if r["population"] == reference_population
                       else float(2 * stats.norm.sf(abs(zval))),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------

def run_battery(rows: pd.DataFrame, calibration_population: Optional[str],
                conf: float = 0.95, n_mc: int = 4999, seed: int = 42) -> Dict:
    """Run every method on one variant's per-population rows.

    Returns a flat dict of scalar results plus a 'per_population' DataFrame.
    """
    rows = rows.sort_values("population").reset_index(drop=True)
    ac = rows["allele_count"].to_numpy(dtype=float)
    an = rows["allele_number"].to_numpy(dtype=float)
    pops = rows["population"].tolist()
    freqs = dict(zip(pops, rows["allele_frequency"]))

    dr = disparity_ratio(freqs)
    ir = interval_ratio(ac, an, conf=conf)
    chi = homogeneity_chi2(ac, an)
    p_mc = homogeneity_mc(ac, an, n_mc=n_mc, seed=seed)
    fst = mean_pairwise_fst(ac, an)

    out: Dict = {
        "n_populations": len(pops),
        "naive_ratio": dr.ratio,
        "naive_ratio_is_infinite": math.isinf(dr.ratio),
        "max_population": dr.max_population, "max_frequency": dr.max_frequency,
        "min_population": dr.min_population, "min_frequency": dr.min_frequency,
        "absolute_diff": dr.absolute_diff,
        "variance": variance_across_populations(freqs),
        "interval_ratio_lb": ir["lb_ratio"],
        "interval_ratio_num_pop": pops[ir["lb_numerator_index"]],
        "interval_ratio_den_pop": pops[ir["lb_denominator_index"]],
        "chi2": chi["chi2"], "chi2_df": chi["df"],
        "chi2_p_asymptotic": chi["p_asymptotic"],
        "chi2_min_expected": chi["min_expected_count"],
        "chi2_asymptotics_valid": chi["asymptotics_valid"],
        "cramers_v": chi["cramers_v"],
        "chi2_p_montecarlo": p_mc,
        "mean_pairwise_fst": fst["fst"],
    }

    per_pop = rows[["population", "allele_count", "allele_number", "allele_frequency"]].copy()
    lo, hi = wilson_bounds(ac, an, conf)
    per_pop["wilson_lo"], per_pop["wilson_hi"] = lo, hi

    if calibration_population and calibration_population in pops:
        ref_i = pops.index(calibration_population)
        kl = kl_divergence_from_reference(freqs, calibration_population)
        lor = log_odds_vs_reference(rows, calibration_population, conf)
        per_pop["kl_vs_calibration"] = per_pop["population"].map(kl)
        per_pop["fst_vs_calibration"] = fst_vs_reference(ac, an, ref_i)
        per_pop = per_pop.merge(lor, on="population")
        out["max_kl_vs_calibration"] = max(kl.values())
    else:
        out["max_kl_vs_calibration"] = float("nan")

    out["per_population"] = per_pop
    return out


def bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (monotone)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out

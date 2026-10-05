"""
metrics.py — statistics for comparing an allele frequency across
reference populations.

Every ratio-based metric here always uses the FULL set of populations
passed in, including any at 0%. A ratio is only ever returned as a
finite number when every population has a nonzero rate; otherwise it
is explicitly reported as inf, never silently computed on a filtered
subset. (This is a hard invariant, checked by tests — see
tests/test_metrics.py::test_no_silent_group_exclusion.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import stats


@dataclass
class DisparityResult:
    max_population: str
    max_frequency: float
    min_population: str
    min_frequency: float
    ratio: float          # max/min; inf if min == 0
    absolute_diff: float  # max - min, in frequency units (0-1)
    n_populations: int


def disparity_ratio(freqs: Dict[str, float]) -> DisparityResult:
    """max/min allele frequency across all supplied populations.

    freqs: {population_code: allele_frequency}. Every key passed in is
    used — there is no filtering step, by design.
    """
    if len(freqs) < 2:
        raise ValueError("disparity_ratio requires at least 2 populations.")
    max_pop = max(freqs, key=freqs.get)
    min_pop = min(freqs, key=freqs.get)
    max_f, min_f = freqs[max_pop], freqs[min_pop]
    ratio = (max_f / min_f) if min_f > 0 else float("inf")
    return DisparityResult(
        max_population=max_pop, max_frequency=max_f,
        min_population=min_pop, min_frequency=min_f,
        ratio=ratio, absolute_diff=max_f - min_f,
        n_populations=len(freqs),
    )


def wilson_ci(count: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal-approximation ("Wald") interval used
    naively for allele frequencies, because Wald intervals become
    invalid (can go negative, or ignore the boundary) for rare alleles
    and small reference panels — exactly the regime pharmacogenomic
    variants in underrepresented populations fall into.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = count / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, float(lo)), min(1.0, float(hi)))


def variance_across_populations(freqs: Dict[str, float]) -> float:
    """Population variance of allele frequency across groups.

    A complementary metric to disparity_ratio: the ratio is dominated
    by whichever single population is most extreme, while variance
    reflects overall spread and is less sensitive to one outlier group.
    """
    vals = np.array(list(freqs.values()), dtype=float)
    return float(np.var(vals, ddof=0))


def kl_divergence_from_reference(freqs: Dict[str, float], reference_pop: str) -> Dict[str, float]:
    """KL divergence of each population's Bernoulli(allele) distribution
    from the reference population's distribution.

    Motivation: many dosing algorithms and clinical decision tools are
    calibrated on a reference cohort (historically European-ancestry-
    majority panels). This quantifies, per population, how much that
    population's carrier-frequency distribution diverges from the
    population the tool was implicitly calibrated on — a proxy for how
    miscalibrated a reference-derived dosing rule may be when applied
    to that population.
    """
    if reference_pop not in freqs:
        raise ValueError(f"reference_pop '{reference_pop}' not in freqs.")
    p_ref = _clip(freqs[reference_pop])
    out = {}
    for pop, f in freqs.items():
        p = _clip(f)
        kl = p * np.log(p / p_ref) + (1 - p) * np.log((1 - p) / (1 - p_ref))
        out[pop] = float(kl)
    return out


def _clip(p: float, eps: float = 1e-6) -> float:
    """Clip a probability away from exactly 0/1 for log-based metrics.

    This clipping is ONLY used inside KL divergence (a smoothing
    requirement of the metric itself, applied identically to every
    population) — it must never be used to avoid reporting an infinite
    disparity_ratio elsewhere.
    """
    return min(max(p, eps), 1 - eps)

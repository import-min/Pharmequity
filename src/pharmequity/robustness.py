"""
robustness.py — systematic error-mode analysis.

A single point estimate of a disparity ratio doesn't tell you whether
that disparity is a stable population-genetic fact or an artifact of
small reference-panel size. This module runs each variant/population
through a fixed battery of error modes so results are comparable
across genes and datasets rather than eyeballed one plot at a time.

Error modes covered:
  1. sampling_uncertainty   — bootstrap + Wilson CI on the observed AF
  2. detection_floor        — is the population's panel size even large
                              enough to reliably detect this AF if it's real?
  3. reference_mismatch     — if a dosing rule's threshold was calibrated
                              on one population's AF, how wrong is the
                              predicted carrier rate in another population?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .metrics import wilson_ci


@dataclass
class DetectionFloorResult:
    population: str
    n_alleles: int
    observed_frequency: float
    min_detectable_frequency_95: float
    below_detection_floor: bool


def detection_floor(n_alleles: int, observed_frequency: float, min_expected_count: int = 5) -> DetectionFloorResult:
    """Flag populations whose panel is too small to reliably characterize
    a variant at the observed (or a plausibly true) frequency.

    Rule of thumb used here: a panel needs at least `min_expected_count`
    expected allele copies to be considered adequately powered for a
    given frequency (standard rare-variant-detection heuristic). We
    invert that to report the minimum frequency this panel could
    confidently characterize, and flag when the observed frequency is
    at or below that floor — i.e. "this population's rate might be an
    artifact of a small reference panel, not a true absence."
    """
    min_detectable = min_expected_count / n_alleles if n_alleles > 0 else float("inf")
    return DetectionFloorResult(
        population="",
        n_alleles=n_alleles,
        observed_frequency=observed_frequency,
        min_detectable_frequency_95=min_detectable,
        below_detection_floor=observed_frequency <= min_detectable,
    )


def bootstrap_frequency_ci(count: int, n: int, n_boot: int = 5000, seed: int = 42) -> Dict[str, float]:
    """Bootstrap CI on an allele frequency by resampling the underlying
    Bernoulli draws implied by (count, n). Complements the closed-form
    Wilson interval as a cross-check.

    At the boundary (count == 0 or count == n), a plug-in parametric
    bootstrap is mathematically degenerate: resampling from
    Binomial(n, 0) can only ever produce 0, so it reports a false
    zero-width interval regardless of n. In that regime the bootstrap
    is not a valid uncertainty estimate and is intentionally returned
    as NaN here rather than as a misleadingly tight [0, 0]; rely on the
    Wilson interval instead for boundary counts.
    """
    if n == 0:
        return {"boot_lo": float("nan"), "boot_hi": float("nan"), "wilson_lo": float("nan"), "wilson_hi": float("nan")}

    wlo, whi = wilson_ci(count, n)

    if count == 0 or count == n:
        return {"boot_lo": float("nan"), "boot_hi": float("nan"), "wilson_lo": wlo, "wilson_hi": whi}

    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, count / n, size=n_boot) / n
    boot_lo, boot_hi = np.percentile(draws, [2.5, 97.5])
    return {
        "boot_lo": float(boot_lo), "boot_hi": float(boot_hi),
        "wilson_lo": wlo, "wilson_hi": whi,
    }

def reference_mismatch_impact(
    calibration_frequency: float,
    target_frequency: float,
    zygosity: str = "recessive",
) -> Dict[str, float]:
    """Estimate the error in predicted carrier/at-risk rate when a
    dosing rule's threshold logic is calibrated using one population's
    allele frequency but applied to a population with a different true
    frequency, under Hardy-Weinberg equilibrium.

    zygosity: 'recessive' compares homozygous-variant rate (p^2);
              'dominant_or_het' compares carrier rate (1 - (1-p)^2).

    Returns the predicted rate under each frequency and the ratio and
    absolute-percentage-point error between them — this is the
    quantity that actually matters for a dosing dashboard: not the
    allele-frequency ratio itself, but how wrong the resulting
    genotype-frequency-based risk estimate becomes.
    """
    def rate(p: float) -> float:
        if zygosity == "recessive":
            return p ** 2
        if zygosity == "dominant_or_het":
            return 1 - (1 - p) ** 2
        raise ValueError("zygosity must be 'recessive' or 'dominant_or_het'")

    r_cal = rate(calibration_frequency)
    r_target = rate(target_frequency)
    ratio = (r_target / r_cal) if r_cal > 0 else float("inf")
    return {
        "calibration_rate": r_cal,
        "target_rate": r_target,
        "rate_ratio": ratio,
        "absolute_pp_error": (r_target - r_cal) * 100,
    }

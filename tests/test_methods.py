import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pharmequity.metrics import wilson_ci
from pharmequity.methods import (
    wilson_bounds, interval_ratio, interval_ratio_batch, homogeneity_chi2,
    homogeneity_mc, mean_pairwise_fst, hudson_terms, log_odds_vs_reference,
    run_battery, bh_adjust,
)

REAL = dict(ac=[483, 227, 180, 2886, 6, 37, 30, 13, 4, 0],
            an=[29564, 60014, 62454, 1179586, 5758, 75038, 91026, 44858, 63920, 912])


def test_vectorised_wilson_matches_scalar():
    for c, n in [(0, 912), (4, 63920), (2886, 1179586), (100, 100), (1, 2)]:
        lo, hi = wilson_bounds([c], [n])
        assert (lo[0], hi[0]) == pytest.approx(wilson_ci(c, n), abs=1e-12)


def test_wilson_zero_n_is_uninformative_not_dropped():
    lo, hi = wilson_bounds([0], [0])
    assert (lo[0], hi[0]) == (0.0, 1.0)


def test_interval_ratio_is_finite_when_naive_ratio_is_infinite():
    r = interval_ratio(REAL["ac"], REAL["an"])
    assert math.isfinite(r["lb_ratio"]) and r["lb_ratio"] > 2


def test_interval_ratio_never_below_one_and_uses_all_populations():
    r = interval_ratio([10, 10, 10], [1000, 1000, 1000])
    assert r["lb_ratio"] == 1.0
    # zero-allele-number population stays in (as [0,1]) and cannot create a disparity
    r2 = interval_ratio([10, 10, 0], [1000, 1000, 0])
    assert r2["lb_ratio"] == 1.0


def test_interval_ratio_batch_matches_single():
    rng = np.random.default_rng(1)
    an = np.array([500, 2000, 100000])
    ac = rng.binomial(an[None, :], [[0.01, 0.02, 0.05]], size=(20, 3))
    batch = interval_ratio_batch(ac, an)
    single = [interval_ratio(row, an)["lb_ratio"] for row in ac]
    assert batch == pytest.approx(single)


def test_chi2_matches_scipy_contingency():
    ac, an = np.array(REAL["ac"]), np.array(REAL["an"])
    ref, p, _, _ = stats.chi2_contingency(np.array([ac, an - ac]), correction=False)
    out = homogeneity_chi2(ac, an)
    assert out["chi2"] == pytest.approx(ref)
    assert out["p_asymptotic"] == pytest.approx(p, abs=1e-12)


def test_chi2_flags_invalid_asymptotics_for_rare_variant():
    assert homogeneity_chi2(REAL["ac"], REAL["an"])["asymptotics_valid"] is False


def test_mc_pvalue_deterministic_and_degenerate():
    a = homogeneity_mc(REAL["ac"], REAL["an"], n_mc=499, seed=3)
    assert a == homogeneity_mc(REAL["ac"], REAL["an"], n_mc=499, seed=3)
    assert homogeneity_mc([0, 0], [100, 100]) == 1.0


def test_hudson_matches_closed_form_for_large_n():
    out = mean_pairwise_fst([2e5, 4e5], [1e6, 1e6])
    assert out["fst"] == pytest.approx(0.04 / (0.2 * 0.6 + 0.4 * 0.8), rel=1e-4)


def test_hudson_unbiased_under_null():
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(4000):
        ac = rng.binomial([200, 300], 0.1)
        vals.append(hudson_terms(ac[0], 200, ac[1], 300))
    num = np.nanmean([v[0] for v in vals]); den = np.nanmean([v[1] for v in vals])
    assert abs(num / den) < 0.01


def test_log_or_finite_with_zero_counts_and_ref_is_one():
    rows = pd.DataFrame({"population": ["a", "b", "c"], "allele_count": [0, 5, 10],
                         "allele_number": [100, 100, 100], "allele_frequency": [0, .05, .1]})
    t = log_odds_vs_reference(rows, "c")
    assert np.isfinite(t["log_or"]).all()
    assert t.loc[t.population == "c", "or"].iloc[0] == pytest.approx(1.0)


def test_bh_adjust_monotone_and_bounded():
    adj = bh_adjust([0.01, 0.04, 0.03, 0.005])
    assert adj.max() <= 1 and (adj >= np.array([0.01, 0.04, 0.03, 0.005])).all()
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_battery_keeps_all_populations():
    df = pd.DataFrame({"population": list("abcdefghij"), "allele_count": REAL["ac"],
                       "allele_number": REAL["an"]})
    df["allele_frequency"] = df.allele_count / df.allele_number
    r = run_battery(df, "d", n_mc=199)
    assert r["n_populations"] == 10 and len(r["per_population"]) == 10
    assert r["naive_ratio_is_infinite"] and math.isfinite(r["interval_ratio_lb"])

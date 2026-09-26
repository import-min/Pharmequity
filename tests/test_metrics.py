import math
import pytest

from pharmequity.metrics import (
    disparity_ratio, wilson_ci, variance_across_populations,
    kl_divergence_from_reference,
)


def test_disparity_ratio_basic():
    r = disparity_ratio({"afr": 0.10, "nfe": 0.20, "eas": 0.05})
    assert r.max_population == "nfe"
    assert r.min_population == "eas"
    assert r.ratio == pytest.approx(4.0)


def test_no_silent_group_exclusion():
    """A single zero-frequency population must make the ratio infinite,
    not be dropped from consideration. This directly encodes the bug
    class found in an earlier project (FairPRS-Clin): a metric that
    quietly filters out the group with the extreme value.
    """
    freqs = {"afr": 0.0, "amr": 0.03, "eas": 0.0, "nfe": 0.49, "sas": 0.02}
    r = disparity_ratio(freqs)
    assert math.isinf(r.ratio)
    assert r.min_frequency == 0.0
    # every population must have been considered, not filtered
    assert r.n_populations == 5


def test_disparity_ratio_requires_two_populations():
    with pytest.raises(ValueError):
        disparity_ratio({"afr": 0.1})


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(5, 100)
    assert 0.0 <= lo < 5 / 100 < hi <= 1.0


def test_wilson_ci_zero_n():
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_variance_across_populations_zero_when_equal():
    assert variance_across_populations({"a": 0.1, "b": 0.1, "c": 0.1}) == pytest.approx(0.0)


def test_kl_divergence_reference_is_zero_to_itself():
    freqs = {"nfe": 0.2, "afr": 0.4}
    kl = kl_divergence_from_reference(freqs, reference_pop="nfe")
    assert kl["nfe"] == pytest.approx(0.0, abs=1e-6)
    assert kl["afr"] > 0

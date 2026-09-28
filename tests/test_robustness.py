import math
import pytest

from pharmequity.robustness import (
    detection_floor, bootstrap_frequency_ci, reference_mismatch_impact,
)


def test_detection_floor_flags_small_panel():
    # 0 observed copies in a panel of 100 alleles: floor = 5/100 = 0.05,
    # so an observed frequency of 0.0 is at/below the floor -> flagged.
    r = detection_floor(n_alleles=100, observed_frequency=0.0)
    assert r.below_detection_floor is True


def test_detection_floor_not_flagged_for_large_panel_and_high_freq():
    r = detection_floor(n_alleles=20000, observed_frequency=0.3)
    assert r.below_detection_floor is False


def test_bootstrap_frequency_ci_matches_wilson_roughly():
    out = bootstrap_frequency_ci(count=50, n=1000)
    assert out["boot_lo"] < 50 / 1000 < out["boot_hi"]
    assert out["wilson_lo"] < 50 / 1000 < out["wilson_hi"]


def test_bootstrap_frequency_ci_zero_n():
    out = bootstrap_frequency_ci(count=0, n=0)
    assert math.isnan(out["boot_lo"])


def test_reference_mismatch_recessive():
    out = reference_mismatch_impact(calibration_frequency=0.1, target_frequency=0.4, zygosity="recessive")
    assert out["calibration_rate"] == pytest.approx(0.01)
    assert out["target_rate"] == pytest.approx(0.16)
    assert out["rate_ratio"] == pytest.approx(16.0)


def test_reference_mismatch_infinite_when_calibration_zero():
    out = reference_mismatch_impact(calibration_frequency=0.0, target_frequency=0.1)
    assert math.isinf(out["rate_ratio"])


def test_reference_mismatch_invalid_zygosity():
    with pytest.raises(ValueError):
        reference_mismatch_impact(0.1, 0.2, zygosity="nonsense")

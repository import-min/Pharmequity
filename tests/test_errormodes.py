import pandas as pd
import pytest

from pharmequity.ancestry import gtex_weights_over_gnomad, GTEX_V8_DONORS, GTEX_V8_N_GENOTYPED_DONORS
from pharmequity.errormodes import (
    build_policies, e1_inclusion_threshold, e2_prevalence_error, e3_pooling_masking,
    e4_crosswalk_sensitivity, e5_decision_stability, e6_dataset_discordance,
)
from pharmequity.ancestry import GNOMAD_TO_HARMONIZED, GTEX_TO_HARMONIZED


def _rows(d):
    r = pd.DataFrame([{"population": p, "allele_count": c, "allele_number": n} for p, (c, n) in d.items()])
    r["allele_frequency"] = r.allele_count / r.allele_number
    return r


# nfe rare (0.2%), afr common (10%) with large panels
ROWS = _rows({"nfe": (2000, 1_000_000), "afr": (10000, 100_000), "eas": (500, 50_000)})


def test_gtex_donor_counts_sum_to_total():
    assert sum(GTEX_V8_DONORS.values()) == GTEX_V8_N_GENOTYPED_DONORS


def test_gtex_weights_sum_to_one_and_disclose_exclusion():
    w, excl = gtex_weights_over_gnomad()
    assert sum(w.values()) == pytest.approx(1.0)
    assert excl == pytest.approx(8 / 838)
    w2, _ = gtex_weights_over_gnomad("split", 0.2)
    assert sum(w2.values()) == pytest.approx(1.0) and "sas" in w2


def test_policies_skip_with_warning_not_silently():
    pol, warn = build_policies(ROWS, "missing_pop")
    assert any("calibration" in w for w in warn)
    assert {p.name for p in pol} >= {"pooled_all"}


def test_e1_supported_false_exclusion_for_afr():
    pol, _ = build_policies(ROWS, "nfe")
    e1 = e1_inclusion_threshold(ROWS, pol, tau=0.01)
    cal = e1[(e1.policy == "calibration:nfe") & (e1.population == "afr")].iloc[0]
    assert cal.false_exclusion and cal.false_exclusion_supported
    assert cal.carriers_missed_per_10k == pytest.approx((1 - 0.9 ** 2) * 1e4)


def test_e1_noise_is_not_a_supported_exclusion():
    # tiny panel: observed 1/20 = 5% >= tau but CI lower bound is far below tau
    rows = _rows({"nfe": (10, 1_000_000), "small": (1, 20)})
    pol, _ = build_policies(rows, "nfe")
    e1 = e1_inclusion_threshold(rows, pol, tau=0.01)
    s = e1[(e1.policy == "calibration:nfe") & (e1.population == "small")].iloc[0]
    assert s.false_exclusion and not s.false_exclusion_supported


def test_e2_under_prediction_direction_and_conservative_columns():
    pol, _ = build_policies(ROWS, "nfe")
    e2 = e2_prevalence_error(ROWS, pol, "recessive")
    afr = e2[(e2.policy == "calibration:nfe") & (e2.population == "afr")].iloc[0]
    assert afr.direction == "under-predicts" and afr.shortfall_lb_per_10k > 0
    assert afr.shortfall_lb_per_10k <= afr.shortfall_per_10k + 1e-9
    nfe = e2[(e2.policy == "calibration:nfe") & (e2.population == "nfe")].iloc[0]
    assert nfe.shortfall_lb_per_10k <= 0   # calibration group vs itself can't be "material"


def test_e2_noise_not_material_for_tiny_panel():
    rows = _rows({"nfe": (200_000, 1_000_000), "small": (5, 20)})   # 25% vs 20% on 20 alleles
    pol, _ = build_policies(rows, "nfe")
    e2 = e2_prevalence_error(rows, pol, "recessive")
    s = e2[(e2.policy == "calibration:nfe") & (e2.population == "small")].iloc[0]
    assert s.shortfall_per_10k > 0 and s.shortfall_lb_per_10k < 0


def test_e3_dominant_population():
    out = e3_pooling_masking(ROWS)
    assert out["dominant_population"] == "nfe" and out["dominant_share"] > 0.8


def test_e4_needs_gnomad_codes_and_varies_with_assumptions():
    rows = _rows({"nfe": (1000, 100_000), "afr": (5000, 100_000), "eas": (100, 100_000), "sas": (3000, 100_000)})
    e4 = e4_crosswalk_sensitivity(rows)
    assert len(e4) == 9 and e4.gtex_weighted_af.nunique() > 1
    assert len(e4_crosswalk_sensitivity(ROWS)) == 3   # no sas -> only asian_to=eas rows computable


def test_e5_statuses():
    pol, _ = build_policies(ROWS, "nfe")
    e5 = e5_decision_stability(ROWS, pol, tau=0.01)
    st = dict(zip(e5.subject, e5.status))
    assert st["afr"] == "include" and st["nfe"] == "exclude"


def test_e6_detects_real_difference_not_noise():
    a = pd.DataFrame({"variant_id": "v", "population": ["nfe", "afr"], "allele_count": [100, 3000],
                      "allele_number": [1000, 10000]})
    b = pd.DataFrame({"variant_id": "v", "population": ["EUR_AMERICAN", "AFR_AMERICAN"],
                      "allele_count": [11, 50], "allele_number": [100, 100]})
    res = e6_dataset_discordance(a, GNOMAD_TO_HARMONIZED, b, GTEX_TO_HARMONIZED)
    d = dict(zip(res.group, res.discordant))
    assert d["AFR"] and not d["EUR"]

import json
from pathlib import Path

import pandas as pd
import pytest

from pharmequity.datasets import Dataset, load_dataset
from pharmequity.evaluate import EvalConfig, run_evaluation
from pharmequity.reporting import write_evaluation
from pharmequity.scorecard import ScorecardConfig, classify
from pharmequity.simulate import (
    ci_coverage_table, fst_vs_ratio_table, gnomad_like_panel, run_validation,
    simulate_disparity_detection, synthetic_gnomad_like, synthetic_gtex_like, gtex_like_panel,
)

REAL = Path(__file__).resolve().parents[1] / "data" / "real_gnomad_frequencies.csv"


def _flags(**kw):
    base = {k: False for k in ["F1_disparity_supported", "F2_heterogeneity_significant",
            "F3_differentiation_moderate", "F4_underpowered_population", "F5_false_exclusion",
            "F6_decision_indeterminate", "F7_material_prevalence_error"]}
    base.update(kw)
    return base


def test_classification_rules():
    assert classify(_flags(F1_disparity_supported=True, F5_false_exclusion=True)) == "ACTIONABLE_DISPARITY"
    assert classify(_flags(F1_disparity_supported=True)) == "DISPARITY_LOW_IMPACT"
    assert classify(_flags(F7_material_prevalence_error=True)) == "ACTIONABLE_DISPARITY"
    assert classify(_flags(F5_false_exclusion=True)) == "NO_EVIDENCE_OF_DISPARITY"  # F5 alone isn't disparity
    assert classify(_flags(F4_underpowered_population=True)) == "INCONCLUSIVE_POWER"
    assert classify(_flags()) == "NO_EVIDENCE_OF_DISPARITY"


def test_config_fingerprint_changes_with_thresholds():
    assert ScorecardConfig().fingerprint() != ScorecardConfig(primary_tau=0.001).fingerprint()


@pytest.fixture(scope="module")
def result():
    real = load_dataset(REAL, name="gnomAD", release="test")
    panel = gnomad_like_panel(real.table)
    sg = Dataset("sg", "x", "synthetic", synthetic_gnomad_like(panel))
    sx = Dataset("sx", "x", "synthetic", synthetic_gtex_like(), calibration_population="EUR_AMERICAN")
    return run_evaluation([real, sg, sx], EvalConfig(n_mc=499))


def test_real_variant_known_behaviour(result):
    v = result.variant_summary.query("dataset == 'gnomAD'").iloc[0]
    assert v.naive_ratio_is_infinite and v.interval_ratio_lb > 10
    assert not v.chi2_asymptotics_valid
    assert v.F5_false_exclusion and v.false_exclusion_populations == "asj"
    assert set(v.populations_below_detection_floor.split(";")) == {"ami", "fin"}


def test_null_archetypes_not_flagged_on_noise(result):
    s = result.variant_summary.set_index(["dataset", "variant_id"])
    assert s.loc[("sg", "SYNTH_COMMON_UNIFORM"), "classification"] == "NO_EVIDENCE_OF_DISPARITY"
    # flat variants must never be scored as supported disparities despite infinite naive ratios
    for vid in ("SYNTH_RARE_FLAT", "SYNTH_VERY_RARE_FLAT"):
        assert not s.loc[("sg", vid), "F1_disparity_supported"]
        assert not s.loc[("sx", vid), "F7_material_prevalence_error"]
    assert not s.loc[("sx", "SYNTH_COMMON_UNIFORM"), "F7_material_prevalence_error"]


def test_provenance_carried_and_warned(result):
    assert set(result.variant_summary.provenance) == {"real", "synthetic"}
    assert any("NOT empirical" in w for w in result.warnings)


def test_every_variant_gets_every_flag_and_all_populations(result):
    assert result.variant_summary["classification"].notna().all()
    n_pop = result.per_population.groupby(["dataset", "variant_id"]).size()
    assert (n_pop.loc["gnomAD"] == 10).all() and (n_pop.loc["sx"] == 4).all()


def test_write_outputs(result, tmp_path):
    paths = write_evaluation(result, tmp_path, validation=None, make_figures=False)
    assert (tmp_path / "REPORT.md").exists()
    blob = json.loads((tmp_path / "evaluation.json").read_text())
    assert blob["schema_version"] == "1.0" and len(blob["variant_summary"]) == 21
    text = (tmp_path / "REPORT.md").read_text()
    for h in ["## 0.", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."]:
        assert h in text


def test_ci_coverage_exact_properties():
    cc = ci_coverage_table([912], [0.0005, 0.2])
    g = lambda m, p: cc[(cc.method == m) & (cc.true_af == p)].iloc[0]
    assert g("wald", 0.0005).coverage < 0.5          # collapses for rare alleles
    assert g("wilson", 0.0005).coverage > 0.9
    assert g("clopper_pearson", 0.0005).coverage >= 0.95
    assert g("bootstrap_plugin", 0.0005).undefined_mass > 0.5
    assert g("wilson", 0.2).coverage == pytest.approx(0.95, abs=0.01)


def test_detection_sim_null_behaviour():
    rows = simulate_disparity_detection(gtex_like_panel(), 0.005, 0.0, reps=300, n_mc=99, seed=1)
    fa = {r["method"]: r["flag_rate"] for r in rows}
    assert fa["naive_ratio"] > 0.8 and fa["interval_ratio_lb"] < 0.05


def test_fst_bounded_by_frequency():
    t = fst_vs_ratio_table()
    rare = t[(t.low_af == 0.001) & (t.ratio == 10)].population_fst.iloc[0]
    assert rare < 0.01


def test_validation_runner_shapes():
    v = run_validation(None, reps=40, seed=0)
    assert set(v) >= {"ci_coverage", "detection", "fst_vs_ratio", "panels"}

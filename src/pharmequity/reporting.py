"""
reporting.py — the standardized evaluation report card.

One fixed section order and one fixed JSON schema, regardless of which
datasets were evaluated. Outputs (under --out):

    REPORT.md            human-readable report card
    evaluation.json      machine-readable (schema_version below)
    tables/*.csv         every result table, tidy / long format
    figures/*.png        (if matplotlib is installed)

REPORT.md sections (always in this order):
    0. Run provenance & reproducibility
    1. Scorecard
    2. Dataset representation
    3. Methods & concordance
    4. Dashboard error modes (E1-E6)
    5. Per-variant detail
    6. Method validation (simulation, ground truth known)
    7. Limitations

Mechanically derived sentences only: this module states numbers that the
tables contain; it does not editorialise beyond the fixed limitations text.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .evaluate import EvaluationResult
from .scorecard import CLASSIFICATIONS, FLAG_DESCRIPTIONS, material_prevalence_error

SCHEMA_VERSION = "1.0"

FLAG_ORDER = list(FLAG_DESCRIPTIONS)
FLAG_SHORT = {k: k.split("_")[0] for k in FLAG_ORDER}  # F1..F7

METHOD_TABLE = [
    ("Naive max/min ratio", "How many times higher is the highest-AF group than the lowest?",
     "Infinite when any group has 0 observed copies; inflated by small panels. Needs a CI."),
    ("Interval ratio (lower bound)", "What ratio can the data *guarantee*? (Wilson, Bonferroni)",
     "Conservative; low power on small panels by design."),
    ("Variance / KL vs calibration", "Overall spread; gap from the calibration group.",
     "Ignores sampling uncertainty; dominated by common variants."),
    ("Chi-square homogeneity (asymptotic)", "Do all groups share one AF?",
     "Invalid when expected counts < 5 (rare variant x small panel)."),
    ("Chi-square homogeneity (Monte Carlo)", "Same statistic; simulated null.",
     "Valid for rare variants; approximate (pooled AF is estimated)."),
    ("Hudson Fst (mean pairwise)", "How strongly does ancestry structure the AF?",
     "Bounded by allele frequency; near 0 for rare alleles even at 10x ratios."),
    ("Log odds ratio vs calibration", "Per-group effect size with CI (Haldane-Anscombe).",
     "Odds ratio approximates frequency ratio only for rare alleles."),
]

LIMITATIONS = [
    "Hardy-Weinberg equilibrium is assumed when converting allele frequency to genotype rates; "
    "admixed or consanguineous populations violate it.",
    "Analyses are per single variant. Real pharmacogenomic phenotypes combine star alleles, "
    "copy-number variation, and (for CYP2D6) structural variants that gnomAD short-read "
    "frequencies do not capture.",
    "tau (the inclusion threshold), the policies, and the F-flag thresholds are parameters of THIS "
    "framework, not properties of any specific commercial or clinical dashboard. Sweep them.",
    "gnomAD ancestry groups are inferred genetic-ancestry clusters; GTEx groups are self-identified "
    "race. The crosswalk between them is lossy (see E4), and neither is a biological category.",
    "GTEx genotypes are controlled-access (dbGaP). Unless an ancestry-stratified GTEx table is "
    "supplied, GTEx enters only through its published donor composition (the 'gtex_weighted' policy).",
    "Catch-all groups (gnomAD 'remaining', GTEx 'other/unreported') are heterogeneous by "
    "construction; their frequencies are hard to interpret and GTEx's 16-allele 'other' bin "
    "alone makes most variants 'underpowered' (F4).",
    "The CPIC gene-drug table in dosing.py is a small illustrative subset; verify before clinical use.",
    "This is a research evaluation framework, not a clinical decision-support tool.",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _fmt(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if math.isinf(v):
            return "∞"
        if float(v).is_integer() and abs(v) >= 1000 and abs(v) < 1e10:
            return f"{int(v):,}"
        if v != 0 and (abs(v) < 10 ** -digits or abs(v) >= 1e6):
            return f"{v:.3g}"
        return f"{v:.{digits}g}"
    return str(v)


def _md(df: pd.DataFrame, digits: int = 4) -> str:
    if df is None or len(df) == 0:
        return "_(no data)_"
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], digits) for c in cols) + " |")
    return "\n".join(out)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    return obj


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _section_provenance(res: EvaluationResult) -> str:
    cfg = res.config
    lines = ["## 0. Run provenance & reproducibility", ""]
    non_real = [d["name"] for d in res.datasets_meta if d["provenance"] != "real"]
    if non_real:
        lines += ["> **Provenance notice.** Datasets marked `synthetic` or `derived` "
                  f"({', '.join(non_real)}) are simulated / computed views, not empirical "
                  "measurements. Their results validate the framework's behaviour; they must "
                  "not be cited as findings about real populations.", ""]
    dm = pd.DataFrame([{
        "dataset": d["name"], "release": d["release"], "provenance": d["provenance"],
        "variants": d["n_variants"], "populations": len(d["populations"]),
        "sha256": (d["sha256"] or "")[:12],
    } for d in res.datasets_meta])
    lines += [_md(dm), ""]
    lines += [f"- Config fingerprint (scorecard): `{cfg.scorecard.fingerprint()}`",
              f"- Seed: {cfg.seed}; Monte Carlo draws: {cfg.n_mc}; CI level: {cfg.conf}",
              f"- Calibration population (default): `{cfg.calibration_population}`; "
              f"zygosity for F7: `{cfg.zygosity}`; tau grid: {list(cfg.taus)}",
              f"- Environment: " + ", ".join(f"{k} {v}" for k, v in res.environment.items()), ""]
    sc = cfg.scorecard.to_dict()
    lines += ["Scorecard thresholds: " + ", ".join(f"`{k}={v}`" for k, v in sc.items()), ""]
    if res.warnings:
        lines += ["**Warnings**", ""] + [f"- {w}" for w in res.warnings] + [""]
    return "\n".join(lines)


def _section_scorecard(res: EvaluationResult) -> str:
    lines = ["## 1. Scorecard", "",
             "Flags (same rules for every variant): "
             + "; ".join(f"**{FLAG_SHORT[k]}** {v}" for k, v in FLAG_DESCRIPTIONS.items()) + ".", "",
             "Classification: `ACTIONABLE_DISPARITY` = supported disparity **and** dashboard impact; "
             "`DISPARITY_LOW_IMPACT`; `INCONCLUSIVE_POWER` (panel too small to say); "
             "`NO_EVIDENCE_OF_DISPARITY` (only meaningful if F4 and F6 are both ✗).", ""]
    s = res.variant_summary
    for ds, sub in s.groupby("dataset", sort=False):
        prov = sub["provenance"].iloc[0]
        counts = sub["classification"].value_counts().reindex(CLASSIFICATIONS, fill_value=0)
        lines += [f"### {ds}  _(provenance: {prov})_", "",
                  "Classification counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()), ""]
        tbl = pd.DataFrame({
            "variant": sub["variant_id"], "gene": sub["gene"],
            "class": sub["classification"],
            **{FLAG_SHORT[k]: sub[k].map(lambda b: "✓" if b else "·") for k in FLAG_ORDER},
            "naive ratio": sub["naive_ratio"], "ratio LB": sub["interval_ratio_lb"],
            "Fst": sub["mean_pairwise_fst"], "p (MC, BH)": sub["chi2_p_montecarlo_bh"],
        })
        lines += [_md(tbl, 3), ""]
    return "\n".join(lines)


def _section_representation(res: EvaluationResult) -> str:
    lines = ["## 2. Dataset representation", ""]
    rm = pd.DataFrame([{"dataset": k, **v} for k, v in res.representation_metrics.items()])
    lines += [_md(rm, 3), ""]
    cmp_ = res.representation_comparison
    if cmp_:
        g, x = cmp_["gnomad_harmonized_shares"], cmp_["gtex_harmonized_shares"]
        idx = sorted(set(g) | set(x))
        t = pd.DataFrame({"group": idx, "gnomAD share": [g.get(i, 0.0) for i in idx],
                          "GTEx v8 donor share": [x.get(i, 0.0) for i in idx]})
        lines += ["Harmonised panel composition (gnomAD allele number vs GTEx v8 donors; "
                  "GTEx composition from Science 369:1318, 'OTHER' derived by subtraction):", "",
                  _md(t, 3), "",
                  f"Total variation distance between the two compositions: "
                  f"**{cmp_['total_variation_distance']:.3f}**.", ""]
    rep = res.representation
    lines += ["Smallest panels (rare-variant resolution = allele frequency at which "
              "5 copies are expected):", ""]
    low = rep.sort_values("allele_number").groupby("dataset").head(3)[
        ["dataset", "population", "allele_number", "share", "min_resolvable_af"]]
    lines += [_md(low, 3), ""]
    return "\n".join(lines)


def _section_methods(res: EvaluationResult) -> str:
    lines = ["## 3. Methods & concordance", "",
             _md(pd.DataFrame(METHOD_TABLE, columns=["method", "question it answers",
                                                     "known failure regime"])), ""]
    c = res.concordance
    if len(c):
        lines += ["Spearman rank correlation between methods across variants (1 = methods rank "
                  "variants identically; negative = they disagree on which variants matter). "
                  "Computed per dataset; needs >= 4 variants.", ""]
        for ds, sub in c.groupby("dataset", sort=False):
            m = sub.drop(columns="dataset").set_index("method").round(2)
            lines += [f"**{ds}**", "", _md(m.reset_index(), 2), ""]
    else:
        lines += ["_Concordance needs a dataset with >= 4 variants; none supplied._", ""]
    return "\n".join(lines)


def _policy_f7_counts(res: EvaluationResult) -> pd.DataFrame:
    cfg = res.config
    rows = []
    e2 = res.e2[res.e2.zygosity == cfg.zygosity]
    for (ds, pol), sub in e2.groupby(["dataset", "policy"]):
        n_var = sub["variant_id"].nunique()
        n_hit = sum(material_prevalence_error(g, cfg.scorecard)[0]
                    for _, g in sub.groupby("variant_id"))
        rows.append({"dataset": ds, "policy": pol, "variants": n_var,
                     "variants_with_material_under_prediction": n_hit})
    return pd.DataFrame(rows)


def _section_error_modes(res: EvaluationResult) -> str:
    cfg = res.config
    lines = ["## 4. Dashboard error modes", "",
             "How frequency disparities become dashboard errors. A genotype-aware tool needs a "
             "population AF only where it filters variants, estimates prevalence, or sets priors; "
             "each is evaluated against three reference-frequency *policies*: the calibration "
             "population's AF, the pooled all-population AF, and a GTEx-composition-weighted AF.", ""]

    # E1
    e1 = res.e1
    if len(e1):
        g = (e1.groupby(["dataset", "policy", "tau", "variant_id"])["false_exclusion_supported"]
               .any().reset_index()
               .groupby(["dataset", "policy", "tau"])["false_exclusion_supported"].sum()
               .reset_index())
        piv = g.pivot_table(index=["dataset", "policy"], columns="tau",
                            values="false_exclusion_supported").reset_index()
        piv.columns = [str(c) for c in piv.columns]
        lines += ["### E1 — Inclusion-threshold exclusion", "",
                  "Number of variants for which a rule *'include the variant iff the policy AF >= tau'* "
                  "wrongly excludes it for at least one population whose AF is **supported** "
                  "(CI lower bound) to be >= tau. Columns are tau.", "", _md(piv, 3), ""]

    # E2
    f7 = _policy_f7_counts(res)
    lines += ["### E2 — Prevalence-prediction error", "",
              f"Variants (per dataset and policy) where a population's actionable-genotype rate "
              f"(`{cfg.zygosity}`, Hardy-Weinberg) is materially under-predicted on the "
              "conservative CI comparison used by F7:", "", _md(f7, 3), ""]
    real = res.variant_summary[res.variant_summary.provenance == "real"]
    if len(real):
        lines += ["Real-data detail (primary policy):", "",
                  _md(real[["dataset", "variant_id", "worst_shortfall_population",
                            "worst_shortfall_per_10k", "worst_shortfall_lb_per_10k"]]
                      .rename(columns={"worst_shortfall_per_10k": "worst shortfall /10k (point)",
                                       "worst_shortfall_lb_per_10k": "worst shortfall /10k (conservative)"}), 3),
                  ""]

    # E3
    if len(res.e3):
        e3 = res.e3.groupby("dataset").agg(
            variants=("variant_id", "nunique"),
            median_dominant_share=("dominant_share", "median"),
            median_pooled_over_dominant_af=("pooled_over_dominant_af", "median"),
            median_max_pop_af_over_pooled=("max_pop_af_over_pooled", "median"),
        ).reset_index()
        lines += ["### E3 — Pooled-AF masking", "",
                  "`pooled_over_dominant_af` near 1 means the pooled AF is essentially the dominant "
                  "population's AF; `max_pop_af_over_pooled` is how many times higher the "
                  "highest-AF group is than the pooled value a dashboard would display.", "",
                  _md(e3, 3), ""]

    # E4
    if len(res.e4):
        e4 = res.e4.groupby(["dataset", "variant_id"]).agg(
            gtex_weighted_af_min=("gtex_weighted_af", "min"),
            gtex_weighted_af_max=("gtex_weighted_af", "max")).reset_index()
        e4["max_over_min"] = e4["gtex_weighted_af_max"] / e4["gtex_weighted_af_min"].replace(0, np.nan)
        keep = e4[e4.dataset.isin(real.dataset.unique())] if len(real) else e4.head(10)
        lines += ["### E4 — Crosswalk sensitivity", "",
                  "Range of the GTEx-weighted AF across 9 label-mapping assumptions "
                  "(Asian American -> East / South / split; 0 / 20 / 30% European admixture assumed "
                  "in African American donors). Shown for real datasets.", "", _md(keep, 3), ""]

    # E5
    if len(res.e5):
        e5 = res.e5[(res.e5.kind == "population") & (res.e5.tau == cfg.scorecard.primary_tau)]
        t = (e5.groupby(["dataset", "status"]).size().unstack(fill_value=0).reset_index())
        lines += [f"### E5 — Decision stability (tau = {cfg.scorecard.primary_tau})", "",
                  "Population x variant AF estimates classified by whether their 95% CI is entirely "
                  "above tau (`include`), entirely below (`exclude`), or straddles it "
                  "(`indeterminate` — the decision is not supported by the data):", "", _md(t, 3), ""]

    # E6
    if len(res.e6):
        e6 = res.e6
        lines += ["### E6 — Cross-dataset discordance", "",
                  f"{len(e6)} variant x harmonised-group comparisons; "
                  f"**{int(e6['discordant'].sum())}** significantly discordant after Bonferroni "
                  "(a dataset effect beyond sampling noise).", ""]
        d = e6[e6.discordant]
        if len(d):
            lines += [_md(d[["dataset_a", "dataset_b", "variant_id", "group", "af_a", "af_b",
                             "n_a", "n_b", "p_bonferroni"]], 3), ""]
    return "\n".join(lines)


def _section_detail(res: EvaluationResult, max_variants: int = 15,
                    include_synthetic: bool = False) -> str:
    lines = ["## 5. Per-variant detail", ""]
    s = res.variant_summary
    for ds, sub in s.groupby("dataset", sort=False):
        if sub["provenance"].iloc[0] != "real" and not include_synthetic:
            lines += [f"_{ds} ({sub['provenance'].iloc[0]}): detail omitted; "
                      "see tables/per_population.csv._", ""]
            continue
        if len(sub) > max_variants:
            lines += [f"_{ds}: {len(sub)} variants; detail omitted (see tables/per_population.csv)._", ""]
            continue
        for _, v in sub.iterrows():
            lines += [f"### {v['gene']} — {v['variant_id']}  ({ds}, {v['provenance']})", ""]
            ratio = "∞ (0 observed copies in >= 1 group)" if v["naive_ratio_is_infinite"] \
                else _fmt(v["naive_ratio"])
            lines += [
                f"- **Classification:** `{v['classification']}`",
                f"- Naive max/min ratio: {ratio} (max {v['max_population']}, min {v['min_population']}); "
                f"conservative lower bound: **{_fmt(v['interval_ratio_lb'], 3)}x** "
                f"({v['interval_ratio_num_pop']} vs {v['interval_ratio_den_pop']})",
                f"- Chi-square: asymptotic p {_fmt(v['chi2_p_asymptotic'])} "
                f"({'valid' if v['chi2_asymptotics_valid'] else '**invalid: min expected count '+_fmt(v['chi2_min_expected'],3)+'**'}); "
                f"Monte Carlo p {_fmt(v['chi2_p_montecarlo'])} (BH {_fmt(v['chi2_p_montecarlo_bh'])}); "
                f"Cramér's V {_fmt(v['cramers_v'])}",
                f"- Mean pairwise Hudson Fst: {_fmt(v['mean_pairwise_fst'])}",
                f"- Flags: " + ", ".join(f"{FLAG_SHORT[k]}={'✓' if v[k] else '·'}" for k in FLAG_ORDER),
            ]
            if v["populations_below_detection_floor"]:
                lines.append(f"- Populations below detection floor: {v['populations_below_detection_floor']}")
            if v["false_exclusion_populations"]:
                lines.append(f"- Falsely excluded at tau={res.config.scorecard.primary_tau} "
                             f"under `{v['primary_policy']}`: {v['false_exclusion_populations']}")
            if v["dosing_drugs"]:
                lines.append(f"- Documented dosing relevance (CPIC, see dosing.py): {v['dosing_drugs'].replace(';', ', ')}")
            lines.append("")
            pp = res.per_population[(res.per_population.dataset == ds)
                                    & (res.per_population.variant_id == v["variant_id"])]
            cols = ["population", "allele_count", "allele_number", "allele_frequency",
                    "wilson_lo", "wilson_hi", "below_detection_floor"]
            for extra in ("or", "or_ci_lo", "or_ci_hi", "fst_vs_calibration"):
                if extra in pp.columns:
                    cols.append(extra)
            lines += [_md(pp[cols], 3), ""]
    return "\n".join(lines)


def _section_validation(validation: Optional[Dict[str, pd.DataFrame]], seed: int) -> str:
    lines = ["## 6. Method validation (simulation; ground truth known)", ""]
    if not validation:
        return "\n".join(lines + ["_Validation not run (use --validate)._", ""])
    d = validation["detection"]
    reps = int(d["reps"].iloc[0])
    lines += [f"Balding-Nichols simulation, {reps} replicates per cell, seed {seed}. "
              "Panels: gnomAD-like (allele numbers from the supplied gnomAD table) and GTEx-like "
              "(2 x v8 donors per group). 'Truth' = whether the TRUE max/min AF ratio is >= 2.", ""]
    M = ["naive_ratio", "interval_ratio_lb", "chi2_asymptotic", "chi2_montecarlo", "hudson_fst"]

    h0 = d[d.fst == 0].pivot_table(index=["panel", "p0"], columns="method", values="flag_rate")[M].reset_index()
    lines += ["### 6.1 False-alarm rate when there is NO disparity (all groups share one AF)", "",
              "Each cell is the fraction of replicates in which the method flagged a disparity "
              "although the true ratio is exactly 1. Nominal tests should sit near 0.05.", "", _md(h0, 3), ""]
    inf = d[(d.fst == 0) & (d.method == "naive_ratio")][
        ["panel", "p0", "naive_ratio_infinite_rate", "naive_ratio_median_inflation"]]
    lines += ["Naive ratio under no disparity: fraction reported as infinite, and median finite "
              "ratio (truth = 1):", "", _md(inf, 3), ""]

    for num, f in (("6.2", 0.01), ("6.3", 0.05)):
        h1 = d[d.fst == f].pivot_table(index=["panel", "p0"], columns="method",
                                       values="sensitivity")[M].reset_index()
        lines += [f"### {num} Sensitivity when populations truly differ (Fst = {f})", "",
                  "Fraction of replicates flagged when the TRUE max/min ratio is >= 2. "
                  "(Fst is scored against the same ratio truth for comparability even though it "
                  "answers a different question.)", "", _md(h1, 3), ""]

    cc = validation["ci_coverage"]
    piv = cc.pivot_table(index=["n_alleles", "true_af"], columns="method", values="coverage").reset_index()
    lines += ["### 6.4 Exact coverage of nominal-95% intervals", "",
              "Enumerated from the binomial pmf (no Monte Carlo error). 'bootstrap_plugin' is the "
              "analytic limit of `robustness.bootstrap_frequency_ci`; it is undefined (NaN) at 0 "
              "copies, counted here as a miss.", "", _md(piv, 3), ""]
    lines += ["### 6.5 Why Fst and the ratio disagree for rare alleles", "",
              "Population-level two-group Fst at a fixed frequency ratio:", "",
              _md(validation["fst_vs_ratio"], 3), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def render_report(res: EvaluationResult, validation: Optional[Dict[str, pd.DataFrame]] = None,
                  figures: Optional[Dict[str, str]] = None) -> str:
    parts = ["# Pharmequity — Allele-Frequency Disparity Evaluation Report Card", ""]
    parts += [_section_provenance(res), _section_scorecard(res), _section_representation(res),
              _section_methods(res)]
    if figures:
        parts += ["### Figures", ""] + [f"![{k}]({v})" for k, v in figures.items()] + [""]
    parts += [_section_error_modes(res), _section_detail(res),
              _section_validation(validation, res.config.seed),
              "## 7. Limitations", ""] + [f"- {l}" for l in LIMITATIONS] + [""]
    return "\n".join(parts)


def write_evaluation(res: EvaluationResult, out_dir: Path,
                     validation: Optional[Dict[str, pd.DataFrame]] = None,
                     make_figures: bool = True) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    tables = {
        "variant_summary": res.variant_summary, "per_population": res.per_population,
        "e1_inclusion_threshold": res.e1, "e2_prevalence_error": res.e2,
        "e3_pooling_masking": res.e3, "e4_crosswalk_sensitivity": res.e4,
        "e5_decision_stability": res.e5, "e6_dataset_discordance": res.e6,
        "representation": res.representation, "method_concordance": res.concordance,
    }
    if validation:
        tables.update({f"validation_{k}": v for k, v in validation.items()})
    paths: Dict[str, Path] = {}
    for name, df in tables.items():
        if df is not None and len(df):
            p = out_dir / "tables" / f"{name}.csv"
            df.to_csv(p, index=False)
            paths[name] = p

    figures = None
    if make_figures:
        try:
            from .plots import make_all_figures
            figures = make_all_figures(res, validation, out_dir / "figures")
            figures = {k: str(Path("figures") / Path(v).name) for k, v in figures.items()}
        except ImportError:
            figures = None

    md = render_report(res, validation, figures)
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")
    paths["report"] = out_dir / "REPORT.md"

    blob = {
        "schema_version": SCHEMA_VERSION,
        "config": res.config.to_dict(),
        "environment": res.environment,
        "datasets": res.datasets_meta,
        "warnings": res.warnings,
        "flag_descriptions": FLAG_DESCRIPTIONS,
        "classifications": list(CLASSIFICATIONS),
        "variant_summary": res.variant_summary.to_dict(orient="records"),
        "representation_metrics": res.representation_metrics,
        "representation_comparison": res.representation_comparison,
        "limitations": LIMITATIONS,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(_jsonable(blob), indent=2), encoding="utf-8")
    paths["json"] = out_dir / "evaluation.json"
    return paths

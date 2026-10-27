"""
evaluate.py — orchestrate datasets x methods x error modes into tidy tables.

    result = run_evaluation([gnomad_ds, synthetic_ds, ...], EvalConfig())

Every variant in every dataset goes through the SAME pipeline:

    methods.run_battery            8 disparity methods (all populations kept)
    errormodes.build_policies      reference-frequency policies
    errormodes.e1..e6              dashboard error modes
    robustness.detection_floor     per-population panel-size power
    scorecard                      7 flags -> 1 classification

and the output is a set of long-format DataFrames (one row per
dataset/variant[/population/policy]) that reporting.py renders and that you
can load directly for your own analysis.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy

from . import __version__
from .datasets import Dataset
from .dosing import DOSING_TABLE
from .errormodes import (
    build_policies, e1_inclusion_threshold, e2_prevalence_error, e3_pooling_masking,
    e4_crosswalk_sensitivity, e5_decision_stability, e6_dataset_discordance,
)
from .methods import bh_adjust, run_battery
from .representation import (
    compare_representation, dataset_representation_metrics, representation_summary,
)
from .robustness import detection_floor
from .scorecard import (
    FLAG_DESCRIPTIONS, ScorecardConfig, classify, material_prevalence_error,
)


@dataclass
class EvalConfig:
    calibration_population: str = "nfe"
    zygosity: str = "recessive"
    taus: tuple = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)
    conf: float = 0.95
    n_mc: int = 4999
    seed: int = 42
    min_expected_count: int = 5
    scorecard: ScorecardConfig = field(default_factory=ScorecardConfig)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["scorecard"] = self.scorecard.to_dict()
        d["scorecard_fingerprint"] = self.scorecard.fingerprint()
        return d


@dataclass
class EvaluationResult:
    config: EvalConfig
    datasets_meta: List[Dict]
    variant_summary: pd.DataFrame
    per_population: pd.DataFrame
    e1: pd.DataFrame
    e2: pd.DataFrame
    e3: pd.DataFrame
    e4: pd.DataFrame
    e5: pd.DataFrame
    e6: pd.DataFrame
    representation: pd.DataFrame
    representation_metrics: Dict[str, Dict]
    representation_comparison: Optional[Dict]
    concordance: pd.DataFrame
    warnings: List[str]
    environment: Dict


def _environment() -> Dict:
    return {
        "pharmequity": __version__, "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
    }


def _tag(df: pd.DataFrame, **cols) -> pd.DataFrame:
    for i, (k, v) in enumerate(cols.items()):
        df.insert(i, k, v)
    return df


def evaluate_dataset(ds: Dataset, cfg: EvalConfig):
    """Evaluate every variant in one dataset. Returns a dict of frames + warnings."""
    cal = ds.calibration_population or cfg.calibration_population
    sc = cfg.scorecard
    warnings: List[str] = []
    taus = sorted(set(cfg.taus) | {sc.primary_tau})  # primary tau always evaluated

    batteries, per_pop_frames = {}, []
    e1s, e2s, e3s, e4s, e5s = [], [], [], [], []
    meta = {}

    for gene, vid in ds.variants():
        rows = ds.rows_for(vid)
        key = (gene, vid)
        bat = run_battery(rows, cal, cfg.conf, cfg.n_mc, cfg.seed)
        policies, w = build_policies(rows, cal)
        warnings += [f"[{ds.name}] {x}" for x in w]

        # detection floor per population (robustness.py)
        floors = []
        for _, r in rows.iterrows():
            fl = detection_floor(int(r["allele_number"]), float(r["allele_frequency"]),
                                 cfg.min_expected_count)
            floors.append((r["population"], fl.below_detection_floor, fl.min_detectable_frequency_95))
        pp = bat["per_population"].copy()
        pp["min_resolvable_af"] = [f[2] for f in floors]
        pp["below_detection_floor"] = [f[1] for f in floors]
        per_pop_frames.append(_tag(pp, dataset=ds.name, gene=gene, variant_id=vid))

        for tau in taus:
            e1s.append(_tag(e1_inclusion_threshold(rows, policies, tau, cfg.conf),
                            dataset=ds.name, gene=gene, variant_id=vid))
            e5s.append(_tag(e5_decision_stability(rows, policies, tau, cfg.conf),
                            dataset=ds.name, gene=gene, variant_id=vid))
        for zyg in ("recessive", "dominant_or_het"):
            e2s.append(_tag(e2_prevalence_error(rows, policies, zyg, cfg.conf),
                            dataset=ds.name, gene=gene, variant_id=vid))
        e3s.append({"dataset": ds.name, "gene": gene, "variant_id": vid, **e3_pooling_masking(rows)})
        e4 = e4_crosswalk_sensitivity(rows)
        if len(e4):
            e4s.append(_tag(e4, dataset=ds.name, gene=gene, variant_id=vid))

        batteries[key] = bat
        meta[key] = {"rows": rows, "policies": policies, "floors": floors,
                     "rsid": str(rows["rsid"].iloc[0])}

    # BH across variants within the dataset (F2)
    keys = list(batteries)
    p_adj = bh_adjust([batteries[k]["chi2_p_montecarlo"] for k in keys])
    p_adj_map = dict(zip(keys, p_adj))

    e1_all = pd.concat(e1s, ignore_index=True) if e1s else pd.DataFrame()
    e2_all = pd.concat(e2s, ignore_index=True) if e2s else pd.DataFrame()
    e5_all = pd.concat(e5s, ignore_index=True) if e5s else pd.DataFrame()

    summary_rows = []
    for (gene, vid) in keys:
        bat = batteries[(gene, vid)]
        m = meta[(gene, vid)]
        prim_name = next((p.name for p in m["policies"] if p.name.startswith(sc.primary_policy_prefix)), None)

        # F5 / F6 / F7 at primary tau and primary policy
        e1v = e1_all[(e1_all.variant_id == vid) & (e1_all.tau == sc.primary_tau)]
        if prim_name:
            e1p = e1v[e1v.policy == prim_name]
            e2p = e2_all[(e2_all.variant_id == vid) & (e2_all.policy == prim_name)
                         & (e2_all.zygosity == cfg.zygosity)]
        else:
            e1p, e2p = e1v.iloc[0:0], e2_all.iloc[0:0]
        e5v = e5_all[(e5_all.variant_id == vid) & (e5_all.tau == sc.primary_tau)
                     & (e5_all.kind == "population")]

        f7, f7_pop = material_prevalence_error(e2p, sc)
        below = [f[0] for f in m["floors"] if f[1]]
        flags = {
            "F1_disparity_supported": bool(bat["interval_ratio_lb"] >= sc.min_supported_ratio),
            "F2_heterogeneity_significant": bool(p_adj_map[(gene, vid)] < sc.alpha),
            "F3_differentiation_moderate": bool(
                not np.isnan(bat["mean_pairwise_fst"]) and bat["mean_pairwise_fst"] >= sc.fst_threshold),
            "F4_underpowered_population": bool(below),
            "F5_false_exclusion": bool(len(e1p) and e1p["false_exclusion_supported"].any()),
            "F6_decision_indeterminate": bool((e5v["status"] == "indeterminate").any()),
            "F7_material_prevalence_error": bool(f7),
        }
        scalar = {k: v for k, v in bat.items() if k != "per_population"}
        row = {
            "dataset": ds.name, "provenance": ds.provenance, "release": ds.release,
            "gene": gene, "variant_id": vid, "rsid": m["rsid"],
            "calibration_population": cal,
            "primary_policy": prim_name or "",
            **scalar,
            "chi2_p_montecarlo_bh": float(p_adj_map[(gene, vid)]),
            **flags,
            "classification": classify(flags),
            "populations_below_detection_floor": ";".join(below),
            "false_exclusion_populations": ";".join(sorted(e1p.loc[e1p.false_exclusion_supported, "population"]))
                                            if len(e1p) else "",
            "worst_shortfall_population": f7_pop,
            "worst_shortfall_per_10k": float(e2p["shortfall_per_10k"].max()) if len(e2p) else float("nan"),
            "worst_shortfall_lb_per_10k": float(e2p["shortfall_lb_per_10k"].max()) if len(e2p) else float("nan"),
            "dosing_drugs": ";".join(DOSING_TABLE[gene].drugs) if gene in DOSING_TABLE else "",
        }
        summary_rows.append(row)

    return {
        "summary": pd.DataFrame(summary_rows),
        "per_population": pd.concat(per_pop_frames, ignore_index=True) if per_pop_frames else pd.DataFrame(),
        "e1": e1_all, "e2": e2_all, "e5": e5_all,
        "e3": pd.DataFrame(e3s),
        "e4": pd.concat(e4s, ignore_index=True) if e4s else pd.DataFrame(),
        "warnings": sorted(set(warnings)),
    }


METRIC_COLUMNS = ["naive_ratio", "interval_ratio_lb", "variance", "max_kl_vs_calibration",
                  "chi2", "cramers_v", "mean_pairwise_fst", "absolute_diff"]


def method_concordance(summary: pd.DataFrame, min_variants: int = 4) -> pd.DataFrame:
    """Spearman rank correlation between methods across variants.

    If methods agreed on which variants are most disparate, this matrix would
    be near 1 everywhere. Low off-diagonals mean the choice of method changes
    which variants look important. Needs >= min_variants variants.
    """
    from scipy.stats import spearmanr
    cols = [c for c in METRIC_COLUMNS if c in summary.columns]
    if len(summary) < min_variants:
        return pd.DataFrame()
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            x, y = summary[a].to_numpy(float), summary[b].to_numpy(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < min_variants or len(set(x[ok])) < 2 or len(set(y[ok])) < 2:
                out.loc[a, b] = np.nan
            else:
                out.loc[a, b] = spearmanr(x[ok], y[ok]).statistic
    return out


def run_evaluation(datasets: List[Dataset], cfg: Optional[EvalConfig] = None) -> EvaluationResult:
    cfg = cfg or EvalConfig()
    parts = [evaluate_dataset(ds, cfg) for ds in datasets]

    def cat(key):
        frames = [p[key] for p in parts if len(p[key])]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    summary = cat("summary")

    # Cross-dataset discordance (E6) for every dataset pair sharing variants
    e6_frames = []
    for a, b in combinations(datasets, 2):
        if a.crosswalk is None or b.crosswalk is None:
            continue
        shared = set(a.table.variant_id) & set(b.table.variant_id)
        if not shared:
            continue
        res = e6_dataset_discordance(
            a.table[a.table.variant_id.isin(shared)], a.crosswalk,
            b.table[b.table.variant_id.isin(shared)], b.crosswalk)
        if len(res):
            e6_frames.append(_tag(res, dataset_a=a.name, dataset_b=b.name))
    e6 = pd.concat(e6_frames, ignore_index=True) if e6_frames else pd.DataFrame()

    # Representation
    rep_frames, rep_metrics = [], {}
    for ds in datasets:
        rs = representation_summary(ds.table, cfg.min_expected_count)
        rep_frames.append(_tag(rs.copy(), dataset=ds.name))
        rep_metrics[ds.name] = dataset_representation_metrics(ds.table, cfg.min_expected_count)

    rep_comp = None
    gn = next((d for d in datasets if d.crosswalk is not None
               and set(d.crosswalk) >= {"nfe", "afr"} and "nfe" in set(d.table.population)), None)
    if gn is not None:
        rep_comp = compare_representation(gn.table, gn.crosswalk)

    # Concordance per dataset (a pooled matrix would confound datasets)
    conc = {}
    for ds in datasets:
        sub = summary[summary.dataset == ds.name]
        c = method_concordance(sub)
        if len(c):
            conc[ds.name] = c
    conc_df = (pd.concat(conc, names=["dataset", "method"]).reset_index()
               if conc else pd.DataFrame())

    warnings = sorted({w for p in parts for w in p["warnings"]})
    for ds in datasets:
        if ds.provenance != "real":
            warnings.append(f"[{ds.name}] provenance='{ds.provenance}': results are NOT "
                            "empirical measurements of real populations.")
        if ds.crosswalk is None:
            warnings.append(f"[{ds.name}] population codes not covered by a known crosswalk; "
                            "cross-dataset (E6) comparison skipped.")

    return EvaluationResult(
        config=cfg,
        datasets_meta=[{
            "name": d.name, "release": d.release, "provenance": d.provenance,
            "source_path": d.source_path, "sha256": d.sha256,
            "n_variants": len(d.variants()), "populations": d.populations, "notes": d.notes,
        } for d in datasets],
        variant_summary=summary,
        per_population=cat("per_population"),
        e1=cat("e1"), e2=cat("e2"), e3=cat("e3"), e4=cat("e4"), e5=cat("e5"), e6=e6,
        representation=pd.concat(rep_frames, ignore_index=True),
        representation_metrics=rep_metrics,
        representation_comparison=rep_comp,
        concordance=conc_df,
        warnings=warnings,
        environment=_environment(),
    )

"""
plots.py — figures for the report card. Requires matplotlib (optional extra:
`pip install pharmequity[plots]`). Plots only draw what the result tables
contain; every figure title carries the dataset provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .evaluate import EvaluationResult  # noqa: E402

PALETTE = {"wilson": "#1b7837", "clopper_pearson": "#2166ac",
           "wald": "#b2182b", "bootstrap_plugin": "#ef8a62"}


def fig_af_forest(res: EvaluationResult, out: Path, max_panels: int = 3) -> Optional[Path]:
    """AF by population with Wilson CIs on a log axis, for real-provenance variants."""
    real = res.variant_summary[res.variant_summary.provenance == "real"]
    if real.empty:
        return None
    real = real.head(max_panels)
    fig, axes = plt.subplots(1, len(real), figsize=(6.5 * len(real), 4.6), squeeze=False)
    for ax, (_, v) in zip(axes[0], real.iterrows()):
        pp = res.per_population[(res.per_population.dataset == v["dataset"])
                                & (res.per_population.variant_id == v["variant_id"])]
        pp = pp.sort_values("allele_frequency").reset_index(drop=True)
        floor = 1e-6
        y = np.arange(len(pp))
        af = pp["allele_frequency"].clip(lower=floor)
        ax.errorbar(af, y, xerr=[(af - pp["wilson_lo"].clip(lower=floor)).clip(lower=0),
                                 (pp["wilson_hi"] - af).clip(lower=0)],
                    fmt="o", color="#333", ecolor="#999", capsize=2)
        zero = pp["allele_count"] == 0
        ax.scatter(af[zero], y[zero], marker="x", color="#b2182b", zorder=3,
                   label="0 observed copies (drawn at 1e-6)")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{p} (n={int(n):,})" for p, n in zip(pp["population"], pp["allele_number"])])
        ax.set_xscale("log")
        ax.set_xlabel("allele frequency (log scale, Wilson 95% CI)")
        ax.set_title(f"{v['variant_id']}\n[{v['dataset']}, {v['provenance']}]", fontsize=10)
        if zero.any():
            ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_representation(res: EvaluationResult, out: Path) -> Optional[Path]:
    cmp_ = res.representation_comparison
    if not cmp_:
        return None
    g, x = cmp_["gnomad_harmonized_shares"], cmp_["gtex_harmonized_shares"]
    groups = sorted(set(g) | set(x))
    xs = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(xs - 0.2, [g.get(k, 0) for k in groups], 0.4, label="gnomAD (allele number)", color="#2166ac")
    ax.bar(xs + 0.2, [x.get(k, 0) for k in groups], 0.4, label="GTEx v8 (donors)", color="#ef8a62")
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_ylabel("share of reference panel")
    ax.set_title(f"Panel composition on harmonised groups\n"
                 f"(total variation distance {cmp_['total_variation_distance']:.2f})", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_ci_coverage(validation: Dict[str, pd.DataFrame], out: Path, n_alleles: int = 912) -> Optional[Path]:
    cc = validation["ci_coverage"]
    sub = cc[cc.n_alleles == n_alleles]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, g in sub.groupby("method"):
        g = g.sort_values("expected_count")
        ax.plot(g["expected_count"], g["coverage"], "o-", label=m, color=PALETTE.get(m))
    ax.axhline(0.95, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"expected allele copies in the panel (n = {n_alleles:,} alleles)")
    ax.set_ylabel("exact coverage of nominal 95% CI")
    ax.set_title("Interval coverage collapses for rare alleles in small panels\n[simulation]", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_false_alarms(validation: Dict[str, pd.DataFrame], out: Path) -> Optional[Path]:
    d = validation["detection"]
    h0 = d[d.fst == 0]
    if h0.empty:
        return None
    methods = ["naive_ratio", "chi2_asymptotic", "chi2_montecarlo", "interval_ratio_lb", "hudson_fst"]
    panels = sorted(h0["panel"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4), sharey=True, squeeze=False)
    for ax, panel in zip(axes[0], panels):
        sub = h0[h0.panel == panel]
        p0s = sorted(sub["p0"].unique())
        xs = np.arange(len(p0s))
        w = 0.16
        for i, m in enumerate(methods):
            vals = [sub[(sub.p0 == p) & (sub.method == m)]["flag_rate"].iloc[0] for p in p0s]
            ax.bar(xs + (i - 2) * w, vals, w, label=m)
        ax.axhline(0.05, color="k", lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p:g}" for p in p0s])
        ax.set_xlabel("true allele frequency (same in all groups)")
        ax.set_title(f"{panel} panel", fontsize=10)
    axes[0][0].set_ylabel("false-alarm rate (no true disparity)")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("False alarms when every group has the same AF [simulation]", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def make_all_figures(res: EvaluationResult, validation: Optional[Dict[str, pd.DataFrame]],
                     out_dir: Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: Dict[str, Optional[Path]] = {
        "AF by population (real data)": fig_af_forest(res, out_dir / "af_forest.png"),
        "Reference-panel composition": fig_representation(res, out_dir / "representation.png"),
    }
    if validation:
        figs["CI coverage"] = fig_ci_coverage(validation, out_dir / "ci_coverage.png")
        figs["False alarms"] = fig_false_alarms(validation, out_dir / "false_alarms.png")
    return {k: v for k, v in figs.items() if v is not None}

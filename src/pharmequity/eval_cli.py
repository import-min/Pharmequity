"""
eval_cli.py — `pharmequity-eval`: the full systematic evaluation.

Examples
--------
# Real gnomAD table + synthetic demo datasets + simulation validation
pharmequity-eval --frequencies data/real_gnomad_frequencies.csv \\
                 --name gnomAD --release "gnomAD v4.1" \\
                 --synthetic-demo --validate --out results/

# Several real tables (repeat --frequencies; --name/--release/--provenance
# are matched by position)
pharmequity-eval --frequencies gnomad.csv --name gnomAD \\
                 --frequencies gtex_stratified.csv --name GTEx --provenance real \\
                 --out results/

The original `pharmequity-cli` (per-variant markdown reports) is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .datasets import Dataset, load_dataset
from .evaluate import EvalConfig, run_evaluation
from .reporting import write_evaluation
from .scorecard import ScorecardConfig
from .simulate import (
    gnomad_like_panel, run_validation, synthetic_gnomad_like, synthetic_gtex_like,
)


def build_demo_datasets(real: Dataset, seed: int = 20260101) -> List[Dataset]:
    """Synthetic gnomAD-like and GTEx-like datasets sharing one set of TRUE frequencies."""
    panel = gnomad_like_panel(real.table)
    return [
        Dataset(name="synthetic_gnomad_like",
                release="synthetic archetypes @ gnomAD-like panel sizes",
                provenance="synthetic", table=synthetic_gnomad_like(panel, seed=seed),
                calibration_population=real.calibration_population,
                notes="Counts drawn from archetype frequencies; not real data."),
        Dataset(name="synthetic_gtex_like",
                release="same archetypes @ GTEx v8 panel sizes (2 x donors)",
                provenance="synthetic", table=synthetic_gtex_like(seed=seed + 1),
                calibration_population="EUR_AMERICAN",
                notes="GTEx-like groups built from gnomAD-coded truth with an ASSUMED 20% "
                      "European admixture in African American donors."),
    ]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Systematic allele-frequency disparity evaluation.")
    p.add_argument("--frequencies", action="append", type=Path, default=[],
                   help="Frequency CSV (repeatable). Columns: gene,variant_id,rsid,population,"
                        "allele_count,allele_number,allele_frequency")
    p.add_argument("--name", action="append", default=[], help="Dataset name (matched by position)")
    p.add_argument("--release", action="append", default=[], help="Release label (matched by position)")
    p.add_argument("--provenance", action="append", default=[],
                   choices=["real", "synthetic", "derived"], help="Provenance (default: real)")
    p.add_argument("--calibration-population", default="nfe")
    p.add_argument("--zygosity", choices=["recessive", "dominant_or_het"], default="recessive")
    p.add_argument("--primary-tau", type=float, default=0.01,
                   help="Inclusion threshold used by flags F5/F6 (a parameter, not a standard).")
    p.add_argument("--synthetic-demo", action="store_true",
                   help="Add synthetic gnomAD-like and GTEx-like datasets (10 archetype variants).")
    p.add_argument("--validate", action="store_true", help="Run the simulation validation suite.")
    p.add_argument("--reps", type=int, default=600, help="Simulation replicates per scenario.")
    p.add_argument("--n-mc", type=int, default=4999, help="Monte Carlo draws for homogeneity test.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)

    if not a.frequencies and not a.synthetic_demo:
        p.error("provide at least one --frequencies file (or --synthetic-demo)")

    datasets: List[Dataset] = []
    for i, path in enumerate(a.frequencies):
        datasets.append(load_dataset(
            path,
            name=a.name[i] if i < len(a.name) else None,
            release=a.release[i] if i < len(a.release) else "unspecified",
            provenance=a.provenance[i] if i < len(a.provenance) else "real",
            calibration_population=a.calibration_population,
        ))
    if a.synthetic_demo:
        if not datasets:
            p.error("--synthetic-demo needs one real --frequencies table to take panel sizes from")
        datasets += build_demo_datasets(datasets[0])

    cfg = EvalConfig(
        calibration_population=a.calibration_population, zygosity=a.zygosity,
        n_mc=a.n_mc, seed=a.seed,
        scorecard=ScorecardConfig(primary_tau=a.primary_tau),
    )
    res = run_evaluation(datasets, cfg)

    validation = None
    if a.validate:
        real_tables = [d.table for d in datasets if d.provenance == "real"]
        validation = run_validation(real_tables[0] if real_tables else None,
                                    reps=a.reps, seed=a.seed)

    paths = write_evaluation(res, a.out, validation=validation, make_figures=not a.no_figures)
    print(f"Evaluated {len(res.variant_summary)} variant(s) across {len(datasets)} dataset(s).")
    counts = res.variant_summary.groupby(["dataset", "classification"]).size()
    print(counts.to_string())
    print(f"Report: {paths['report']}")


if __name__ == "__main__":
    main()

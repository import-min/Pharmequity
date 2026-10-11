"""
representation.py — how well does a reference dataset represent each group?

Variant-independent. Answers: which populations dominate the panel, how
many "effective" populations does it contain, and what is the smallest
allele frequency each group's panel can resolve? These feed the dataset-
level section of the report and the F4 (underpowered population) flag.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .ancestry import GTEX_V8_DONORS, GTEX_TO_HARMONIZED, harmonize_population


def panel_sizes(table: pd.DataFrame) -> pd.DataFrame:
    """Per-population panel size = median allele_number across variants.

    (AN varies by variant with coverage; the median is a stable summary.)
    """
    g = table.groupby("population")["allele_number"].median().rename("allele_number")
    return g.reset_index()


def representation_summary(table: pd.DataFrame, min_expected_count: int = 5) -> pd.DataFrame:
    """Per-population share of the panel and rare-variant resolution."""
    ps = panel_sizes(table)
    total = ps["allele_number"].sum()
    ps["share"] = ps["allele_number"] / total
    ps["min_resolvable_af"] = min_expected_count / ps["allele_number"]
    return ps.sort_values("allele_number", ascending=False).reset_index(drop=True)


def effective_number_of_populations(shares) -> float:
    """Inverse Simpson index of the share distribution (1 = one group only)."""
    s = np.asarray(list(shares), dtype=float)
    s = s / s.sum()
    return float(1.0 / np.sum(s ** 2))


def dataset_representation_metrics(table: pd.DataFrame, min_expected_count: int = 5) -> Dict:
    rs = representation_summary(table, min_expected_count)
    top = rs.iloc[0]
    return {
        "n_populations": int(len(rs)),
        "total_allele_number": float(rs["allele_number"].sum()),
        "dominant_population": str(top["population"]),
        "dominant_share": float(top["share"]),
        "effective_n_populations": effective_number_of_populations(rs["share"]),
        "smallest_population": str(rs.iloc[-1]["population"]),
        "smallest_panel_alleles": float(rs.iloc[-1]["allele_number"]),
        "largest_to_smallest_panel_ratio": float(top["allele_number"] / rs.iloc[-1]["allele_number"]),
    }


def harmonized_shares(rs: pd.DataFrame, crosswalk: Dict[str, str]) -> pd.Series:
    """Share of panel per harmonised group (UNMAPPED reported, not dropped)."""
    mapped = rs.assign(group=rs["population"].map(lambda c: harmonize_population(c, crosswalk)))
    return mapped.groupby("group")["share"].sum().sort_index()


def total_variation_distance(a: pd.Series, b: pd.Series) -> float:
    """TVD between two share distributions over the union of their groups."""
    idx = a.index.union(b.index)
    a, b = a.reindex(idx, fill_value=0.0), b.reindex(idx, fill_value=0.0)
    return float(0.5 * np.abs(a - b).sum())


def gtex_representation() -> pd.DataFrame:
    """GTEx v8 donor-level representation (alleles = 2 x donors)."""
    rows = [{"population": k, "allele_number": 2 * v} for k, v in GTEX_V8_DONORS.items()]
    df = pd.DataFrame(rows)
    df["share"] = df["allele_number"] / df["allele_number"].sum()
    df["min_resolvable_af"] = 5 / df["allele_number"]
    return df.sort_values("allele_number", ascending=False).reset_index(drop=True)


def compare_representation(gnomad_table: pd.DataFrame, gnomad_crosswalk: Dict[str, str],
                           ) -> Optional[Dict]:
    """gnomAD-vs-GTEx representation on harmonised groups."""
    g_shares = harmonized_shares(representation_summary(gnomad_table), gnomad_crosswalk)
    x_shares = harmonized_shares(gtex_representation(), GTEX_TO_HARMONIZED)
    return {
        "gnomad_harmonized_shares": g_shares.to_dict(),
        "gtex_harmonized_shares": x_shares.to_dict(),
        "total_variation_distance": total_variation_distance(g_shares, x_shares),
    }

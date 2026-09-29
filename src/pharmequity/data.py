"""
data.py — loading and validating population allele-frequency tables.

Expected schema (matches gnomAD's own per-population export columns,
so a real gnomAD download can be dropped in with no reshaping):

    gene, variant_id, rsid, population, allele_count, allele_number, allele_frequency

`population` should use gnomAD's standard ancestry-group codes:
afr, amr, asj, eas, fin, nfe, sas, mid, remaining (gnomAD v4 codes vary
by release; whatever codes are present in the file are used as-is).

IMPORTANT: the file shipped at data/synthetic_demo_frequencies.csv is
SYNTHETIC placeholder data with a realistic shape (realistic gnomAD
cohort sizes per population, plausible allele frequencies for known
pharmacogenes) — it is NOT real gnomAD data and must not be cited or
reported as if it were. See data/README.md for how to substitute a
real gnomAD export.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "gene", "variant_id", "rsid", "population",
    "allele_count", "allele_number", "allele_frequency",
]


def load_frequency_table(path: Path) -> pd.DataFrame:
    """Load and validate a population allele-frequency table.

    Raises ValueError if required columns are missing, if allele_frequency
    is not consistent with allele_count/allele_number, or if any row has
    allele_count > allele_number.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Frequency table missing required columns: {missing}")

    df["allele_count"] = pd.to_numeric(df["allele_count"], errors="raise")
    df["allele_number"] = pd.to_numeric(df["allele_number"], errors="raise")
    df["allele_frequency"] = pd.to_numeric(df["allele_frequency"], errors="raise")

    if (df["allele_count"] > df["allele_number"]).any():
        bad = df[df["allele_count"] > df["allele_number"]]
        raise ValueError(f"allele_count exceeds allele_number in rows: {bad.index.tolist()}")

    # Recompute frequency from counts rather than trusting the stored column,
    # and flag rows where the two disagree beyond floating-point tolerance —
    # this catches stale/corrupted exports rather than silently using them.
    recomputed = df["allele_count"] / df["allele_number"].replace(0, pd.NA)
    mismatch = (recomputed - df["allele_frequency"]).abs() > 1e-4
    mismatch = mismatch.fillna(False)
    if mismatch.any():
        raise ValueError(
            f"allele_frequency inconsistent with allele_count/allele_number "
            f"in {int(mismatch.sum())} row(s), e.g. index {df[mismatch].index[0]}."
        )

    return df


def frequencies_for_variant(df: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    """Return the per-population rows for one variant, sorted by population."""
    sub = df[df["variant_id"] == variant_id].sort_values("population")
    if sub.empty:
        raise ValueError(f"variant_id '{variant_id}' not found in frequency table.")
    return sub.reset_index(drop=True)

"""
datasets.py — a thin Dataset wrapper that carries provenance with the data.

The evaluation runs across several datasets (real gnomAD export, a
GTEx-composition view, ancestry-stratified GTEx or other cohorts if you have
access, and synthetic data with known ground truth). Mixing those up is the
easiest way to misreport a result, so every Dataset carries:

    provenance : 'real' | 'synthetic' | 'derived'
    release    : free-text release label (e.g. 'gnomAD v4.1')
    sha256     : hash of the source file, for reproducibility

and every report prints these. Synthetic results are always labelled as such.

All datasets share the canonical schema from data.py:
    gene, variant_id, rsid, population, allele_count, allele_number, allele_frequency
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .ancestry import GNOMAD_TO_HARMONIZED, GTEX_TO_HARMONIZED
from .data import load_frequency_table

PROVENANCE_VALUES = ("real", "synthetic", "derived")


@dataclass
class Dataset:
    name: str
    release: str
    provenance: str
    table: pd.DataFrame
    source_path: Optional[str] = None
    sha256: Optional[str] = None
    calibration_population: Optional[str] = None  # overrides config default
    notes: str = ""
    crosswalk: Optional[Dict[str, str]] = None    # population code -> harmonised group

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_VALUES:
            raise ValueError(f"provenance must be one of {PROVENANCE_VALUES}")
        if self.crosswalk is None:
            self.crosswalk = infer_crosswalk(set(self.table["population"]))
        self.table = self.table.copy()
        self.table["dataset"] = self.name

    def variants(self) -> List[Tuple[str, str]]:
        sub = self.table[["gene", "variant_id"]].drop_duplicates()
        return [(g, v) for g, v in sub.itertuples(index=False)]

    def rows_for(self, variant_id: str) -> pd.DataFrame:
        sub = self.table[self.table["variant_id"] == variant_id]
        if sub.empty:
            raise ValueError(f"variant_id '{variant_id}' not in dataset '{self.name}'.")
        return sub.sort_values("population").reset_index(drop=True)

    @property
    def populations(self) -> List[str]:
        return sorted(self.table["population"].unique())


def infer_crosswalk(populations) -> Optional[Dict[str, str]]:
    """Pick a known crosswalk if every population code is covered by it."""
    pops = set(populations)
    if pops <= set(GNOMAD_TO_HARMONIZED):
        return GNOMAD_TO_HARMONIZED
    if pops <= set(GTEX_TO_HARMONIZED):
        return GTEX_TO_HARMONIZED
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(
    path: Path,
    name: Optional[str] = None,
    release: str = "unspecified",
    provenance: str = "real",
    calibration_population: Optional[str] = None,
    notes: str = "",
) -> Dataset:
    """Load and validate a frequency CSV into a Dataset.

    Validation (counts <= allele number, AF consistent with counts, required
    columns) is delegated to data.load_frequency_table.
    """
    path = Path(path)
    table = load_frequency_table(path)
    return Dataset(
        name=name or path.stem,
        release=release,
        provenance=provenance,
        table=table,
        source_path=str(path),
        sha256=_sha256(path),
        calibration_population=calibration_population,
        notes=notes,
    )


def concat_variant_tables(datasets: List[Dataset]) -> pd.DataFrame:
    return pd.concat([d.table for d in datasets], ignore_index=True)

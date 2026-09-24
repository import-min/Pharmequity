from pathlib import Path

import pandas as pd
import pytest

from pharmequity.data import load_frequency_table, frequencies_for_variant, REQUIRED_COLUMNS


def _write(tmp_path: Path, rows: str) -> Path:
    p = tmp_path / "freqs.csv"
    p.write_text(rows)
    return p


def test_load_valid_table(tmp_path):
    p = _write(tmp_path,
        "gene,variant_id,rsid,population,allele_count,allele_number,allele_frequency\n"
        "CYP2C19,v1,rs1,afr,10,100,0.1\n"
        "CYP2C19,v1,rs1,nfe,5,100,0.05\n"
    )
    df = load_frequency_table(p)
    assert list(df.columns[:len(REQUIRED_COLUMNS)]) == REQUIRED_COLUMNS
    assert len(df) == 2


def test_missing_column_raises(tmp_path):
    p = _write(tmp_path, "gene,variant_id,rsid,population,allele_count,allele_number\nA,v,r,afr,1,10\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_frequency_table(p)


def test_count_exceeds_number_raises(tmp_path):
    p = _write(tmp_path,
        "gene,variant_id,rsid,population,allele_count,allele_number,allele_frequency\n"
        "A,v,r,afr,150,100,1.5\n"
    )
    with pytest.raises(ValueError, match="exceeds allele_number"):
        load_frequency_table(p)


def test_frequency_mismatch_raises(tmp_path):
    p = _write(tmp_path,
        "gene,variant_id,rsid,population,allele_count,allele_number,allele_frequency\n"
        "A,v,r,afr,10,100,0.99\n"  # should be 0.1, not 0.99
    )
    with pytest.raises(ValueError, match="inconsistent"):
        load_frequency_table(p)


def test_frequencies_for_variant_not_found(tmp_path):
    p = _write(tmp_path,
        "gene,variant_id,rsid,population,allele_count,allele_number,allele_frequency\n"
        "A,v1,r,afr,10,100,0.1\n"
    )
    df = load_frequency_table(p)
    with pytest.raises(ValueError, match="not found"):
        frequencies_for_variant(df, "does_not_exist")

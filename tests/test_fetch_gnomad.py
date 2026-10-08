import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_gnomad", Path(__file__).resolve().parents[1] / "scripts" / "fetch_gnomad.py")
fg = importlib.util.module_from_spec(spec); spec.loader.exec_module(fg)

MOCK = {"data": {"variant": {"variant_id": "1-1-A-G", "rsids": ["rs1"],
    "exome": {"populations": [{"id": "nfe", "ac": 10, "an": 1000}, {"id": "afr", "ac": 1, "an": 100},
                              {"id": "afr_XX", "ac": 1, "an": 50}, {"id": "XY", "ac": 0, "an": 10}]},
    "genome": {"populations": [{"id": "nfe", "ac": 5, "an": 500}]}}}}


def test_sums_callsets_and_drops_strata():
    out = fg.parse_variant_response(MOCK)
    assert out == {"nfe": {"ac": 15, "an": 1500}, "afr": {"ac": 1, "an": 100}}


def test_missing_variant_and_errors_raise():
    with pytest.raises(LookupError):
        fg.parse_variant_response({"data": {"variant": None}})
    with pytest.raises(RuntimeError):
        fg.parse_variant_response({"errors": [{"message": "x"}]})

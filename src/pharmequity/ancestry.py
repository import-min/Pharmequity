"""
ancestry.py — population-label crosswalks and reference-panel composition.

Different resources label ancestry differently:

  * gnomAD uses genetic-ancestry-group codes (afr, amr, asj, eas, fin, nfe, ...)
    inferred from genotype PCs and a trained classifier.
  * GTEx v8 reports *self-identified* race categories for its 838 genotyped
    donors (European American, African American, Asian American, ...).

Comparing the two requires a crosswalk, and every crosswalk is lossy. This
module makes those choices explicit and parameterised, so they can be
varied as a sensitivity analysis (see errormodes.e4_crosswalk_sensitivity)
rather than buried in an analysis script.

GTEx composition source:
    GTEx Consortium, "The GTEx Consortium atlas of genetic regulatory
    effects across human tissues", Science 369:1318-1330 (2020).
    "Of the 838 donors, 715 (85.3%) were European American, 103 (12.3%)
    African American, and 12 (1.4%) Asian American, with 16 (1.9%)
    reporting Hispanic or Latino ethnicity."
The remaining 8 donors are NOT stated in the paper; they are derived here by
subtraction (838 - 715 - 103 - 12) and labelled as such.
"""

from __future__ import annotations

from typing import Dict, Tuple

GTEX_V8_N_GENOTYPED_DONORS = 838

GTEX_V8_DONORS: Dict[str, int] = {
    "EUR_AMERICAN": 715,
    "AFR_AMERICAN": 103,
    "ASIAN_AMERICAN": 12,
    "OTHER_OR_UNREPORTED": 8,  # DERIVED by subtraction; not stated in the paper
}

# Harmonised super-groups used when two datasets must be compared directly.
# These are deliberately coarse; every mapping below is a judgement call.
GNOMAD_TO_HARMONIZED: Dict[str, str] = {
    "nfe": "EUR",
    "fin": "EUR",
    "afr": "AFR",
    "eas": "ASN",   # GTEx "Asian American" cannot separate East/South Asian,
    "sas": "ASN",   # so both collapse to ASN
    "amr": "AMR",
    "asj": "OTHER",
    "ami": "OTHER",
    "mid": "OTHER",
    "remaining": "OTHER",
}

GTEX_TO_HARMONIZED: Dict[str, str] = {
    "EUR_AMERICAN": "EUR",
    "AFR_AMERICAN": "AFR",
    "ASIAN_AMERICAN": "ASN",
    "OTHER_OR_UNREPORTED": "OTHER",
}
# NOTE: 16 GTEx donors self-report Hispanic/Latino ethnicity, but that is an
# ethnicity field overlapping the race categories above, so an "AMR" group
# cannot be reconstructed from the published summary. GTEx therefore has no
# AMR entry in the harmonised space.

UNMAPPED = "UNMAPPED"


def harmonize_population(code: str, crosswalk: Dict[str, str]) -> str:
    """Map a source population code to a harmonised group.

    Unknown codes map to 'UNMAPPED' (never dropped silently), so callers can
    count and report them.
    """
    return crosswalk.get(code, UNMAPPED)


def gtex_weights_over_gnomad(
    asian_to: str = "eas",
    aa_european_fraction: float = 0.0,
) -> Tuple[Dict[str, float], float]:
    """GTEx donor-composition weights expressed over gnomAD ancestry codes.

    This builds the weights for the "GTEx-implied pooled AF": the allele
    frequency a tool would see if it were calibrated on a cohort with GTEx's
    composition but each ancestry group carried gnomAD's frequency.

    Parameters
    ----------
    asian_to : 'eas' | 'sas' | 'split'
        GTEx 'Asian American' cannot be resolved into East/South Asian.
    aa_european_fraction : float in [0, 1]
        Fraction of African American donors' alleles treated as European-
        derived (admixture). 0.0 is a pure label mapping. This is an
        ASSUMPTION knob for sensitivity analysis, not a measured value.

    Returns
    -------
    (weights, excluded_fraction)
        weights sum to 1 over the covered donors (EUR/AFR/Asian);
        excluded_fraction is the share of donors (the 8 'other/unreported')
        that has no gnomAD counterpart and is therefore left out. It is
        returned so callers can disclose it, never silently discarded.
    """
    if asian_to not in ("eas", "sas", "split"):
        raise ValueError("asian_to must be 'eas', 'sas' or 'split'")
    if not 0.0 <= aa_european_fraction <= 1.0:
        raise ValueError("aa_european_fraction must be in [0, 1]")

    d = GTEX_V8_DONORS
    covered = d["EUR_AMERICAN"] + d["AFR_AMERICAN"] + d["ASIAN_AMERICAN"]
    excluded_fraction = d["OTHER_OR_UNREPORTED"] / GTEX_V8_N_GENOTYPED_DONORS

    w_eur = d["EUR_AMERICAN"] / covered
    w_afr = d["AFR_AMERICAN"] / covered
    w_asn = d["ASIAN_AMERICAN"] / covered

    weights: Dict[str, float] = {
        "nfe": w_eur + aa_european_fraction * w_afr,
        "afr": (1 - aa_european_fraction) * w_afr,
        "eas": 0.0,
        "sas": 0.0,
    }
    if asian_to == "eas":
        weights["eas"] = w_asn
    elif asian_to == "sas":
        weights["sas"] = w_asn
    else:
        weights["eas"] = weights["sas"] = w_asn / 2

    weights = {k: v for k, v in weights.items() if v > 0}
    return weights, excluded_fraction

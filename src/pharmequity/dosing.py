"""
dosing.py — maps genes/variants to their documented pharmacogenomic
dosing relevance.

The gene-drug relevance facts below are well-established and already
published by CPIC (Clinical Pharmacogenetics Implementation Consortium)
and PharmGKB — this module does not claim novelty for those facts. What
this module adds is a structured way to attach a population's disparity
metrics (from metrics.py / robustness.py) to a specific, named clinical
consequence, so a report reads as "this frequency gap affects THIS drug
THIS way" rather than a generic disparity number.

Sources for the gene-drug mappings (verify/update before publishing —
these are widely cited standing CPIC guidelines, not primary results
of this project):
  - CPIC guidelines: https://cpicpgx.org/guidelines/
  - PharmGKB clinical annotations: https://www.pharmgkb.org/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DosingRelevance:
    gene: str
    drugs: List[str]
    clinical_action: str
    cpic_guideline_url: str


# NOTE: this is a small, illustrative reference table, not an exhaustive
# CPIC crawl. Extend it deliberately, with a citation per row, rather
# than scraping — wrong gene-drug mappings here are a patient-safety-
# adjacent error, not just a data-quality one.
DOSING_TABLE = {
    "CYP2C19": DosingRelevance(
        gene="CYP2C19",
        drugs=["clopidogrel", "voriconazole", "proton pump inhibitors"],
        clinical_action=(
            "Poor/intermediate metabolizer status reduces clopidogrel "
            "bioactivation (reduced antiplatelet effect) and alters "
            "voriconazole/PPI exposure; CPIC recommends genotype-guided "
            "alternative therapy or dose adjustment."
        ),
        cpic_guideline_url="https://cpicpgx.org/guidelines/guideline-for-clopidogrel-and-cyp2c19/",
    ),
    "CYP2D6": DosingRelevance(
        gene="CYP2D6",
        drugs=["codeine", "tamoxifen", "tricyclic antidepressants"],
        clinical_action=(
            "Poor metabolizers get little analgesic effect from codeine "
            "(fails to convert to morphine); ultrarapid metabolizers risk "
            "toxicity. CPIC recommends avoiding codeine or adjusting dose "
            "based on phenotype."
        ),
        cpic_guideline_url="https://cpicpgx.org/guidelines/guideline-for-codeine-and-cyp2d6/",
    ),
    "VKORC1": DosingRelevance(
        gene="VKORC1",
        drugs=["warfarin"],
        clinical_action=(
            "VKORC1 promoter variants (combined with CYP2C9 genotype) "
            "shift the therapeutic warfarin dose substantially; CPIC "
            "provides a genotype-based starting-dose algorithm."
        ),
        cpic_guideline_url="https://cpicpgx.org/guidelines/guideline-for-warfarin-and-cyp2c9-and-vkorc1/",
    ),
    "DPYD": DosingRelevance(
        gene="DPYD",
        drugs=["fluorouracil", "capecitabine"],
        clinical_action=(
            "Reduced-function DPYD alleles impair fluoropyrimidine "
            "clearance and raise risk of severe/fatal toxicity; CPIC "
            "recommends upfront dose reduction or avoidance."
        ),
        cpic_guideline_url="https://cpicpgx.org/guidelines/guideline-for-fluoropyrimidines-and-dpyd/",
    ),
    "TPMT": DosingRelevance(
        gene="TPMT",
        drugs=["azathioprine", "mercaptopurine", "thioguanine"],
        clinical_action=(
            "Reduced TPMT activity increases thiopurine myelosuppression "
            "risk; CPIC recommends genotype-guided starting dose."
        ),
        cpic_guideline_url="https://cpicpgx.org/guidelines/guideline-for-thiopurines-and-tpmt/",
    ),
    "SLCO1B1": DosingRelevance(
        gene="SLCO1B1",
        drugs=["simvastatin"],
        clinical_action=(
            "Reduced-function SLCO1B1 alleles raise simvastatin plasma "
            "exposure and myopathy risk; CPIC recommends dose limits or "
            "an alternative statin."
        ),
        cpic_guideline_url="https://cpicpgx.org/guidelines/guideline-for-simvastatin-and-slco1b1/",
    ),
    "CYP3A5": DosingRelevance(
        gene="CYP3A5",
        drugs=["tacrolimus"],
        clinical_action=(
            "CYP3A5 expressers (common in African-ancestry populations, "
            "rare in European-ancestry populations) clear tacrolimus "
            "faster and are systematically underdosed by weight-based "
            "protocols calibrated on non-expresser cohorts; CPIC "
            "recommends a higher starting dose for expressers."
        ),
        cpic_guideline_url="https://cpicpgx.org/guidelines/guideline-for-tacrolimus-and-cyp3a5/",
    ),
}


def lookup(gene: str) -> DosingRelevance:
    if gene not in DOSING_TABLE:
        raise KeyError(f"No dosing-relevance entry for gene '{gene}'. Known genes: {list(DOSING_TABLE)}")
    return DOSING_TABLE[gene]

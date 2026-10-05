"""
report.py — standardized per-variant equity report.

One function, one schema, applied identically to every gene/variant so
results are comparable across a whole dataset rather than hand-tuned
per plot. This is the "systematic evaluation" layer: run it over N
variants and you get N directly comparable report dicts / rows, not
N bespoke analyses.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd

from .dosing import lookup as dosing_lookup
from .metrics import disparity_ratio, variance_across_populations, wilson_ci
from .robustness import bootstrap_frequency_ci, detection_floor, reference_mismatch_impact

def _fmt_ci(ci: list) -> str:
    import math
    if any(isinstance(v, float) and math.isnan(v) for v in ci):
        return "n/a (bootstrap undefined at 0%/100%; see Wilson CI)"
    return str(ci)

def build_variant_report(
    gene: str,
    variant_id: str,
    freq_rows: pd.DataFrame,
    calibration_population: str = "nfe",
    zygosity: str = "recessive",
    min_expected_count: int = 5,
) -> Dict:
    """freq_rows must have columns: population, allele_count, allele_number, allele_frequency
    (i.e. the output of data.frequencies_for_variant for one variant).
    """
    freqs = dict(zip(freq_rows["population"], freq_rows["allele_frequency"]))
    dr = disparity_ratio(freqs)
    variance = variance_across_populations(freqs)

    per_population: List[Dict] = []
    for _, row in freq_rows.iterrows():
        pop = row["population"]
        count, n = int(row["allele_count"]), int(row["allele_number"])
        wlo, whi = wilson_ci(count, n)
        boot = bootstrap_frequency_ci(count, n)
        floor = detection_floor(n_alleles=n, observed_frequency=row["allele_frequency"],
                                 min_expected_count=min_expected_count)

        entry = {
            "population": pop,
            "allele_count": count,
            "allele_number": n,
            "allele_frequency": round(float(row["allele_frequency"]), 5),
            "wilson_ci_95": [round(wlo, 5), round(whi, 5)],
            "bootstrap_ci_95": [round(boot["boot_lo"], 5), round(boot["boot_hi"], 5)],
            "below_detection_floor": floor.below_detection_floor,
            "min_detectable_frequency_95": round(floor.min_detectable_frequency_95, 5),
        }

        if pop != calibration_population and calibration_population in freqs:
            mismatch = reference_mismatch_impact(
                calibration_frequency=freqs[calibration_population],
                target_frequency=row["allele_frequency"],
                zygosity=zygosity,
            )
            entry["reference_mismatch_vs_calibration"] = {
                k: (round(v, 6) if v not in (float("inf"),) else None)
                for k, v in mismatch.items()
            }

        per_population.append(entry)

    dosing_info: Optional[Dict] = None
    try:
        rel = dosing_lookup(gene)
        dosing_info = asdict(rel)
    except KeyError:
        pass

    return {
        "gene": gene,
        "variant_id": variant_id,
        "calibration_population": calibration_population,
        "disparity": {
            "max_population": dr.max_population, "max_frequency": round(dr.max_frequency, 5),
            "min_population": dr.min_population, "min_frequency": round(dr.min_frequency, 5),
            "ratio": (round(dr.ratio, 3) if dr.ratio != float("inf") else None),
            "ratio_is_infinite": dr.ratio == float("inf"),
            "absolute_diff": round(dr.absolute_diff, 5),
            "variance_across_populations": round(variance, 8),
        },
        "per_population": per_population,
        "dosing_relevance": dosing_info,
    }


def report_to_markdown(report: Dict) -> str:
    lines = []
    lines.append(f"# {report['gene']} — {report['variant_id']}")
    lines.append("")
    d = report["disparity"]
    ratio_str = "∞ (0% observed in ≥1 population)" if d["ratio_is_infinite"] else f"{d['ratio']}x"
    lines.append(f"- **Max/min disparity ratio:** {ratio_str}")
    lines.append(f"  - Max: {d['max_population']} ({d['max_frequency']})")
    lines.append(f"  - Min: {d['min_population']} ({d['min_frequency']})")
    lines.append(f"- **Absolute frequency difference:** {d['absolute_diff']}")
    lines.append(f"- **Variance across populations:** {d['variance_across_populations']}")
    lines.append("")

    if report["dosing_relevance"]:
        dr = report["dosing_relevance"]
        lines.append(f"**Documented dosing relevance ({', '.join(dr['drugs'])}):** {dr['clinical_action']}")
        lines.append(f"Source: {dr['cpic_guideline_url']}")
        lines.append("")

    lines.append("| Population | AF | Wilson 95% CI | Bootstrap 95% CI | Below detection floor? |")
    lines.append("|---|---|---|---|---|")
    for p in report["per_population"]:
        lines.append(
            f"| {p['population']} | {p['allele_frequency']} | "
            f"{p['wilson_ci_95']} | {_fmt_ci(p['bootstrap_ci_95'])} | "
            f"{'**yes**' if p['below_detection_floor'] else 'no'} |"
        )
    lines.append("")

    flagged = [p for p in report["per_population"] if p["below_detection_floor"]]
    if flagged:
        names = ", ".join(p["population"] for p in flagged)
        lines.append(
            f"⚠️ **{names}** may be underpowered to characterize this variant "
            f"at its observed frequency — a reported 0% or near-0% rate here "
            f"could reflect panel size rather than true absence."
        )
        lines.append("")

    return "\n".join(lines)

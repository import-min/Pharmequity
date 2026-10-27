#!/usr/bin/env python
"""
fetch_gnomad.py — build a Pharmequity frequency CSV from the gnomAD GraphQL API.

    python scripts/fetch_gnomad.py --variants data/pgx_variants_template.csv \
        --dataset gnomad_r4 --out data/my_gnomad_frequencies.csv

STATUS: the response PARSER is unit-tested against a mock payload
(tests/test_fetch_gnomad.py). The LIVE API call has NOT been run by the
author (the development sandbox could not reach gnomad.broadinstitute.org).
Before trusting output, spot-check 2-3 variants against the gnomAD browser.
If the API schema has changed, only QUERY and parse_variant_response need
editing.

Notes
-----
* Input CSV columns: gene, variant_id (gnomAD format chrom-pos-ref-alt,
  matching the chosen genome build/dataset), rsid, label (optional name used
  as the output variant_id).
* Exome and genome counts are SUMMED per ancestry group. For gnomAD v4 the
  exome and genome callsets can contain overlapping individuals; if you need
  de-duplicated joint frequencies, use the browser's joint view and supply
  the table via the CSV loader instead. The script records this in a comment
  header of the output.
* Sex-specific and subset rows (ids containing '_' or XX/XY) are dropped
  explicitly; the ancestry groups kept are the ones present in the response.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

API = "https://gnomad.broadinstitute.org/api"

QUERY = """
query Variant($variantId: String!, $dataset: DatasetId!) {
  variant(variantId: $variantId, dataset: $dataset) {
    variant_id
    rsids
    exome   { populations { id ac an } }
    genome  { populations { id ac an } }
  }
}
"""


def parse_variant_response(payload: Dict) -> Dict[str, Dict[str, int]]:
    """Sum exome+genome (ac, an) per ancestry group from one API response."""
    if payload.get("errors"):
        raise RuntimeError(f"gnomAD API error: {payload['errors']}")
    v = (payload.get("data") or {}).get("variant")
    if v is None:
        raise LookupError("variant not found in this dataset")
    totals: Dict[str, Dict[str, int]] = {}
    for callset in ("exome", "genome"):
        block = v.get(callset)
        if not block:
            continue
        for pop in block.get("populations", []):
            pid = pop["id"]
            if "_" in pid or pid in ("XX", "XY"):   # sex / subset strata
                continue
            t = totals.setdefault(pid, {"ac": 0, "an": 0})
            t["ac"] += int(pop["ac"])
            t["an"] += int(pop["an"])
    if not totals:
        raise ValueError("no ancestry populations in response")
    return totals


def fetch_variant(variant_id: str, dataset: str) -> Dict:
    body = json.dumps({"query": QUERY, "variables": {"variantId": variant_id, "dataset": dataset}}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, type=Path)
    ap.add_argument("--dataset", default="gnomad_r4")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    rows: List[Dict] = []
    with open(a.variants, newline="") as fh:
        for rec in csv.DictReader(fh):
            if not rec.get("variant_id") or rec["variant_id"].startswith("#"):
                continue
            try:
                pops = parse_variant_response(fetch_variant(rec["variant_id"], a.dataset))
            except Exception as e:  # report, never silently skip
                print(f"FAILED {rec['variant_id']}: {e}", file=sys.stderr)
                continue
            name = rec.get("label") or rec["variant_id"]
            for pid, t in sorted(pops.items()):
                rows.append({"gene": rec["gene"], "variant_id": name, "rsid": rec.get("rsid", "NA"),
                             "population": pid, "allele_count": t["ac"], "allele_number": t["an"],
                             "allele_frequency": (t["ac"] / t["an"]) if t["an"] else 0.0})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene", "variant_id", "rsid", "population",
                                           "allele_count", "allele_number", "allele_frequency"])
        w.writeheader()
        w.writerows(rows)
    a.out.with_suffix(".provenance.txt").write_text(
        f"source: gnomAD API, dataset={a.dataset}\nexome+genome allele counts summed per ancestry group "
        "(possible sample overlap in v4; verify against the browser)\n")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()

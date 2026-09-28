from __future__ import annotations

import argparse
from pathlib import Path

from .data import frequencies_for_variant, load_frequency_table
from .report import build_variant_report, report_to_markdown


def main() -> None:
    p = argparse.ArgumentParser(description="Systematic allele-frequency disparity evaluation.")
    p.add_argument("--frequencies", required=True, type=Path,
                    help="CSV with columns: gene,variant_id,rsid,population,allele_count,allele_number,allele_frequency")
    p.add_argument("--out", required=True, type=Path, help="Output directory")
    p.add_argument("--calibration-population", default="nfe",
                    help="Population code treated as the dosing-rule calibration reference.")
    p.add_argument("--zygosity", choices=["recessive", "dominant_or_het"], default="recessive")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_frequency_table(args.frequencies)

    index_lines = ["# Allele Frequency Disparity Report", ""]
    for (gene, variant_id) in df[["gene", "variant_id"]].drop_duplicates().itertuples(index=False):
        rows = frequencies_for_variant(df, variant_id)
        report = build_variant_report(
            gene=gene, variant_id=variant_id, freq_rows=rows,
            calibration_population=args.calibration_population,
            zygosity=args.zygosity,
        )
        md = report_to_markdown(report)
        out_file = args.out / f"{gene}_{variant_id}.md"
        out_file.write_text(md, encoding="utf-8")
        index_lines.append(f"- [{gene} — {variant_id}]({out_file.name})")
        print(f"Wrote {out_file}")

    (args.out / "index.md").write_text("\n".join(index_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

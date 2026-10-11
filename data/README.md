# Data

| File | What it is | Provenance |
|---|---|---|
| `real_gnomad_frequencies.csv` | gnomAD allele counts for CYP2C19 M1V (start-lost), 10 ancestry groups. gnomAD release not recorded in the file. | **real** |
| `gtex_v8_donor_ancestry.csv` | GTEx v8 donor composition (838 genotyped donors). The 8 "other/unreported" are derived by subtraction. | published summary (Science 369:1318, 2020) |
| `pgx_variants_template.csv` | Input template for `scripts/fetch_gnomad.py` (placeholders only). | n/a |

## Canonical schema (every dataset)
`gene, variant_id, rsid, population, allele_count, allele_number, allele_frequency`

## Adding datasets
* **More gnomAD variants:** fill `pgx_variants_template.csv` and run `scripts/fetch_gnomad.py`
  (parser tested; the live API call was not run by the author — spot-check against the browser).
  Or export tables from the gnomAD browser and reshape to the schema above.
* **GTEx:** individual genotypes are controlled-access (dbGaP). Public GTEx gives only pooled MAFs and
  donor composition, so by default GTEx enters via the `gtex_weighted` policy. If you have approved
  access, compute per-ancestry allele counts (e.g. PLINK `--freq counts --within`) with groups named
  `EUR_AMERICAN`, `AFR_AMERICAN`, `ASIAN_AMERICAN`, `OTHER_OR_UNREPORTED` and pass the CSV with
  `--frequencies ... --provenance real`; the crosswalk is inferred from those names.
* Any other cohort (1000 Genomes, All of Us, UK Biobank): same schema; add a crosswalk in
  `ancestry.py` if its group names differ.

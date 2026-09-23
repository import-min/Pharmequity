# pharmequity

A systematic evaluation framework for **allele-frequency disparities across
reference populations** (e.g. gnomAD ancestry groups) and their downstream
effect on **pharmacogenomic dosing logic** (CPIC-guideline gene/drug pairs
like CYP2C19/clopidogrel, CYP3A5/tacrolimus, DPYD/fluoropyrimidines).

This is a *tool*, not a claimed discovery. The gene-drug relevance facts
it uses are established CPIC/PharmGKB guidelines (see `src/pharmequity/dosing.py`
for citations) — the contribution here is a standardized, testable pipeline
for turning a population-frequency table into a disparity report with
honest uncertainty and error-mode handling, run identically across every
variant rather than a one-off analysis per gene.

## ⚠️ About the shipped example data

`data/synthetic_demo_frequencies.csv` is **synthetic placeholder data**,
not real gnomAD data — it exists so the pipeline runs out of the box.
See `data/README.md` for how to plug in a real gnomAD or PharmGKB export.
**Do not report numbers from the demo run as findings.**

## What this does differently from "plot the allele frequencies"

1. **No silent exclusion of extreme groups.** `disparity_ratio()` always
   uses every population passed in. If one population has 0% observed
   frequency, the ratio is reported as infinite — never quietly computed
   on a filtered subset. This is enforced by a dedicated regression test
   (`tests/test_metrics.py::test_no_silent_group_exclusion`).
2. **Every frequency ships with two independent uncertainty estimates**
   (closed-form Wilson interval + resampling bootstrap), because the
   normal ("Wald") approximation people often reach for breaks down
   exactly in the rare-allele / small-panel regime pharmacogenomics
   variants tend to live in.
3. **A detection-floor check, separate from the frequency itself.** A
   population showing 0% for a variant with only a few hundred alleles
   sampled is a different finding than 0% in a panel of 20,000 — the
   report flags when a population's panel is too small to distinguish
   "truly absent" from "just not sampled."
4. **The frequency gap is translated into a clinical-consequence number**,
   not left as an abstract ratio: `reference_mismatch_impact()` computes,
   under Hardy-Weinberg, how wrong a dosing rule's predicted carrier rate
   becomes when calibrated on one population and applied to another.
5. **One schema for every variant.** `build_variant_report()` runs the
   same battery of checks on any gene/variant, so results are directly
   comparable across a whole dataset instead of a bespoke plot per gene.

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
pharmequity-cli --frequencies data/synthetic_demo_frequencies.csv --out out/demo
```

Writes one Markdown report per variant plus an `index.md`, to `out/demo/`.

## Test

```bash
pytest -q
```

## Layout

```
src/pharmequity/
  data.py         # load + validate a population frequency table
  metrics.py      # disparity ratio, Wilson CI, variance, KL divergence
  robustness.py   # detection floor, bootstrap CI, reference-mismatch impact
  dosing.py       # CPIC gene→drug relevance lookup (cited, not novel)
  report.py       # standardized per-variant report (dict + Markdown)
  cli.py          # command-line entry point
tests/            # pytest suite, incl. the anti-exclusion regression test
data/             # synthetic demo table + instructions for real data
```

## Known limitations (read before presenting this)

- The shipped dataset is synthetic. Results are only meaningful once a
  real gnomAD/PharmGKB export is substituted.
- The gene→drug table in `dosing.py` is a small illustrative set, not an
  exhaustive CPIC crawl — extend it deliberately, with a citation per row.
- `reference_mismatch_impact()` assumes Hardy-Weinberg equilibrium and a
  simple recessive/dominant model; real phenotype prediction (e.g. CYP2D6
  metabolizer status) is more nuanced and gene-specific.
- This has not been validated against a real published disparity finding.
  Treat it as a well-tested pipeline you now need to point at real data,
  not as a completed research result.

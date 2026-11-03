# Professor follow-up: minimum 5% Group B approval

The first common passing coefficient is **+0.0003125B**. Group B receives
**383/3,030 loans (12.64%)** without systemic differences and **154/3,030
(5.08%)** with the existing systemic difference, below Group A in both scenarios.
Five candidates were examined; the sixth was not tried. All fitted models and
four control/Gradient Boosting policies remain frozen.

- [Editable 14-slide PowerPoint](delivery/professor_five_percent_penalty.pptx)
- [Matching PDF handout](delivery/professor_five_percent_penalty.pdf)
- [Detailed protocol, results and limitations](../../../docs/professor_five_percent_penalty.md)

Training remains **80,000 applicants**; all decisions, group denominators and
profits use only the **20,000 prediction applicants**. Selecting a penalty from
prediction-group approval counts is a disclosed illustrative policy calibration.
True profit never enters penalty selection.

## Results to communicate

The selected penalty lowers true expected profit versus OLS control by
**$23,312.55** in Scenario 1 and **$354,515.28** in Scenario 2. It increases
profit versus the preserved +0.005B policy by **$3,710,054.65/$2,628,740.43**.
These are synthetic expected values under the same stylized contracts, not
realized bank returns or empirical evidence about actual lenders.

For Gradient Boosting, Scenario 1 OLS is closer to approval parity (absolute gaps
0.155 versus 0.725 pp); Scenario 2 ML is closer (0.073 versus 14.565 pp).
ML profit is materially lower than OLS control: **28.1%/13.3%**, with capital
utilization **72.4%/87.2%**. Approval parity is one descriptive measure. No
quantified decomposition of the profit difference or general equity ranking
is established.

## Compact evidence snapshot

- [Group outcomes](evidence/tables/group_results.csv)
- [Portfolio results](evidence/tables/portfolio_results.csv)
- [Method comparisons](evidence/tables/method_comparisons.csv)
- [Penalty search: all ten candidate/scenario attempts](evidence/tables/penalty_search.csv)
- [Changes versus previous +0.005B](evidence/tables/previous_penalty_comparisons.csv)
- [Training/prediction audit](evidence/tables/split_audit.csv)
- [Four research plots, PNG and vector PDF](evidence/figures/)
- [Snapshot manifest](evidence/manifest.json)
- [Scientific and presentation provenance](evidence/provenance/)

Scientific snapshot copies match the completed run's artifact hashes. The full
applicant-level CSV/Parquet datasets and large cutoff tables remain local under
the repository's existing ignore rules. This snapshot does not replace or alter
original evidence.

## Validation and editability

**73 new tests**, **269 focused/dependency tests**, and **720 professor-model
tests** passed without warnings. Independent rational enumeration reproduces both
selected cutoffs. All previous scientific artifacts and presentations are
preserved. The unrelated full suite was not rerun; four historical V2 lookup
failures remain unresolved and untouched.

All fourteen slides were rendered and inspected. The new content includes four
native charts with embedded workbook snapshots, two method tables and fourteen
speaker-note sections. Slides 9–14 preserve earlier source pages as explicitly
historical images. Chart workbook values use documented six-decimal snapshots.
The PDF is an image-backed appearance-matched handout; use PowerPoint for editing.
No native Microsoft PowerPoint application-opening check is claimed.

The [presentation receipt](evidence/provenance/delivery_validation.json) records
file hashes, package structure, plotted-value checks and matching PDF pages.

## Reproduction

```bash
# Verify existing scientific outputs; no fitting, optimization or writes.
python3 experiments/run_professor_five_percent_penalty.py --summarize-only
```

To rebuild the presentation, use a new revision with the bundled Node/artifact-
tool runtime and bundled Python. Neither command fits models or selects loans:

```bash
FIVE_PERCENT_PENALTY_SLIDE_REVISION=02 /Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node experiments/build_professor_five_percent_penalty_slides.mjs
/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiments/export_professor_five_percent_penalty_pdf.py --revision 02
```

The builder refuses existing revisions. Runtime/reference paths can be changed
through its documented environment variables. Prior studies and deliveries remain
unchanged. Scientific execution performed no commit, push or email; repository
publication is a separate operation.

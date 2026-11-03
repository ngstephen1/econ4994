# Final professor update: smaller penalty

This isolated successor preserves the completed November 2 study and reuses
its fitted models and four unadjusted control/Gradient Boosting policies.
Only the selected manual-penalty policies are recalculated.

- [Editable 14-slide PowerPoint](delivery/professor_small_penalty.pptx)
- [Matching read-only PDF handout](delivery/professor_small_penalty.pdf)

The slides define Group A (B=0) and minority Group B (B=1), and make the split
explicit: **80,000 training applicants; 20,000 prediction applicants; all
reported lending and profit results use prediction applicants only**.
The models retain their original training-only selection and fitting history.

## Penalty selection

Start at +0.005B and follow the predeclared finite halving sequence only if
needed. Choose one common coefficient giving positive Group B lending at a
lower within-group approval rate than A in both risk scenarios. Selection
uses bank-side approval counts, not hidden risk or true profit. This is an
illustrative policy calibration on the prediction cohort, not an untouched
confirmatory comparison. No model refitting, new noise or new population is
needed.

The first candidate, **+0.005B**, satisfied both scenarios; no smaller value
was tried. It adds **0.5 percentage points** to perceived default risk.
Group B receives **3/3,030 loans (0.0990%)** without systemic differences and
**1/3,030 (0.0330%)** with the existing systemic shift. These rates are positive
but very low. True expected portfolio profit falls by **$3,733,367.21** and
**$2,983,255.71**, respectively, relative to the matching OLS controls.

## Presentation

Slides 1–8 use native editable text, a training/prediction workflow, two method
tables and four data charts with embedded workbook snapshots. Slides 9–14
preserve the earlier supplied PDF pages as clearly labeled historical images.
The PDF is an appearance-matched image-backed handout; use the PowerPoint to
edit the new content. All prior presentations and the source PDF remain
unchanged.

The charts use total evaluator true expected funded-loan profit by group.
Group sizes differ, and companion tables include per-eligible-applicant values.
Small positive approval rates show sufficient precision and exact approved
counts. ML performance remains empirical, including any unused capital from
indivisible prediction ties.

## Evidence and validation

The [compact table snapshot](evidence/tables/),
[research figures](evidence/figures/), and
[provenance records](evidence/provenance/) accompany this delivery.
The [snapshot manifest](evidence/manifest.json) binds eight CSV tables, eight
figure files, scientific provenance, presentation checks and speaker notes.
Every scientific copy matches the completed run's artifact hashes.
The detailed [presentation validation receipt](evidence/provenance/delivery_validation.json)
records package structure, numerical source checks, all-slide visual review,
PDF matching, and final file hashes. Charts are editable in the PowerPoint;
the six historical appendix pages retain their original appearance as images.
No native PowerPoint application-opening check is claimed.

The fresh professor-model suite passed **647 tests with no warnings**, including
**63 new tests**; the focused/dependency run passed **297 tests**. The unrelated
full suite was not rerun. Four previously recorded V2 checkpoint lookup failures
remain unresolved. See the [study report](../../../docs/professor_small_penalty.md)
for complete results, information boundaries and validation scope.

## Rebuilding

The builder requires the completed validated local `small_penalty` artifacts
and the original reference PDF. It performs no fitting or lending search:

```bash
# Use the bundled @oai/artifact-tool Node runtime and its dependencies.
SMALL_PENALTY_SLIDE_REVISION=04 node experiments/build_professor_small_penalty_slides.mjs
python3 experiments/export_professor_small_penalty_pdf.py --revision 04
```

Each revision writes new filenames and refuses silent overwrites. Override
`RUNTIME_NODE_MODULES`, `PRESENTATIONS_SKILL_DIR`, `RUNTIME_PYTHON` and
`REFERENCE_PDF` if their original local locations differ.

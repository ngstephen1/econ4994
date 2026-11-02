# November 2 professor follow-up

The delivery contains the 14-slide update for the two matched synthetic
scenarios and three methods:

- [Editable PowerPoint with speaker notes](delivery/professor_nov2_followup.pptx)
- [Matching PDF handout](delivery/professor_nov2_followup.pdf)

Slides 1-8 describe the new +0.01B penalty, one-standard-deviation systemic
epsilon shift, OLS controls, and gradient-boosting regression. The four new
charts are native editable PowerPoint charts, with embedded data workbooks.
Slides 9-14 preserve the supplied earlier pages as explicitly labeled
historical reference images. Their original results were not rerun.

The PDF is an image-backed handout matching the rendered final PowerPoint.
Use the PowerPoint to edit text, tables, the overview diagram, or new charts.
The source PDF and earlier presentations remain unchanged.

## Evidence

The new charts read the validated `group_results.csv` and
`portfolio_results.csv` under
`results/tables/professor_model/nov2_followup/`. Source hashes must match the
completed scientific run manifest before authoring begins.

The Git-tracked [evidence snapshot](evidence/manifest.json) binds these compact
copies to the original completed run without changing the original files:

| Table | Contents |
|---|---|
| [Group results](evidence/tables/group_results.csv) | Twelve scenario/method/group rows: approvals, true profit, and normalization |
| [Portfolio results](evidence/tables/portfolio_results.csv) | Six policy rows: total true/perceived profit, capital use, cutoffs and ties |
| [Penalty comparisons](evidence/tables/penalty_comparisons.csv) | Displaced/replacement approvals and their true-profit contributions |
| [Systemic risk audit](evidence/tables/systemic_risk_audit.csv) | Epsilon, credit-signal and true-risk shifts by group |
| [ML validation](evidence/tables/ml_validation.csv) | The four predeclared candidate settings per world and selected rows |
| [Risk diagnostics](evidence/tables/risk_diagnostics.csv) | Held-out PN errors and evaluator-only true-PD recovery |

| Figure | PNG | Vector PDF |
|---|---|---|
| Scenario 1 approval | [View](evidence/figures/scenario_1_no_systemic_approval_rates.png) | [PDF](evidence/figures/scenario_1_no_systemic_approval_rates.pdf) |
| Scenario 1 true expected profit | [View](evidence/figures/scenario_1_no_systemic_true_profit.png) | [PDF](evidence/figures/scenario_1_no_systemic_true_profit.pdf) |
| Scenario 2 approval | [View](evidence/figures/scenario_2_systemic_approval_rates.png) | [PDF](evidence/figures/scenario_2_systemic_approval_rates.pdf) |
| Scenario 2 true expected profit | [View](evidence/figures/scenario_2_systemic_true_profit.png) | [PDF](evidence/figures/scenario_2_systemic_true_profit.pdf) |

The full generated applicant/decision datasets, models, 76,743-row candidate
table and private build receipts remain local according to existing ignore
rules. This compact snapshot supports manual inspection, not full execution
from a fresh clone without the original inputs and local artifacts.

The figures describe one matched noise realization on the same population,
with the original 80,000/20,000 split, contracts, and capital. Profit bars sum
evaluator true expected profit from funded loans by group. The group sizes
are unequal. ML leaves capital unused when its next indivisible tied-score
block is infeasible; that limitation appears on the result slides.

## Rebuilding

The tracked authoring entry points are:

```bash
# Use the bundled Node executable and its @oai/artifact-tool installation.
# Set RUNTIME_NODE_MODULES, PRESENTATIONS_SKILL_DIR, RUNTIME_PYTHON and
# REFERENCE_PDF when those locations differ from the original environment.
NOV2_SLIDE_REVISION=04 node experiments/build_professor_nov2_slides.mjs
python3 experiments/export_professor_nov2_pdf.py --revision 04
```

The source reference is the supplied `nov_2_followup_slide_update.pptx.pdf`.
The builder requires that PDF and the completed local scientific artifacts.
It performs no simulation, fitting, or lending optimization. Each revision
uses new filenames and refuses to overwrite an existing finalized output.
Review the new renders before replacing the curated delivery copies.

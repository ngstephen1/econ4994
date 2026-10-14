# Professor-Model Progress Report — Eight Slides

| Artifact | Use |
|---|---|
| [oct15_progress.pptx](oct15_progress.pptx) | Editable PowerPoint: native text, diagrams, and chart objects |
| [Matching eight-page PDF](professor_model_progress_8_slides.pdf) | Appearance-matched sharing/printing copy; image-backed at 144 dpi |
| [Contact sheet](contact_sheet.png) | Overview of all eight slides |
| [Evidence manifest](evidence_manifest.json) | Claim-to-artifact references |
| [Verified evidence](verified_evidence.json) | Unrounded values and evaluator-only five-row preview |
| [Delivery validation](delivery_validation.json) | Original rendering, preservation checks, and final hashes |

The author renamed the original generated
`professor_model_progress_8_slides.pptx` to `oct15_progress.pptx`.
Its SHA-256 is unchanged:
`bb0f7545f767d63e7128f7d7d1d9e1f332438e3217354b88aa74402fc2990996`.
Historical validation JSON retains the original output filename rather than
rewriting the delivery record. The PDF keeps its original filename.

## Scope and caveats

The deck summarizes professor-model Steps 1–8B, including the completed
200-noise-replication study. It distinguishes canonical single-noise findings
from the full study. The applicants and true risk are synthetic, and the
experiment conditions on one population. No empirical discrimination claim,
flexible-model comparison, or unrestricted optimality guarantee is implied.

Test counts in the deck are documented results at report creation.
See [current project status](../../../docs/PROJECT_STATUS.md) for the latest
validation, including the four known unrelated V2 checkpoint lookup failures.

## Sources and rebuilds

[Build instructions](BUILD_INSTRUCTIONS.txt) describe the artifact authoring
runtime, scripts, evidence audit, and matching-PDF export route. Source files
are retained here alongside the three scientific chart PNG/SVG pairs, slide
previews, and PDF-page previews. The PDF is not an editable slide deck; the
PowerPoint remains the editable artifact.

Rebuilding requires the authoring runtime and local scientific inputs absent
from a fresh Git clone. Use a new output stem; do not overwrite the reviewed
deck or original validation evidence. Build caches, runtime symlinks,
LibreOffice profiles, and large data ZIPs are intentionally not committed.

The five-row `assets/applicant_preview_EVALUATOR_ONLY.csv` is an intentional
small report excerpt, not a bank training table or full dataset.

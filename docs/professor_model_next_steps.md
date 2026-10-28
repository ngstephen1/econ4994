# Professor Model — Step 10A Roadmap for Professor Review

Date: **2026-10-28**. This is a research roadmap, not authorization to run an
experiment. See the [evidence synthesis](professor_model_evidence_synthesis.md)
for quantitative findings, claim boundaries, source paths and the proposed
canonical-fit preregistration. Step 10B has not been implemented or run.

## Completed work and what it establishes

| Status | Work | Boundary |
|---|---|---|
| Completed: Steps 5–7B | Frozen aware/blind OLS, held-out errors, stylized contracts, capital-constrained common cutoffs, true-risk reference | Canonical aware/blind comparisons include refitting other coefficients; oracle is not unrestricted optimization |
| Completed: Steps 8A–8B | Five pilot draws included in 200 fixed-population noise replications; coefficient/group/profit/tie diagnostics | Conditional eta uncertainty, not 205 draws or new-population evidence |
| Completed: Steps 9A–9C | Revised score branches, matched lending comparison, zero-only positive-floor audit | −0.1B manual adjustment is distinct from fitted beta_B; D relocates, rather than breaks, the tie |
| Completed: Step 10A | Evidence-backed synthesis, scientific-claim audit, traceable evidence index and next-study design | Documentation/read-only audit only; no new estimates or policy outcomes |
| Unresolved, separate engineering issue | Four existing V2 corrected-checkpoint artifact lookup failures | Not professor-model experiment failures; no recovery repair is authorized here |

## Ordered roadmap

1. **Professor review — next decision.** Confirm that the target question is
   the fitted B contribution versus full blind refitting. Acknowledge the
   current conditional-per-period PD, 5%-per-period growing payment schedule,
   PN scaling/noise, manual −10 pp adjustment and indivisible-tie rule.
   If those equations are intended differently, freeze a separate revised
   specification before changing any implementation. Do not retrofit old results.

2. **Proposed Step 10B — canonical fixed-coefficient audit.** Compare the saved
   aware model W, Z formed as `saved_aware_raw - saved_beta_B*B`, and the saved
   full blind refit F. Only Z receives new scores/accounting/cutoff decisions;
   W/F/oracle stay frozen. Keep the aware intercept and Y/D/C coefficients
   fixed without recentering. Apply the existing clipping, accounting, exact
   budget and common-cutoff rules. The detailed preregistration specifies
   Z−W, F−W and F−Z contrasts, raw-score identities, information boundaries,
   outputs and tests. No Z values have been generated in Step 10A.

3. **Proposed follow-up — conditional noise stability.** If the canonical
   intervention is understood and approved, separately specify the same
   coefficient intervention over the existing 200 saved fits. This could
   reuse the same population and draws without new noise or refitting, but
   would still require new accounting/policy evaluations. Do not silently
   extend the canonical study, count pilot reuse twice, or launch it now.

4. **Proposed robustness families — separate scientific choices.** After
   review, prioritize economics/event interpretation, alternative valid
   scoring/clipping/tie/capital policies, and measurement-noise or estimator
   specifications according to the research question. Each changes a
   different mechanism; do not combine them into one unexplained intervention.
   New-population uncertainty requires its own design and resources.

5. **Proposed capstone synthesis and communication.** Develop the paper and
   presentation around prediction versus hidden-risk recovery, perceived versus
   true economic value, and budget-mediated group allocation. Dashboard updates,
   flexible-model comparisons and eventual HMDA application remain separately
   scoped work. Synthetic evidence does not automatically transfer to real
   lending or establish causal discrimination.

No unsupported runtime or completion-date estimates are assigned.

## Go/no-go conditions for proposed Step 10B

| Gate | Go condition | No-go / response |
|---|---|---|
| Scientific contrast | Professor/researcher accepts canonical coefficient removal with all other aware coefficients and intercept frozen; separate implementation request exists | If intended treatment is instead −0.1B, coefficient sign reversal, recentering, or a new fit, revise the design before execution |
| Accounting interpretation | Current contracts are explicitly accepted as conditional stylized assumptions for this mechanism audit | If realism or different period/rate/payment semantics are required, write a separate model specification; do not alter frozen economics |
| Inputs and provenance | Original fit/score/split/contract/budget/decision hashes match, canonical references are reproducible from their saved identities | Stop on missing/mutated inputs; preserve evidence rather than regenerate or relabel it |
| Estimator boundary | No refit or new target; only the exact saved B contribution changes; B=0 raw scores remain identical | Reject any new intercept, other-slope change, scale correction, tuned clipping or hidden-truth input |
| Decision integrity | Only Z optimized; same exact capital, full requests, common cutoff, indivisible ties, reject-all and tie rule; independent candidate audit passes | Stop on budget slack/guard changes, skipping, split ties, hidden oracle input or desired-result tuning |
| Evaluator separation | Z artifacts saved/frozen before truth or oracle access; truth used only for final evaluation | Stop on leakage or outcome-based treatment/cutoff selection |
| Validation and reporting | Focused deterministic tests, signed metrics, null handling, complete artifact hashes, preservation and interpretation limits pass | Do not manufacture a desired gap/profit sign or present conditional results as real-world causality |

**Current next-experiment status: proposed, awaiting review and a separate
implementation instruction.** Documentation completion does not authorize
Step 10B. The unrelated four V2 lookup failures remain disclosed; they neither
invalidate the preserved professor evidence automatically nor justify claiming
the full repository is green.

## Reuse and engineering cautions

Use the existing saved coefficient/prediction artifacts, bank-only profit API,
exact cutoff search, and post-freeze evaluator patterns. No public API or
scientific implementation changes are needed for Step 10A. Future experiment
files must have a separate namespace and may not overwrite Steps 5–9C.

The Step 8B runner fingerprints a full source-directory inventory. Eight later
9A–9C modules enlarge that inventory even though every originally recorded
source file still matches. A current-tree old-CLI resume can therefore refuse
its identity check. This is a static compatibility finding, not a failed
replication or permission to bypass the guard. If reuse across saved fits is
later requested, specify provenance-safe adoption using the recorded source
set, and validate saved records without changing old manifests. Do not run
resume or summary CLIs merely to inspect results.

The latest documented Step 9C scientific suite is 437 professor tests passed,
and 717 passed / 4 failed / 32 warnings repository-wide. Step 10A uses only
targeted read-only evidence and documentation checks. No scientific suite,
experiment, figure generation, staging, commit or push is part of this roadmap.

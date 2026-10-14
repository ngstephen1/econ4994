# Professor Model — Step 2 Applicant Generation

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

Step 2 implements only synthetic applicant characteristics and the noisy credit
signal. The canonical run contains **100,000 applicants**, using master seed
**4994**. All structural and reproducibility checks passed. Default-risk
components, PN/eta, model fitting, and lending remain later steps.

## Purpose and distributions

This population makes the professor's working distribution assumptions
inspectable before risk estimation or economic allocation. One row represents
one applicant, and the shared `applicant_id` joins bank and simulator views.
The supplied equations are working transcriptions, not empirical calibration.

| Variable | Generation | Interpretation |
|---|---|---|
| `log_Y` | Normal(mean=11, SD=0.7) | Natural log of annual income in synthetic dollars. |
| `Y` | `exp(log_Y)` | Positive annual income; not a Normal draw in income levels. |
| `D` | Normal(mean=0.35, SD=0.16) | DTI in ratio units; untruncated, with no clipping or redrawing. |
| `B` | Bernoulli(p=0.15) | 0 is reference, 1 is comparison; counts are sampled, not forced. |
| `epsilon` | Normal(mean=0, SD=1) | Hidden borrower factor, unavailable in the bank view. |
| `omega` | Normal(mean=0, SD=0.2) | Independent credit-signal noise, unavailable in the bank view. |
| `C` | `0.4 * epsilon + omega` | Continuous observed credit/risk signal; not a conventional credit score. |

The second Normal parameter always means **standard deviation**. The working
independence assumption is explicit: log_Y, D, B, epsilon, and omega are drawn
independently. C has only its declared dependence on epsilon and omega. No
additional V2 correlations, group shifts, or latent factors are used.

All numerical parameters and paths are configurable in
`configs/professor_model/baseline.yaml`. The canonical configuration records
the requested values unchanged.

## Implementation and reproducibility

Reusable code lives in `src/fair_lending/professor_model/`:

- `config.py`: YAML validation, schema contracts, and configuration hashing.
- `population.py`: vectorized NumPy draws, separate information views, preview,
  and CSV/Parquet exports.
- `validation.py`: structural identities, distribution summaries, group audit,
  and tail warnings.

`experiments/generate_professor_population.py` orchestrates one generation and
checks regeneration before saving. It uses `SeedSequence(4994).spawn(5)` in
the fixed order `log_Y, D, B, epsilon, omega`, with a separate PCG64 generator
for each child. Stream order is part of `professor-population-v1`; changing
group probability does not change the other draws.

The saved metadata contains the exact resolved configuration and SHA-256
fingerprint, schema, seed, stream spawn keys, generator name, actual source
hashes including uncommitted code, Git revision/dirty status, Python and
dependency versions, output checksums, and information roles. CSV exports
were read back with round-trip float parsing and matched Parquet values.
Regeneration from saved configuration matched both primary datasets exactly.

From the repository root:

```bash
python3 experiments/generate_professor_population.py
```

Existing outputs require an explicit overwrite flag. To regenerate this same
baseline after restarting:

```bash
python3 experiments/generate_professor_population.py --overwrite
```

`--rows`, `--seed`, and `--config` are optional overrides. The resolved settings
are saved with the output; canonical YAML is never silently rewritten. The
configured output paths must remain inside professor-model namespaces.

## Data schema and exact locations

The base directory is:

```text
/Users/macbook/Hack/econ4994/data/synthetic/professor_model/baseline/
```

| File | Rows | Columns / role |
|---|---:|---|
| `applicants_bank_view.csv` | 100,000 | Exactly `applicant_id, Y, D, B, C`; bank-observed variables. |
| `applicants_bank_view.parquet` | 100,000 | Same bank schema and values, preserving data types. |
| `applicants_simulator_private.csv` | 100,000 | Exactly `applicant_id, log_Y, epsilon, omega`; simulator/evaluator information. |
| `applicants_simulator_private.parquet` | 100,000 | Same private schema and values, preserving data types. |
| `applicants_audit_preview.csv` | 150 | `information_scope, applicant_id, Y, log_Y, D, B, epsilon, omega, C`. |
| `population_metadata.json` | One manifest | Resolved configuration, seeds, source provenance, and output checksums. |

Every preview row has
`information_scope = EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING`. It contains the
first 150 applicants for manual inspection; it is not a representative or
group-balanced analysis sample. The bank-view allowlist excludes log_Y,
epsilon, omega, and all future truth, outcome, and risk-estimation fields.
Although log_Y is mathematically derivable from observed Y, it is stored only
in the private table to follow the requested persisted schema.

The complete numerical report and group table are:

```text
/Users/macbook/Hack/econ4994/results/metrics/professor_model/population_validation.json
/Users/macbook/Hack/econ4994/results/tables/professor_model/population_group_comparison.csv
```

The report and group table are evaluator artifacts and include private-variable
diagnostics. Generated datasets and numerical results remain ignored by Git
under the repository's existing policy.

## Observed validation results

The run has 100,000 unique IDs in each primary table, identical ID sets,
positive income, binary group values, and no missing or infinite values.
Both the exponential income identity and credit-signal identity passed with
floating-point tolerances. Repeating the same configuration/seed reproduced
both tables exactly; seed 4995 changed both tables. Different-seed draws used
for this check were not persisted as a second research population.

All SD figures below are sample standard deviations (`ddof=1`).

| Variable | Mean | Median | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| Y ($/year) | 76,547.80 | 60,082.88 | 61,063.82 | 2,202.47 | 1,481,292.72 |
| log_Y | 10.999955 | 11.003480 | 0.701210 | 7.697335 | 14.208426 |
| D | 0.350487 | 0.351091 | 0.159973 | -0.356832 | 1.009353 |
| epsilon | 0.003299 | 0.003870 | 0.997868 | -4.392724 | 3.924996 |
| omega | 0.000271 | 0.000632 | 0.199336 | -0.852582 | 0.870303 |
| C | 0.001590 | 0.001197 | 0.446312 | -2.118867 | 1.888165 |

| Income percentile | Annual income ($) |
|---|---:|
| p01 | 11,494.86 |
| p05 | 18,888.06 |
| p25 | 37,277.85 |
| p50 | 60,082.88 |
| p75 | 96,065.62 |
| p95 | 189,954.29 |
| p99 | 307,318.75 |

Observed `corr(C, epsilon)` is **0.894719**, and `corr(C, omega)` is
**0.447421**. Under the specified independent draws, the corresponding
population values are approximately 0.894427 and 0.447214; population SD(C)
is approximately 0.447214. These are mathematical benchmarks, not fitted
model results.

The largest observed absolute correlation among primitive variables was
0.007610. Small empirical correlations and group differences are expected
under independent sampling; structural stream/group-invariance tests provide
the direct check that extra dependencies were not introduced.

### Group comparison

| Group | Applicants | Share | Mean Y ($/year) | Mean D | Mean epsilon | Mean C |
|---|---:|---:|---:|---:|---:|---:|
| B=0 | 84,930 | 84.93% | 76,507.70 | 0.350621 | 0.001841 | 0.001033 |
| B=1 | 15,070 | 15.07% | 76,773.81 | 0.349733 | 0.011514 | 0.004731 |
| B=1 minus B=0 | — | — | +266.11 | -0.000888 | +0.009673 | +0.003698 |

No group equality constraint or group-specific financial effect was applied.
These realized differences are sampling differences in a group-independent
population, not evidence of an inserted discrimination mechanism.

### Warnings and limitations

- **D < 0:** 1,489 applicants, or **1.4890%**. Negative DTI is economically
  problematic but allowed by the requested untruncated Normal. Values remain
  unchanged so the working specification can be discussed with the professor.
- **D > 1:** one applicant, or **0.0010%**. A debt burden above income can occur
  conceptually; this tail value is retained and flagged for review.
- There are **zero structural validation errors**. The two DTI warnings are
  reported separately and do not invalidate the mathematically specified run.
- The income distribution has an unbounded right tail and an observed range
  of $2,202.47–$1,481,292.72 per year. No empirical representativeness claim or
  truncation is imposed.
- C is a continuous noisy latent-risk signal, not a conventional credit score.
  Its direction and scale have been preserved.
- Independence is a simple experimental assumption, not a description of real
  borrower finances. Group labels are abstract synthetic categories.

## Tests and next step

`tests/test_professor_population.py` covers configuration and SD interpretation,
invalid settings, exact schemas/counts/IDs, valid groups, income and C
identities, missing/infinite data, stream isolation and group invariance,
reproducibility, distribution moments with predeclared broad tolerances,
unclipped DTI tails, group audits, evaluator preview labels, export round trips,
source/output hashes, overwrite protection, and failure-before-export behavior.
The focused suite passed **45 tests**.

The complete repository suite finished with **325 passed, 4 failed, and 32
existing scikit-learn/SciPy deprecation warnings**. All four failures are in
`tests/test_systemic_corrected_checkpoint.py` and concern existing V2 artifact
lookup rather than professor-population tests. The saved corrected successes
use revision `8a9f71f`, while the checkpoint loader includes current HEAD
`7a7e590` in its run ID and therefore looks for different filenames. Read-only
inspection found all five corrected success records, identical current/saved
scientific source and dependency fingerprints, and successful integrity checks
under each record's saved identity. No recovery code or records were modified,
and no old experiments were rerun to conceal the lookup failures.

Step 3 should implement only the exact professor true-risk components and
true_PD in a separate evaluator/private artifact, with identity/bounds tests
and distribution diagnostics. The risk-event horizon still needs clarification
before multi-period economic accounting. PN, bank regression, threshold
lending, and profit allocation should follow in their separately scoped steps.

# Professor Model — Step 4 Noisy Risk Measurement

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

Step 4 adds a raw measurement for the existing **100,000 applicants** and saved
true-risk values. The inputs were read from Parquet, joined one-to-one by ID,
and their shapes, columns, and first 10 joined rows printed before generation.
Applicants and true risk were not regenerated. All 31 registered original
data/configuration/source artifacts retained their checksums after export.

## Equation, units, and random stream

```text
eta ~ Normal(mean=0, standard_deviation=0.1)
PN = 0.8 * true_PD + eta
```

All arithmetic uses **raw probability units**. A true_PD of 0.0014 represents
0.14%, whereas noise SD 0.1 represents **10 percentage points**. PN is an
unbounded noisy measurement, not a true probability or a realized default
indicator. Negative values are permitted by the equation; no transformation
or calibration to [0,1] was applied. The factor 0.8 is retained.

The separate `configs/professor_model/noisy_signal.yaml` declares the settings.
The population and true-risk configurations/metadata remain unchanged.
For eta, the implementation constructs:

```python
child = numpy.random.SeedSequence(entropy=4994, spawn_key=(5,))
rng = numpy.random.Generator(numpy.random.PCG64(child))
eta = rng.normal(loc=0.0, scale=0.1, size=100000)
```

This is exactly equivalent to child 5 from `SeedSequence(4994).spawn(6)`,
while avoiding dependence on a previously advanced generator. Population
children [0]–[4] are neither reused nor modified. One independent eta draw is
attached to each applicant in ascending **lexicographic string applicant-ID
order**. This order is deterministic for the zero-padded existing IDs.
Group membership and row order do not change the ID-to-noise mapping.

The raw draws are not recentered, clipped, redrawn, or rescaled. Zero-noise
tests supply a zero vector to the equation helper; they do not change the
canonical SD. The required toy example, true_PD=0.05 and eta=-0.10, returns
PN=-0.06.

## Files and information boundary

The base directory is:

```text
/Users/macbook/Hack/econ4994/data/synthetic/professor_model/baseline/
```

| File | Rows | Exact schema / role |
|---|---:|---|
| `training_signals.csv` | 100,000 | `applicant_id, PN`; future bank regression target only. |
| `training_signals.parquet` | 100,000 | Same target schema and values. |
| `risk_measurement_noise_private.csv` | 100,000 | `applicant_id, eta`; simulator/evaluator only. |
| `risk_measurement_noise_private.parquet` | 100,000 | Same private schema and values. |
| `noisy_signal_audit_preview_EVALUATOR_ONLY.csv` | 150 | `information_scope, applicant_id, Y, D, B, C, true_PD, eta, PN`. |
| `noisy_signal_metadata.json` | One manifest | Resolved settings, stream identity/ID-order hash, input/output checksums, source/configuration fingerprints, dependencies, and information roles. |

The new preview uses the exact same 150 IDs and order as Step 3. Every row
carries `EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING`. The target table excludes
eta, epsilon, true_PD, and both true-risk components. It should eventually be
joined to bank-observed predictors by ID, without joining evaluator/private
files into the bank's feature matrix.

Exports retain numerical precision. Rounding occurs only in printed displays
and this documentation. CSV read back with round-trip float parsing matches
Parquet and the original generated floating-point values exactly.

Additional evaluator reports are:

```text
results/metrics/professor_model/noisy_signal_validation.json
results/tables/professor_model/noisy_signal_summary.csv
results/tables/professor_model/noisy_signal_group_comparison.csv
```

The summary CSV has one row per eta/PN/scaled-truth quantity; the group CSV
has one row per B category. The JSON includes correlations, range counts,
mean identity, noise/signal ratio, and preservation checks.

## Actual distributions

These figures use raw probability units and sample SD (`ddof=1`):

| Quantity | Mean | Median | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| eta | 0.000361539 | 0.000246206 | 0.100108158 | -0.449832029 | 0.435346528 |
| PN | 0.001499200 | 0.001369507 | 0.100108677 | -0.448083054 | 0.436024244 |
| 0.8 * true_PD | 0.001137661 | 0.001022592 | 0.000548081 | 0.000129305 | 0.007552224 |

| Percentile | eta | PN | 0.8 * true_PD |
|---|---:|---:|---:|
| p01 | -0.234532802 | -0.233344628 | 0.000353224 |
| p05 | -0.164686309 | -0.163476337 | 0.000483625 |
| p25 | -0.067357831 | -0.066190226 | 0.000753948 |
| p50 | 0.000246206 | 0.001369507 | 0.001022592 |
| p75 | 0.067941092 | 0.069128475 | 0.001394480 |
| p95 | 0.164767261 | 0.165914203 | 0.002173369 |
| p99 | 0.232827964 | 0.233933641 | 0.002992623 |

| PN range | Applicants | Percentage |
|---|---:|---:|
| Below 0 | 49,453 | 49.453% |
| Within [0,1] | 50,547 | 50.547% |
| Above 1 | 0 | 0% |

Negative PN values were retained; they are not negative true default
probabilities. The absence of values above 1 in this sample does not make
the Gaussian measurement equation bounded.

## Signal/noise comparison and group audit

- `corr(PN, true_PD) = 0.003683797`.
- `corr(eta, true_PD) = -0.001791069`, a small finite-sample correlation despite
  independent generation.
- `sample_SD(eta) / sample_SD(0.8 * true_PD) = 182.652237`.

The sample means satisfy:

```text
mean(PN) = 0.001499199590167
         = 0.001137660682935 + 0.000361538907232
         = 0.8 * mean(true_PD) + mean(eta)
```

The numerical residual is approximately -3.25e-19. The realized eta mean is
not forced to zero, so mean(PN) need not equal the scaled truth mean exactly.

| Group | Applicants | Mean eta | Mean PN | Mean 0.8 * true_PD |
|---|---:|---:|---:|---:|
| B=0 | 84,930 | 0.000372652 | 0.001510203 | 0.001137551 |
| B=1 | 15,070 | 0.000298908 | 0.001437188 | 0.001138281 |

Both groups receive the same noise distribution and an ID-based draw. The
observed differences are sampling diagnostics, not inserted group treatments.
The large noise/signal ratio and tiny individual correlation do **not** imply
that a future regression cannot learn systematic patterns. With independent
mean-zero noise, conditional measurement means retain the 0.8-scaled signal;
estimation precision and predictive performance require the later fitting step.

## Figures, implementation, and validation

Only two new figures were created and visually inspected:

1. `results/figures/professor_model/pn_histogram.png`: all PN values, linear
   horizontal axis, zero marked, negative measurements and full tails retained.
2. `results/figures/professor_model/pn_vs_true_pd.png`: all 100,000 observations
   aggregated into hexagonal counts; both data axes linear in raw probability
   units. Log color scaling indicates counts only, not a log measurement axis.

Reusable code is in `noisy_signal.py`, `signal_io.py`, and `signal_figures.py`.
The runner is `experiments/generate_professor_noisy_signal.py`:

```bash
python3 experiments/generate_professor_noisy_signal.py
# Deliberately regenerate only the Step 4 outputs, if they already exist:
python3 experiments/generate_professor_noisy_signal.py --overwrite
```

Validation passed exact counts and schemas, unique/matching IDs, finite and
nonmissing values, the raw equation and sample-mean identity, exact reserved
RNG draws, same-input/configuration reproducibility, CSV/Parquet agreement,
unchanged original artifacts, and the shared Step 3 preview IDs. Negative PN
and the noise scale are interpretation notices, not structural errors.

`tests/test_professor_noisy_signal.py` adds **35 tests**, including the negative
hand example, zero-noise toy, above-one raw measurements, reserved-child
equivalence, row-order stability, changed noise seed, B invariance, no noise
recentering/reuse, exact schemas, invalid joins/outputs, group calculations,
export/checksum preservation, and assertions that existing generation functions
are not called. All **112 professor-model tests passed without warnings**.

The complete suite finished with **392 passed, 4 failed, and 32 existing
deprecation warnings** in 100.08 seconds. There are no new failures. The four
unchanged failures are in `tests/test_systemic_corrected_checkpoint.py`, where
current-HEAD run IDs do not locate the older saved V2 checkpoint filenames.
They remain visible; unrelated recovery code and records were not modified.

Step 4 ends here. Bank regression, bank predictions, realized default outcomes,
lending, profit, flexible ML comparisons, and Monte Carlo remain unimplemented
for this professor-model family.

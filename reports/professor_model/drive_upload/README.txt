ECON 4994: Professor-model CSV sharing bundle, through Step 8B

This bundle copies existing files without regeneration, fitting, or rounding.
All applicant records are synthetic. No HMDA or real borrower data is included.
Repository-relative folder paths are preserved inside the ZIP.

START HERE
data/synthetic/professor_model/baseline/applicants_bank_view.csv
  100,000 applicants: applicant_id, Y (annual income), D (DTI ratio),
  B (abstract binary group), C (noisy credit signal).

data/synthetic/professor_model/baseline/simulation_truth.csv
  Hidden risk components and true_PD. EVALUATOR ONLY.

data/synthetic/professor_model/baseline/training_signals.csv
  Raw continuous noisy PN measurements, including negatives.
  These are regression targets, not observed default outcomes.

data/processed/professor_model/loan_contracts.csv
  Contracts for the 20,000 saved evaluation applicants.

data/processed/professor_model/lending_decisions.csv
data/processed/professor_model/lending_decisions_group_blind.csv
data/processed/professor_model/oracle_lending_decisions.csv
  Canonical group-aware, group-blind, and evaluator-oracle funding decisions.

results/tables/professor_model/group_aware_vs_blind_portfolio.csv
  The original single-noise-draw policy comparison, not Monte Carlo averages.

results/tables/professor_model/noise_robustness_full/noise_full_results.csv
results/tables/professor_model/noise_robustness_full/noise_full_summary.csv
  Latest completed 200-noise-replication study: 400 policy rows, 200 pairs.
  The result table repeats paired metrics on both policy rows. For paired
  analysis, use one observation per replication, not 400 independent pairs.

INFORMATION BOUNDARY
applicants_simulator_private.csv, risk_measurement_noise_private.csv,
simulation_truth.csv, true_expected_profit.csv, oracle decisions, and
evaluator/audit previews contain evaluator-only information. Do not merge
hidden truth or noise into lender prediction features.

WHAT ELSE IS INCLUDED
All current professor-model dataset CSVs and result-table CSVs, including
clearly named small previews, historical stage summaries, and Step 8A pilot
tables. Data metadata and Step 8B manifest/completion/reference JSONs preserve
context. FILE_MANIFEST.json records exact paths, row counts, column names,
file sizes, and SHA-256 hashes for every copied source file.

The 200 Monte Carlo draws do not have 200 separate raw applicant CSVs.
The population was fixed; the study saved compact replication results instead.
This professor-model family has no realized default or repayment-history table.

UPLOAD
The current local archive was renamed by the author to csv_bundle.zip.
Upload csv_bundle.zip to Google Drive to share the whole set.
The packaging script's default output name is professor_model_csv_bundle.zip;
this name difference does not change the preserved bundle contents.
ZIP archives stay local and are excluded from Git. This README and the file
manifest are committed; a fresh clone does not contain the datasets or ZIP.
For individual CSV preview/import, extract the ZIP locally and upload its
folders/files. Google Sheets does not directly import the ZIP as a worksheet.

PARQUET
Canonical Parquet files remain unchanged in the repository. They are omitted
from this CSV-only sharing bundle to avoid duplicate datasets. Metadata may
reference those original Parquet files and scientific source paths that are
not bundled here; this is a sharing package, not the complete runnable repo.

"""Run the deterministic machine-learning benchmark across synthetic worlds."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from fair_lending.models.evaluation import calibration_bins, evaluate_oracle, evaluate_predictions
from fair_lending.models.features import (
    RACE_AWARE,
    RACE_BLIND,
    get_feature_spec,
    select_features,
)
from fair_lending.models.group_audit import black_white_disparities, group_prediction_metrics
from fair_lending.models.training import (
    MODEL_LABELS,
    MODEL_ORDER,
    deterministic_split,
    tune_on_validation,
)
from fair_lending.simulation.calibration import load_calibration_artifact
from fair_lending.simulation.config import (
    PROJECT_ROOT,
    git_revision,
    package_version,
    resolve_simulation_config,
    stable_fingerprint,
)
from fair_lending.simulation.generator import (
    dataset_filename,
    generate_synthetic_data,
    save_synthetic_dataset,
)
from fair_lending.simulation.validation import validate_generated_data


SCENARIOS = (
    "fair_baseline",
    "direct_discrimination",
    "upstream_inequality",
    "mixed_mechanism",
)
SCENARIO_LABELS = {
    "fair_baseline": "Fair baseline",
    "direct_discrimination": "Direct discrimination",
    "upstream_inequality": "Upstream inequality",
    "mixed_mechanism": "Mixed mechanism",
}
REGIMES = (RACE_BLIND, RACE_AWARE)
REGIME_LABELS = {
    RACE_BLIND: "Race-blind",
    RACE_AWARE: "Race-aware sensitivity",
}
N_ROWS = 100_000
SEED = 4_994
EFFECT_LEVEL = "moderate"
THRESHOLD = 0.50
COLORS = {
    "observed": "#861F41",
    RACE_BLIND: "#159A9C",
    RACE_AWARE: "#E87722",
    "logistic_regression": "#3A506B",
    "random_forest": "#7083A5",
    "hist_gradient_boosting": "#159A9C",
}


def _load_experiment_data(scenario: str, intercept: float) -> tuple[pd.DataFrame, dict]:
    """Reuse Prompt 6 data when valid or regenerate it without calibration."""
    filename = dataset_filename(scenario, EFFECT_LEVEL, N_ROWS, SEED)
    data_path = PROJECT_ROOT / "data" / "synthetic" / filename
    metadata_path = data_path.with_suffix(".metadata.json")
    expected_config = resolve_simulation_config(scenario, EFFECT_LEVEL, N_ROWS, SEED)
    expected_fingerprint = stable_fingerprint(expected_config)

    if data_path.exists() and metadata_path.exists():
        data = pd.read_parquet(data_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["config_fingerprint"] != expected_fingerprint:
            raise RuntimeError(f"Existing dataset config drift for {scenario}")
        if float(metadata["intercept"]) != intercept:
            raise RuntimeError(f"Existing dataset intercept drift for {scenario}")
    else:
        data, metadata = generate_synthetic_data(
            scenario, EFFECT_LEVEL, N_ROWS, SEED, intercept=intercept
        )
        _, _, metadata = save_synthetic_dataset(data, metadata, data_path)

    validation = validate_generated_data(data, metadata)
    if not validation["valid"] or validation["warnings"]:
        raise RuntimeError(
            f"Generator validation failed for {scenario}: "
            f"errors={validation['errors']}, warnings={validation['warnings']}"
        )
    return data, metadata


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _performance_figure(benchmark: pd.DataFrame, path: Path) -> None:
    frame = benchmark.loc[benchmark["feature_regime"] == RACE_BLIND]
    x = np.arange(len(SCENARIOS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for offset, model in enumerate(MODEL_ORDER):
        values = frame.loc[frame["model"] == model].set_index("scenario").loc[
            list(SCENARIOS), "brier_score"
        ]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
        )
    ax.set_xticks(x, [SCENARIO_LABELS[item] for item in SCENARIOS], rotation=16, ha="right")
    ax.set_ylabel("Test Brier score (lower is better)")
    ax.set_title("Race-blind probability performance by model and scenario")
    ax.legend(frameon=False, ncols=3, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _gap_figure(benchmark: pd.DataFrame, main_model: str, path: Path) -> None:
    frame = benchmark.loc[benchmark["model"] == main_model]
    blind = frame.loc[frame["feature_regime"] == RACE_BLIND].set_index("scenario")
    aware = frame.loc[frame["feature_regime"] == RACE_AWARE].set_index("scenario")
    x = np.arange(len(SCENARIOS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(
        x - width,
        100.0 * blind.loc[list(SCENARIOS), "observed_black_white_gap"],
        width,
        label="Observed label gap",
        color=COLORS["observed"],
    )
    ax.bar(
        x,
        100.0 * blind.loc[list(SCENARIOS), "predicted_probability_gap"],
        width,
        label="Race-blind mean-prediction gap",
        color=COLORS[RACE_BLIND],
    )
    ax.bar(
        x + width,
        100.0 * aware.loc[list(SCENARIOS), "predicted_probability_gap"],
        width,
        label="Race-aware mean-prediction gap",
        color=COLORS[RACE_AWARE],
    )
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(x, [SCENARIO_LABELS[item] for item in SCENARIOS], rotation=16, ha="right")
    ax.set_ylabel("Black − White gap (percentage points)")
    ax.set_title(f"Observed and predicted gaps: {MODEL_LABELS[main_model]}")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _regime_comparison_figure(comparison: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    for ax, scenario in zip(
        axes, ("direct_discrimination", "mixed_mechanism"), strict=True
    ):
        frame = comparison.loc[comparison["scenario"] == scenario].set_index("model")
        x = np.arange(len(MODEL_ORDER))
        width = 0.32
        ax.bar(
            x - width / 2,
            100.0 * frame.loc[list(MODEL_ORDER), "race_blind_predicted_gap"],
            width,
            label="Race-blind",
            color=COLORS[RACE_BLIND],
        )
        ax.bar(
            x + width / 2,
            100.0 * frame.loc[list(MODEL_ORDER), "race_aware_predicted_gap"],
            width,
            label="Race-aware sensitivity",
            color=COLORS[RACE_AWARE],
        )
        ax.axhline(
            100.0 * frame["observed_gap"].iloc[0],
            color=COLORS["observed"],
            linestyle="--",
            linewidth=1.2,
            label="Observed label gap",
        )
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(x, ["Logit", "RF", "HistGB"])
        ax.set_title(SCENARIO_LABELS[scenario])
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].set_ylabel("Mean-prediction gap (percentage points)")
    fig.legend(handles, labels, frameon=False, loc="lower center", ncols=3)
    fig.suptitle("Access to race changes reproduced disparity")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _true_recovery_figure(bins: pd.DataFrame, main_model: str, path: Path) -> None:
    frame = bins.loc[
        (bins["feature_regime"] == RACE_BLIND) & (bins["model"] == main_model)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5), sharex=True, sharey=True)
    for ax, scenario in zip(axes.flat, SCENARIOS, strict=True):
        values = frame.loc[frame["scenario"] == scenario]
        ax.plot(
            values["mean_predicted_probability"],
            values["mean_true_probability"],
            marker="o",
            color=COLORS[RACE_BLIND],
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.0)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    fig.supxlabel("Mean predicted probability")
    fig.supylabel("Mean synthetic true probability")
    fig.suptitle(
        f"Race-blind recovery of synthetic true probability: {MODEL_LABELS[main_model]}"
    )
    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _group_calibration_figure(group_bins: pd.DataFrame, main_model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for race, color in (("White", "#7083A5"), ("Black", "#E87722")):
        values = group_bins.loc[group_bins["race"] == race]
        ax.plot(
            values["mean_predicted_probability"],
            values["observed_approval_rate"],
            marker="o",
            label=race,
            color=color,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed approval rate")
    ax.set_title(
        "Group reliability in mixed mechanism\n"
        f"{MODEL_LABELS[main_model]}, race-aware sensitivity"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_experiment() -> dict[str, Path]:
    """Run all controlled model/regime/scenario combinations."""
    tables = PROJECT_ROOT / "results" / "tables"
    figures = PROJECT_ROOT / "results" / "figures"
    metrics = PROJECT_ROOT / "results" / "metrics"
    for directory in (tables, figures, metrics):
        directory.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration_artifact()
    intercept = float(calibration["intercept"])
    benchmark_rows: list[dict] = []
    group_rows: list[dict] = []
    selection_rows: list[dict] = []
    selected_rows: list[dict] = []
    calibration_rows: list[dict] = []
    oracle_rows: list[dict] = []
    split_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    prediction_cache: dict[tuple[str, str, str], tuple[pd.DataFrame, np.ndarray]] = {}
    shared_ids: dict[str, list[str]] | None = None

    for scenario in SCENARIOS:
        data, metadata = _load_experiment_data(scenario, intercept)
        split = deterministic_split(data, SEED)
        current_ids = {
            name: list(getattr(split, name)["application_id"])
            for name in ("train", "validation", "test")
        }
        if shared_ids is None:
            shared_ids = current_ids
        elif shared_ids != current_ids:
            raise RuntimeError("Application-level split assignments differ by scenario")
        split_rows.append(
            {
                "scenario": scenario,
                "n_train": len(split.train),
                "n_validation": len(split.validation),
                "n_test": len(split.test),
                "train_prevalence": float(split.train["approved"].mean()),
                "validation_prevalence": float(split.validation["approved"].mean()),
                "test_prevalence": float(split.test["approved"].mean()),
                "config_fingerprint": metadata["config_fingerprint"],
            }
        )
        oracle_rows.append({"scenario": scenario, **evaluate_oracle(split.test)})

        for regime in REGIMES:
            for model_name in MODEL_ORDER:
                selected = tune_on_validation(split, regime, model_name, SEED)
                probability = selected["pipeline"].predict_proba(
                    select_features(split.test, regime)
                )[:, 1]
                overall = evaluate_predictions(
                    split.test["approved"].to_numpy(dtype=int),
                    probability,
                    split.test["approval_probability_true"].to_numpy(dtype=float),
                    THRESHOLD,
                )
                audit = group_prediction_metrics(split.test, probability, THRESHOLD)
                disparities = black_white_disparities(audit)
                common = {
                    "scenario": scenario,
                    "feature_regime": regime,
                    "feature_regime_label": REGIME_LABELS[regime],
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                }
                benchmark_rows.append(
                    {
                        **common,
                        "n_train": len(split.train),
                        "n_validation": len(split.validation),
                        "n_test": len(split.test),
                        "selected_hyperparameters": json.dumps(
                            selected["selected_parameters"], sort_keys=True
                        ),
                        **overall,
                        **disparities,
                    }
                )
                group_rows.extend(audit.assign(**common).to_dict("records"))
                for record in selected["selection_records"]:
                    selection_rows.append(
                        {
                            **common,
                            "candidate_index": record["candidate_index"],
                            "parameters": json.dumps(record["parameters"], sort_keys=True),
                            "validation_log_loss": record["validation_log_loss"],
                            "validation_brier_score": record["validation_brier_score"],
                            "validation_roc_auc": record["validation_roc_auc"],
                            "selected": record["selected"],
                        }
                    )
                selected_rows.append(
                    {
                        **common,
                        "selected_hyperparameters": json.dumps(
                            selected["selected_parameters"], sort_keys=True
                        ),
                        "selection_metric": selected["selection_metric"],
                    }
                )
                binned = calibration_bins(
                    split.test["approved"],
                    probability,
                    split.test["approval_probability_true"],
                ).assign(**common)
                calibration_rows.extend(binned.to_dict("records"))
                prediction_cache[(scenario, regime, model_name)] = (split.test, probability)
                prediction_frames.append(
                    split.test.loc[
                        :,
                        [
                            "application_id",
                            "race",
                            "approved",
                            "approval_probability_true",
                        ],
                    ]
                    .assign(
                        predicted_probability=probability,
                        scenario=scenario,
                        feature_regime=regime,
                        model=model_name,
                    )
                    .reset_index(drop=True)
                )

    benchmark = pd.DataFrame(benchmark_rows)
    group_audit = pd.DataFrame(group_rows)
    selection = pd.DataFrame(selection_rows)
    selected_parameters = pd.DataFrame(selected_rows)
    calibration_table = pd.DataFrame(calibration_rows)
    oracle = pd.DataFrame(oracle_rows)
    split_table = pd.DataFrame(split_rows)
    test_predictions = pd.concat(prediction_frames, ignore_index=True)

    validation_summary = (
        selection.loc[selection["selected"]]
        .groupby("model", as_index=False)
        .agg(
            mean_validation_log_loss=("validation_log_loss", "mean"),
            mean_validation_brier_score=("validation_brier_score", "mean"),
        )
        .sort_values(["mean_validation_log_loss", "mean_validation_brier_score", "model"])
    )
    main_model = str(validation_summary.iloc[0]["model"])

    comparisons = []
    for scenario in ("direct_discrimination", "mixed_mechanism"):
        for model_name in MODEL_ORDER:
            subset = benchmark.loc[
                (benchmark["scenario"] == scenario) & (benchmark["model"] == model_name)
            ].set_index("feature_regime")
            blind = subset.loc[RACE_BLIND]
            aware = subset.loc[RACE_AWARE]
            comparisons.append(
                {
                    "scenario": scenario,
                    "model": model_name,
                    "race_blind_predicted_gap": blind["predicted_probability_gap"],
                    "race_aware_predicted_gap": aware["predicted_probability_gap"],
                    "observed_gap": blind["observed_black_white_gap"],
                    "true_probability_gap": blind["true_probability_black_white_gap"],
                    "race_blind_reproduction_error": blind[
                        "probability_gap_reproduction_error"
                    ],
                    "race_aware_reproduction_error": aware[
                        "probability_gap_reproduction_error"
                    ],
                    "race_aware_minus_blind_gap": aware["predicted_probability_gap"]
                    - blind["predicted_probability_gap"],
                }
            )
    comparison = pd.DataFrame(comparisons)

    mixed_test, mixed_probability = prediction_cache[
        ("mixed_mechanism", RACE_AWARE, main_model)
    ]
    group_calibration_frames = []
    for race in ("White", "Black"):
        mask = mixed_test["race"].astype(object).eq(race).to_numpy()
        group_calibration_frames.append(
            calibration_bins(
                mixed_test.loc[mask, "approved"],
                mixed_probability[mask],
                mixed_test.loc[mask, "approval_probability_true"],
            ).assign(
                scenario="mixed_mechanism",
                feature_regime=RACE_AWARE,
                model=main_model,
                race=race,
            )
        )
    group_calibration = pd.concat(group_calibration_frames, ignore_index=True)

    statistical = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "statistical_recovery.csv")
    statistical = statistical.loc[
        statistical["model"] == "model_2",
        [
            "scenario",
            "black_coefficient",
            "adjusted_probability_gap",
            "true_direct_log_odds",
            "true_direct_probability_gap",
        ],
    ]
    ml_main = benchmark.loc[
        benchmark["model"] == main_model,
        ["scenario", "feature_regime", "predicted_probability_gap", "observed_black_white_gap"],
    ].pivot(index="scenario", columns="feature_regime")
    ml_main.columns = [f"{metric}__{regime}" for metric, regime in ml_main.columns]
    statistical_comparison = statistical.merge(
        ml_main.reset_index(), on="scenario", how="left", validate="one_to_one"
    )
    statistical_comparison["main_ml_model"] = main_model

    paths = {
        "benchmark": tables / "ml_benchmark.csv",
        "group_audit": tables / "ml_group_audit.csv",
        "selection": tables / "ml_hyperparameter_selection.csv",
        "selected_parameters": tables / "ml_selected_hyperparameters.csv",
        "regime_comparison": tables / "ml_race_regime_comparison.csv",
        "oracle": tables / "ml_oracle_benchmark.csv",
        "calibration": tables / "ml_calibration_bins.csv",
        "group_calibration": tables / "ml_group_calibration_bins.csv",
        "split": tables / "ml_split_summary.csv",
        "statistical_comparison": tables / "statistical_ml_comparison.csv",
        "figure_performance": figures / "ml_brier_performance.png",
        "figure_gaps": figures / "ml_observed_predicted_gaps.png",
        "figure_regimes": figures / "ml_race_regime_comparison.png",
        "figure_true_recovery": figures / "ml_true_probability_recovery.png",
        "figure_group_calibration": figures / "ml_group_calibration.png",
        "metadata": metrics / "ml_benchmark_metadata.json",
        "test_predictions": metrics / "ml_test_predictions.parquet",
    }
    for frame, key in (
        (benchmark, "benchmark"),
        (group_audit, "group_audit"),
        (selection, "selection"),
        (selected_parameters, "selected_parameters"),
        (comparison, "regime_comparison"),
        (oracle, "oracle"),
        (calibration_table, "calibration"),
        (group_calibration, "group_calibration"),
        (split_table, "split"),
        (statistical_comparison, "statistical_comparison"),
    ):
        frame.to_csv(paths[key], index=False)
    test_predictions.to_parquet(paths["test_predictions"], index=False, engine="pyarrow")

    _configure_plot_style()
    _performance_figure(benchmark, paths["figure_performance"])
    _gap_figure(benchmark, main_model, paths["figure_gaps"])
    _regime_comparison_figure(comparison, paths["figure_regimes"])
    _true_recovery_figure(calibration_table, main_model, paths["figure_true_recovery"])
    _group_calibration_figure(group_calibration, main_model, paths["figure_group_calibration"])

    revision, dirty = git_revision()
    metadata = {
        "experiment": "first_machine_learning_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": list(SCENARIOS),
        "feature_regimes": list(REGIMES),
        "models": list(MODEL_ORDER),
        "effect_level": EFFECT_LEVEL,
        "rows_per_scenario": N_ROWS,
        "split_seed": SEED,
        "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "threshold": THRESHOLD,
        "intercept": intercept,
        "intercept_calibration_fingerprint": calibration["config_fingerprint"],
        "hyperparameter_selection": "minimum validation log loss; test unseen",
        "final_fit_partition": "training only",
        "main_model": main_model,
        "main_model_selection": "lowest mean selected-candidate validation log loss",
        "feature_sets": {
            regime: list(get_feature_spec(regime).columns) for regime in REGIMES
        },
        "proxy_regime": "deferred",
        "package_version": package_version(),
        "sklearn_version": sklearn.__version__,
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "artifacts": {
            name: str(path.resolve()) for name, path in paths.items() if name != "metadata"
        },
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def main() -> None:
    paths = run_experiment()
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

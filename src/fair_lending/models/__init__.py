"""Machine-learning benchmark features, training, and evaluation."""

from fair_lending.models.evaluation import (
    calibration_bins,
    evaluate_oracle,
    evaluate_predictions,
    true_probability_recovery,
)
from fair_lending.models.features import (
    RACE_AWARE,
    RACE_BLIND,
    get_feature_spec,
    select_features,
)
from fair_lending.models.group_audit import (
    black_white_disparities,
    group_prediction_metrics,
)
from fair_lending.models.training import (
    MODEL_ORDER,
    deterministic_split,
    tune_on_validation,
)

__all__ = [
    "MODEL_ORDER",
    "RACE_AWARE",
    "RACE_BLIND",
    "black_white_disparities",
    "calibration_bins",
    "deterministic_split",
    "evaluate_oracle",
    "evaluate_predictions",
    "get_feature_spec",
    "group_prediction_metrics",
    "select_features",
    "true_probability_recovery",
    "tune_on_validation",
]

"""Prediction models, validation, persistence and drift detection."""

from signalforge.models.conditional import (
    ConditionalEdge,
    ConditionSlice,
    build_conditional_edge,
    session_bucket,
)
from signalforge.models.drift import DriftReport, assess, population_stability_index
from signalforge.models.ensemble import CLASSES, ModelReport, SignalModel, wilson_interval
from signalforge.models.registry import ModelEntry, ModelRegistry
from signalforge.models.significance import (
    SignificanceResult,
    assess_batch,
    benjamini_hochberg,
    binomial_p_value,
    describe_batch,
    minimum_accuracy_for_significance,
)
from signalforge.models.validation import (
    Fold,
    PurgedWalkForward,
    check_no_leakage,
    walk_forward_predict,
)

__all__ = [
    "SignalModel",
    "ModelReport",
    "CLASSES",
    "wilson_interval",
    "ModelRegistry",
    "ModelEntry",
    "ConditionalEdge",
    "ConditionSlice",
    "build_conditional_edge",
    "session_bucket",
    "SignificanceResult",
    "assess_batch",
    "benjamini_hochberg",
    "binomial_p_value",
    "describe_batch",
    "minimum_accuracy_for_significance",
    "PurgedWalkForward",
    "Fold",
    "walk_forward_predict",
    "check_no_leakage",
    "DriftReport",
    "assess",
    "population_stability_index",
]

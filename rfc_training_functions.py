"""Reusable RFC baseline training helpers for Azure Databricks Runtime 15.4 LTS.

No Databricks widgets, Spark actions, MLflow runs or model registration at import.
Feature groups must be supplied by the caller from an approved, versioned contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


@dataclass(frozen=True)
class RFCFeatureContract:
    """Mutually exclusive feature groups selected from training data only."""

    binary_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    discount_column: str = "policy_org_discount_per"
    discount_missing_column: str = "policy_org_discount_per_missing"

    def __post_init__(self) -> None:
        groups = (self.binary_columns, self.categorical_columns, self.numeric_columns)
        flattened = [column for group in groups for column in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Feature groups contain duplicate columns")
        if self.discount_column not in self.numeric_columns:
            raise ValueError("Discount column must be included in numeric_columns")
        if self.discount_missing_column in flattened:
            raise ValueError("Derived missingness indicator must not be in source feature groups")
        if "target" in flattened or "policy_id" in flattened or "effective_dttm" in flattened:
            raise ValueError("Target or split identifiers must not be model features")

    @property
    def numeric_inputs(self) -> list[str]:
        return list(self.binary_columns + self.numeric_columns) + [self.discount_missing_column]

    @property
    def feature_columns(self) -> list[str]:
        return list(self.binary_columns + self.categorical_columns + self.numeric_columns) + [
            self.discount_missing_column
        ]


def prepare_rfc_features(df: DataFrame, contract: RFCFeatureContract) -> DataFrame:
    """Apply notebook-equivalent Spark transformations to any dataset split."""
    required = set(contract.binary_columns + contract.categorical_columns + contract.numeric_columns)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing RFC source features: {missing}")
    for column in contract.binary_columns:
        normalized = F.upper(F.trim(F.col(column)))
        df = df.withColumn(
            column,
            F.when(normalized == "Y", F.lit(1.0))
            .when(normalized == "N", F.lit(0.0))
            .otherwise(F.lit(None).cast("double")),
        )
    df = df.withColumn(contract.discount_column, F.expr(
        f"try_cast(`{contract.discount_column}` AS DOUBLE)"
    ))
    df = df.withColumn(
        contract.discount_missing_column,
        F.when(F.col(contract.discount_column).isNull(), F.lit(1.0)).otherwise(F.lit(0.0)),
    )
    return df


def to_rfc_pandas(df: DataFrame, contract: RFCFeatureContract) -> tuple[pd.DataFrame, pd.Series]:
    """Collect selected features to driver; caller must manage driver memory."""
    selected = df.select(*contract.feature_columns, "target").toPandas()
    x = selected[contract.feature_columns].copy()
    for column in contract.numeric_inputs:
        x[column] = pd.to_numeric(x[column], errors="coerce").astype("float32")
    y = selected["target"].astype("int8")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")
    return x, y


def build_rfc_pipeline(
    contract: RFCFeatureContract,
    *,
    model_params: dict | None = None,
) -> Pipeline:
    """Construct unfitted preprocessing + XGBoost baseline pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
             list(contract.categorical_columns)),
            ("numeric", "passthrough", contract.numeric_inputs),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    defaults = {
        "objective": "binary:logistic",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": 16,
    }
    if model_params:
        defaults.update(model_params)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(**defaults)),
    ])


def correct_rfc_probabilities(
    raw_probabilities: Sequence[float] | np.ndarray,
    *,
    negative_sampling_rate: float,
) -> np.ndarray:
    """Correct posterior odds when positives retained and negatives sampled uniformly.

    Assumes sampling is independent of features conditional on the label.
    This is not a substitute for checking calibration on natural-prevalence validation.
    """
    if not np.isfinite(negative_sampling_rate) or not 0 < negative_sampling_rate <= 1:
        raise ValueError("negative_sampling_rate must be in (0, 1]")
    p = np.asarray(raw_probabilities, dtype=np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must be finite and between 0 and 1")
    return negative_sampling_rate * p / (1.0 - p + negative_sampling_rate * p)


def evaluate_rfc_validation(
    y_true: Sequence[int] | np.ndarray,
    raw_probabilities: Sequence[float] | np.ndarray,
    *,
    negative_sampling_rate: float,
    top_percentages: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Evaluate on natural-prevalence validation, never select using test labels."""
    y = np.asarray(y_true)
    p = np.asarray(raw_probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Expected non-empty equal-length 1D targets and predictions")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Validation target must contain both binary classes")
    corrected = correct_rfc_probabilities(p, negative_sampling_rate=negative_sampling_rate)
    prevalence = float(y.mean())
    result = {
        "validation_rows": float(len(y)),
        "validation_prevalence": prevalence,
        "validation_roc_auc": float(roc_auc_score(y, p)),
        "validation_average_precision": float(average_precision_score(y, p)),
        "validation_brier_corrected": float(brier_score_loss(y, corrected)),
        "validation_logloss_corrected": float(log_loss(y, corrected, labels=[0, 1])),
    }
    order = np.argsort(-p, kind="stable")
    ranked_labels = y[order]
    for pct in top_percentages:
        if not 0 < pct <= 100:
            raise ValueError("Top percentages must be in (0, 100]")
        k = int(np.ceil(len(y) * pct / 100.0))
        result[f"validation_lift_top_{pct}pct"] = float(ranked_labels[:k].mean() / prevalence)
        result[f"validation_recall_top_{pct}pct"] = float(ranked_labels[:k].sum() / y.sum())
    return result

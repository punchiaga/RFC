# Databricks notebook source
# MAGIC %md
# MAGIC # RFC | XGBoost full training, Optuna HPO and SHAP
# MAGIC Reads **existing deterministic RFC Delta splits**; never recreates or overwrites them.
# MAGIC Validation selects HPO parameters. Holdout test is optional and evaluated only after selection.
# MAGIC SHAP explains the final selected model and logs plots to its MLflow run.

# COMMAND ----------

# MAGIC %pip install xgboost==2.1.4 scikit-learn==1.3.0 optuna==4.5.0 shap==0.49.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration and persisted RFC tables

# COMMAND ----------

import gc
import time
from pathlib import Path
from importlib.metadata import version
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import optuna
import shap
from matplotlib import pyplot as plt
from scipy import sparse
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, NumericType, StringType, TimestampType
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

WIDGET_DEFAULTS = {
    "batch_process_id": "20261119034822",
    "ml_catalog": "bdp4_prod_ml_build",
    "ml_schema": "hi_customer_right_fit_cover_data",
    "negative_sample_pct": "5",
    "collection_batches": "10",
    "evaluation_batches": "24",
    "run_baseline": "true",
    "run_hpo": "true",
    "hpo_trial_count": "30",
    "hpo_timeout_minutes": "0",
    "hpo_early_stopping_rounds": "50",
    "hpo_n_jobs": "16",
    "score_test_after_hpo": "false",
    "run_shap": "true",
    "shap_sample_rows": "1000",
    "shap_dependence_top_n": "5",
    "mlflow_experiment": "/Users/nadeep.silva@bupa.com.au/RFC2026/rfc_experiments_hpo",
}
for key, default in WIDGET_DEFAULTS.items():
    dbutils.widgets.text(key, default)

def widget_bool(key):
    value = dbutils.widgets.get(key).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{key} must be true or false")
    return value == "true"

batch_process_id = dbutils.widgets.get("batch_process_id").strip()
ml_catalog = dbutils.widgets.get("ml_catalog").strip()
ml_schema = dbutils.widgets.get("ml_schema").strip()
negative_sample_pct = int(dbutils.widgets.get("negative_sample_pct"))
collection_batches = int(dbutils.widgets.get("collection_batches"))
evaluation_batches = int(dbutils.widgets.get("evaluation_batches"))
run_baseline = widget_bool("run_baseline")
run_hpo = widget_bool("run_hpo")
hpo_trial_count = int(dbutils.widgets.get("hpo_trial_count"))
hpo_timeout_minutes = int(dbutils.widgets.get("hpo_timeout_minutes"))
hpo_early_stopping_rounds = int(dbutils.widgets.get("hpo_early_stopping_rounds"))
hpo_n_jobs = int(dbutils.widgets.get("hpo_n_jobs"))
score_test_after_hpo = widget_bool("score_test_after_hpo")
run_shap = widget_bool("run_shap")
shap_sample_rows = int(dbutils.widgets.get("shap_sample_rows"))
shap_dependence_top_n = int(dbutils.widgets.get("shap_dependence_top_n"))
mlflow_experiment = dbutils.widgets.get("mlflow_experiment").strip()
assert batch_process_id and ml_catalog and ml_schema and mlflow_experiment
assert negative_sample_pct in {5, 10, 20, 100}
assert min(collection_batches, evaluation_batches, hpo_trial_count,
           hpo_early_stopping_rounds, hpo_n_jobs, shap_sample_rows) > 0
assert hpo_timeout_minutes >= 0 and shap_dependence_top_n >= 0
assert run_baseline or run_hpo, "Enable at least baseline or HPO"

RANDOM_SEED = 42
RAW_THRESHOLD = 0.5
negative_sampling_rate = negative_sample_pct / 100.0
prefix = f"{ml_catalog}.{ml_schema}.rfc_training_{batch_process_id}"
train_table = f"{prefix}_train_sampled"
validation_table = f"{prefix}_validation"
test_table = f"{prefix}_test"
required_tables = [train_table, validation_table] + ([test_table] if score_test_after_hpo else [])
for table_name in required_tables:
    assert spark.catalog.tableExists(table_name), f"Missing persisted Delta table: {table_name}"
sampled_train_df = spark.table(train_table)
sampled_validation_df = spark.table(validation_table)
# Test is deliberately not read until after HPO model selection.
mlflow.set_experiment(mlflow_experiment)
mlflow.autolog(disable=True)
print("Train:", train_table, "Validation:", validation_table)
print("Package versions:", {p: version(p) for p in ["xgboost", "scikit-learn", "optuna", "shap"]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RFC feature contract and preprocessing
# MAGIC Uses the feature exclusions, Y/N detection, numeric conversion and one-hot rules from the uploaded 7.4 notebook.

# COMMAND ----------

RFC_EXCLUDE_COLUMNS = [
    "policy_id", "effective_dttm", "target", "target_first_event_date",
    "target_first_source", "days_from_snapshot_to_first_event",
    "policy_latest_coverchange_date", "policy_package_code_prev",
    "policy_package_code", "policy_package_cd", "policy_package_name_prev",
    "policy_package_name", "policy_package_dsc", "blr_first_login_date",
    "top3_lifetime_last_order_date", "blr_recent_event_date",
    "top1_lifetime_last_order_date", "top2_lifetime_last_order_date",
    "policy_date_paid_to", "policy_latest_coverchange_dtls",
    "policy_last_payment_date", "top1_24m_last_order_date",
    "top3_24m_last_order_date", "top2_24m_last_order_date",
    "policy_latest_call_date", "policy_is_hi_suspension_fh_start_date",
    "policy_is_hi_suspension_fh_end_date", "policy_future_org_id",
    "policy_future_org_name", "policy_future_org_discount_per",
    "policy_overseas_type", "policy_hosp_prod_desc",
    "policy_is_ancl_prod_desc", "policy_member_financial_status",
    "policy_portfolio_dsc", "policy_ambulance_product_id",
    "policy_ambulance_product_rank", "policy_fund_no", "state",
    "MEMBERSHIP_STATUS_SUSPENDED", "policy_ancillary_product_id",
    "policy_extras_product_id", "policy_hospital_product_id",
    "policy_national_pharmacy_product_id", "policy_pharmacy_product_id",
    "policy_future_org_from_date", "policy_future_org_to_date",
    "policy_ancillary_product_id_prev",
    "policy_hospital_product_id_prev",
    "policy_national_pharm_product_id_prev",
]
RFC_HIGH_CARDINALITY_COLUMNS = [
    "blr_recent_order_item", "policy_latest_call_app_tag", "top1_24m_order_item",
    "top1_lifetime_order_item", "top2_24m_order_item", "top2_lifetime_order_item",
    "top3_24m_order_item", "top3_lifetime_order_item",
]
RFC_NUMERIC_STRING_COLUMNS = ["policy_org_discount_per"]
RFC_ADDITIONAL_BINARY_COLUMNS = ["Hosp_group_OverseasHospitalOnsale"]
RFC_EXCLUDED_PRODUCT_ID_COLUMNS = [
    "policy_national_pharmacy_product_id", "policy_pharmacy_product_id"
]
RFC_RAW_DATE_COLUMNS = ["policy_org_from_date", "policy_org_to_date"]

rfc_train_df = sampled_train_df
candidate_features = [
    column for column in rfc_train_df.columns if column not in RFC_EXCLUDE_COLUMNS
]
schema_types = {field.name: field.dataType for field in rfc_train_df.schema.fields}
temporal_columns = [
    column
    for column in candidate_features
    if isinstance(schema_types[column], (DateType, TimestampType))
]
model_candidate_features = [
    column for column in candidate_features if column not in temporal_columns
]
categorical_columns = [
    column
    for column in model_candidate_features
    if isinstance(schema_types[column], StringType)
]
numeric_columns = [
    column
    for column in model_candidate_features
    if isinstance(schema_types[column], NumericType)
]
unsupported_columns = sorted(
    set(model_candidate_features) - set(categorical_columns) - set(numeric_columns)
)
assert not unsupported_columns, f"Unsupported model feature types: {unsupported_columns}"

# Apply the proven Y/N classification logic to the full sampled training data.
# This is feature-contract preparation, not a prototype model run.
binary_profile_expressions = []
for column in categorical_columns:
    normalised = F.upper(F.trim(F.col(column)))
    binary_profile_expressions.extend([
        F.max(F.when(normalised.isNotNull() & ~normalised.isin("Y", "N"), 1).otherwise(0)).alias(
            f"{column}__has_other"
        ),
        F.max(F.when(normalised == "Y", 1).otherwise(0)).alias(f"{column}__has_y"),
        F.max(F.when(normalised == "N", 1).otherwise(0)).alias(f"{column}__has_n"),
    ])

binary_profile = (
    rfc_train_df.agg(*binary_profile_expressions).first().asDict()
    if binary_profile_expressions else {}
)
binary_candidate_columns = [
    column
    for column in categorical_columns
    if binary_profile.get(f"{column}__has_other", 0) == 0
    and (
        binary_profile.get(f"{column}__has_y", 0) == 1
        or binary_profile.get(f"{column}__has_n", 0) == 1
    )
]

binary_columns = sorted(
    set(binary_candidate_columns)
    | (set(RFC_ADDITIONAL_BINARY_COLUMNS) & set(categorical_columns))
)
onehot_columns = sorted(
    set(categorical_columns)
    - set(binary_columns)
    - set(RFC_HIGH_CARDINALITY_COLUMNS)
    - set(RFC_NUMERIC_STRING_COLUMNS)
)
numeric_model_columns = sorted(
    (set(numeric_columns) | set(RFC_NUMERIC_STRING_COLUMNS))
    - set(RFC_EXCLUDED_PRODUCT_ID_COLUMNS)
    - set(RFC_RAW_DATE_COLUMNS)
)
numeric_pipeline_columns = (
    binary_columns + numeric_model_columns + ["policy_org_discount_per_missing"]
)
active_feature_columns = (
    binary_columns
    + onehot_columns
    + numeric_model_columns
    + ["policy_org_discount_per_missing"]
)
assert len(active_feature_columns) == len(set(active_feature_columns))
assert active_feature_columns

print("Binary features:", len(binary_columns))
print("One-hot features:", len(onehot_columns))
print("Numeric features:", len(numeric_model_columns) + 1)
print("Total active features:", len(active_feature_columns))
print("Deferred high-cardinality features:", len(RFC_HIGH_CARDINALITY_COLUMNS))


def prepare_rfc_features(df: DataFrame) -> DataFrame:
    prepared_df = df
    for column in binary_columns:
        value = F.upper(F.trim(F.col(column)))
        prepared_df = prepared_df.withColumn(
            column,
            F.when(value == "Y", F.lit(1.0))
            .when(value == "N", F.lit(0.0))
            .otherwise(F.lit(None).cast("double")),
        )
    prepared_df = prepared_df.withColumn(
        "policy_org_discount_per",
        F.expr("try_cast(policy_org_discount_per AS DOUBLE)"),
    ).withColumn(
        "policy_org_discount_per_missing",
        F.when(F.col("policy_org_discount_per").isNull(), F.lit(1.0)).otherwise(F.lit(0.0)),
    )
    return prepared_df


prepared_schema = prepare_rfc_features(rfc_train_df).limit(0).columns
missing_features = set(active_feature_columns) - set(prepared_schema)
assert not missing_features, f"Missing prepared model features: {sorted(missing_features)}"

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            onehot_columns,
        ),
        ("numeric", "passthrough", numeric_pipeline_columns),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Collect the full sampled training dataset
# MAGIC Deterministic batches only control Spark-to-pandas collection; they do not resample rows.
# MAGIC All collected batches are concatenated in driver memory.

# COMMAND ----------

def add_batch_id(df: DataFrame, batch_column: str, batch_count: int) -> DataFrame:
    return df.withColumn(
        batch_column,
        F.pmod(
            F.xxhash64(
                F.col("policy_id"),
                F.col("effective_dttm"),
                F.lit(RANDOM_SEED),
            ),
            F.lit(batch_count),
        ),
    )


def prepare_pandas_batch(df: DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    batch_pdf = (
        prepare_rfc_features(df)
        .select(*active_feature_columns, "target")
        .toPandas()
    )
    X_batch = batch_pdf[active_feature_columns].copy()
    y_batch = batch_pdf["target"].to_numpy(dtype=np.int8)
    for column in numeric_pipeline_columns:
        X_batch[column] = pd.to_numeric(X_batch[column], errors="coerce").astype(np.float32)
    remaining_object_columns = set(X_batch.select_dtypes(include=["object"]).columns)
    assert remaining_object_columns == set(onehot_columns), (
        "Unexpected object columns: "
        f"{sorted(remaining_object_columns - set(onehot_columns))}"
    )
    return X_batch, y_batch


batched_train_df = add_batch_id(rfc_train_df, "_collection_batch", collection_batches)
X_train_parts = []
y_train_parts = []
for batch_id in range(collection_batches):
    X_part, y_part = prepare_pandas_batch(
        batched_train_df.filter(F.col("_collection_batch") == batch_id)
    )
    X_train_parts.append(X_part)
    y_train_parts.append(y_part)
    print(f"Training batch {batch_id + 1}/{collection_batches}: {len(y_part):,} rows")

X_train = pd.concat(X_train_parts, ignore_index=True)
y_train = np.concatenate(y_train_parts)
del X_train_parts, y_train_parts
gc.collect()

persisted_train_counts = target_counts(rfc_train_df)

assert len(y_train) == sum(persisted_train_counts.values())
assert int((y_train == 1).sum()) == persisted_train_counts.get(1, 0)
assert list(X_train.columns) == active_feature_columns

print("Full sampled Train matrix:", X_train.shape)
print("Target counts:", persisted_train_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluation functions and baseline model

# COMMAND ----------

model = Pipeline([
    ("preprocessor", clone(preprocessor)),
    (
        "classifier",
        XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=42,
            n_jobs=16,
        ),
    ),
])

METRIC_ORDER = [
    "ROC-AUC",
    "PR-AUC",
    "Precision @ raw 0.5",
    "Recall @ raw 0.5",
    "F1 @ raw 0.5",
    "Lift @ Top 1%",
    "Lift @ Top 5%",
    "Lift @ Top 10%",
    "Lift @ Top 20%",
    "Corrected Brier",
    "Corrected Log Loss",
]


def correct_probabilities(raw_scores: np.ndarray, negative_sampling_rate: float) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if not 0 < negative_sampling_rate <= 1:
        raise ValueError("negative_sampling_rate must be in (0, 1].")
    if not np.isfinite(raw_scores).all():
        raise ValueError("Probabilities must be finite.")
    if np.any((raw_scores < 0) | (raw_scores > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")
    return (
        negative_sampling_rate * raw_scores
        / (1.0 - raw_scores + negative_sampling_rate * raw_scores)
    )


def calculate_lift(y_true: np.ndarray, scores: np.ndarray, pct: int) -> float:
    prevalence = float(np.mean(y_true))
    if prevalence <= 0:
        raise ValueError("Lift requires positive observations.")
    top_count = max(1, int(np.ceil(len(y_true) * pct / 100)))
    top_indices = np.argsort(-scores, kind="stable")[:top_count]
    return float(np.mean(y_true[top_indices]) / prevalence)


def evaluate_scores(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    negative_sampling_rate: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    assert set(np.unique(y_true)) == {0, 1}, "Evaluation requires both target classes"
    predictions = (raw_scores >= RAW_THRESHOLD).astype(np.int8)
    corrected_scores = correct_probabilities(raw_scores, negative_sampling_rate)
    return {
        "ROC-AUC": roc_auc_score(y_true, raw_scores),
        "PR-AUC": average_precision_score(y_true, raw_scores),
        "Precision @ raw 0.5": precision_score(y_true, predictions, zero_division=0),
        "Recall @ raw 0.5": recall_score(y_true, predictions, zero_division=0),
        "F1 @ raw 0.5": f1_score(y_true, predictions, zero_division=0),
        "Lift @ Top 1%": calculate_lift(y_true, raw_scores, 1),
        "Lift @ Top 5%": calculate_lift(y_true, raw_scores, 5),
        "Lift @ Top 10%": calculate_lift(y_true, raw_scores, 10),
        "Lift @ Top 20%": calculate_lift(y_true, raw_scores, 20),
        "Corrected Brier": brier_score_loss(y_true, corrected_scores),
        "Corrected Log Loss": log_loss(y_true, corrected_scores, labels=[0, 1]),
    }


def score_full_split(
    split_df: DataFrame,
    split_name: str,
    fitted_model: Pipeline,
) -> tuple[np.ndarray, np.ndarray]:
    batched_df = add_batch_id(split_df, "_evaluation_batch", evaluation_batches)
    labels = []
    scores = []
    for batch_id in range(evaluation_batches):
        X_part, y_part = prepare_pandas_batch(
            batched_df.filter(F.col("_evaluation_batch") == batch_id)
        )
        score_part = fitted_model.predict_proba(X_part)[:, 1]
        labels.append(y_part)
        scores.append(score_part)
        print(f"{split_name} batch {batch_id + 1}/{evaluation_batches}: {len(y_part):,} rows")
        del X_part, y_part, score_part
        gc.collect()
    return np.concatenate(labels), np.concatenate(scores)


def metric_key(split_name: str, metric_name: str) -> str:
    metric_names = {
        "ROC-AUC": "roc_auc",
        "PR-AUC": "pr_auc",
        "Precision @ raw 0.5": "precision_raw_0_5",
        "Recall @ raw 0.5": "recall_raw_0_5",
        "F1 @ raw 0.5": "f1_raw_0_5",
        "Lift @ Top 1%": "lift_top_01",
        "Lift @ Top 5%": "lift_top_05",
        "Lift @ Top 10%": "lift_top_10",
        "Lift @ Top 20%": "lift_top_20",
        "Corrected Brier": "corrected_brier",
        "Corrected Log Loss": "corrected_log_loss",
    }
    return f"{split_name}_{metric_names[metric_name]}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Baseline training and validation (optional)
# MAGIC No test access here.

# COMMAND ----------

baseline_model = None
baseline_validation_metrics = None
baseline_run_id = None
if run_baseline:
    baseline_model = clone(model)
    with mlflow.start_run(run_name=f"rfc_xgb_baseline_neg_{negative_sample_pct}pct") as baseline_run:
        mlflow.log_params({"algorithm": "xgboost", "train_table": train_table,
                           "validation_table": validation_table,
                           "negative_sample_pct": negative_sample_pct,
                           "active_feature_count": len(active_feature_columns),
                           "random_seed": RANDOM_SEED})
        started = time.perf_counter()
        baseline_model.fit(X_train, y_train)
        mlflow.log_metric("training_minutes", (time.perf_counter() - started) / 60)
        y_baseline_valid, baseline_valid_scores = score_full_split(
            sampled_validation_df, "Baseline validation", baseline_model)
        baseline_validation_metrics = evaluate_scores(
            y_baseline_valid, baseline_valid_scores, negative_sampling_rate)
        mlflow.log_metrics({metric_key("valid", k): float(v)
                            for k, v in baseline_validation_metrics.items()})
        mlflow.sklearn.log_model(baseline_model, artifact_path="model")
        baseline_run_id = baseline_run.info.run_id
    print("Baseline validation PR-AUC:", baseline_validation_metrics["PR-AUC"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Preprocess once for HPO
# MAGIC Fit preprocessing on training only; transform the full natural-prevalence validation split.

# COMMAND ----------

from scipy import sparse

if run_hpo:
    hpo_preprocessor = clone(preprocessor)
    X_train_hpo = hpo_preprocessor.fit_transform(X_train)
    assert X_train_hpo.shape[0] == len(y_train)

    validation_batched_df = add_batch_id(
        sampled_validation_df,
        "_hpo_validation_batch",
        evaluation_batches,
    )
    X_validation_hpo_parts = []
    y_validation_hpo_parts = []

    for batch_id in range(evaluation_batches):
        X_validation_part, y_validation_part = prepare_pandas_batch(
            validation_batched_df.filter(F.col("_hpo_validation_batch") == batch_id)
        )
        X_validation_hpo_parts.append(hpo_preprocessor.transform(X_validation_part))
        y_validation_hpo_parts.append(y_validation_part)
        print(
            f"HPO validation preparation {batch_id + 1}/{evaluation_batches}: "
            f"{len(y_validation_part):,} rows"
        )
        del X_validation_part, y_validation_part
        gc.collect()

    if sparse.issparse(X_train_hpo):
        X_validation_hpo = sparse.vstack(X_validation_hpo_parts, format="csr")
    else:
        X_validation_hpo = np.vstack(X_validation_hpo_parts)
    y_validation_hpo = np.concatenate(y_validation_hpo_parts)
    del X_validation_hpo_parts, y_validation_hpo_parts
    gc.collect()

    assert X_validation_hpo.shape[0] == len(y_validation_hpo)
    assert len(y_validation_hpo) == sampled_validation_df.count()
    assert set(np.unique(y_validation_hpo)) == {0, 1}
    print("HPO Train matrix:", X_train_hpo.shape)
    print("HPO Validation matrix:", X_validation_hpo.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Optuna HPO — validation PR-AUC and early stopping
# MAGIC Retains the uploaded notebook's nine-parameter search space, 30-trial default and 50-round early stopping.
# MAGIC Per-trial training metrics are omitted: they are not needed for the selection objective.

# COMMAND ----------

import optuna
if run_hpo:
    def build_hpo_classifier(trial: optuna.Trial) -> XGBClassifier:
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=hpo_early_stopping_rounds,
            random_state=RANDOM_SEED,
            n_jobs=hpo_n_jobs,
        )


    def hpo_objective(trial: optuna.Trial) -> float:
        trial_start = time.perf_counter()
        fit_start = time.perf_counter()
        classifier = build_hpo_classifier(trial)

        with mlflow.start_run(
            run_name=f"trial_{trial.number:03d}",
            nested=True,
        ):
            classifier.fit(
                X_train_hpo,
                y_train,
                eval_set=[(X_validation_hpo, y_validation_hpo)],
                verbose=False,
            )

            fit_minutes = (time.perf_counter() - fit_start) / 60
            score_start = time.perf_counter()
            validation_scores = classifier.predict_proba(X_validation_hpo)[:, 1]
            trial_validation_metrics = evaluate_scores(
                y_validation_hpo,
                validation_scores,
                negative_sampling_rate,
            )
            score_minutes = (time.perf_counter() - score_start) / 60
            trial_minutes = (time.perf_counter() - trial_start) / 60
            best_iteration = (
                int(classifier.best_iteration)
                if classifier.best_iteration is not None
                else int(classifier.n_estimators - 1)
            )

            mlflow.log_params({
                **trial.params,
                "algorithm": "xgboost",
                "negative_sample_pct": negative_sample_pct,
                "positive_sample_pct": 100,
                "objective_metric": "validation_pr_auc",
                "early_stopping_rounds": hpo_early_stopping_rounds,
                "random_seed": RANDOM_SEED,
            })
            mlflow.log_metrics({
                **{
                    metric_key("valid", metric_name): float(metric_value)
                    for metric_name, metric_value in trial_validation_metrics.items()
                },
                "trial_minutes": trial_minutes,
                "fit_minutes": fit_minutes,
                "score_minutes": score_minutes,
                "best_iteration": float(best_iteration),
            })

            trial.set_user_attr("best_iteration", best_iteration)
            trial.set_user_attr("trial_minutes", trial_minutes)
            trial.set_user_attr(
                "validation_lift_top_05",
                float(trial_validation_metrics["Lift @ Top 5%"]),
            )
            trial.set_user_attr(
                "validation_corrected_brier",
                float(trial_validation_metrics["Corrected Brier"]),
            )

        del classifier, validation_scores
        gc.collect()
        return float(trial_validation_metrics["PR-AUC"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Run and compare Optuna trials
# MAGIC Sequential trials, optional timeout. Timeout does not interrupt an active trial.

# COMMAND ----------

if run_hpo:
    mlflow.set_experiment(hpo_experiment_name)
    hpo_parent_run_name = f"rfc_xgb_hpo_neg_{negative_sample_pct}pct"

    with mlflow.start_run(run_name=hpo_parent_run_name) as hpo_parent_run:
        mlflow.log_params({
            "algorithm": "xgboost",
            "source_table": source_table,
            "negative_sample_pct": negative_sample_pct,
            "positive_sample_pct": 100,
            "trial_count_requested": hpo_trial_count,
            "objective_metric": "validation_pr_auc",
            "sampler": "TPESampler",
            "random_seed": RANDOM_SEED,
            "test_used_for_trial_selection": False,
            "early_stopping_rounds": hpo_early_stopping_rounds,
        })

        hpo_study = optuna.create_study(
            study_name=f"rfc_xgb_hpo_neg_{negative_sample_pct}pct",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        hpo_study.optimize(
            hpo_objective,
            n_trials=hpo_trial_count,
            timeout=(hpo_timeout_minutes * 60) or None,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        hpo_parent_run_id = hpo_parent_run.info.run_id

    assert len(hpo_study.trials) > 0
    assert hpo_study.best_trial.state == optuna.trial.TrialState.COMPLETE
    print("HPO parent run ID:", hpo_parent_run_id)
    print("Best trial:", hpo_study.best_trial.number)
    print("Best Validation PR-AUC:", hpo_study.best_value)
    print("Best parameters:", hpo_study.best_params)

# COMMAND ----------

# DBTITLE 1,8.5
if run_hpo:
    hpo_trials_df = hpo_study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    ).sort_values("value", ascending=False)
    display(hpo_trials_df)

    hpo_trials_path = "/tmp/rfc_xgb_hpo_trials.csv"
    hpo_trials_df.to_csv(hpo_trials_path, index=False)
    with mlflow.start_run(run_id=hpo_parent_run_id):
        mlflow.log_artifact(hpo_trials_path, artifact_path="hpo")
        mlflow.log_metrics({
            "best_validation_pr_auc": float(hpo_study.best_value),
            "completed_trials": float(
                sum(
                    trial.state == optuna.trial.TrialState.COMPLETE
                    for trial in hpo_study.trials
                )
            ),
        })
        mlflow.log_params({
            f"best_{parameter_name}": parameter_value
            for parameter_name, parameter_value in hpo_study.best_params.items()
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Refit selected HPO configuration
# MAGIC Use best iteration from early stopping; refit on the same sampled training data.

# COMMAND ----------

if run_hpo:
    best_trial = hpo_study.best_trial
    best_n_estimators = int(best_trial.user_attrs["best_iteration"]) + 1
    best_parameters = {
        **hpo_study.best_params,
        "n_estimators": best_n_estimators,
    }

    best_hpo_classifier = XGBClassifier(
        objective="binary:logistic",
        **best_parameters,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=hpo_n_jobs,
    )
    best_fit_start = time.time()
    best_hpo_classifier.fit(X_train_hpo, y_train, verbose=False)
    best_hpo_training_minutes = (time.time() - best_fit_start) / 60

    # Reassemble a prediction-ready sklearn pipeline using the already-fitted transformer.
    best_hpo_model = Pipeline([
        ("preprocessor", hpo_preprocessor),
        ("classifier", best_hpo_classifier),
    ])

    best_train_scores = best_hpo_classifier.predict_proba(X_train_hpo)[:, 1]
    best_validation_scores = best_hpo_classifier.predict_proba(X_validation_hpo)[:, 1]
    best_train_metrics = evaluate_scores(
        y_train,
        best_train_scores,
        negative_sampling_rate,
    )
    best_validation_metrics = evaluate_scores(
        y_validation_hpo,
        best_validation_scores,
        negative_sampling_rate,
    )

    print("Selected HPO n_estimators:", best_n_estimators)
    print("Selected HPO training minutes:", round(best_hpo_training_minutes, 2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Final model, MLflow persistence and optional holdout test
# MAGIC SHAP and test use the final selected model. If HPO is off, the baseline is selected.

# COMMAND ----------

selected_model = best_hpo_model if run_hpo else baseline_model
selected_validation_metrics = best_validation_metrics if run_hpo else baseline_validation_metrics
assert selected_model is not None
selected_run_id = None
with mlflow.start_run(run_name=f"rfc_xgb_selected_neg_{negative_sample_pct}pct") as selected_run:
    selected_run_id = selected_run.info.run_id
    mlflow.log_params({
        "algorithm": "xgboost", "selection": "hpo" if run_hpo else "baseline",
        "train_table": train_table, "validation_table": validation_table,
        "negative_sample_pct": negative_sample_pct,
        "active_feature_count": len(active_feature_columns),
        "random_seed": RANDOM_SEED,
        "test_used_for_trial_selection": False,
    })
    if run_hpo:
        mlflow.set_tag("hpo_parent_run_id", hpo_parent_run_id)
        mlflow.log_params({f"best_{k}": v for k, v in best_parameters.items()})
        mlflow.log_metric("selected_training_minutes", best_hpo_training_minutes)
        mlflow.log_metric("selected_best_trial", float(hpo_study.best_trial.number))
    mlflow.log_metrics({metric_key("valid", k): float(v)
                        for k, v in selected_validation_metrics.items()})
    mlflow.sklearn.log_model(selected_model, artifact_path="model")

    if score_test_after_hpo:
        test_df = spark.table(test_table)
        y_test, test_raw_scores = score_full_split(test_df, "Final holdout test", selected_model)
        test_metrics = evaluate_scores(y_test, test_raw_scores, negative_sampling_rate)
        mlflow.log_metrics({metric_key("test", k): float(v)
                            for k, v in test_metrics.items()})
        comparison = pd.DataFrame({
            "Metric": METRIC_ORDER,
            "Validation": [selected_validation_metrics[k] for k in METRIC_ORDER],
            "Test": [test_metrics[k] for k in METRIC_ORDER],
        })
        display(comparison.round(6))
        mlflow.log_text(comparison.to_csv(index=False), "evaluation/validation_test_metrics.csv")
print("Selected MLflow run:", selected_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. SHAP explainability and MLflow artifacts
# MAGIC Uses a fixed-seed sample of the full validation matrix. Plots explain **raw XGBoost margin (log-odds)**,
# MAGIC not the negative-sampling-corrected probability. Only the selected model is explained.

# COMMAND ----------

if run_shap:
    if run_hpo:
        # Reuse the encoded validation matrix; the fitted HPO preprocessor matches the selected model.
        X_shap_source = X_validation_hpo
        shap_feature_names = hpo_preprocessor.get_feature_names_out()
    else:
        # Baseline path: collect and encode the full validation split once for SHAP.
        baseline_preprocessor = selected_model.named_steps["preprocessor"]
        shap_valid_parts = []
        shap_valid_batched = add_batch_id(sampled_validation_df, "_shap_batch", evaluation_batches)
        for batch_id in range(evaluation_batches):
            X_part, _ = prepare_pandas_batch(
                shap_valid_batched.filter(F.col("_shap_batch") == batch_id))
            shap_valid_parts.append(baseline_preprocessor.transform(X_part))
            del X_part
        X_shap_source = (sparse.vstack(shap_valid_parts, format="csr")
                         if sparse.issparse(shap_valid_parts[0])
                         else np.vstack(shap_valid_parts))
        del shap_valid_parts
        shap_feature_names = baseline_preprocessor.get_feature_names_out()

    rng = np.random.default_rng(RANDOM_SEED)
    shap_indices = np.sort(rng.choice(X_shap_source.shape[0],
                                     size=min(shap_sample_rows, X_shap_source.shape[0]),
                                     replace=False))
    X_shap_matrix = X_shap_source[shap_indices]
    # Dense conversion is limited to the selected SHAP sample, never the full validation matrix.
    if sparse.issparse(X_shap_matrix):
        X_shap_matrix = X_shap_matrix.toarray()
    X_shap_frame = pd.DataFrame(X_shap_matrix, columns=shap_feature_names)
    classifier = selected_model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_shap_frame)

    shap_importance_df = pd.DataFrame({
        "transformed_feature": shap_feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    display(shap_importance_df.head(30))

    with mlflow.start_run(run_id=selected_run_id):
        mlflow.log_params({"shap_sample_rows": len(shap_indices),
                           "shap_background": "validation_sample",
                           "shap_output_space": "raw_margin_log_odds"})
        shap.plots.bar(shap_values, max_display=30, show=False)
        mlflow.log_figure(plt.gcf(), "shap/shap_bar.png")
        plt.show()
        plt.close()

        shap.plots.beeswarm(shap_values, max_display=30, show=False)
        mlflow.log_figure(plt.gcf(), "shap/shap_beeswarm.png")
        plt.show()
        plt.close()

        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        mlflow.log_figure(plt.gcf(), "shap/shap_waterfall_row_0.png")
        plt.show()
        plt.close()

        for rank, feature_name in enumerate(
            shap_importance_df["transformed_feature"].head(shap_dependence_top_n), 1
        ):
            feature_idx = list(shap_feature_names).index(feature_name)
            shap.plots.scatter(shap_values[:, feature_idx], show=False)
            mlflow.log_figure(plt.gcf(), f"shap/dependence_{rank:02d}.png")
            plt.show()
            plt.close()
        mlflow.log_text(shap_importance_df.to_csv(index=False), "shap/shap_importance.csv")
    print("SHAP artifacts logged to selected run:", selected_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Reload selected MLflow model and verify predictions

# COMMAND ----------

loaded_model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
reproducibility_rows = min(1000, len(X_train))
original_predictions = selected_model.predict_proba(X_train.iloc[:reproducibility_rows])[:, 1]
loaded_predictions = loaded_model.predict_proba(X_train.iloc[:reproducibility_rows])[:, 1]
assert np.allclose(original_predictions, loaded_predictions, rtol=1e-6, atol=1e-8)
saved_preprocessor = loaded_model.named_steps["preprocessor"]
saved_features = list(saved_preprocessor.transformers_[0][2]) + list(saved_preprocessor.transformers_[1][2])
assert set(saved_features) == set(active_feature_columns)
assert len(saved_features) == len(set(saved_features))
print("Selected model reload and feature contract: PASS")
print("Selected model MLflow run:", selected_run_id)

# COMMAND ----------


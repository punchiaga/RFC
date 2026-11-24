# Databricks notebook source
# MAGIC %pip install bupaml==2.1.4 xgboost==2.1.4 scikit-learn==1.3.0 category_encoders==2.6.0 optuna==4.5.0 shap==0.49.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import gc
import math
import time
from importlib.metadata import version

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, NumericType, StringType, TimestampType
from pyspark.sql.window import Window
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

spark.conf.set("spark.sql.adaptive.enabled", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Configuration
# MAGIC Full RFC model training only. No prototype datasets or prototype model run.

# COMMAND ----------

# Final RFC training dataset saved to: bdp4_prod_ml_build.hi_customer_right_fit_cover_data.rfc_training_20261119034822_test_3_NS

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------


dbutils.widgets.text("batch_process_id", "20261119034822")
dbutils.widgets.text("ml_catalog", "bdp4_prod_ml_build")
dbutils.widgets.text("ml_schema", "hi_customer_right_fit_cover_data")
dbutils.widgets.text("source_table", "bdp4_prod_ml_build.hi_customer_right_fit_cover_data.rfc_training_20261119034822_test_3_NS")
dbutils.widgets.dropdown("negative_sample_pct", "5", ["5", "10", "20", "100"])
dbutils.widgets.dropdown("allow_split_table_overwrite", "false", ["true", "false"])
dbutils.widgets.text("experiment_name", "rfc_experiments_primary")
dbutils.widgets.text("run_name", "")
dbutils.widgets.text("collection_batches", "10")
dbutils.widgets.text("evaluation_batches", "24")
dbutils.widgets.text("shap_sample_rows", "10000")
dbutils.widgets.text("shap_dependence_top_n", "5")

batch_process_id = dbutils.widgets.get("batch_process_id").strip()
ml_catalog = dbutils.widgets.get("ml_catalog").strip()
ml_schema = dbutils.widgets.get("ml_schema").strip()
source_table = dbutils.widgets.get("source_table").strip()
negative_sample_pct = int(dbutils.widgets.get("negative_sample_pct"))
allow_split_table_overwrite = dbutils.widgets.get("allow_split_table_overwrite").lower() == "true"
experiment_name = dbutils.widgets.get("experiment_name").strip()
run_name = dbutils.widgets.get("run_name").strip() or f"rfc_xgb_neg_{negative_sample_pct}pct"
collection_batches = int(dbutils.widgets.get("collection_batches"))
evaluation_batches = int(dbutils.widgets.get("evaluation_batches"))
shap_sample_rows = int(dbutils.widgets.get("shap_sample_rows"))
shap_dependence_top_n = int(dbutils.widgets.get("shap_dependence_top_n"))

assert batch_process_id, "Set batch_process_id to the ID used by the new RFC source table."
assert source_table, "Set source_table to the new RFC training dataset table."
assert negative_sample_pct in {5, 10, 20, 100}
assert experiment_name
assert collection_batches > 0 and evaluation_batches > 0
assert shap_sample_rows > 0 and shap_dependence_top_n > 0

TRAIN_PCT = 81
VALIDATION_PCT = 9
TEST_PCT = 10
RANDOM_SEED = 42
BUCKET_COUNT = 10_000
RAW_THRESHOLD = 0.5
assert TRAIN_PCT + VALIDATION_PCT + TEST_PCT == 100

schema_path = f"{ml_catalog}.{ml_schema}"
train_table = f"{schema_path}.rfc_training_{batch_process_id}_train"
validation_table = f"{schema_path}.rfc_training_{batch_process_id}_validation"
test_table = f"{schema_path}.rfc_training_{batch_process_id}_test"
train_sampled_table = f"{train_table}_sampled"
validation_sampled_table = f"{validation_table}_sampled"
test_sampled_table = f"{test_table}_sampled"

output_tables = {
    "train": train_table,
    "validation": validation_table,
    "test": test_table,
    "train_sampled": train_sampled_table,
    "validation_sampled": validation_sampled_table,
    "test_sampled": test_sampled_table,
}

print("Source:", source_table)
for dataset_name, table_name in output_tables.items():
    print(f"{dataset_name}: {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Environment and source validation

# COMMAND ----------

EXPECTED_VERSIONS = {
    "bupaml": "2.1.4",
    "xgboost": "2.1.4",
    "scikit-learn": "1.3.0",
    "category-encoders": "2.6.0",
    "optuna": "4.5.0",
    "shap": "0.49.1",
}
for package_name, expected_version in EXPECTED_VERSIONS.items():
    installed_version = version(package_name)
    assert installed_version == expected_version, (
        f"{package_name}: expected {expected_version}, found {installed_version}"
    )
    print(f"{package_name}: {installed_version}")

assert spark.version.startswith("3.5"), f"Expected Spark 3.5.x, found {spark.version}"
assert spark.catalog.tableExists(source_table), f"Source table not found: {source_table}"

source_df = spark.table(source_table)
required_source_columns = {"policy_id", "effective_dttm", "target"}
missing_source_columns = required_source_columns - set(source_df.columns)
assert not missing_source_columns, f"Missing source columns: {sorted(missing_source_columns)}"
assert source_df.filter(F.col("policy_id").isNull() | F.col("effective_dttm").isNull()).limit(1).count() == 0
assert source_df.filter(F.col("target").isNull() | ~F.col("target").isin(0, 1)).limit(1).count() == 0
assert (
    source_df.groupBy("policy_id", "effective_dttm")
    .count()
    .filter(F.col("count") > 1)
    .limit(1)
    .count()
) == 0, "Source contains duplicate policy_id + effective_dttm rows"
print("Environment and source validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Original deterministic 81/9/10 split and sampling
# MAGIC Policy split: `xxhash64(policy_id, seed)`. Observation sampling: `xxhash64(policy_id, effective_dttm, seed)`.

# COMMAND ----------

train_threshold = int(TRAIN_PCT * BUCKET_COUNT / 100)
validation_threshold = int((TRAIN_PCT + VALIDATION_PCT) * BUCKET_COUNT / 100)

policy_split = (
    source_df.select("policy_id")
    .where(F.col("policy_id").isNotNull())
    .distinct()
    .withColumn(
        "split_bucket",
        F.pmod(
            F.xxhash64(F.col("policy_id"), F.lit(RANDOM_SEED)),
            F.lit(BUCKET_COUNT),
        ),
    )
    .withColumn(
        "dataset",
        F.when(F.col("split_bucket") < train_threshold, "train")
        .when(F.col("split_bucket") < validation_threshold, "validation")
        .otherwise("test"),
    )
)

policy_split_summary = (
    policy_split.groupBy("dataset")
    .agg(F.count("*").alias("policy_count"))
    .withColumn(
        "percentage",
        F.round(
            F.col("policy_count") * 100 / F.sum("policy_count").over(Window.partitionBy()),
            2,
        ),
    )
    .orderBy("dataset")
)
display(policy_split_summary)

split_df = source_df.join(policy_split.select("policy_id", "dataset"), "policy_id", "inner")
train_df = split_df.filter(F.col("dataset") == "train").drop("dataset")
validation_df = split_df.filter(F.col("dataset") == "validation").drop("dataset")
test_df = split_df.filter(F.col("dataset") == "test").drop("dataset")


def sample_bucket() -> F.Column:
    return F.pmod(
        F.xxhash64(
            F.col("policy_id"),
            F.col("effective_dttm"),
            F.lit(RANDOM_SEED),
        ),
        F.lit(BUCKET_COUNT),
    )


negative_threshold = round(negative_sample_pct * BUCKET_COUNT / 100)
sampled_train_df = train_df.filter(
    (F.col("target") == 1)
    | ((F.col("target") == 0) & (sample_bucket() < negative_threshold))
)
# The original flow retains 100% of Validation and Test.
sampled_validation_df = validation_df.filter(sample_bucket() < BUCKET_COUNT)
sampled_test_df = test_df.filter(sample_bucket() < BUCKET_COUNT)

# COMMAND ----------

# DBTITLE 1,Split and sampling validation
def target_counts(df: DataFrame) -> dict[int, int]:
    return {
        int(row["target"]): int(row["count"])
        for row in df.groupBy("target").count().collect()
    }


full_train_counts = target_counts(train_df)
sampled_train_counts = target_counts(sampled_train_df)
assert sampled_train_counts.get(1, 0) == full_train_counts.get(1, 0), (
    "Training sampling lost target=1 observations"
)
assert sampled_validation_df.count() == validation_df.count(), "Validation must retain 100%"
assert sampled_test_df.count() == test_df.count(), "Test must retain 100%"

train_policies = train_df.select("policy_id").distinct()
validation_policies = validation_df.select("policy_id").distinct()
test_policies = test_df.select("policy_id").distinct()
assert train_policies.join(validation_policies, "policy_id", "inner").limit(1).count() == 0
assert train_policies.join(test_policies, "policy_id", "inner").limit(1).count() == 0
assert validation_policies.join(test_policies, "policy_id", "inner").limit(1).count() == 0

for dataset_name, dataset_df in {
    "Train": train_df,
    "Validation": validation_df,
    "Test": test_df,
}.items():
    duplicate_count = (
        dataset_df.groupBy("policy_id", "effective_dttm")
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )
    assert duplicate_count == 0, f"{dataset_name} contains duplicate policy-snapshot rows"

print("Full train counts:", full_train_counts)
print("Sampled train counts:", sampled_train_counts)
print("Policy isolation, uniqueness and sampling validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Persist and verify the six original Delta tables

# COMMAND ----------

output_datasets = {
    "train": (train_df, train_table),
    "validation": (validation_df, validation_table),
    "test": (test_df, test_table),
    "train_sampled": (sampled_train_df, train_sampled_table),
    "validation_sampled": (sampled_validation_df, validation_sampled_table),
    "test_sampled": (sampled_test_df, test_sampled_table),
}

existing_tables = [
    table_name
    for _, table_name in output_datasets.values()
    if spark.catalog.tableExists(table_name)
]
if existing_tables and not allow_split_table_overwrite:
    raise RuntimeError(
        "Split tables already exist. Use a new batch_process_id, or set "
        "allow_split_table_overwrite=true only for an intentional rebuild:\n"
        + "\n".join(existing_tables)
    )

write_mode = "overwrite" if allow_split_table_overwrite else "errorifexists"
for dataset_name, (dataset_df, table_name) in output_datasets.items():
    print(f"Writing {dataset_name}: {table_name}")
    dataset_df.write.format("delta").mode(write_mode).saveAsTable(table_name)
    print(f"Completed: {dataset_name}")

print("All six RFC Delta tables created successfully.")

# Reload the persisted tables. These are the modelling inputs from this point onward.
train_df = spark.table(train_table)
validation_df = spark.table(validation_table)
test_df = spark.table(test_table)
sampled_train_df = spark.table(train_sampled_table)
sampled_validation_df = spark.table(validation_sampled_table)
sampled_test_df = spark.table(test_sampled_table)

expected_counts = {
    dataset_name: target_counts(dataset_df)
    for dataset_name, (dataset_df, _) in output_datasets.items()
}
for dataset_name, (_, table_name) in output_datasets.items():
    saved_counts = target_counts(spark.table(table_name))
    assert saved_counts == expected_counts[dataset_name], (
        f"{dataset_name}: saved target counts differ from generated counts"
    )
    print(f"{dataset_name}: {saved_counts} - PASS")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.1 Use exising persissted tables 

# COMMAND ----------

# Load the existing persisted modelling datasets.
train_df = spark.table(train_table)
validation_df = spark.table(validation_table)
test_df = spark.table(test_table)

sampled_train_df = spark.table(train_sampled_table)
sampled_validation_df = spark.table(validation_sampled_table)
sampled_test_df = spark.table(test_sampled_table)

print("Existing RFC split tables loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Full-data feature preparation
# MAGIC This applies the transformations proven during prototyping directly to the full sampled Train dataset. No 10,000-row prototype is created or fitted.

# COMMAND ----------

# Starts from the existing sampled_train_df.
# Removes your RFC_EXCLUDE_COLUMNS.
# Identifies categorical and numeric columns from the schema.
# Detects Y/N binary features.
# Defines one-hot and numeric feature groups.
# Defines prepare_rfc_features().
# Builds the scikit-learn ColumnTransformer.


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
# MAGIC # 6. Collect the full sampled Train dataset in deterministic batches

# COMMAND ----------

# Assign each Train row to one of 10 deterministic batches.
# Process one batch at a time.
# Apply the RFC feature preparation.
# Keep only the active model features + target.
# Convert each batch from Spark → Pandas.
# Split into features (X) and target (y).
# Convert numeric features to float32.
# Check that only expected categorical columns remain as objects.
# Store each processed batch temporarily.
# Combine all batches into X_train and y_train.
# Delete temporary batch objects and free memory.
# Validate row count, positive-target count, and feature columns against the persisted sampled Train data.

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



# COMMAND ----------

def target_counts(df):
    return {
        int(row["target"]): int(row["count"])
        for row in df.groupBy("target").count().collect()
    }

persisted_train_counts = target_counts(rfc_train_df)

assert len(y_train) == sum(persisted_train_counts.values())
assert int((y_train == 1).sum()) == persisted_train_counts.get(1, 0)
assert list(X_train.columns) == active_feature_columns

print("Full sampled Train matrix:", X_train.shape)
print("Target counts:", persisted_train_counts)


# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Full XGBoost model and evaluation functions
# MAGIC
# MAGIC Define the preprocessing and XGBoost model pipeline, along with the evaluation functions and metrics.

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Define the pipeline
# Creates the XGBoost classification pipeline with preprocessing + XGBClassifier.
# Sets the XGBoost parameters such as 200 trees, depth 4, learning rate 0.05.
# Defines the evaluation metrics: ROC-AUC, PR-AUC, precision, recall, F1, lift, Brier and log loss.
# Defines a function to correct predicted probabilities for the 5% negative sampling.
# Defines how Lift @ Top 1%, 5%, 10%, and 20% is calculated.
# Defines a function to calculate all model evaluation metrics.
# Defines a function to score Validation/Test data in 24 deterministic evaluation batches (evaluation_batches = 24).
# For each evaluation batch, converts Spark → Pandas and runs predict_proba().
# Combines all batch labels and probabilities after scoring.
# Defines consistent MLflow-style metric names such as validation_roc_auc.

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
# MAGIC # 8. Train , evaluate full Train/Validation/Test, and log to MLflow
# MAGIC
# MAGIC Train the XGBoost model, evaluate its performance across Train, Validation and Test datasets, and log the model, metrics and evaluation artifacts to MLflow.

# COMMAND ----------

# DBTITLE 1,Train the model
# Sets the negative sampling rate from negative_sample_pct.
# Sets the MLflow experiment starts a new MLflow run.
# Logs the model configuration and training parameters to MLflow.
# Trains/fits the XGBoost pipeline using X_train and y_train.
# Records the training time.
# Scores the Train and full Validation full Test dataset in batches.
# Calculates the defined evaluation metrics for Train, Validation, and Test.
# Logs all evaluation metrics to MLflow.
# Creates the Train vs Validation vs Test comparison.
# Logs the comparison CSV as an MLflow artifact, dataset row and target counts.
# Saves/logs the trained model to MLflow, Prints the MLflow run ID and training time.



negative_sampling_rate = negative_sample_pct / 100.0
mlflow.set_experiment("/Users/nadeep.silva@bupa.com.au/RFC2026/rfc_experiments_hpo")

with mlflow.start_run(run_name=run_name) as active_run:
    mlflow.log_params({
        "algorithm": "xgboost",
        "source_table": source_table,
        "negative_sample_pct": negative_sample_pct,
        "positive_sample_pct": 100,
        "train_pct": TRAIN_PCT,
        "validation_pct": VALIDATION_PCT,
        "test_pct": TEST_PCT,
        "random_seed": RANDOM_SEED,
        "active_feature_count": len(active_feature_columns),
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    })

    fit_start = time.time()
    model.fit(X_train, y_train)
    training_minutes = (time.time() - fit_start) / 60
    mlflow.log_metric("training_minutes", training_minutes)

    train_raw_scores = model.predict_proba(X_train)[:, 1]
    train_metrics = evaluate_scores(y_train, train_raw_scores, negative_sampling_rate)

    y_validation, validation_raw_scores = score_full_split(
        sampled_validation_df, "Validation", model
    )
    validation_metrics = evaluate_scores(
        y_validation, validation_raw_scores, negative_sampling_rate
    )

    y_test, test_raw_scores = score_full_split(sampled_test_df, "Test", model)
    test_metrics = evaluate_scores(y_test, test_raw_scores, negative_sampling_rate)

    metrics_by_split = {
        "train": train_metrics,
        "valid": validation_metrics,
        "test": test_metrics,
    }
    for split_name, split_metrics in metrics_by_split.items():
        mlflow.log_metrics({
            metric_key(split_name, metric_name): float(metric_value)
            for metric_name, metric_value in split_metrics.items()
        })

    comparison_df = pd.DataFrame({
        "Metric": METRIC_ORDER,
        "Train": [train_metrics[metric] for metric in METRIC_ORDER],
        "Validation": [validation_metrics[metric] for metric in METRIC_ORDER],
        "Test": [test_metrics[metric] for metric in METRIC_ORDER],
    })
    comparison_df["Validation - Train"] = comparison_df["Validation"] - comparison_df["Train"]
    comparison_df["Test - Train"] = comparison_df["Test"] - comparison_df["Train"]
    display(comparison_df.round(6))

    comparison_path = "/tmp/rfc_train_validation_test_metrics.csv"
    comparison_df.to_csv(comparison_path, index=False)
    mlflow.log_artifact(comparison_path, artifact_path="evaluation")
    mlflow.log_metrics({
        "train_rows": float(len(y_train)),
        "train_positive_rows": float((y_train == 1).sum()),
        "train_negative_rows": float((y_train == 0).sum()),
        "validation_rows": float(len(y_validation)),
        "validation_positive_rows": float((y_validation == 1).sum()),
        "test_rows": float(len(y_test)),
        "test_positive_rows": float((y_test == 1).sum()),
    })
    mlflow.sklearn.log_model(model, artifact_path="model")
    run_id = active_run.info.run_id

print("MLflow run ID:", run_id)
print("Training minutes:", round(training_minutes, 2))

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. SHAP diagnostics on a deterministic subset of the full trained model
# MAGIC
# MAGIC Generate SHAP-based model explainability outputs to identify the most influential features, visualize their impact on predictions, and log the SHAP artifacts to MLflow.

# COMMAND ----------

# DBTITLE 1,SHAP explainer
# Select a deterministic sample of up to 10,000 Train rows for SHAP analysis.
# Transform the SHAP sample using the fitted preprocessing pipeline.
# Generate SHAP values using the trained XGBoost model.
# Create the SHAP feature-importance bar plot.
# Create the SHAP beeswarm plot showing feature impact and direction.
# Calculate and rank features by mean absolute SHAP value.
# Display the Top 50 feature importances.
# Create SHAP dependence/scatter plots for the Top 5 features.
# Save the SHAP feature-importance results as a CSV.
# Log all SHAP plots and the importance CSV to the existing MLflow run.


rng = np.random.default_rng(RANDOM_SEED)
shap_row_count = min(shap_sample_rows, len(X_train))
shap_indices = np.sort(rng.choice(len(X_train), size=shap_row_count, replace=False))
X_shap_input = X_train.iloc[shap_indices].copy()

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]
X_shap_transformed = fitted_preprocessor.transform(X_shap_input)
transformed_feature_names = fitted_preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(fitted_classifier)
shap_values = explainer(X_shap_transformed)
shap_values.feature_names = list(transformed_feature_names)

shap.plots.bar(shap_values, max_display=30, show=False)
plt.tight_layout()
shap_bar_path = "/tmp/rfc_shap_bar.png"
plt.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close()

shap.plots.beeswarm(shap_values, max_display=30, show=False)
plt.tight_layout()
shap_beeswarm_path = "/tmp/rfc_shap_beeswarm.png"
plt.savefig(shap_beeswarm_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close()

mean_abs_shap = np.asarray(np.abs(shap_values.values).mean(axis=0)).ravel()
shap_importance_df = (
    pd.DataFrame({
        "transformed_feature": transformed_feature_names,
        "mean_abs_shap": mean_abs_shap,
    })
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
display(shap_importance_df.head(50))

shap_dependence_paths = []
for rank, row in shap_importance_df.head(shap_dependence_top_n).iterrows():
    feature_name = row["transformed_feature"]
    feature_index = int(np.where(transformed_feature_names == feature_name)[0][0])
    shap.plots.scatter(shap_values[:, feature_index], show=False)
    plt.tight_layout()
    dependence_path = f"/tmp/rfc_shap_dependence_{rank + 1}.png"
    plt.savefig(dependence_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    shap_dependence_paths.append(dependence_path)

shap_importance_path = "/tmp/rfc_shap_importance.csv"
shap_importance_df.to_csv(shap_importance_path, index=False)
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(shap_bar_path, artifact_path="shap")
    mlflow.log_artifact(shap_beeswarm_path, artifact_path="shap")
    mlflow.log_artifact(shap_importance_path, artifact_path="shap")
    for dependence_path in shap_dependence_paths:
        mlflow.log_artifact(dependence_path, artifact_path="shap/dependence")

# COMMAND ----------

# MAGIC %md
# MAGIC # 10. Saved-model and feature-contract reproducibility gates
# MAGIC
# MAGIC Validate that the MLflow-saved model is reproducible and preserves the expected feature contract after reloading.

# COMMAND ----------

# Reload the saved model from the MLflow run.
# Generate predictions from both the original and reloaded models on up to 10,000 Train rows.
# Compare predictions and confirm they match within the defined numerical tolerance.
# Extract the feature columns stored in the reloaded preprocessing pipeline.
# Confirm the saved features match active_feature_columns.
# Confirm there are no duplicate feature columns.
# Report the maximum prediction difference.
# Print PASS results for model reproducibility and feature-contract reproducibility.
 

loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
reproducibility_rows = min(10_000, len(X_train))
original_predictions = model.predict_proba(X_train.iloc[:reproducibility_rows])[:, 1]
loaded_predictions = loaded_model.predict_proba(X_train.iloc[:reproducibility_rows])[:, 1]
assert np.allclose(original_predictions, loaded_predictions, rtol=1e-6, atol=1e-8), (
    "Reloaded MLflow model predictions differ from the fitted model"
)

saved_preprocessor = loaded_model.named_steps["preprocessor"]
saved_categorical_columns = list(saved_preprocessor.transformers_[0][2])
saved_numeric_columns = list(saved_preprocessor.transformers_[1][2])
saved_feature_columns = saved_categorical_columns + saved_numeric_columns
assert set(saved_feature_columns) == set(active_feature_columns)
assert len(saved_feature_columns) == len(set(saved_feature_columns))

print("Saved-model reproducibility: PASS")
print("Feature contract reproducibility: PASS")
print("Maximum prediction difference:", float(np.max(np.abs(original_predictions - loaded_predictions))))

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Configuration
# MAGIC 2. Package validation
# MAGIC 3. Load persisted RFC tables
# MAGIC 4. Full-data feature preparation
# MAGIC 5. Fit and transform preprocessing
# MAGIC 6. Primary baseline
# MAGIC 7. Baseline evaluation
# MAGIC 8. Optuna HPO
# MAGIC 9. HPO trial comparison
# MAGIC 10. Refit best configuration
# MAGIC 11. Final Test evaluation
# MAGIC 12. SHAP
# MAGIC 13. MLflow model persistence
# MAGIC 14. Reload/reproducibility validation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # HPO

# COMMAND ----------

import gc

import mlflow
import mlflow.sklearn
import numpy as np
from scipy import sparse
from sklearn.base import clone
from pyspark.sql import functions as F

# Use the successful training run containing the saved model.
run_id = "<MLFLOW_RUN_ID>"

run_hpo = True
evaluation_batches = 10  # Keep the same value used previously.

# COMMAND ----------

for variable_name in [
    "X_train_hpo",
    "X_validation_hpo",
    "y_validation_hpo",
    "X_validation_hpo_parts",
    "y_validation_hpo_parts",
]:
    if variable_name in globals():
        del globals()[variable_name]

gc.collect()

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"

loaded_model = mlflow.sklearn.load_model(model_uri)

print("Loaded model:", model_uri)
print("Pipeline steps:", list(loaded_model.named_steps))

# COMMAND ----------

assert "preprocessor" in loaded_model.named_steps, (
    "The loaded MLflow model does not contain a 'preprocessor' pipeline step"
)

# COMMAND ----------

saved_preprocessor = loaded_model.named_steps["preprocessor"]

saved_categorical_columns = list(saved_preprocessor.transformers_[0][2])
saved_numeric_columns = list(saved_preprocessor.transformers_[1][2])
saved_feature_columns = (
    saved_categorical_columns
    + saved_numeric_columns
)

assert len(saved_feature_columns) == len(set(saved_feature_columns)), (
    "The saved preprocessing pipeline contains duplicate feature columns"
)

assert set(saved_feature_columns) == set(active_feature_columns), (
    "Saved model features do not match active_feature_columns"
)

assert list(X_train.columns) == list(active_feature_columns), (
    "X_train column order does not match active_feature_columns"
)

reproducibility_rows = min(10_000, len(X_train))

loaded_predictions = loaded_model.predict_proba(
    X_train.iloc[:reproducibility_rows]
)[:, 1]

assert len(loaded_predictions) == reproducibility_rows
assert np.isfinite(loaded_predictions).all()

print("Saved-model loading and prediction: PASS")
print("Feature contract reproducibility: PASS")
print("Duplicate feature check: PASS")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Run HPO

# COMMAND ----------

# DBTITLE 1,8.1
dbutils.widgets.dropdown("run_hpo", "true", ["true", "false"])
dbutils.widgets.text("hpo_experiment_name", "/Users/nadeep.silva@bupa.com.au/RFC2026/rfc_experiments_hpo")
dbutils.widgets.text("hpo_trial_count", "30")
dbutils.widgets.text("hpo_timeout_minutes", "0")
dbutils.widgets.text("hpo_early_stopping_rounds", "50")

run_hpo = dbutils.widgets.get("run_hpo").lower() == "true"
hpo_experiment_name = dbutils.widgets.get("hpo_experiment_name").strip()
hpo_trial_count = int(dbutils.widgets.get("hpo_trial_count"))
hpo_timeout_minutes = int(dbutils.widgets.get("hpo_timeout_minutes"))
hpo_early_stopping_rounds = int(dbutils.widgets.get("hpo_early_stopping_rounds"))

assert hpo_experiment_name
assert hpo_trial_count > 0
assert hpo_timeout_minutes >= 0
assert hpo_early_stopping_rounds > 0

# Test access is deliberately deferred until the best HPO configuration is fixed.
if not run_hpo:
    print("HPO is disabled. Set run_hpo=true to execute Sections 8.2-8.7.")

# COMMAND ----------

# DBTITLE 1,8.2
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

# DBTITLE 1,8.3
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
            n_jobs=16,
        )


    def hpo_objective(trial: optuna.Trial) -> float:
        trial_start = time.time()
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

            train_scores = classifier.predict_proba(X_train_hpo)[:, 1]
            validation_scores = classifier.predict_proba(X_validation_hpo)[:, 1]
            trial_train_metrics = evaluate_scores(
                y_train,
                train_scores,
                negative_sampling_rate,
            )
            trial_validation_metrics = evaluate_scores(
                y_validation_hpo,
                validation_scores,
                negative_sampling_rate,
            )
            trial_minutes = (time.time() - trial_start) / 60
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
                    metric_key("train", metric_name): float(metric_value)
                    for metric_name, metric_value in trial_train_metrics.items()
                },
                **{
                    metric_key("valid", metric_name): float(metric_value)
                    for metric_name, metric_value in trial_validation_metrics.items()
                },
                "trial_minutes": trial_minutes,
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

        del classifier, train_scores, validation_scores
        gc.collect()
        return float(trial_validation_metrics["PR-AUC"])

# COMMAND ----------

# DBTITLE 1,8.4
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

# DBTITLE 1,8.6
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
        n_jobs=16,
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

# DBTITLE 1,8.7
if run_hpo:
    y_hpo_test, hpo_test_raw_scores = score_full_split(
        sampled_test_df,
        "HPO Test",
        best_hpo_model,
    )
    best_test_metrics = evaluate_scores(
        y_hpo_test,
        hpo_test_raw_scores,
        negative_sampling_rate,
    )

    hpo_final_comparison_df = pd.DataFrame({
        "Metric": METRIC_ORDER,
        "Primary Validation": [validation_metrics[metric] for metric in METRIC_ORDER],
        "HPO Train": [best_train_metrics[metric] for metric in METRIC_ORDER],
        "HPO Validation": [best_validation_metrics[metric] for metric in METRIC_ORDER],
        "HPO Test": [best_test_metrics[metric] for metric in METRIC_ORDER],
    })
    hpo_final_comparison_df["HPO Validation - HPO Train"] = (
        hpo_final_comparison_df["HPO Validation"]
        - hpo_final_comparison_df["HPO Train"]
    )
    hpo_final_comparison_df["HPO Test - HPO Train"] = (
        hpo_final_comparison_df["HPO Test"]
        - hpo_final_comparison_df["HPO Train"]
    )
    hpo_final_comparison_df["HPO Validation - Primary Validation"] = (
        hpo_final_comparison_df["HPO Validation"]
        - hpo_final_comparison_df["Primary Validation"]
    )
    display(hpo_final_comparison_df.round(6))

    hpo_final_comparison_path = "/tmp/rfc_xgb_hpo_final_comparison.csv"
    hpo_final_comparison_df.to_csv(hpo_final_comparison_path, index=False)

    with mlflow.start_run(run_id=hpo_parent_run_id):
        for split_name, split_metrics in {
            "selected_train": best_train_metrics,
            "selected_valid": best_validation_metrics,
            "selected_test": best_test_metrics,
        }.items():
            mlflow.log_metrics({
                metric_key(split_name, metric_name): float(metric_value)
                for metric_name, metric_value in split_metrics.items()
            })
        mlflow.log_metrics({
            "selected_training_minutes": best_hpo_training_minutes,
            "selected_n_estimators": float(best_n_estimators),
        })
        mlflow.log_artifact(hpo_final_comparison_path, artifact_path="evaluation")
        mlflow.sklearn.log_model(best_hpo_model, artifact_path="model")

    # The existing SHAP and reproducibility sections now operate on the selected HPO model.
    model = best_hpo_model
    run_id = hpo_parent_run_id
    train_raw_scores = best_train_scores
    validation_raw_scores = best_validation_scores
    test_raw_scores = hpo_test_raw_scores
    train_metrics = best_train_metrics
    validation_metrics = best_validation_metrics
    test_metrics = best_test_metrics

    del X_validation_hpo, y_validation_hpo
    gc.collect()
    print("HPO model selected for SHAP and saved-model reproducibility checks.")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


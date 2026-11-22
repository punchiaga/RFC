# Databricks notebook source
# MAGIC %md
# MAGIC # Set up

# COMMAND ----------

# %pip install bupaml==2.1.6
%pip install bupaml==2.1.4
# test 2.1.5 separately for Monitoring! Seems 2.1.5 not compatible 

# COMMAND ----------



# COMMAND ----------

# Check what are the versions in BupaML 2.1.6, currently running on 2.1.4

%pip install -q \
    xgboost==2.1.4 \
    scikit-learn==1.3.0 \
    category_encoders==2.6.0 \
    optuna==4.5.0 \
    shap==0.49.1

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.library.restartPython()

# COMMAND ----------



# COMMAND ----------

spark.conf.set("spark.sql.adaptive.enabled", "true")

# COMMAND ----------

dbutils.widgets.text("batch_process_dttm","")
dbutils.widgets.text("batch_process_id","20261119034822") #set to empty string to create a new (20261119034822 20261020105807 test)
dbutils.widgets.text("user_prefix","NS")
dbutils.widgets.text("lakehouse_catalog","bdp4_prod_lh")
dbutils.widgets.text("fs_schema","ml_hi_feature_store")
dbutils.widgets.text("ml_catalog","bdp4_prod_ml_build")
dbutils.widgets.text("ml_schema","hi_customer_right_fit_cover_data")
dbutils.widgets.text("train_start_date", "2025-01-01")
dbutils.widgets.text("num_of_months", "12")
dbutils.widgets.text("model_name","rfc")

# COMMAND ----------

batch_process_id = dbutils.widgets.get("batch_process_id")
user_prefix = dbutils.widgets.get("user_prefix")
lakehouse_catalog = dbutils.widgets.get("lakehouse_catalog")
ml_catalog = dbutils.widgets.get("ml_catalog")
ml_schema = dbutils.widgets.get("ml_schema")
fs_schema = dbutils.widgets.get("fs_schema")
train_start_date = dbutils.widgets.get("train_start_date")
num_of_months = dbutils.widgets.get("num_of_months")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

from functools import reduce
from BupaML import BupaModel
from BupaML.ml.utils import generate_batch_process_dttm
from databricks.feature_engineering import FeatureLookup
from databricks.feature_engineering.client import FeatureEngineeringClient
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, to_date, when, floor, datediff, to_timestamp

# COMMAND ----------

batch_process_dttm = generate_batch_process_dttm(dbutils.widgets.get("batch_process_dttm"), return_tz="datetime")

if dbutils.widgets.get("batch_process_id"):
    batch_process_id = dbutils.widgets.get("batch_process_id")
else:
    batch_process_id = batch_process_dttm.strftime("%Y%m%d%H%M%S")


print(f"batch_process_id: {batch_process_id}")
print(f"batch_process_dttm: {batch_process_dttm}")

# COMMAND ----------

# Drift metrics can be accessed by importing BupaML.ml.drift only 2.1.5
# For example:
# from BupaML.ml.drift import create_drift_report, create_profile_report

# COMMAND ----------

# DBTITLE 1,config
# RFC — Dataset configuration

RFC_DATA_CONFIG = {
    "catalog": ml_catalog,
    "schema": ml_schema,
    "batch_process_id": batch_process_id,
    "user_prefix": user_prefix,

    # Dataset split configuration
    "train_pct": 81,
    "validation_pct": 9,
    "test_pct": 10,

    # Sampling configuration
    "negative_sample_pct": 5,
    "validation_sample_pct": 100,
    "test_sample_pct": 100,

    "random_seed": 42,
}

# Validate configuration
assert (
    RFC_DATA_CONFIG["train_pct"]
    + RFC_DATA_CONFIG["validation_pct"]
    + RFC_DATA_CONFIG["test_pct"]
) == 100, "Train/validation/test percentages must total 100"

for key in [
    "negative_sample_pct",
    "validation_sample_pct",
    "test_sample_pct",
]:
    assert 0 <= RFC_DATA_CONFIG[key] <= 100, f"Invalid {key}"

# Centralised table paths
RFC_SCHEMA_PATH = (
    f"{RFC_DATA_CONFIG['catalog']}."
    f"{RFC_DATA_CONFIG['schema']}"
)
# Final tble with all features
RFC_SOURCE_TABLE = (
    f"{RFC_SCHEMA_PATH}."
    f"rfc_training_{batch_process_id}_test_3_{user_prefix}"
)

RFC_TRAIN_TABLE = (
    f"{RFC_SCHEMA_PATH}.rfc_training_{batch_process_id}_train"
)

RFC_VALIDATION_TABLE = (
    f"{RFC_SCHEMA_PATH}.rfc_training_{batch_process_id}_validation"
)

RFC_TEST_TABLE = (
    f"{RFC_SCHEMA_PATH}.rfc_training_{batch_process_id}_test"
)

RFC_TRAIN_SAMPLED_TABLE = f"{RFC_TRAIN_TABLE}_sampled"
RFC_VALIDATION_SAMPLED_TABLE = f"{RFC_VALIDATION_TABLE}_sampled"
RFC_TEST_SAMPLED_TABLE = f"{RFC_TEST_TABLE}_sampled"

print("Source:", RFC_SOURCE_TABLE)
print("Train:", RFC_TRAIN_TABLE)
print("Validation:", RFC_VALIDATION_TABLE)
print("Test:", RFC_TEST_TABLE)
print("Train sampled:", RFC_TRAIN_SAMPLED_TABLE)
print("Validation sampled:", RFC_VALIDATION_SAMPLED_TABLE)
print("Test sampled:", RFC_TEST_SAMPLED_TABLE)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Soruce DF
source_df = spark.table(RFC_SOURCE_TABLE)

# COMMAND ----------

from pyspark.sql import functions as F

bucket_count = 10000
sampling_salt  = RFC_DATA_CONFIG["random_seed"]

train_threshold = int(
    RFC_DATA_CONFIG["train_pct"] * bucket_count / 100
)

validation_threshold = int(
    (RFC_DATA_CONFIG["train_pct"]
     + RFC_DATA_CONFIG["validation_pct"]) * bucket_count / 100
)

policy_split = (
    source_df
    .select("policy_id")
    .where(F.col("policy_id").isNotNull())
    .distinct()
    .withColumn(
        "split_bucket",
        F.pmod(
            F.xxhash64(
                F.col("policy_id"),
                F.lit(sampling_salt )
            ),
            F.lit(bucket_count)
        )
    )
    .withColumn(
        "dataset",
        F.when(F.col("split_bucket") < train_threshold, "train")
         .when(F.col("split_bucket") < validation_threshold, "validation")
         .otherwise("test")
    )
)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window 

# Policy-level split distribution
policy_split_summary = (
    policy_split
    .groupBy("dataset")
    .agg(F.count("*").alias("policy_count"))
    .withColumn(
        "percentage",
        F.round(
            F.col("policy_count") * 100
            / F.sum("policy_count").over(
                Window.partitionBy()
            ),
            2
        )
    )
    .orderBy("dataset")
)

display(policy_split_summary)

# COMMAND ----------

# DBTITLE 1,Configurable sampling thresholds
# RFC — Configurable sampling thresholds

bucket_count = 10000

negative_threshold = round(
    RFC_DATA_CONFIG["negative_sample_pct"] * bucket_count / 100
)

validation_threshold = round(
    RFC_DATA_CONFIG["validation_sample_pct"] * bucket_count / 100
)

test_threshold = round(
    RFC_DATA_CONFIG["test_sample_pct"] * bucket_count / 100
)

print(f"Training negatives: {negative_threshold}/{bucket_count} buckets")
print(f"Validation: {validation_threshold}/{bucket_count} buckets")
print(f"Test: {test_threshold}/{bucket_count} buckets")

# COMMAND ----------


from pyspark.sql import functions as F

seed = RFC_DATA_CONFIG["random_seed"]

# Restore all snapshots after policy-level assignment
split_df = source_df.join(
    policy_split.select("policy_id", "dataset"),
    on="policy_id",
    how="inner"
)

# Full train / validation / test datasets
train_df = split_df.filter(F.col("dataset") == "train").drop("dataset")
validation_df = split_df.filter(F.col("dataset") == "validation").drop("dataset")
test_df = split_df.filter(F.col("dataset") == "test").drop("dataset")

# Deterministic sampling at the observation level
def sample_bucket(df):
    return F.pmod(
        F.xxhash64(
            F.col("policy_id"),
            F.col("effective_dttm"),
            F.lit(seed)
        ),
        F.lit(bucket_count)
    )

# Keep all training positives and configured percentage of negatives
sampled_train_df = train_df.filter(
    (F.col("target") == 1)
    | (
        (F.col("target") == 0)
        & (sample_bucket(train_df) < negative_threshold)
    )
)

# Sample validation and test without class-based filtering
sampled_validation_df = validation_df.filter(
    sample_bucket(validation_df) < validation_threshold
)

sampled_test_df = test_df.filter(
    sample_bucket(test_df) < test_threshold
)

print("Configurable RFC sampling DataFrames created successfully")

# COMMAND ----------


from pyspark.sql import functions as F

# Full and sampled dataset summaries
datasets = {
    "Train": (train_df, sampled_train_df),
    "Validation": (validation_df, sampled_validation_df),
    "Test": (test_df, sampled_test_df),
}

for name, (full_df, sample_df) in datasets.items():

    full_counts = {
        row["target"]: row["count"]
        for row in full_df.groupBy("target").count().collect()
    }

    sample_counts = {
        row["target"]: row["count"]
        for row in sample_df.groupBy("target").count().collect()
    }

    full_total = sum(full_counts.values())
    sample_total = sum(sample_counts.values())

    full_positives = full_counts.get(1, 0)
    sample_positives = sample_counts.get(1, 0)

    print(f"\n{name}")
    print(f"  Full rows:       {full_total:,}")
    print(f"  Sampled rows:    {sample_total:,}")
    print(f"  Full positives:  {full_positives:,}")
    print(f"  Sample positives:{sample_positives:,}")

    print(
        f"  Full target rate: "
        f"{100 * full_positives / full_total:.2f}%"
        if full_total else "  Full dataset empty"
    )

    print(
        f"  Sample target rate: "
        f"{100 * sample_positives / sample_total:.2f}%"
        if sample_total else "  Sample dataset empty"
    )

    if name == "Train":
        assert sample_positives == full_positives, (
            "Training sampling lost positive observations!"
        )
    else:
        assert sample_total <= full_total, (
            f"{name} sampled rows exceed full rows!"
        )

        if (
            RFC_DATA_CONFIG[
                f"{name.lower()}_sample_pct"
            ] == 100
        ):
            assert sample_total == full_total, (
                f"{name} should retain all observations!"
            )

print("\nSampling validation completed successfully.")

# COMMAND ----------

# DBTITLE 1,Test overlap and uniqueness

from pyspark.sql import functions as F

# 1. Check cross-dataset policy overlap
train_policies = train_df.select("policy_id").distinct()
val_policies = validation_df.select("policy_id").distinct()
test_policies = test_df.select("policy_id").distinct()

overlaps = {
    "Train vs Validation": train_policies.join(
        val_policies, "policy_id", "inner"
    ).count(),

    "Train vs Test": train_policies.join(
        test_policies, "policy_id", "inner"
    ).count(),

    "Validation vs Test": val_policies.join(
        test_policies, "policy_id", "inner"
    ).count()
}

for name, count in overlaps.items():
    print(f"{name} overlap: {count:,}")
    assert count == 0, f"Policy leakage detected: {name}"

# 2. Check uniqueness of policy + snapshot
for name, df in {
    "Train": train_df,
    "Validation": validation_df,
    "Test": test_df
}.items():

    duplicates = (
        df.groupBy("policy_id", "effective_dttm")
          .count()
          .filter(F.col("count") > 1)
          .limit(1)
          .count()
    )

    print(f"{name} duplicate policy-snapshot keys: {duplicates}")

    assert duplicates == 0, (
        f"{name} contains duplicate policy-snapshot observations"
    )

print("\nPolicy isolation and snapshot uniqueness checks passed.")

# COMMAND ----------

RFC_OUTPUT_DATASETS = {
    "train": (train_df, RFC_TRAIN_TABLE),
    "validation": (validation_df, RFC_VALIDATION_TABLE),
    "test": (test_df, RFC_TEST_TABLE),
    "train_sampled": (sampled_train_df, RFC_TRAIN_SAMPLED_TABLE),
    "validation_sampled": (
        sampled_validation_df,
        RFC_VALIDATION_SAMPLED_TABLE
    ),
    "test_sampled": (
        sampled_test_df,
        RFC_TEST_SAMPLED_TABLE
    ),
}

# COMMAND ----------

# DBTITLE 1,Delat persist test - Do NOT Run
# Step 8: Persist RFC datasets

RFC_OUTPUT_DATASETS = {
    "train": (train_df, RFC_TRAIN_TABLE),
    "validation": (validation_df, RFC_VALIDATION_TABLE),
    "test": (test_df, RFC_TEST_TABLE),
    "train_sampled": (sampled_train_df, RFC_TRAIN_SAMPLED_TABLE),
    "validation_sampled": (
        sampled_validation_df,
        RFC_VALIDATION_SAMPLED_TABLE
    ),
    "test_sampled": (
        sampled_test_df,
        RFC_TEST_SAMPLED_TABLE
    ),
}

# Prevent accidental overwrites
existing_tables = [
    table_name
    for _, table_name in RFC_OUTPUT_DATASETS.values()
    if spark.catalog.tableExists(table_name)
]

if existing_tables:
    raise RuntimeError(
        "Output tables already exist. No tables will be written:\n"
        + "\n".join(existing_tables)
    )

# Check for duplicate output destinations
output_names = [
    table_name for _, table_name in RFC_OUTPUT_DATASETS.values()
]

assert len(output_names) == len(set(output_names)), (
    "Duplicate output table names detected"
)

print("Output table validation passed.")
print("Tables to create:")

for dataset_name, (_, table_name) in RFC_OUTPUT_DATASETS.items():
    print(f"  {dataset_name}: {table_name}")

# COMMAND ----------

# DBTITLE 1,Step 9 Persist Delata for train test valid - do n0t RUN
# Step 9: Persist RFC datasets to Delta tables

for dataset_name, (df, table_name) in RFC_OUTPUT_DATASETS.items():

    print(f"Writing {dataset_name}: {table_name}")

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(table_name)
    )

    print(f"Completed: {dataset_name}")

print("\nAll six RFC Delta tables created successfully.")

for dataset_name, (_, table_name) in RFC_OUTPUT_DATASETS.items():
    print(f"  {dataset_name}: {table_name}")

# COMMAND ----------

# Step 10: Verify persisted RFC Delta tables
# numbers slightly change for this run/
RFC_EXPECTED_COUNTS = {
    "train": (17_934_370, 565_699),
    "validation": (1_998_344, 62_920),
    "test": (2_216_483, 70_396),
    "train_sampled": (1_433_482, 565_699),
    "validation_sampled": (1_998_344, 62_920),
    "test_sampled": (2_216_483, 70_396),
}

for dataset_name, (_, table_name) in RFC_OUTPUT_DATASETS.items():

    saved_df = spark.table(table_name)

    counts = {
        row["target"]: row["count"]
        for row in saved_df.groupBy("target").count().collect()
    }

    actual_rows = sum(counts.values())
    actual_positives = counts.get(1, 0)

    expected_rows, expected_positives = RFC_EXPECTED_COUNTS[dataset_name]

    assert actual_rows == expected_rows, (
        f"{dataset_name}: Row count mismatch! "
        f"Expected {expected_rows}, got {actual_rows}"
    )

    assert actual_positives == expected_positives, (
        f"{dataset_name}: Positive count mismatch! "
        f"Expected {expected_positives}, got {actual_positives}"
    )

    print(
        f"{dataset_name}: "
        f"Rows={actual_rows:,}, "
        f"Positives={actual_positives:,} — PASS"
    )

print("\nAll six persisted RFC datasets verified successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Data prep Steps

# COMMAND ----------

# MAGIC %md
# MAGIC initital rfc_train_df dataset crated

# COMMAND ----------

# DBTITLE 1,Model Train- prototype
from pyspark.sql import functions as F

# Read persisted sampled training data
rfc_train_df = spark.table(RFC_TRAIN_SAMPLED_TABLE)

# Prototype configuration
RFC_PROTOTYPE_SIZE = 10_000
RFC_PROTOTYPE_SEED = 42

# Approximate deterministic sampling
total_rows = rfc_train_df.count()

sample_fraction = min(
    1.0,
    RFC_PROTOTYPE_SIZE / total_rows
)

rfc_train_prototype_df = rfc_train_df.sample(
    withReplacement=False,
    fraction=sample_fraction,
    seed=RFC_PROTOTYPE_SEED
)

print("Full sampled training rows:", f"{total_rows:,}")
print("Prototype rows:", f"{rfc_train_prototype_df.count():,}")

rfc_train_prototype_df.groupBy("target").count().show()

# COMMAND ----------

# DBTITLE 1,Feature Exclusions
# Define initial RFC feature exclusions

# good candidate to calculate days_paid till EOF - ***policy_date_paid_to

RFC_EXCLUDE_COLUMNS = [
    "policy_id",
    "effective_dttm",
    "target",
    "target_first_event_date",
    "target_first_source",
    "days_from_snapshot_to_first_event",
    "policy_latest_coverchange_date",
    "policy_package_code_prev",
    "policy_package_code",
    "policy_package_cd",
    "policy_package_name_prev",
    "policy_package_name",
    "policy_package_dsc",
    "blr_first_login_date",
    "top3_lifetime_last_order_date",
    "blr_recent_event_date",
    "top1_lifetime_last_order_date",
    "top2_lifetime_last_order_date",
    "policy_date_paid_to",
    "policy_latest_coverchange_dtls",
    "policy_last_payment_date",
    "top1_24m_last_order_date",
    "top3_24m_last_order_date",
    "top2_24m_last_order_date",
    # "policy_org_discount_per",
    "policy_latest_call_date",
    "policy_is_hi_suspension_fh_start_date",
    "policy_is_hi_suspension_fh_end_date",

    "policy_future_org_id", "policy_future_org_name", "policy_future_org_discount_per", "policy_overseas_type", "policy_hosp_prod_desc", "policy_is_ancl_prod_desc", "policy_member_financial_status", "policy_portfolio_dsc", "policy_ambulance_product_id", "policy_ambulance_product_rank", "policy_fund_no", "state", "MEMBERSHIP_STATUS_SUSPENDED",

    "policy_ancillary_product_id",
    "policy_extras_product_id",
    "policy_hospital_product_id",
    "policy_national_pharmacy_product_id",
    "policy_pharmacy_product_id",
    "policy_future_org_from_date",
    "policy_future_org_to_date",

]

# policy_future_org_id
# policy_future_org_name
# policy_future_org_discount_per
# policy_overseas_type
# policy_hosp_prod_desc
# policy_is_ancl_prod_desc
# policy_member_financial_status
# policy_portfolio_dsc
# policy_ambulance_product_id
# policy_ambulance_product_rank
# policy_fund_no
# state
# MEMBERSHIP_STATUS_SUSPENDED


rfc_feature_columns = [
    c for c in rfc_train_df.columns
    if c not in RFC_EXCLUDE_COLUMNS
]

print("Total source columns:", len(rfc_train_df.columns))
print("Excluded columns:", RFC_EXCLUDE_COLUMNS)
print("Remaining candidate features:", len(rfc_feature_columns))

print("\nMissing exclusion columns:")
print([
    c for c in RFC_EXCLUDE_COLUMNS
    if c not in rfc_train_df.columns
])

# COMMAND ----------

# DBTITLE 1,Feature datatypes profile
# Profile RFC candidate feature data types

from collections import Counter

feature_types = {
    field.name: field.dataType.simpleString()
    for field in rfc_train_df.schema.fields
    if field.name in rfc_feature_columns
}

type_counts = Counter(feature_types.values())

print("Candidate features:", len(feature_types))
print("\nFeature data types:")

for dtype, count in sorted(type_counts.items()):
    print(f"{dtype}: {count}")

print("\nNon-numeric candidate features:")

numeric_types = {
    "int", "bigint", "double", "float",
    "smallint", "tinyint"
}

non_numeric_features = {
    col: dtype
    for col, dtype in feature_types.items()
    if dtype not in numeric_types
    and not dtype.startswith("decimal")
}

for col, dtype in non_numeric_features.items():
    print(f"{col}: {dtype}")

print("\nNon-numeric feature count:", len(non_numeric_features))

# COMMAND ----------

# Cell 21: Profile categorical feature cardinality
# profiles categorical feature cardinality using PySpark, calculating the approximate count of distinct values for string features and printing the top 20 highest-cardinality features.

from pyspark.sql import functions as F

string_features = [
    col for col, dtype in feature_types.items()
    if dtype == "string"
]

cardinality_exprs = [
    F.approx_count_distinct(F.col(c)).alias(c)
    for c in string_features
]

cardinality_row = (
    rfc_train_df
    .agg(*cardinality_exprs)
    .first()
    .asDict()
)

cardinality_summary = sorted(
    cardinality_row.items(),
    key=lambda x: x[1],
    reverse=True
)

print("Total string features:", len(string_features))
print("\nTop 20 highest-cardinality string features:")

for col, count in cardinality_summary[:20]:
    print(f"{col}: {count}")

# COMMAND ----------

# Cell 22: Refresh candidate features after exclusions

rfc_feature_columns = [
    c for c in rfc_train_df.columns
    if c not in RFC_EXCLUDE_COLUMNS
]

remaining_string_features = [
    c for c in rfc_feature_columns
    if feature_types.get(c) == "string"
]

print("Total candidate features:", len(rfc_feature_columns))
print("Remaining string features:", len(remaining_string_features))

print("\nRemaining high-cardinality strings (>100 distinct values):")

for col, count in cardinality_summary:
    if col in remaining_string_features and count > 100:
        print(f"{col}: {count}")

# COMMAND ----------

# DBTITLE 1,Temporal features
# Identify remaining temporal features

temporal_features = [
    (field.name, field.dataType.simpleString())
    for field in rfc_train_df.schema.fields
    if field.name in rfc_feature_columns
    and field.dataType.simpleString() in ("date", "timestamp")
]

print("Remaining date/timestamp features:", len(temporal_features))

for name, dtype in temporal_features:
    print(f"{name}: {dtype}")

# COMMAND ----------

# Cell 27: Identify remaining temporal features

temporal_features = [
    (field.name, field.dataType.simpleString())
    for field in rfc_train_df.schema.fields
    if field.name in rfc_feature_columns
    and field.dataType.simpleString() in ("date", "timestamp")
]

print("Remaining date/timestamp features:", len(temporal_features))

for name, dtype in temporal_features:
    print(f"{name}: {dtype}")

# COMMAND ----------

RFC_DATE_ENGINEERING_PENDING = True

rfc_temporal_columns = [
    field.name
    for field in rfc_train_df.schema.fields
    if field.name in rfc_feature_columns
    and field.dataType.simpleString() in ("date", "timestamp")
]

rfc_model_feature_columns = [
    c for c in rfc_feature_columns
    if c not in rfc_temporal_columns
]

print("Date engineering pending:", RFC_DATE_ENGINEERING_PENDING)
print("Temporarily excluded date columns:", len(rfc_temporal_columns))
print("Model candidate features:", len(rfc_model_feature_columns))

# COMMAND ----------

# # Cell 23: Inspect organisation discount percentage

# from pyspark.sql import functions as F

# rfc_train_df.select(
#     "policy_org_discount_per"
# ).groupBy(
#     "policy_org_discount_per"
# ).count().orderBy(
#     F.desc("count")
# ).show(20, truncate=False)

# # Check whether non-null values can be converted to double
# rfc_train_df.select(
#     F.count(
#         F.when(
#             F.col("policy_org_discount_per").isNotNull()
#             & F.expr(
#                 "try_cast(policy_org_discount_per AS DOUBLE)"
#             ).isNull(),
#             1
#         )
#     ).alias("non_numeric_values")
# ).show()

# COMMAND ----------

rfc_train_df = (
    rfc_train_df
    .withColumn(
        "policy_org_discount_per",
        F.col("policy_org_discount_per").cast("double")
    )
    .withColumn(
        "policy_org_discount_per_missing",
        F.when(
            F.col("policy_org_discount_per").isNull(), 1.0
        ).otherwise(0.0)
    )
)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Step 12: Inspect RFC prototype feature types
# Only the 10k Sample reflection here, do another inspection once full dataset is used for the model

from pyspark.sql.types import (
    StringType,
    NumericType,
    DateType,
    TimestampType
)

rfc_schema = {
    field.name: field.dataType
    for field in rfc_train_prototype_df.schema.fields
}

rfc_categorical_columns = [
    c for c in rfc_model_feature_columns
    if isinstance(rfc_schema[c], StringType)
]

rfc_numeric_columns = [
    c for c in rfc_model_feature_columns
    if isinstance(rfc_schema[c], NumericType)
]

rfc_temporal_columns_remaining = [
    c for c in rfc_model_feature_columns
    if isinstance(rfc_schema[c], (DateType, TimestampType))
]

print("Categorical features:", len(rfc_categorical_columns))
print("Numeric features:", len(rfc_numeric_columns))
print("Temporal features:", len(rfc_temporal_columns_remaining))

print(
    "Total classified features:",
    len(rfc_categorical_columns)
    + len(rfc_numeric_columns)
    + len(rfc_temporal_columns_remaining)
)

# COMMAND ----------

# DBTITLE 1,Categorical cols
print(f"Categorical feature count: {len(rfc_categorical_columns)}")

for i, column in enumerate(sorted(rfc_categorical_columns), start=1):
    print(f"{i:02d}. {column}")

# COMMAND ----------

# Step 14: Inspect categorical indicator values - only few selected just for observation

rfc_indicator_check_columns = [
    "policy_org_bupa_staff",
    "policy_org_corp_voluntary",
    "policy_is_hosp_prod_corp",
    "policy_is_ancl_prod_off_sale",
    "policy_latest_coverchange_30days",
    "Ancl_group_NoAncillary",
    "policy_org_discount_per"
]

for column in rfc_indicator_check_columns:
    print(f"\n{column}")

    (
        rfc_train_prototype_df
        .groupBy(column)
        .count()
        .orderBy(F.desc("count"))
        .show(10, truncate=False)
    )

    

# COMMAND ----------

# DBTITLE 1,Binary candidates
# Step 15: Identify Y/N indicator candidates
# Inspect Which one further manipulate to be binalry feature - 36 found

from pyspark.sql import functions as F

rfc_binary_candidate_columns = []

for column in rfc_categorical_columns:

    distinct_values = (
        rfc_train_prototype_df
        .select(
            F.upper(
                F.trim(F.col(column))
            ).alias("value")
        )
        .filter(
            F.col("value").isNotNull()
            & (F.col("value") != "")
        )
        .distinct()
        .limit(3)
        .collect()
    )

    values = {row["value"] for row in distinct_values}

    if values and values.issubset({"Y", "N"}):
        rfc_binary_candidate_columns.append(column)

print(
    "Potential binary indicator columns:",
    len(rfc_binary_candidate_columns)
)

for column in sorted(rfc_binary_candidate_columns):
    print(column)

# COMMAND ----------

# DBTITLE 1,Profile Binary
# Step 16: Profile binary indicator values

from pyspark.sql import functions as F

binary_profile_exprs = []

for c in rfc_binary_candidate_columns:
    value = F.upper(F.trim(F.col(c)))

    binary_profile_exprs.extend([
        F.sum(F.when(value == "Y", 1).otherwise(0)).alias(f"{c}__Y"),
        F.sum(F.when(value == "N", 1).otherwise(0)).alias(f"{c}__N"),
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}__NULL"),
        F.sum(F.when(value == "", 1).otherwise(0)).alias(f"{c}__BLANK"),
    ])

profile = rfc_train_prototype_df.agg(*binary_profile_exprs).first().asDict()

binary_summary = [
    (
        c,
        profile[f"{c}__Y"],
        profile[f"{c}__N"],
        profile[f"{c}__NULL"],
        profile[f"{c}__BLANK"]
    )
    for c in rfc_binary_candidate_columns
]

display(
    spark.createDataFrame(
        binary_summary,
        ["feature", "Y", "N", "NULL", "BLANK"]
    ).orderBy("feature")
)

# COMMAND ----------

# Step 17: Finalise preliminary RFC feature groups

rfc_binary_columns = sorted(rfc_binary_candidate_columns)

rfc_remaining_categorical_columns = sorted(
    set(rfc_categorical_columns) - set(rfc_binary_columns)
)

print("Binary indicators:", len(rfc_binary_columns))
print(
    "Remaining categorical features:",
    len(rfc_remaining_categorical_columns)
)
print("Numeric features:", len(rfc_numeric_columns))

print("\nRemaining categorical columns:")
for c in rfc_remaining_categorical_columns:
    print(c)

# COMMAND ----------

# DBTITLE 1,Feature group counts
# Step 18: Refine RFC feature groups

RFC_HIGH_CARDINALITY_COLUMNS = [
    "blr_recent_order_item",
    "policy_latest_call_app_tag",
    "top1_24m_order_item",
    "top1_lifetime_order_item",
    "top2_24m_order_item",
    "top2_lifetime_order_item",
    "top3_24m_order_item",
    "top3_lifetime_order_item",
]

RFC_NUMERIC_STRING_COLUMNS = [
    "policy_org_discount_per"
]

RFC_ADDITIONAL_BINARY_COLUMNS = [
    "Hosp_group_OverseasHospitalOnsale"
]

rfc_binary_columns = sorted(
    set(rfc_binary_candidate_columns)
    | set(RFC_ADDITIONAL_BINARY_COLUMNS)
)

rfc_onehot_columns = sorted(
    set(rfc_categorical_columns)
    - set(rfc_binary_columns)
    - set(RFC_HIGH_CARDINALITY_COLUMNS)
    - set(RFC_NUMERIC_STRING_COLUMNS)
)

rfc_numeric_model_columns = sorted(
    set(rfc_numeric_columns)
    | set(RFC_NUMERIC_STRING_COLUMNS)
)

print("Binary indicators:", len(rfc_binary_columns))
print("One-hot categorical features:", len(rfc_onehot_columns))
print("Numeric features:", len(rfc_numeric_model_columns))
print("Deferred high-cardinality features:", len(RFC_HIGH_CARDINALITY_COLUMNS))

assert (
    len(rfc_binary_columns)
    + len(rfc_onehot_columns)
    + len(rfc_numeric_model_columns)
    + len(RFC_HIGH_CARDINALITY_COLUMNS)
) == len(rfc_model_feature_columns)

print("Feature grouping validated successfully.")

# COMMAND ----------

# DBTITLE 1,Feature transformation
# Step 19: Reusable RFC feature transformations

from pyspark.sql import functions as F

def prepare_rfc_features(df):

    # Convert Y/N indicators to numeric
    for c in rfc_binary_columns:
        value = F.upper(F.trim(F.col(c)))

        df = df.withColumn(
            c,
            F.when(value == "Y", F.lit(1.0))
             .when(value == "N", F.lit(0.0))
             .otherwise(F.lit(None).cast("double"))
        )

    # Convert numeric string safely
    df = df.withColumn(
        "policy_org_discount_per",
        F.expr("try_cast(policy_org_discount_per AS DOUBLE)")
    )

    # Add missingness indicator
    df = df.withColumn(
        "policy_org_discount_per_missing",
        F.when(
            F.col("policy_org_discount_per").isNull(),
            F.lit(1.0)
        ).otherwise(F.lit(0.0))
    )

    return df


# Apply to prototype only
rfc_train_prototype_prepared_df = prepare_rfc_features(
    rfc_train_prototype_df
)

print("RFC prototype feature transformations completed.")

rfc_train_prototype_prepared_df.select(
    "policy_org_bupa_staff",
    "policy_latest_coverchange_30days",
    "policy_org_discount_per",
    "policy_org_discount_per_missing"
).show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,Final feature list count
# Step 20: Finalise RFC prototype feature columns

RFC_ACTIVE_FEATURE_COLUMNS = (
    rfc_binary_columns
    + rfc_onehot_columns
    + rfc_numeric_model_columns
    + ["policy_org_discount_per_missing"]
)

# Validate no duplicate feature names
assert len(RFC_ACTIVE_FEATURE_COLUMNS) == len(
    set(RFC_ACTIVE_FEATURE_COLUMNS)
), "Duplicate feature names detected"

# Validate all features exist after transformation
missing_features = (
    set(RFC_ACTIVE_FEATURE_COLUMNS)
    - set(rfc_train_prototype_prepared_df.columns)
)

assert not missing_features, (
    f"Missing features: {sorted(missing_features)}"
)

print("Binary features:", len(rfc_binary_columns))
print("One-hot features:", len(rfc_onehot_columns))
print("Numeric features:", len(rfc_numeric_model_columns) + 1)
print("Total active features:", len(RFC_ACTIVE_FEATURE_COLUMNS))
print("Deferred high-cardinality features:", len(RFC_HIGH_CARDINALITY_COLUMNS))
print("Feature validation passed.")

# COMMAND ----------

# print(RFC_ACTIVE_FEATURE_COLUMNS)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Train

# COMMAND ----------

# Step 21: Convert prepared RFC prototype to pandas

rfc_prototype_pd = (
    rfc_train_prototype_prepared_df
    .select(*RFC_ACTIVE_FEATURE_COLUMNS, "target")
    .toPandas()
)

X_train = rfc_prototype_pd[RFC_ACTIVE_FEATURE_COLUMNS].copy()
y_train = rfc_prototype_pd["target"].astype(int)

print("Training features:", X_train.shape)
print("Training target:", y_train.shape)

print("\nTarget distribution:")
print(y_train.value_counts())

print("\nData types:")
print(X_train.dtypes.value_counts())

# COMMAND ----------

# DBTITLE 1,Object conversion test

# Step 22: Inspect pandas object column types

from collections import Counter

object_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

expected_categorical = set(rfc_onehot_columns)

unexpected_object_columns = [
    c for c in object_columns
    if c not in expected_categorical
]

print("Total object columns:", len(object_columns))
print("Expected categorical columns:", len(expected_categorical))
print("Unexpected object columns:", len(unexpected_object_columns))

# Inspect actual Python types in unexpected object columns
type_summary = Counter()

for c in unexpected_object_columns:
    non_null = X_train[c].dropna()

    if not non_null.empty:
        type_summary[type(non_null.iloc[0]).__name__] += 1
    else:
        type_summary["all_null"] += 1

print("\nUnexpected object column types:")
for dtype, count in type_summary.items():
    print(f"{dtype}: {count}")

print("\nExample unexpected columns:")
print(unexpected_object_columns[:15])

# COMMAND ----------


# Step 23: Fix numeric pandas data types

import pandas as pd

# All numeric model inputs, including binary indicators
rfc_all_numeric_columns = (
    rfc_binary_columns
    + rfc_numeric_model_columns
    + ["policy_org_discount_per_missing"]
)

# Convert Spark Decimal objects and other numeric representations
for c in rfc_all_numeric_columns:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")

# Validate that only intended categorical columns remain as objects
remaining_object_columns = set(
    X_train.select_dtypes(include=["object"]).columns
)

assert remaining_object_columns == set(rfc_onehot_columns), (
    "Unexpected object columns remain: "
    f"{sorted(remaining_object_columns - set(rfc_onehot_columns))}"
)

print("Numeric columns:", len(rfc_all_numeric_columns))
print("Categorical columns:", len(rfc_onehot_columns))
print("Total features:", X_train.shape[1])

print("\nPandas data types:")
print(X_train.dtypes.value_counts())

print("\nAll-null numeric columns:")
print(
    X_train[rfc_all_numeric_columns]
    .columns[
        X_train[rfc_all_numeric_columns].isna().all()
    ].tolist()
)

# COMMAND ----------

# Replace X_train with your actual training feature DataFrame name
flagged = sorted(
    c for c in X_train.columns
    if (
        "future" in c.lower()
        or "target" in c.lower()
        or c.lower().endswith("_id")
        or c.lower() in {
            "policy_id",
            "effective_dttm",
            "target_first_event_date",
            "target_first_source",
            "days_from_snapshot_to_first_event"
        }
    )
)

print("Total features:", len(X_train.columns))
print("Potentially excluded features:", len(flagged))

for c in flagged:
    print(c)

# COMMAND ----------

# DBTITLE 1,Build feature set
# Step 24: Exclude product identifiers from RFC prototype
# 691 identified as potential candidtes for the model build

RFC_EXCLUDED_PRODUCT_ID_COLUMNS = [
    "policy_national_pharmacy_product_id",
    "policy_pharmacy_product_id",
]

# Update numeric feature group
rfc_numeric_model_columns = sorted(
    set(rfc_numeric_model_columns)
    - set(RFC_EXCLUDED_PRODUCT_ID_COLUMNS)
)

# Rebuild active feature list
RFC_ACTIVE_FEATURE_COLUMNS = (
    rfc_binary_columns
    + rfc_onehot_columns
    + rfc_numeric_model_columns
    + ["policy_org_discount_per_missing"]
)

# Update existing pandas prototype
X_train = X_train[RFC_ACTIVE_FEATURE_COLUMNS].copy()

assert X_train.shape[1] == len(RFC_ACTIVE_FEATURE_COLUMNS)
assert len(RFC_ACTIVE_FEATURE_COLUMNS) == len(set(RFC_ACTIVE_FEATURE_COLUMNS))

print("Excluded product IDs:", RFC_EXCLUDED_PRODUCT_ID_COLUMNS)
print("Numeric features:", len(rfc_numeric_model_columns) + 1 + len(rfc_binary_columns))
print("Categorical features:", len(rfc_onehot_columns))
print("Total active features:", len(RFC_ACTIVE_FEATURE_COLUMNS))
print("Training shape:", X_train.shape)

# COMMAND ----------

remaining = [
    c for c in [
        "policy_ancillary_product_id",
        "policy_extras_product_id",
        "policy_hospital_product_id",
        "policy_future_org_from_date",
        "policy_future_org_to_date",
    ]
    if c in X_train.columns
]

print("Remaining unwanted features:", remaining)

# COMMAND ----------

# DBTITLE 1,Preprocesser step 1
# Step 25: Build RFC preprocessing pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

rfc_numeric_pipeline_columns = (
    rfc_binary_columns
    + rfc_numeric_model_columns
    + ["policy_org_discount_per_missing"]
)

rfc_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            rfc_onehot_columns
        ),
        (
            "numeric",
            "passthrough",
            rfc_numeric_pipeline_columns
        )
    ],
    remainder="drop",
    sparse_threshold=0.3
)

print("RFC preprocessing pipeline created.")
print("Categorical columns:", len(rfc_onehot_columns))
print("Numeric columns:", len(rfc_numeric_pipeline_columns))
print("Total input features:", X_train.shape[1])

# COMMAND ----------

excluded = {
    "policy_ancillary_product_id",
    "policy_extras_product_id",
    "policy_hospital_product_id",
    "policy_national_pharmacy_product_id",
    "policy_pharmacy_product_id",
    "policy_future_org_from_date",
    "policy_future_org_to_date",
}

active = set(rfc_numeric_pipeline_columns) | set(rfc_onehot_columns)

print("Excluded columns still selected:", sorted(active & excluded))
print("cash_rate_target retained:", "cash_rate_target" in active)

# COMMAND ----------

# DBTITLE 1,Test preprocessor
# Step 26: Validate RFC preprocessing pipeline

from scipy import sparse
import numpy as np

X_train_transformed = rfc_preprocessor.fit_transform(X_train)

print("Original shape:", X_train.shape)
print("Transformed shape:", X_train_transformed.shape)

print(
    "Output format:",
    "Sparse" if sparse.issparse(X_train_transformed) else "Dense"
)

print(
    "Transformed feature count:",
    len(rfc_preprocessor.get_feature_names_out())
)

# Check for infinite values without densifying a sparse matrix
values = (
    X_train_transformed.data
    if sparse.issparse(X_train_transformed)
    else X_train_transformed
)

print("Infinite values:", np.isinf(values).sum())
print("Missing values:", np.isnan(values).sum())

# COMMAND ----------

# DBTITLE 1,test valid sample prototype only set
# Step 26: Prepare RFC validation and test prototypes

RFC_EVALUATION_PROTOTYPE_SIZE = 10_000
RFC_EVALUATION_SEED = 42

rfc_validation_df = spark.table(RFC_VALIDATION_SAMPLED_TABLE)
rfc_test_df = spark.table(RFC_TEST_SAMPLED_TABLE)

# Small prototypes for technical pipeline validation only
rfc_validation_prototype_df = rfc_validation_df.sample(
    withReplacement=False,
    fraction=min(
        1.0,
        RFC_EVALUATION_PROTOTYPE_SIZE / rfc_validation_df.count()
    ),
    seed=RFC_EVALUATION_SEED
)

rfc_test_prototype_df = rfc_test_df.sample(
    withReplacement=False,
    fraction=min(
        1.0,
        RFC_EVALUATION_PROTOTYPE_SIZE / rfc_test_df.count()
    ),
    seed=RFC_EVALUATION_SEED
)

# Apply the SAME feature transformation function
rfc_validation_prototype_prepared_df = prepare_rfc_features(
    rfc_validation_prototype_df
)

rfc_test_prototype_prepared_df = prepare_rfc_features(
    rfc_test_prototype_df
)

print("Validation prototype:", rfc_validation_prototype_prepared_df.count())
print("Test prototype:", rfc_test_prototype_prepared_df.count())

# COMMAND ----------


# Step 27: Prepare validation and test data for sklearn

import pandas as pd

def prepare_rfc_pandas_dataset(spark_df):
    pdf = (
        spark_df
        .select(*RFC_ACTIVE_FEATURE_COLUMNS, "target")
        .toPandas()
    )

    X = pdf[RFC_ACTIVE_FEATURE_COLUMNS].copy()
    y = pdf["target"].astype(int)

    # Use the same numeric conversions as training
    numeric_columns = (
        rfc_binary_columns
        + rfc_numeric_model_columns
        + ["policy_org_discount_per_missing"]
    )

    for c in numeric_columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, y


X_valid, y_valid = prepare_rfc_pandas_dataset(
    rfc_validation_prototype_prepared_df
)

X_test, y_test = prepare_rfc_pandas_dataset(
    rfc_test_prototype_prepared_df
)

assert list(X_train.columns) == list(X_valid.columns)
assert list(X_train.columns) == list(X_test.columns)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_valid.shape, y_valid.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nPositive rates:")
print(f"Training:   {y_train.mean():.2%}")
print(f"Validation: {y_valid.mean():.2%}")
print(f"Test:       {y_test.mean():.2%}")

print("\nFeature schemas match.")

# COMMAND ----------

# DBTITLE 1,Build pipline XGB Classifier
# Step 28: Build RFC XGBoost prototype pipeline

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

rfc_xgb_prototype = Pipeline(
    steps=[
        ("preprocessor", rfc_preprocessor),
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
                n_jobs=4
            )
        )
    ]
)

print("RFC XGBoost prototype pipeline created.")
print("Pipeline steps:", list(rfc_xgb_prototype.named_steps.keys()))
print("Model:", type(rfc_xgb_prototype.named_steps["classifier"]).__name__)

# COMMAND ----------

# DBTITLE 1,Fit
# Step 29: Fit RFC XGBoost prototype

import time

start_time = time.time()

# Fits preprocessing and XGBoost on TRAINING data only
rfc_xgb_prototype.fit(X_train, y_train)

training_seconds = time.time() - start_time

print("RFC XGBoost prototype training completed.")
print(f"Training time: {training_seconds:.2f} seconds")

print("\nModel details:")
print(
    "Boosted trees:",
    rfc_xgb_prototype.named_steps["classifier"].n_estimators
)

print(
    "Transformed feature count:",
    len(
        rfc_xgb_prototype.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )
)

print("\nTraining observations:", len(X_train))
print("Validation observations:", len(X_valid))
print("Test observations:", len(X_test))

# COMMAND ----------

# DBTITLE 1,Evaluate
# Step 30: Evaluate RFC prototype on validation data

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

# Use the fitted pipeline — no refitting
rfc_valid_probabilities = (
    rfc_xgb_prototype.predict_proba(X_valid)[:, 1]
)

rfc_valid_roc_auc = roc_auc_score(
    y_valid,
    rfc_valid_probabilities
)

rfc_valid_pr_auc = average_precision_score(
    y_valid,
    rfc_valid_probabilities
)

print("RFC validation results")
print("----------------------")
print(f"Observations: {len(y_valid):,}")
print(f"Positive rate: {y_valid.mean():.4%}")
print(f"ROC-AUC:       {rfc_valid_roc_auc:.4f}")
print(f"PR-AUC (AP):   {rfc_valid_pr_auc:.4f}")
print(f"PR baseline:   {y_valid.mean():.4f}")

print("\nPredicted probability range:")
print(
    f"Min: {rfc_valid_probabilities.min():.4f}, "
    f"Mean: {rfc_valid_probabilities.mean():.4f}, "
    f"Max: {rfc_valid_probabilities.max():.4f}"
)

# COMMAND ----------

# Step 30: Comprehensive RFC validation evaluation

import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Predict on validation only — no refitting
rfc_valid_probabilities = (
    rfc_xgb_prototype.predict_proba(X_valid)[:, 1]
)

# Diagnostic threshold only — not production-approved
RFC_EVALUATION_THRESHOLD = 0.50

rfc_valid_predictions = (
    rfc_valid_probabilities >= RFC_EVALUATION_THRESHOLD
).astype(int)

rfc_validation_metrics = {
    "ROC-AUC": roc_auc_score(
        y_valid, rfc_valid_probabilities
    ),
    "PR-AUC (Average Precision)": average_precision_score(
        y_valid, rfc_valid_probabilities
    ),
    "Accuracy": accuracy_score(
        y_valid, rfc_valid_predictions
    ),
    "Precision": precision_score(
        y_valid, rfc_valid_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_valid, rfc_valid_predictions, zero_division=0
    ),
    "F1-score": f1_score(
        y_valid, rfc_valid_predictions, zero_division=0
    ),
}

print("RFC VALIDATION METRICS")
print("----------------------")
print(f"Observations: {len(y_valid):,}")
print(f"Positive rate: {y_valid.mean():.2%}")
print(f"Classification threshold: {RFC_EVALUATION_THRESHOLD}")

for metric, value in rfc_validation_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nCONFUSION MATRIX")
print("Rows = Actual, Columns = Predicted")

print(
    confusion_matrix(
        y_valid,
        rfc_valid_predictions,
        labels=[0, 1]
    )
)

print("\nPREDICTED PROBABILITY SUMMARY")
print(
    pd.Series(rfc_valid_probabilities)
    .describe()
)

# COMMAND ----------


# Step 31: RFC validation ranking performance

import numpy as np
import pandas as pd

rfc_validation_ranking = pd.DataFrame({
    "actual": np.asarray(y_valid),
    "score": rfc_valid_probabilities
}).sort_values("score", ascending=False).reset_index(drop=True)

total_positives = rfc_validation_ranking["actual"].sum()
baseline_rate = rfc_validation_ranking["actual"].mean()

results = []

for pct in [1, 5, 10, 20]:
    k = int(np.ceil(len(rfc_validation_ranking) * pct / 100))

    selected = rfc_validation_ranking.head(k)
    captured = int(selected["actual"].sum())

    precision_at_k = captured / k
    recall_at_k = captured / total_positives
    lift_at_k = precision_at_k / baseline_rate

    results.append({
        "Top %": pct,
        "Customers": k,
        "Actual reviewers": captured,
        "Precision@K": round(precision_at_k, 4),
        "Recall@K": round(recall_at_k, 4),
        "Lift@K": round(lift_at_k, 2)
    })

rfc_lift_results = pd.DataFrame(results)

print("RFC VALIDATION — TOP-K PERFORMANCE")
print(f"Population review rate: {baseline_rate:.2%}")
display(rfc_lift_results)

# COMMAND ----------

# Step 31A: Visualise RFC validation lift by targeting proportion

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    rfc_lift_results["Top %"].astype(str) + "%",
    rfc_lift_results["Lift@K"],
    color="#2563EB",
    width=0.55
)

# Baseline: random targeting produces 1x lift
ax.axhline(
    y=1.0,
    color="#DC2626",
    linestyle="--",
    linewidth=1.5,
    label="Random targeting (1× lift)"
)

# Display lift values above bars
for bar in bars:
    height = bar.get_height()

    ax.annotate(
        f"{height:.2f}×",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

ax.set_title(
    "RFC Validation — Lift by Targeting Proportion",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Top % of Scored Observations")
ax.set_ylabel("Lift over Population Review Rate")

ax.set_ylim(0, rfc_lift_results["Lift@K"].max() * 1.2)

ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()



# COMMAND ----------

# Step 32: Inspect RFC XGBoost feature importance

import pandas as pd

rfc_fitted_preprocessor = (
    rfc_xgb_prototype.named_steps["preprocessor"]
)

rfc_fitted_classifier = (
    rfc_xgb_prototype.named_steps["classifier"]
)

rfc_transformed_feature_names = (
    rfc_fitted_preprocessor.get_feature_names_out()
)

rfc_feature_importance = pd.DataFrame({
    "feature": rfc_transformed_feature_names,
    "importance": rfc_fitted_classifier.feature_importances_
}).sort_values(
    "importance", ascending=False
).reset_index(drop=True)

print("Transformed features:", len(rfc_transformed_feature_names))
print("\nTop 20 features:")
display(rfc_feature_importance.head(20))

# COMMAND ----------

# DBTITLE 1,Potential leakage identify
# Step 33: Inventory potentially leakage-sensitive RFC features

RFC_LEAKAGE_REVIEW_TERMS = [
    "review",
    "coverchange",
    "cover_change",
    "retention",
    "future",
    "call"
]

rfc_leakage_review_columns = sorted([
    c for c in RFC_ACTIVE_FEATURE_COLUMNS
    if any(
        term in c.lower()
        for term in RFC_LEAKAGE_REVIEW_TERMS
    )
])

print(
    "Features requiring point-in-time review:",
    len(rfc_leakage_review_columns)
)

for c in rfc_leakage_review_columns:
    print(c)

# COMMAND ----------

# Step 34: Prioritise leakage review using feature importance

rfc_top20_features = (
    rfc_feature_importance
    .head(20)["feature"]
    .tolist()
)

rfc_priority_leakage_features = []

for feature in rfc_top20_features:
    for candidate in rfc_leakage_review_columns:

        # ColumnTransformer adds numeric__ or categorical__ prefixes
        if feature == f"numeric__{candidate}" or (
            feature.startswith(f"categorical__{candidate}_")
        ):
            rfc_priority_leakage_features.append({
                "feature": feature,
                "source_column": candidate
            })
            break

rfc_priority_leakage_df = pd.DataFrame(
    rfc_priority_leakage_features,
    columns=["feature", "source_column"]
)

print(
    "Top-20 features flagged for leakage review:",
    len(rfc_priority_leakage_df)
)

display(rfc_priority_leakage_df)

# COMMAND ----------

# Step 36: RFC validation review rate by score decile

import pandas as pd

rfc_decile_df = pd.DataFrame({
    "actual": y_valid.to_numpy(),
    "score": rfc_valid_probabilities
})

# Decile 1 = highest predicted propensity
rfc_decile_df["decile"] = (
    pd.qcut(
        rfc_decile_df["score"].rank(method="first", ascending=False),
        q=10,
        labels=False
    ) + 1
)

rfc_decile_results = (
    rfc_decile_df
    .groupby("decile")
    .agg(
        observations=("actual", "size"),
        reviewers=("actual", "sum"),
        avg_predicted_score=("score", "mean")
    )
    .reset_index()
)

rfc_decile_results["review_rate"] = (
    rfc_decile_results["reviewers"]
    / rfc_decile_results["observations"]
)

rfc_decile_results["lift"] = (
    rfc_decile_results["review_rate"]
    / y_valid.mean()
)

display(rfc_decile_results)

# COMMAND ----------

# DBTITLE 1,RAM tests
# Step 37: Estimate RFC full-training memory requirements

import pandas as pd

RFC_FULL_TRAIN_ROWS = rfc_train_df.count()

# Actual memory used by the prepared pandas prototype
prototype_memory_bytes = X_train.memory_usage(
    deep=True
).sum()

prototype_memory_gb = prototype_memory_bytes / (1024 ** 3)

# Linear estimate for the full sampled training dataset
estimated_full_memory_gb = (
    prototype_memory_bytes
    / len(X_train)
    * RFC_FULL_TRAIN_ROWS
    / (1024 ** 3)
)

print(f"Prototype rows: {len(X_train):,}")
print(f"Prototype memory: {prototype_memory_gb:.3f} GB")

print(f"\nFull sampled training rows: {RFC_FULL_TRAIN_ROWS:,}")
print(
    f"Estimated pandas feature memory: "
    f"{estimated_full_memory_gb:.2f} GB"
)

print(
    f"Estimated with 3x working-memory allowance: "
    f"{estimated_full_memory_gb * 3:.2f} GB"
)

# COMMAND ----------

# Step 38: Inspect Databricks driver resources

import os
import psutil

memory = psutil.virtual_memory()
process = psutil.Process(os.getpid())

print("RFC DRIVER RESOURCE CHECK")
print("-------------------------")

print(f"Total system memory:     {memory.total / (1024**3):.2f} GB")
print(f"Available system memory: {memory.available / (1024**3):.2f} GB")
print(f"Current Python RSS:      {process.memory_info().rss / (1024**3):.2f} GB")
print(f"Logical CPU count:       {psutil.cpu_count(logical=True)}")

# Container/cgroup memory limits, where exposed
for path in [
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes"
]:
    if os.path.exists(path):
        with open(path) as f:
            print(f"Cgroup memory limit ({path}): {f.read().strip()}")

# COMMAND ----------

# Step 40: Check numeric precision before float32 conversion

import numpy as np
import pandas as pd

precision_results = []

for col in rfc_numeric_pipeline_columns:
    original = pd.to_numeric(X_train[col], errors="coerce")
    valid = original.dropna()

    if valid.empty:
        continue

    converted = valid.astype(np.float32).astype(np.float64)

    absolute_error = np.abs(valid.to_numpy(dtype=np.float64) - converted)

    relative_error = absolute_error / np.maximum(
        np.abs(valid.to_numpy(dtype=np.float64)), 1.0
    )

    precision_results.append({
        "feature": col,
        "max_abs_value": np.abs(valid).max(),
        "max_abs_error": absolute_error.max(),
        "max_rel_error": relative_error.max(),
        "values_changed": int((absolute_error > 0).sum())
    })

rfc_precision_check = pd.DataFrame(precision_results)

print("RFC FLOAT32 PRECISION CHECK")
print("---------------------------")

print("Numeric features checked:", len(rfc_precision_check))

print(
    "Features with relative error > 1e-5:",
    (rfc_precision_check["max_rel_error"] > 1e-5).sum()
)

print("\nTop 10 features by absolute conversion error:")

display(
    rfc_precision_check
    .sort_values("max_abs_error", ascending=False)
    .head(10)
)

# COMMAND ----------

# Step 41: Inspect encoded date features

date_columns = [
    "policy_org_from_date",
    "policy_org_to_date"
]

for col in date_columns:
    print(f"\nFEATURE: {col}")
    print("-" * 50)

    print("Data type:", X_train[col].dtype)
    print("Missing values:", X_train[col].isna().sum())

    print("\nMost frequent values:")
    print(X_train[col].value_counts(dropna=False).head(10))

# COMMAND ----------

# Step 42: Exclude raw YYYYMMDD organisation dates

RFC_RAW_DATE_COLUMNS = [
    "policy_org_from_date",
    "policy_org_to_date"
]

# Remove from the numeric feature list
rfc_numeric_pipeline_columns = [
    col for col in rfc_numeric_pipeline_columns
    if col not in RFC_RAW_DATE_COLUMNS
]

# Remove from the active feature list
RFC_ACTIVE_FEATURE_COLUMNS = [
    col for col in RFC_ACTIVE_FEATURE_COLUMNS
    if col not in RFC_RAW_DATE_COLUMNS
]

# Rebuild the preprocessor using the updated feature lists
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

rfc_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            rfc_onehot_columns
        ),
        (
            "numeric",
            "passthrough",
            rfc_numeric_pipeline_columns
        )
    ],
    remainder="drop",
    sparse_threshold=0.3
)

print("Excluded raw date features:", RFC_RAW_DATE_COLUMNS)
print("Active input features:", len(RFC_ACTIVE_FEATURE_COLUMNS))
print("Numeric features:", len(rfc_numeric_pipeline_columns))
print("Categorical features:", len(rfc_onehot_columns))

# COMMAND ----------

# DBTITLE 1,Nummrical feature tests
# Step 43: Identify potentially problematic numeric features

import numpy as np
import pandas as pd

rfc_large_numeric_features = []

for col in rfc_numeric_pipeline_columns:
    values = pd.to_numeric(X_train[col], errors="coerce")
    valid = values.dropna()

    if valid.empty:
        continue

    max_abs = valid.abs().max()

    if max_abs > 1_000_000:
        rfc_large_numeric_features.append({
            "feature": col,
            "max_abs_value": max_abs,
            "unique_values": valid.nunique(),
            "sample_values": valid.head(3).tolist()
        })

rfc_large_numeric_results = pd.DataFrame(
    rfc_large_numeric_features
)

if rfc_large_numeric_results.empty:
    print("Features with absolute values above 1 million:",
        len(rfc_large_numeric_results))
else:
    display(rfc_large_numeric_results)

# COMMAND ----------

# Step 44: Convert numeric model inputs to float32

X_train[rfc_numeric_pipeline_columns] = (
    X_train[rfc_numeric_pipeline_columns]
    .astype("float32")
)

print("Training feature shape:", X_train.shape)

print(
    "Active model feature count:",
    len(RFC_ACTIVE_FEATURE_COLUMNS)
)

print(
    "Numeric float32 columns:",
    X_train[rfc_numeric_pipeline_columns]
    .select_dtypes(include=["float32"])
    .shape[1]
)

print(
    "Prototype memory:",
    round(X_train.memory_usage(deep=True).sum() / 1024**2, 2),
    "MB"
)

# COMMAND ----------

# Step 45: Prepare full sampled RFC training data in Spark

rfc_full_train_prepared_df = prepare_rfc_features(
    rfc_train_df
).select(
    *RFC_ACTIVE_FEATURE_COLUMNS,
    "target"
)

print("Full training columns:", len(rfc_full_train_prepared_df.columns))
print("Active model features:", len(RFC_ACTIVE_FEATURE_COLUMNS))

print("\nTarget distribution:")
rfc_full_train_prepared_df.groupBy("target").count().show()

# COMMAND ----------

# DBTITLE 1,Full training data prep - Error full dataset run

# # Step 46: Collect full sampled training data into pandas

# import gc
# import numpy as np

# # Collect only the approved model features and target
# rfc_full_train_pd = rfc_full_train_prepared_df.toPandas()

# print("Collected rows:", f"{len(rfc_full_train_pd):,}")

# # Convert numeric features to float32
# rfc_full_train_pd[rfc_numeric_pipeline_columns] = (
#     rfc_full_train_pd[rfc_numeric_pipeline_columns]
#     .astype(np.float32)
# )

# # Separate features and target
# X_train_full = rfc_full_train_pd[RFC_ACTIVE_FEATURE_COLUMNS].copy()
# y_train_full = rfc_full_train_pd["target"].astype(np.int8).copy()

# # Release the intermediate pandas DataFrame
# del rfc_full_train_pd
# gc.collect()

# print("X_train_full shape:", X_train_full.shape)
# print("y_train_full shape:", y_train_full.shape)

# print(
#     "Positive observations:",
#     f"{int(y_train_full.sum()):,}"
# )

# print(
#     "Negative observations:",
#     f"{int((y_train_full == 0).sum()):,}"
# )

# print(
#     "Feature memory:",
#     f"{X_train_full.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
# )

# COMMAND ----------

# DBTITLE 1,Batch Sizing

# Step 46A: Test one bounded training-data batch

from pyspark.sql import functions as F

RFC_COLLECTION_BATCHES = 10

rfc_train_batched_df = (
    rfc_train_df
    .withColumn(
        "_collection_batch",
        F.pmod(
            F.xxhash64(
                F.col("policy_id"),
                F.col("effective_dttm"),
                F.lit(42)
            ),
            F.lit(RFC_COLLECTION_BATCHES)
        )
    )
)

rfc_batch_0_df = (
    prepare_rfc_features(
        rfc_train_batched_df.filter(
            F.col("_collection_batch") == 0
        )
    )
    .select(*RFC_ACTIVE_FEATURE_COLUMNS, "target")
)

rfc_batch_0_pd = rfc_batch_0_df.toPandas()

print("Batch 0 rows:", len(rfc_batch_0_pd))
print(
    "Batch 0 memory:",
    round(
        rfc_batch_0_pd.memory_usage(deep=True).sum() / 1024**3,
        2
    ),
    "GB"
)

# COMMAND ----------

# DBTITLE 1,Batches
# Step 46B: Collect full RFC training data in bounded batches

import gc
import numpy as np
import pandas as pd

rfc_train_batches = []

# Reuse the successfully collected first batch
rfc_batch_0_pd[rfc_numeric_pipeline_columns] = (
    rfc_batch_0_pd[rfc_numeric_pipeline_columns]
    .astype(np.float32)
)

rfc_train_batches.append(rfc_batch_0_pd)

print(f"Batch 0: {len(rfc_batch_0_pd):,} rows")

# Collect remaining batches
for batch_id in range(1, RFC_COLLECTION_BATCHES):

    batch_df = (
        prepare_rfc_features(
            rfc_train_batched_df.filter(
                F.col("_collection_batch") == batch_id
            )
        )
        .select(*RFC_ACTIVE_FEATURE_COLUMNS, "target")
    )

    batch_pd = batch_df.toPandas()

    batch_pd[rfc_numeric_pipeline_columns] = (
        batch_pd[rfc_numeric_pipeline_columns]
        .astype(np.float32)
    )

    rfc_train_batches.append(batch_pd)

    print(
        f"Batch {batch_id}: {len(batch_pd):,} rows | "
        f"{batch_pd.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
    )

    del batch_pd
    gc.collect()

# Combine batches into one pandas DataFrame
rfc_full_train_pd = pd.concat(
    rfc_train_batches,
    ignore_index=True
)

# Release individual batch references
del rfc_train_batches
del rfc_batch_0_pd
gc.collect()

# Separate model features and target
X_train_full = rfc_full_train_pd[RFC_ACTIVE_FEATURE_COLUMNS].copy()
y_train_full = rfc_full_train_pd["target"].astype(np.int8).copy()

# Validate row counts and class distribution
assert len(X_train_full) == 1_434_220
assert int(y_train_full.sum()) == 565_589
assert int((y_train_full == 0).sum()) == 868_631

print("\nFULL RFC TRAINING DATA READY")
print("----------------------------")
print("X_train_full shape:", X_train_full.shape)
print("y_train_full shape:", y_train_full.shape)

print(
    "Feature memory:",
    f"{X_train_full.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
)

# Release combined intermediate DataFrame
del rfc_full_train_pd
gc.collect()

# COMMAND ----------

# DBTITLE 1,Full Model preprocessor
# Step 48 — Configure Full RFC model fit

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Fresh, unfitted preprocessor using our approved 689 features
rfc_xgb_full = Pipeline([
    ("preprocessor", clone(rfc_preprocessor)),
    ("classifier", XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=16
    ))
])

assert len(RFC_ACTIVE_FEATURE_COLUMNS) == 690
assert len(X_train_full) == len(y_train_full) == 1_433_482
assert set(rfc_xgb_full.named_steps["preprocessor"].feature_names_in_) if hasattr(
    rfc_xgb_full.named_steps["preprocessor"], "feature_names_in_"
) else True

print("RFC full-training pipeline configured.")
print("Training rows:", len(X_train_full))
print("Active features:", len(RFC_ACTIVE_FEATURE_COLUMNS))



# COMMAND ----------

# # Validate test_3_NS training data

# # 1. Validate row counts
# assert len(X_train_full) == len(y_train_full), \
#     "Training features and target row counts do not match"

# # 2. Validate target distribution
# positive_count = int((y_train_full == 1).sum())
# negative_count = int((y_train_full == 0).sum())

# assert positive_count + negative_count == len(y_train_full), \
#     "Unexpected target values found"

# assert positive_count > 0, "No positive training examples"
# assert negative_count > 0, "No negative training examples"

# # 3. Validate approved feature count
# assert len(RFC_ACTIVE_FEATURE_COLUMNS) == 690, \
#     "Unexpected RFC feature count"

# assert X_train_full.shape[1] == 690, \
#     "Training feature count does not match approved contract"

# # 4. Validate exact model feature contract
# model_preprocessor = rfc_xgb_full.named_steps["preprocessor"]

# if hasattr(model_preprocessor, "feature_names_in_"):
#     assert list(model_preprocessor.feature_names_in_) == list(X_train_full.columns), \
#         "Model preprocessing feature contract mismatch"

# # 5. Display results
# print("test_3_NS training validation PASSED")
# print("Training rows:", len(X_train_full))
# print("Positive rows:", positive_count)
# print("Negative rows:", negative_count)
# print("Active features:", len(RFC_ACTIVE_FEATURE_COLUMNS))

# COMMAND ----------

# Step 48 — Full RFC model fit

import time

start = time.time()

rfc_xgb_full.fit(X_train_full, y_train_full)

print(
    f"Training completed in {(time.time() - start) / 60:.2f} minutes"
)

# COMMAND ----------

# Step 49: Evaluate full RFC model on validation prototype

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss
)

# Predict using the full trained pipeline
rfc_valid_probabilities_full = (
    rfc_xgb_full.predict_proba(X_valid)[:, 1]
)

# Ranking and probability metrics
rfc_full_valid_roc_auc = roc_auc_score(
    y_valid,
    rfc_valid_probabilities_full
)

rfc_full_valid_pr_auc = average_precision_score(
    y_valid,
    rfc_valid_probabilities_full
)

rfc_full_valid_log_loss = log_loss(
    y_valid,
    rfc_valid_probabilities_full
)

print("RFC FULL MODEL — VALIDATION PROTOTYPE")
print("-------------------------------------")
print(f"ROC-AUC: {rfc_full_valid_roc_auc:.4f}")
print(f"PR-AUC:  {rfc_full_valid_pr_auc:.4f}")
print(f"Log loss: {rfc_full_valid_log_loss:.4f}")
print(f"Actual positive rate: {y_valid.mean():.4%}")
print(
    f"Mean predicted probability: "
    f"{rfc_valid_probabilities_full.mean():.4%}"
)

# COMMAND ----------

# Step 50: Full-model top-decile lift

import numpy as np
import pandas as pd

rfc_full_lift_df = pd.DataFrame({
    "actual": np.asarray(y_valid),
    "score": rfc_valid_probabilities_full
})

# Highest predicted scores first
rfc_full_lift_df = rfc_full_lift_df.sort_values(
    "score", ascending=False
).reset_index(drop=True)

baseline_rate = rfc_full_lift_df["actual"].mean()

top_n = int(np.ceil(len(rfc_full_lift_df) * 0.10))
top_decile = rfc_full_lift_df.iloc[:top_n]

top_decile_precision = top_decile["actual"].mean()
top_decile_lift = top_decile_precision / baseline_rate

print("RFC FULL MODEL — TOP-DECILE LIFT")
print("--------------------------------")
print(f"Validation observations: {len(rfc_full_lift_df):,}")
print(f"Overall review rate: {baseline_rate:.4%}")
print(f"Top-decile observations: {top_n:,}")
print(f"Top-decile reviewers: {int(top_decile['actual'].sum()):,}")
print(f"Top-decile review rate: {top_decile_precision:.4%}")
print(f"Top-decile lift: {top_decile_lift:.2f}x")

print("\nPrototype top-decile lift: 2.55x")

# COMMAND ----------

# Step 51: Prepare full validation dataset for batch scoring

from pyspark.sql import functions as F

RFC_VALIDATION_BATCHES = 20

rfc_validation_batched_df = (
    rfc_validation_df
    .withColumn(
        "_validation_batch",
        F.pmod(
            F.xxhash64(
                F.col("policy_id"),
                F.col("effective_dttm"),
                F.lit(42)
            ),
            F.lit(RFC_VALIDATION_BATCHES)
        )
    )
)

print("Validation batch distribution:")

rfc_validation_batched_df.groupBy(
    "_validation_batch"
).count().orderBy(
    "_validation_batch"
).show(20)

print(
    "Total validation rows:",
    rfc_validation_batched_df.count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Full model validation

# COMMAND ----------

# Step 52: Score full RFC validation data in bounded batches

import gc
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

rfc_validation_predictions = []

for batch_id in range(RFC_VALIDATION_BATCHES):

    # Select one deterministic validation batch
    batch_spark_df = (
        rfc_validation_batched_df
        .filter(F.col("_validation_batch") == batch_id)
    )

    # Apply the same feature transformations used for training
    batch_prepared_df = (
        prepare_rfc_features(batch_spark_df)
        .select(*RFC_ACTIVE_FEATURE_COLUMNS, "target")
    )

    # Collect only this batch
    batch_pd = batch_prepared_df.toPandas()

    # Match training numeric types
    batch_pd[rfc_numeric_pipeline_columns] = (
        batch_pd[rfc_numeric_pipeline_columns]
        .astype(np.float32)
    )

    # Generate probabilities using the fitted full pipeline
    batch_scores = rfc_xgb_full.predict_proba(
        batch_pd[RFC_ACTIVE_FEATURE_COLUMNS]
    )[:, 1]

    # Retain only actual labels and predictions
    rfc_validation_predictions.append(
        pd.DataFrame({
            "actual": batch_pd["target"].to_numpy(dtype=np.int8),
            "score": batch_scores
        })
    )

    print(
        f"Batch {batch_id + 1}/{RFC_VALIDATION_BATCHES}: "
        f"{len(batch_pd):,} observations scored"
    )

    del batch_pd, batch_prepared_df, batch_spark_df, batch_scores
    gc.collect()

# Combine lightweight prediction results
rfc_full_validation_scores = pd.concat(
    rfc_validation_predictions,
    ignore_index=True
)

del rfc_validation_predictions
gc.collect()

# Validate completeness
assert len(rfc_full_validation_scores) == 1_998_356
assert int(rfc_full_validation_scores["actual"].sum()) == 62_908

print("\nFULL VALIDATION SCORING COMPLETE")
print("--------------------------------")
print("Observations:", f"{len(rfc_full_validation_scores):,}")
print(
    "Actual reviewers:",
    f"{int(rfc_full_validation_scores['actual'].sum()):,}"
)
print(
    "Mean predicted score:",
    f"{rfc_full_validation_scores['score'].mean():.4%}"
)

# COMMAND ----------

# DBTITLE 1,Full data validation
# Step 53: Evaluate full RFC validation population

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

y_true = rfc_full_validation_scores["actual"].to_numpy()
y_score = rfc_full_validation_scores["score"].to_numpy()

# Overall ranking metrics
rfc_full_validation_roc_auc = roc_auc_score(y_true, y_score)
rfc_full_validation_pr_auc = average_precision_score(y_true, y_score)

# Top-decile lift
top_n = int(np.ceil(len(y_true) * 0.10))

top_indices = np.argpartition(y_score, -top_n)[-top_n:]

top_reviewers = int(y_true[top_indices].sum())

overall_review_rate = y_true.mean()
top_decile_review_rate = top_reviewers / top_n
top_decile_lift = top_decile_review_rate / overall_review_rate

print("RFC FULL VALIDATION — MODEL PERFORMANCE")
print("---------------------------------------")
print(f"Observations: {len(y_true):,}")
print(f"Actual reviewers: {int(y_true.sum()):,}")
print(f"Overall review rate: {overall_review_rate:.4%}")

print(f"\nROC-AUC: {rfc_full_validation_roc_auc:.4f}")
print(f"PR-AUC:  {rfc_full_validation_pr_auc:.4f}")

print("\nTOP-DECILE PERFORMANCE")
print("----------------------")
print(f"Top 10% observations: {top_n:,}")
print(f"Reviewers in top 10%: {top_reviewers:,}")
print(f"Top-decile review rate: {top_decile_review_rate:.4%}")
print(f"Top-decile lift: {top_decile_lift:.2f}x")

print("\nPREVIOUS RESULTS")
print("----------------")
print("Prototype top-decile lift: 2.55x")
print("Full model on 10K validation sample: 2.95x")

# COMMAND ----------

# display(rfc_full_validation_metrics.round(6))
print("Columns:", rfc_full_validation_scores.columns.tolist())
print("Rows:", len(rfc_full_validation_scores))

display(rfc_full_validation_scores.head(3))

# COMMAND ----------

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss
)

df = rfc_full_validation_scores

assert len(df) == 1_998_344, "Unexpected validation row count"

y = df["actual"].to_numpy(dtype=int)
p_raw = df["score"].to_numpy(dtype=float)
p_corrected = df["corrected_score"].to_numpy(dtype=float)

assert np.isfinite(p_raw).all()
assert np.isfinite(p_corrected).all()
assert np.isin(y, [0, 1]).all()

predicted = (p_raw >= 0.5).astype(int)

def calculate_lift(y_true, scores, pct):
    n_top = int(np.ceil(len(y_true) * pct / 100))
    top_indices = np.argsort(-scores, kind="stable")[:n_top]
    return y_true[top_indices].mean() / y_true.mean()

validation_metrics = {
    "ROC-AUC": roc_auc_score(y, p_raw),
    "PR-AUC": average_precision_score(y, p_raw),
    "Precision @ raw 0.5": precision_score(y, predicted, zero_division=0),
    "Recall @ raw 0.5": recall_score(y, predicted, zero_division=0),
    "F1 @ raw 0.5": f1_score(y, predicted, zero_division=0),
    "Lift @ Top 1%": calculate_lift(y, p_raw, 1),
    "Lift @ Top 5%": calculate_lift(y, p_raw, 5),
    "Lift @ Top 10%": calculate_lift(y, p_raw, 10),
    "Lift @ Top 20%": calculate_lift(y, p_raw, 20),
    "Corrected Brier": brier_score_loss(y, p_corrected),
    "Corrected Log Loss": log_loss(y, p_corrected, labels=[0, 1])
}

rfc_full_validation_metrics = pd.DataFrame({
    "Metric": list(validation_metrics.keys()),
    "Validation": list(validation_metrics.values())
})

print("Full validation rows:", len(df))
print("Actual reviewers:", int(y.sum()))

display(rfc_full_validation_metrics.round(6))

# COMMAND ----------

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss
)

# Use the existing full validation scoring output
df = rfc_full_validation_scores

assert len(df) == 1_998_344, "Unexpected validation row count"

required_columns = {"actual", "score", "corrected_score"}
assert required_columns.issubset(df.columns), (
    f"Missing columns: {required_columns - set(df.columns)}"
)

y = df["actual"].to_numpy(dtype=int)
p_raw = df["score"].to_numpy(dtype=float)
p_corrected = df["corrected_score"].to_numpy(dtype=float)

assert np.isfinite(p_raw).all()
assert np.isfinite(p_corrected).all()
assert ((p_raw >= 0) & (p_raw <= 1)).all()
assert ((p_corrected >= 0) & (p_corrected <= 1)).all()

# Classification metrics at raw 0.5 threshold
predicted = (p_raw >= 0.5).astype(int)

# Lift at top K% of ranked customers
def calculate_lift(y_true, scores, pct):
    n_top = int(np.ceil(len(y_true) * pct / 100))
    top_indices = np.argsort(-scores, kind="stable")[:n_top]

    return y_true[top_indices].mean() / y_true.mean()

validation_metrics = {
    "ROC-AUC": roc_auc_score(y, p_raw),
    "PR-AUC": average_precision_score(y, p_raw),
    "Precision @ raw 0.5": precision_score(y, predicted, zero_division=0),
    "Recall @ raw 0.5": recall_score(y, predicted, zero_division=0),
    "F1 @ raw 0.5": f1_score(y, predicted, zero_division=0),
    "Lift @ Top 1%": calculate_lift(y, p_raw, 1),
    "Lift @ Top 5%": calculate_lift(y, p_raw, 5),
    "Lift @ Top 10%": calculate_lift(y, p_raw, 10),
    "Lift @ Top 20%": calculate_lift(y, p_raw, 20),
    "Corrected Brier": brier_score_loss(y, p_corrected),
    "Corrected Log Loss": log_loss(y, p_corrected, labels=[0, 1])
}

rfc_full_validation_metrics = pd.DataFrame({
    "Metric": list(validation_metrics.keys()),
    "Validation": list(validation_metrics.values())
})

print("Full validation observations:", len(df))
print("Actual reviewers:", int(y.sum()))

display(rfc_full_validation_metrics.round(6))

# COMMAND ----------

# Step 53: Full RFC validation performance

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

y_true = rfc_full_validation_scores["actual"].to_numpy()
y_score = rfc_full_validation_scores["score"].to_numpy()

# Ranking metrics
roc_auc = roc_auc_score(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)

# Classification metrics (diagnostic threshold)
threshold = 0.5
y_pred = (y_score >= threshold).astype(int)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Top-decile lift
top_n = int(np.ceil(len(y_true) * 0.10))
top_indices = np.argpartition(y_score, -top_n)[-top_n:]

top_reviewers = int(y_true[top_indices].sum())
baseline_rate = y_true.mean()
top_decile_rate = top_reviewers / top_n
top_decile_lift = top_decile_rate / baseline_rate

print("RFC FULL VALIDATION — MODEL PERFORMANCE")
print("---------------------------------------")
print(f"Observations: {len(y_true):,}")
print(f"Actual reviewers: {int(y_true.sum()):,}")
print(f"Overall review rate: {baseline_rate:.4%}")

print("\nRANKING METRICS")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print(f"\nCLASSIFICATION METRICS (threshold = {threshold})")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\nTOP-DECILE PERFORMANCE")
print(f"Top 10% observations: {top_n:,}")
print(f"Reviewers in top 10%: {top_reviewers:,}")
print(f"Top-decile review rate: {top_decile_rate:.4%}")
print(f"Top-decile lift: {top_decile_lift:.2f}x")

# COMMAND ----------


# Step 54: Full-validation Top-K targeting metrics

import numpy as np
import pandas as pd

ranked = rfc_full_validation_scores.sort_values(
    "score", ascending=False
).reset_index(drop=True)

total_reviewers = int(ranked["actual"].sum())
baseline_rate = ranked["actual"].mean()

results = []

for pct in [1, 5, 10, 20]:

    k = int(np.ceil(len(ranked) * pct / 100))

    selected = ranked.iloc[:k]
    tp = int(selected["actual"].sum())

    precision = tp / k
    recall = tp / total_reviewers

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    results.append({
        "Top %": pct,
        "Observations": k,
        "Reviewers": tp,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Lift": precision / baseline_rate
    })

rfc_full_topk_results = pd.DataFrame(results)

display(rfc_full_topk_results)

# COMMAND ----------

# Step 54A: RFC Full Validation — Lift Chart

import matplotlib.pyplot as plt
import numpy as np

lift_df = rfc_full_topk_results.sort_values("Top %")

labels = [f"Top {int(x)}%" for x in lift_df["Top %"]]
lift_values = lift_df["Lift"].to_numpy()

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    labels,
    lift_values,
    color="#2878B5",
    width=0.55
)

# Random targeting baseline
ax.axhline(
    y=1.0,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Random targeting (1.0x)"
)

# Reference benchmark
ax.axhline(
    y=3.5,
    color="orange",
    linestyle=":",
    linewidth=1.8,
    label="Reference benchmark (3.5x)"
)

# Value labels above bars
for bar, lift in zip(bars, lift_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.10,
        f"{lift:.2f}x",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

ax.set_title(
    "RFC Propensity Model — Full Validation Lift",
    fontsize=15,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Targeting Population", fontsize=12)
ax.set_ylabel("Lift vs Random Targeting", fontsize=12)

ax.set_ylim(0, max(lift_values) + 0.9)
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.set_axisbelow(True)
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()

# COMMAND ----------


# Step 55: Record RFC full-model validation baseline

rfc_validation_baseline = {
    "model": "RFC_XGB_Full_Baseline",
    "validation_rows": len(y_true),
    "positive_rate": float(y_true.mean()),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "precision_at_0_5": float(
        precision_score(y_true, y_score >= 0.5, zero_division=0)
    ),
    "recall_at_0_5": float(
        recall_score(y_true, y_score >= 0.5, zero_division=0)
    ),
    "f1_at_0_5": float(
        f1_score(y_true, y_score >= 0.5, zero_division=0)
    ),
    "lift_at_1pct": float(
        rfc_full_topk_results.loc[
            rfc_full_topk_results["Top %"] == 1, "Lift"
        ].iloc[0]
    ),
    "lift_at_5pct": float(
        rfc_full_topk_results.loc[
            rfc_full_topk_results["Top %"] == 5, "Lift"
        ].iloc[0]
    ),
    "lift_at_10pct": float(
        rfc_full_topk_results.loc[
            rfc_full_topk_results["Top %"] == 10, "Lift"
        ].iloc[0]
    ),
    "lift_at_20pct": float(
        rfc_full_topk_results.loc[
            rfc_full_topk_results["Top %"] == 20, "Lift"
        ].iloc[0]
    )
}

rfc_validation_baseline_df = pd.DataFrame(
    [rfc_validation_baseline]
)

display(rfc_validation_baseline_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Calibration

# COMMAND ----------

# Step 56: Raw probability calibration diagnostics

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

y_true = rfc_full_validation_scores["actual"].to_numpy()
y_score = rfc_full_validation_scores["score"].to_numpy()

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true,
    y_score,
    n_bins=10,
    strategy="quantile"
)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    mean_predicted_value,
    fraction_of_positives,
    marker="o",
    linewidth=2,
    label="RFC XGBoost — Raw scores"
)

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Perfect calibration"
)

ax.set_title("RFC Model — Raw Probability Calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed Review Rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# Step 57: Prior-probability correction for negative downsampling

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

y_true = rfc_full_validation_scores["actual"].to_numpy()
raw_scores = rfc_full_validation_scores["score"].to_numpy()

# Training retained all positives and 5% of negatives
negative_sampling_rate = RFC_DATA_CONFIG["negative_sample_pct"] / 100.0

# Adjust sampled-training odds to population odds
corrected_scores = (
    negative_sampling_rate * raw_scores
    / (
        (1.0 - raw_scores)
        + negative_sampling_rate * raw_scores
    )
)

rfc_full_validation_scores["corrected_score"] = corrected_scores

print("RFC PROBABILITY CORRECTION")
print("--------------------------")
print(f"Negative sampling rate: {negative_sampling_rate:.2%}")
print(f"Actual review rate:     {y_true.mean():.4%}")
print(f"Mean raw score:         {raw_scores.mean():.4%}")
print(f"Mean corrected score:   {corrected_scores.mean():.4%}")

print("\nPROBABILITY QUALITY")
print("-------------------")
print(
    f"Raw Brier score:       "
    f"{brier_score_loss(y_true, raw_scores):.6f}"
)
print(
    f"Corrected Brier score: "
    f"{brier_score_loss(y_true, corrected_scores):.6f}"
)
print(
    f"Raw log loss:          "
    f"{log_loss(y_true, raw_scores):.6f}"
)
print(
    f"Corrected log loss:    "
    f"{log_loss(y_true, corrected_scores):.6f}"
)

# COMMAND ----------


# Step 58: Compare raw vs corrected calibration

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

raw_fraction, raw_mean = calibration_curve(
    y_true,
    raw_scores,
    n_bins=10,
    strategy="quantile"
)

corrected_fraction, corrected_mean = calibration_curve(
    y_true,
    corrected_scores,
    n_bins=10,
    strategy="quantile"
)

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    raw_mean,
    raw_fraction,
    marker="o",
    linewidth=2,
    label="Raw XGBoost"
)

ax.plot(
    corrected_mean,
    corrected_fraction,
    marker="o",
    linewidth=2,
    label="Prior-corrected XGBoost"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Perfect calibration"
)

# Zoom into the range relevant to observed review probabilities
ax.set_xlim(0, 0.15)
ax.set_ylim(0, 0.15)

ax.set_title(
    "RFC Model — Probability Calibration Comparison",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed 90-Day Cover Review Rate")

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------


# Step 59: Corrected probability calibration by decile

import numpy as np
import pandas as pd

rfc_calibration_df = pd.DataFrame({
    "actual": y_true,
    "corrected_probability": corrected_scores
})

# Decile 1 = highest predicted probability
rfc_calibration_df["decile"] = (
    pd.qcut(
        rfc_calibration_df["corrected_probability"].rank(method="first"),
        q=10,
        labels=False
    )
)

rfc_calibration_df["decile"] = (
    10 - rfc_calibration_df["decile"]
)

rfc_calibration_results = (
    rfc_calibration_df
    .groupby("decile")
    .agg(
        observations=("actual", "size"),
        reviewers=("actual", "sum"),
        mean_predicted_probability=("corrected_probability", "mean"),
        observed_review_rate=("actual", "mean")
    )
    .reset_index()
    .sort_values("decile")
)

rfc_calibration_results["calibration_gap_pp"] = (
    rfc_calibration_results["mean_predicted_probability"]
    - rfc_calibration_results["observed_review_rate"]
) * 100

display(rfc_calibration_results)

# COMMAND ----------

# MAGIC %md
# MAGIC # Test population

# COMMAND ----------

# Step 60: Prepare full test population for bounded scoring

from pyspark.sql import functions as F

RFC_TEST_BATCHES = 24

rfc_test_batched_df = (
    rfc_test_df
    .withColumn(
        "_test_batch",
        F.pmod(
            F.xxhash64(
                F.col("policy_id"),
                F.col("effective_dttm"),
                F.lit(42)
            ),
            F.lit(RFC_TEST_BATCHES)
        )
    )
)

print("RFC TEST BATCH DISTRIBUTION")
print("---------------------------")

rfc_test_batched_df.groupBy(
    "_test_batch"
).count().orderBy(
    "_test_batch"
).show(RFC_TEST_BATCHES)

print(
    "Total test observations:",
    f"{rfc_test_batched_df.count():,}"
)

# COMMAND ----------

rfc_full_test_scores["actual"].sum()

# COMMAND ----------


# Step 61: Score full RFC test population in batches

import gc
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

rfc_test_predictions = []

negative_sampling_rate = (
    RFC_DATA_CONFIG["negative_sample_pct"] / 100.0
)

for batch_id in range(RFC_TEST_BATCHES):

    batch_spark_df = rfc_test_batched_df.filter(
        F.col("_test_batch") == batch_id
    )

    batch_prepared_df = (
        prepare_rfc_features(batch_spark_df)
        .select(*RFC_ACTIVE_FEATURE_COLUMNS, "target")
    )

    batch_pd = batch_prepared_df.toPandas()

    batch_pd[rfc_numeric_pipeline_columns] = (
        batch_pd[rfc_numeric_pipeline_columns]
        .astype(np.float32)
    )

    raw_scores = rfc_xgb_full.predict_proba(
        batch_pd[RFC_ACTIVE_FEATURE_COLUMNS]
    )[:, 1]

    corrected_scores_batch = (
        negative_sampling_rate * raw_scores
        / (
            (1.0 - raw_scores)
            + negative_sampling_rate * raw_scores
        )
    )

    rfc_test_predictions.append(
        pd.DataFrame({
            "actual": batch_pd["target"].to_numpy(dtype=np.int8),
            "raw_score": raw_scores,
            "corrected_score": corrected_scores_batch
        })
    )

    print(
        f"Batch {batch_id + 1}/{RFC_TEST_BATCHES}: "
        f"{len(batch_pd):,} observations scored"
    )

    del batch_pd, batch_prepared_df, batch_spark_df
    del raw_scores, corrected_scores_batch
    gc.collect()

rfc_full_test_scores = pd.concat(
    rfc_test_predictions,
    ignore_index=True
)

del rfc_test_predictions
gc.collect()

assert len(rfc_full_test_scores) == 2_216_521
assert int(rfc_full_test_scores["actual"].sum()) == 70_375

print("\nRFC FULL TEST SCORING COMPLETE")
print("--------------------------------")
print(f"Observations: {len(rfc_full_test_scores):,}")
print(f"Actual reviewers: {int(rfc_full_test_scores['actual'].sum()):,}")
print(f"Mean raw score: {rfc_full_test_scores['raw_score'].mean():.4%}")
print(f"Mean corrected score: {rfc_full_test_scores['corrected_score'].mean():.4%}")
print(f"Actual review rate: {rfc_full_test_scores['actual'].mean():.4%}")

# COMMAND ----------

print("\nRFC FULL TEST SCORING COMPLETE")
print("--------------------------------")
print(f"Observations: {len(rfc_full_test_scores):,}")
print(f"Actual reviewers: {int(rfc_full_test_scores['actual'].sum()):,}")
print(f"Mean raw score: {rfc_full_test_scores['raw_score'].mean():.4%}")
print(f"Mean corrected score: {rfc_full_test_scores['corrected_score'].mean():.4%}")
print(f"Actual review rate: {rfc_full_test_scores['actual'].mean():.4%}")

# COMMAND ----------

# Step 62: RFC full test evaluation

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss
)

# Test labels and scores
y_test_full = rfc_full_test_scores["actual"].to_numpy()
raw_test = rfc_full_test_scores["raw_score"].to_numpy()
corrected_test = rfc_full_test_scores["corrected_score"].to_numpy()

# Ranking metrics
test_roc_auc = roc_auc_score(y_test_full, raw_test)
test_pr_auc = average_precision_score(y_test_full, raw_test)

# Classification metrics — raw-score threshold 0.5
test_threshold = 0.5
test_predictions = (raw_test >= test_threshold).astype(int)

test_precision = precision_score(
    y_test_full, test_predictions, zero_division=0
)
test_recall = recall_score(
    y_test_full, test_predictions, zero_division=0
)
test_f1 = f1_score(
    y_test_full, test_predictions, zero_division=0
)

# Probability quality
test_brier_raw = brier_score_loss(y_test_full, raw_test)
test_brier_corrected = brier_score_loss(
    y_test_full, corrected_test
)

test_logloss_raw = log_loss(y_test_full, raw_test)
test_logloss_corrected = log_loss(
    y_test_full, corrected_test
)

print("RFC FULL TEST — MODEL PERFORMANCE")
print("---------------------------------")
print(f"Observations: {len(y_test_full):,}")
print(f"Actual reviewers: {int(y_test_full.sum()):,}")
print(f"Actual review rate: {y_test_full.mean():.4%}")

print("\nRANKING METRICS")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"PR-AUC:  {test_pr_auc:.4f}")

print(f"\nCLASSIFICATION METRICS (raw threshold = {test_threshold})")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-score:  {test_f1:.4f}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test_full, test_predictions))

print("\nPROBABILITY QUALITY")
print(f"Mean raw score:       {raw_test.mean():.4%}")
print(f"Mean corrected score: {corrected_test.mean():.4%}")
print(f"Raw Brier score:      {test_brier_raw:.6f}")
print(f"Corrected Brier:      {test_brier_corrected:.6f}")
print(f"Raw log loss:         {test_logloss_raw:.6f}")
print(f"Corrected log loss:   {test_logloss_corrected:.6f}")

# Top-K targeting metrics
ranked_test = rfc_full_test_scores.sort_values(
    "raw_score", ascending=False
).reset_index(drop=True)

total_reviewers = int(ranked_test["actual"].sum())
baseline_rate = ranked_test["actual"].mean()

test_topk_results = []

for pct in [1, 5, 10, 20]:
    k = int(np.ceil(len(ranked_test) * pct / 100))
    selected = ranked_test.iloc[:k]

    tp = int(selected["actual"].sum())
    precision_k = tp / k
    recall_k = tp / total_reviewers

    f1_k = (
        2 * precision_k * recall_k / (precision_k + recall_k)
        if precision_k + recall_k > 0 else 0
    )

    test_topk_results.append({
        "Top %": pct,
        "Observations": k,
        "Reviewers": tp,
        "Precision": precision_k,
        "Recall": recall_k,
        "F1-score": f1_k,
        "Lift": precision_k / baseline_rate
    })

rfc_test_topk_results = pd.DataFrame(test_topk_results)

print("\nTOP-K TEST PERFORMANCE")
display(rfc_test_topk_results)

# COMMAND ----------

print("Full validation DataFrame available:", "rfc_validation_df" in globals())

# COMMAND ----------

print("Full validation rows:", rfc_validation_df.count())
print("Validation columns:", len(rfc_validation_df.columns))

# COMMAND ----------

# Step 62A: test_3_NS — validation statistics for Step 63

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss
)

# Use the current test_3_NS validation predictions
y = np.asarray(y_valid).ravel().astype(int)
p = np.asarray(rfc_valid_probabilities).ravel().astype(float)

assert len(y) == len(p), "Validation labels and predictions do not match"
assert set(np.unique(y)) == {0, 1}, "Validation must contain both classes"
assert np.isfinite(p).all(), "Invalid validation probabilities"
assert ((p >= 0) & (p <= 1)).all(), "Probabilities outside [0, 1]"

# Raw 0.5 threshold — same definition as previous baseline
pred = (p >= 0.5).astype(int)

# Prior correction for 5% negative sampling
negative_sample_fraction = 0.05

p_corrected = (
    negative_sample_fraction * p
    / (1 - p + negative_sample_fraction * p)
)

# Top-K lift using raw probabilities
# Ranking is unchanged by prior correction.
order = np.argsort(-p, kind="stable")
n = len(y)
prevalence = y.mean()

lift_by_k = {}

for pct in [1, 5, 10, 20]:
    k = max(1, int(np.ceil(n * pct / 100)))
    precision_at_k = y[order[:k]].mean()
    lift_by_k[pct] = precision_at_k / prevalence

# Assemble the 11 validation metrics
rfc_validation_step63 = pd.Series({
    "ROC-AUC": roc_auc_score(y, p),
    "PR-AUC": average_precision_score(y, p),
    "Precision @ raw 0.5": precision_score(y, pred, zero_division=0),
    "Recall @ raw 0.5": recall_score(y, pred, zero_division=0),
    "F1 @ raw 0.5": f1_score(y, pred, zero_division=0),
    "Lift @ Top 1%": lift_by_k[1],
    "Lift @ Top 5%": lift_by_k[5],
    "Lift @ Top 10%": lift_by_k[10],
    "Lift @ Top 20%": lift_by_k[20],
    "Corrected Brier": brier_score_loss(y, p_corrected),
    "Corrected Log Loss": log_loss(y, p_corrected),
})

print("Validation observations:", n)
print("Validation prevalence:", round(prevalence, 6))

display(
    rfc_validation_step63
    .rename("Validation")
    .to_frame()
    .round(6)
)

# COMMAND ----------

# Step 63: RFC baseline — validation vs test comparison

rfc_final_comparison = pd.DataFrame({
    "Metric": [
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
        "Corrected Log Loss"
    ],
    "Validation": [
        
0.685268,
0.085582,
0.076281,
0.361582,
0.125984,
4.509139,
3.388499,
2.767274,
1.976624,
0.032929,
0.143282
    ],
    "Test": [
        test_roc_auc,
        test_pr_auc,
        test_precision,
        test_recall,
        test_f1,
        *rfc_test_topk_results["Lift"].to_list(),
        test_brier_corrected,
        test_logloss_corrected
    ]
})

rfc_final_comparison["Difference"] = (
    rfc_final_comparison["Test"]
    - rfc_final_comparison["Validation"]
)

display(rfc_final_comparison.round(6))

# COMMAND ----------

# Step 63: RFC baseline — full validation vs full test - Updated

rfc_final_comparison = pd.DataFrame({
    "Metric": [
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
        "Corrected Log Loss"
    ],
    "Test": [
        test_roc_auc,
        test_pr_auc,
        test_precision,
        test_recall,
        test_f1,
        *rfc_test_topk_results["Lift"].to_list(),
        test_brier_corrected,
        test_logloss_corrected
    ]
})

# Match validation metrics by name, not row position
rfc_final_comparison = rfc_final_comparison.merge(
    rfc_full_validation_metrics,
    on="Metric",
    how="left",
    validate="one_to_one"
)

assert rfc_final_comparison["Validation"].notna().all()

rfc_final_comparison["Difference"] = (
    rfc_final_comparison["Test"]
    - rfc_final_comparison["Validation"]
)

rfc_final_comparison = rfc_final_comparison[
    ["Metric", "Validation", "Test", "Difference"]
]

display(rfc_final_comparison.round(6))

# COMMAND ----------

# MAGIC %md
# MAGIC # Set up MLflow Run

# COMMAND ----------

# Step 64: Identify the RFC full-model MLflow run

import mlflow

rfc_full_model_run = mlflow.last_active_run()

if rfc_full_model_run is None:
    print("No active MLflow run found in this session.")
else:
    print("RFC FULL MODEL — MLFLOW RUN")
    print("---------------------------")
    print("Run ID:", rfc_full_model_run.info.run_id)
    print("Run name:", rfc_full_model_run.info.run_name)
    print("Experiment ID:", rfc_full_model_run.info.experiment_id)
    print("Status:", rfc_full_model_run.info.status)
    print("Artifact URI:", rfc_full_model_run.info.artifact_uri)

# COMMAND ----------

# Step 65: Verify the MLflow run before logging metrics

import mlflow

rfc_run_id = rfc_full_model_run.info.run_id
rfc_run = mlflow.get_run(rfc_run_id)

print("RFC MLFLOW RUN VERIFICATION")
print("---------------------------")
print("Run ID:", rfc_run.info.run_id)
print("Run name:", rfc_run.info.run_name)
print("Status:", rfc_run.info.status)

print("\nLogged model parameters:")
for key, value in sorted(rfc_run.data.params.items()):
    if any(term in key.lower() for term in [
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample", "random_state"
    ]):
        print(f"{key}: {value}")

print("\nLogged artifacts:")
for artifact in mlflow.artifacts.list_artifacts(
    run_id=rfc_run_id
):
    print(artifact.path)

# COMMAND ----------


# Step 66: Log RFC Baseline v1 evaluation metrics to MLflow

import mlflow

rfc_run_id = "8a12f0170d494479a33772f48f7e1d9b"

rfc_metrics = {
    # Full validation metrics
    "validation_roc_auc": float(roc_auc),
    "validation_pr_auc": float(pr_auc),
    "validation_precision_raw_0_5": float(
        precision_score(y_true, y_score >= 0.5, zero_division=0)
    ),
    "validation_recall_raw_0_5": float(
        recall_score(y_true, y_score >= 0.5, zero_division=0)
    ),
    "validation_f1_raw_0_5": float(
        f1_score(y_true, y_score >= 0.5, zero_division=0)
    ),
    "validation_brier_corrected": float(
        brier_score_loss(y_true, corrected_scores)
    ),
    "validation_logloss_corrected": float(
        log_loss(y_true, corrected_scores)
    ),

    # Independent test metrics
    "test_roc_auc": float(test_roc_auc),
    "test_pr_auc": float(test_pr_auc),
    "test_precision_raw_0_5": float(test_precision),
    "test_recall_raw_0_5": float(test_recall),
    "test_f1_raw_0_5": float(test_f1),
    "test_brier_corrected": float(test_brier_corrected),
    "test_logloss_corrected": float(test_logloss_corrected)
}

# Log targeting lift for both datasets
for pct in [1, 5, 10, 20]:
    rfc_metrics[f"validation_lift_top_{pct}pct"] = float(
        rfc_full_topk_results.loc[
            rfc_full_topk_results["Top %"] == pct, "Lift"
        ].iloc[0]
    )

    rfc_metrics[f"test_lift_top_{pct}pct"] = float(
        rfc_test_topk_results.loc[
            rfc_test_topk_results["Top %"] == pct, "Lift"
        ].iloc[0]
    )

# Log to the verified run
with mlflow.start_run(run_id=rfc_run_id):
    mlflow.log_metrics(rfc_metrics)
    mlflow.set_tags({
        "model_name": "RFC_XGBoost_Baseline_test_3_NS",
        "model_stage": "baseline_evaluation",
        "evaluation_scope": "full_validation_and_test",
        "negative_sampling_rate": "0.05",
        "probability_correction": "prior_odds_correction",
        "classification_threshold": "raw_score_0.5"
    })

print("RFC BASELINE V1 — MLFLOW METRICS LOGGED")
print("---------------------------------------")
print("Run ID:", rfc_run_id)
print("Metrics logged:", len(rfc_metrics))

# COMMAND ----------

# Step 67: Inspect MLflow model artifacts

import mlflow

rfc_run_id = "8a12f0170d494479a33772f48f7e1d9b"

print("RFC BASELINE Test 3 NS — MLFLOW ARTIFACT INSPECTION")
print("-------------------------------------------")

def inspect_artifacts(path="", level=0):
    artifacts = mlflow.artifacts.list_artifacts(
        run_id=rfc_run_id,
        artifact_path=path
    )

    for artifact in artifacts:
        print("  " * level + artifact.path)

        if artifact.is_dir:
            inspect_artifacts(artifact.path, level + 1)

inspect_artifacts()

print("\nMLFLOW LOGGED MODELS")
print("-------------------")

try:
    models = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{rfc_run_id}'"
    )

    print("Logged models found:", len(models))

    for model in models:
        print("Model ID:", model.model_id)
        print("Model URI:", model.model_uri)

except Exception as e:
    print("Logged-model lookup unavailable:", str(e)[:300])

# COMMAND ----------

# Step 67A: Find the saved model using our existing MLflow run
import mlflow

run_id = "8a12f0170d494479a33772f48f7e1d9b"

client = mlflow.tracking.MlflowClient()

models = client.search_logged_models(
    experiment_ids=["1100835868419499"]
)

matching_models = [
    model for model in models
    if model.source_run_id == run_id
]

print("Models associated with run:", len(matching_models))

for model in matching_models:
    print("Model ID:", model.model_id)
    print("Name:", model.name)
    print("Model URI:", model.model_uri)
    print("Status:", model.status)

# COMMAND ----------

# Step 67B: Inspect MLflow logged models safely

import mlflow
import pandas as pd

rfc_run_id = "8a12f0170d494479a33772f48f7e1d9b"
experiment_id = mlflow.get_run(rfc_run_id).info.experiment_id

models = mlflow.search_logged_models(
    experiment_ids=[experiment_id],
    max_results=10,
    output_format="pandas"
)

print("Logged models found:", len(models))

# Display only columns that exist in this MLflow version
columns_to_show = [
    "model_id",
    "name",
    "model_uri",
    "source_run_id",
    "status"
]

available_columns = [
    col for col in columns_to_show
    if col in models.columns
]

print("Available columns:", models.columns.tolist())

display(models[available_columns])

# COMMAND ----------

# Step 68: Load and inspect the existing RFC MLflow model

import mlflow
import mlflow.sklearn

rfc_run_id = "8a12f0170d494479a33772f48f7e1d9b"

# Reuse the models DataFrame from Step 67B
matching_models = models.loc[
    models["source_run_id"] == rfc_run_id
]

assert len(matching_models) == 1, (
    f"Expected 1 RFC model, found {len(matching_models)}"
)

rfc_logged_model_id = matching_models.iloc[0]["model_id"]
rfc_logged_model_uri = f"models:/{rfc_logged_model_id}"

print("RFC SAVED MODEL VERIFICATION")
print("----------------------------")
print("Model ID:", rfc_logged_model_id)
print("Model URI:", rfc_logged_model_uri)

# Load the already-trained pipeline
rfc_loaded_model = mlflow.sklearn.load_model(
    rfc_logged_model_uri
)

print("\nLoaded model type:", type(rfc_loaded_model))

print("\nPipeline steps:")
if hasattr(rfc_loaded_model, "named_steps"):
    for name, step in rfc_loaded_model.named_steps.items():
        print(f"{name}: {type(step).__name__}")
else:
    print("Loaded object is not an sklearn Pipeline.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Verify reproducibility

# COMMAND ----------

# Step 69: Verify saved RFC model prediction reproducibility

import numpy as np

# Use the existing validation prototype
original_scores = rfc_xgb_full.predict_proba(X_valid)[:, 1]

loaded_scores = rfc_loaded_model.predict_proba(X_valid)[:, 1]

# Compare predictions
max_difference = np.max(
    np.abs(original_scores - loaded_scores)
)

predictions_match = np.allclose(
    original_scores,
    loaded_scores,
    rtol=1e-6,
    atol=1e-8
)

print("RFC MLFLOW MODEL — REPRODUCIBILITY CHECK")
print("----------------------------------------")
print("Observations tested:", len(X_valid))
print("Maximum prediction difference:", max_difference)
print("Predictions match:", predictions_match)

assert predictions_match, (
    "Loaded MLflow model predictions differ from original model!"
)

print("\nPASS: MLflow model reproduces original predictions.")

# COMMAND ----------

# Step 70: Verify saved RFC model feature contract

saved_preprocessor = rfc_loaded_model.named_steps["preprocessor"]

saved_numeric_columns = list(
    saved_preprocessor.transformers_[1][2]
)

saved_categorical_columns = list(
    saved_preprocessor.transformers_[0][2]
)

saved_feature_columns = (
    saved_numeric_columns + saved_categorical_columns
)

print("RFC SAVED MODEL — FEATURE CONTRACT")
print("----------------------------------")
print("Numeric features:", len(saved_numeric_columns))
print("Categorical features:", len(saved_categorical_columns))
print("Total input features:", len(saved_feature_columns))

print("\nFeature contract matches notebook:",
      set(saved_feature_columns) == set(RFC_ACTIVE_FEATURE_COLUMNS))

print("Duplicate features:",
      len(saved_feature_columns) - len(set(saved_feature_columns)))

assert set(saved_feature_columns) == set(RFC_ACTIVE_FEATURE_COLUMNS)
assert len(saved_feature_columns) == len(set(saved_feature_columns))

print("\nPASS: Saved model feature contract verified.")

# PASS: MLflow model reproduces original predictions.

# COMMAND ----------

# Step 71: Reusable RFC probability correction

import numpy as np

def correct_rfc_probabilities(raw_probabilities, negative_sampling_rate=0.05):
    """
    Adjust XGBoost probabilities for negative-class downsampling.

    Assumes:
      - All positive training observations were retained.
      - Negative observations were sampled independently at the
        specified rate.
    """
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)

    if not 0 < negative_sampling_rate <= 1:
        raise ValueError("negative_sampling_rate must be in (0, 1].")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    return (
        negative_sampling_rate * probabilities
        / (
            1.0 - probabilities
            + negative_sampling_rate * probabilities
        )
    )


# Verify against the previously calculated full-test scores
recalculated = correct_rfc_probabilities(
    rfc_full_test_scores["raw_score"].to_numpy(),
    negative_sampling_rate=RFC_DATA_CONFIG["negative_sample_pct"] / 100.0
)

original = rfc_full_test_scores["corrected_score"].to_numpy()

max_difference = np.max(np.abs(recalculated - original))

print("RFC PROBABILITY CORRECTION — REPRODUCIBILITY")
print("--------------------------------------------")
print(f"Test observations: {len(recalculated):,}")
print(f"Maximum difference: {max_difference:.12f}")
print(
    "Predictions match:",
    np.allclose(recalculated, original, rtol=1e-6, atol=1e-8)
)

assert np.allclose(recalculated, original, rtol=1e-6, atol=1e-8)

print("\nPASS: Probability correction is reproducible.")

# COMMAND ----------

# Step 72: Verify end-to-end RFC inference

import numpy as np

# 1. Prepare raw validation features using our existing function
rfc_inference_prepared_df = prepare_rfc_features(
    rfc_validation_prototype_df
).select(*RFC_ACTIVE_FEATURE_COLUMNS)

# 2. Convert to pandas
rfc_inference_pd = rfc_inference_prepared_df.toPandas()

# 3. Apply the same numeric types used during training
rfc_inference_pd[rfc_numeric_pipeline_columns] = (
    rfc_inference_pd[rfc_numeric_pipeline_columns]
    .astype(np.float32)
)

# 4. Score using the saved MLflow model
rfc_inference_raw_scores = rfc_loaded_model.predict_proba(
    rfc_inference_pd[RFC_ACTIVE_FEATURE_COLUMNS]
)[:, 1]

# 5. Apply the verified probability correction
rfc_inference_corrected_scores = correct_rfc_probabilities(
    rfc_inference_raw_scores,
    negative_sampling_rate=RFC_DATA_CONFIG["negative_sample_pct"] / 100.0
)

# 6. Compare with our original validation predictions
raw_match = np.allclose(
    rfc_inference_raw_scores,
    rfc_valid_probabilities_full,
    rtol=1e-6,
    atol=1e-8
)

print("RFC END-TO-END INFERENCE VERIFICATION")
print("-------------------------------------")
print("Observations:", len(rfc_inference_raw_scores))
print("Raw predictions match:", raw_match)
print("Mean corrected probability:",
      f"{rfc_inference_corrected_scores.mean():.4%}")

assert raw_match, "End-to-end inference predictions do not match!"

print("\nPASS: RFC inference pipeline is reproducible.")

# COMMAND ----------

# Step 72A: Correct RFC MLflow inference reproducibility check

import numpy as np

# Use the SAME validation input used for original predictions
rfc_inference_raw_scores = rfc_loaded_model.predict_proba(
    X_valid
)[:, 1]

# Apply our reusable probability correction
rfc_inference_corrected_scores = correct_rfc_probabilities(
    rfc_inference_raw_scores,
    negative_sampling_rate=RFC_DATA_CONFIG["negative_sample_pct"] / 100.0
)

# Verify matching observation counts
assert len(rfc_inference_raw_scores) == len(
    rfc_valid_probabilities_full
)

# Compare predictions
max_difference = np.max(
    np.abs(
        rfc_inference_raw_scores
        - rfc_valid_probabilities_full
    )
)

raw_match = np.allclose(
    rfc_inference_raw_scores,
    rfc_valid_probabilities_full,
    rtol=1e-6,
    atol=1e-8
)

print("RFC MLFLOW INFERENCE — REPRODUCIBILITY")
print("--------------------------------------")
print("Observations:", len(rfc_inference_raw_scores))
print("Maximum prediction difference:", max_difference)
print("Raw predictions match:", raw_match)
print(
    "Mean corrected probability:",
    f"{rfc_inference_corrected_scores.mean():.4%}"
)

assert raw_match, "MLflow predictions do not match!"

print("\nPASS: Saved model and probability correction verified.")

# COMMAND ----------


# Step 73: Validate RFC inference feature preparation

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType

# Use a small deterministic subset of the existing test data
rfc_feature_check_df = prepare_rfc_features(
    rfc_test_batched_df.filter(F.col("_test_batch") == 0)
).select(*RFC_ACTIVE_FEATURE_COLUMNS).limit(100)

schema = {
    field.name: field.dataType
    for field in rfc_feature_check_df.schema.fields
}

# Validate required columns
missing_columns = [
    c for c in RFC_ACTIVE_FEATURE_COLUMNS
    if c not in schema
]

# Validate numeric feature types
invalid_numeric = [
    c for c in rfc_numeric_pipeline_columns
    if c in schema and not isinstance(schema[c], NumericType)
]

# Validate categorical feature types
invalid_categorical = [
    c for c in rfc_onehot_columns
    if c in schema and not isinstance(schema[c], StringType)
]

print("RFC FEATURE PREPARATION — CONTRACT CHECK")
print("----------------------------------------")
print("Expected features:", len(RFC_ACTIVE_FEATURE_COLUMNS))
print("Missing features:", len(missing_columns))
print("Invalid numeric types:", len(invalid_numeric))
print("Invalid categorical types:", len(invalid_categorical))

if missing_columns:
    print("Missing:", missing_columns)

if invalid_numeric:
    print("Invalid numeric:", invalid_numeric)

if invalid_categorical:
    print("Invalid categorical:", invalid_categorical)

assert not missing_columns
assert not invalid_numeric
assert not invalid_categorical

print("\nPASS: RFC feature preparation schema verified.")

# COMMAND ----------

# Step 74: Inspect RFC Delta dataset versions
# ERROR: Databricks shared cluster blocks direct JVM access through _jdf.
rfc_dataset_tables = {
    "train": rfc_train_df,
    "validation": rfc_validation_df,
    "test": rfc_test_df
}

print("RFC DATASET LINEAGE CHECK")
print("-------------------------")

for dataset_name, dataset_df in rfc_dataset_tables.items():

    print(f"\nDataset: {dataset_name}")

    print("Spark source plan:")
    print(dataset_df._jdf.queryExecution().analyzed().toString()[:1000])

    print("Columns:", len(dataset_df.columns))

# COMMAND ----------

# Step 74: Inspect RFC dataset lineage

rfc_dataset_tables = {
    "train": rfc_train_df,
    "validation": rfc_validation_df,
    "test": rfc_test_df
}

print("RFC DATASET LINEAGE CHECK")
print("-------------------------")

for dataset_name, dataset_df in rfc_dataset_tables.items():

    print(f"\nDataset: {dataset_name}")

    print("Spark source plan:")
    dataset_df.explain(mode="formatted")

    print("Columns:", len(dataset_df.columns))

# COMMAND ----------

rfc_dataset_tables = {
    "train": f"bdp4_prod_ml_build.hi_customer_right_fit_cover_data.rfc_training_{batch_process_id}_train_sampled",
    "validation": f"bdp4_prod_ml_build.hi_customer_right_fit_cover_data.rfc_training_{batch_process_id}_validation",
    "test": f"bdp4_prod_ml_build.hi_customer_right_fit_cover_data.rfc_training_{batch_process_id}_test"
}

print("RFC DATASET LINEAGE CHECK")
print("-------------------------")

for dataset_name, table_name in rfc_dataset_tables.items():

    dataset_df = spark.table(table_name)

    print(f"\nDataset: {dataset_name}")
    print("Source table:", table_name)
    print("Columns:", len(dataset_df.columns))

# COMMAND ----------

# Step 74A: Inspect RFC dataset sources
# Compatible with Databricks Standard access mode

rfc_dataset_tables = {
    "train": rfc_train_df,
    "validation": rfc_validation_df,
    "test": rfc_test_df
}

print("RFC DATASET SOURCE INSPECTION")
print("-----------------------------")

for dataset_name, dataset_df in rfc_dataset_tables.items():

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Columns: {len(dataset_df.columns)}")
    print(f"{'=' * 60}")

    dataset_df.explain(mode="formatted")

# COMMAND ----------

# Step 74B: Inspect training Delta table history

rfc_train_table = (
    "bdp4_prod_ml_build."
    "hi_customer_right_fit_cover_data."
    f"rfc_training_{batch_process_id}_train_sampled"
)

history_df = spark.sql(
    f"DESCRIBE HISTORY {rfc_train_table}"
)

display(
    history_df.select(
        "version",
        "timestamp",
        "operation"
    ).orderBy("version", ascending=False)
)

# COMMAND ----------

# Step 75: Inspect latest Delta operations for RFC datasets

rfc_tables = {
    "train": (
        "bdp4_prod_ml_build.hi_customer_right_fit_cover_data."
        f"rfc_training_{batch_process_id}_train_sampled"
    )
}

for name, table_name in rfc_tables.items():
    history = spark.sql(
        f"DESCRIBE HISTORY {table_name}"
    )

    print(f"\n{name.upper()} — Latest Delta operations")

    display(
        history.select("version", "timestamp", "operation")
               .orderBy("version", ascending=False)
               .limit(3)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Important

# COMMAND ----------


# Restore RFC model after Databricks session restart

import mlflow
import mlflow.sklearn

# Verified MLflow model ID from our previous session
rfc_logged_model_id = "m-21c40bf17de64d7f82f672d0468ffdfa"
rfc_logged_model_uri = f"models:/{rfc_logged_model_id}"

# Load the existing trained model — no retraining
rfc_loaded_model = mlflow.sklearn.load_model(
    rfc_logged_model_uri
)

# Recover fitted model components
preprocessor = rfc_loaded_model.named_steps["preprocessor"]
classifier = rfc_loaded_model.named_steps["classifier"]

print("RFC MODEL RESTORED")
print("------------------")
print("Model URI:", rfc_logged_model_uri)
print("Model type:", type(rfc_loaded_model).__name__)
print("Preprocessor:", type(preprocessor).__name__)
print("Classifier:", type(classifier).__name__)
print("Encoded features:", len(preprocessor.get_feature_names_out()))
print("Importance values:", len(classifier.feature_importances_))

assert len(preprocessor.get_feature_names_out()) == len(
    classifier.feature_importances_
)

print("\nPASS: Saved RFC model and feature importance restored.")

# COMMAND ----------

# Step 76: Check fitted XGBoost feature importance

preprocessor = rfc_loaded_model.named_steps["preprocessor"]
classifier = rfc_loaded_model.named_steps["classifier"]

encoded_feature_names = preprocessor.get_feature_names_out()
importance_values = classifier.feature_importances_

print("RFC FEATURE IMPORTANCE CHECK")
print("----------------------------")
print("Encoded features:", len(encoded_feature_names))
print("Importance values:", len(importance_values))
print("Importance method:", classifier.importance_type)
print("Total importance:", importance_values.sum())

assert len(encoded_feature_names) == len(importance_values)

print("\nPASS: Feature importance is available.")

# COMMAND ----------


import pandas as pd
import numpy as np

# Fitted components from the restored MLflow model
preprocessor = rfc_loaded_model.named_steps["preprocessor"]
classifier = rfc_loaded_model.named_steps["classifier"]

# Original feature groups saved inside the fitted preprocessor
categorical_cols = list(preprocessor.transformers_[0][2])
numeric_cols = list(preprocessor.transformers_[1][2])

# XGBoost total gain, aligned to transformed feature positions
booster = classifier.get_booster()
gain_map = booster.get_score(importance_type="total_gain")

encoded_names = preprocessor.get_feature_names_out()

# XGBoost uses f0, f1, ... when trained without feature names
encoded_gain = np.array([
    gain_map.get(f"f{i}", 0.0)
    for i in range(len(encoded_names))
])

# Map transformed features to original business features
ohe = preprocessor.named_transformers_["categorical"]
category_counts = [len(cats) for cats in ohe.categories_]

original_features = []

for col, count in zip(categorical_cols, category_counts):
    original_features.extend([col] * count)

original_features.extend(numeric_cols)

assert len(original_features) == len(encoded_gain), (
    "Feature mapping does not match XGBoost input dimensions."
)

rfc_feature_importance = (
    pd.DataFrame({
        "Feature": original_features,
        "Total_Gain": encoded_gain
    })
    .groupby("Feature", as_index=False)["Total_Gain"]
    .sum()
    .sort_values("Total_Gain", ascending=False)
    .reset_index(drop=True)
)

rfc_feature_importance["Rank"] = (
    np.arange(1, len(rfc_feature_importance) + 1)
)

rfc_feature_importance["Importance_Pct"] = (
    100 * rfc_feature_importance["Total_Gain"]
    / rfc_feature_importance["Total_Gain"].sum()
)

rfc_top100_features = rfc_feature_importance.head(100)[
    ["Rank", "Feature", "Total_Gain", "Importance_Pct"]
]

display(rfc_top100_features)

print("Original features ranked:", len(rfc_feature_importance))
print("Top 100 features extracted:", len(rfc_top100_features))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnostic correlation

# COMMAND ----------

from pyspark.sql.types import NumericType
from itertools import combinations
from pyspark.sql import functions as F

# Numeric columns available in the training data
numeric_cols = [
    field.name
    for field in rfc_train_df.schema.fields
    if isinstance(field.dataType, NumericType)
]

# Keep only numeric features actually used by the model
model_features = set(original_features)

numeric_model_features = [
    col_name
    for col_name in numeric_cols
    if col_name in model_features
]

print("Numeric model features:", len(numeric_model_features))

# Use a deterministic sample to keep the diagnostic efficient
corr_df = (
    rfc_train_df
    .select(numeric_model_features)
    .sample(withReplacement=False, fraction=0.05, seed=42)
)

# Calculate pairwise correlations
correlation_results = []

for col_1, col_2 in combinations(numeric_model_features, 2):
    corr_value = corr_df.stat.corr(col_1, col_2)

    if corr_value is not None and abs(corr_value) >= 0.80:
        correlation_results.append(
            (col_1, col_2, corr_value, abs(corr_value))
        )

correlation_results = sorted(
    correlation_results,
    key=lambda x: x[3],
    reverse=True
)

print("\nHIGH CORRELATION FEATURE PAIRS")
print("------------------------------")
print("Pairs with |correlation| >= 0.80:", len(correlation_results))

for col_1, col_2, corr_value, _ in correlation_results:
    flag = "VERY HIGH" if abs(corr_value) >= 0.90 else "HIGH"

    print(
        f"{flag:9} | "
        f"{col_1} <-> {col_2} | "
        f"corr = {corr_value:.4f}"
    )

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


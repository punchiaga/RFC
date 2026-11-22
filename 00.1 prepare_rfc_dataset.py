# Databricks notebook source
"""RFC dataset preparation (Databricks Runtime 15.4 LTS).

Run independently of training. Default is dry-run: no Delta writes.
This script mirrors the approved RFC notebook's deterministic split and sampling.
"""

# COMMAND ----------
from dataclasses import dataclass
from pyspark.sql import functions as F

# COMMAND ----------
# Databricks notebook widgets can be overridden by DAB notebook_task.base_parameters.
# Never removeAll(): that can interfere with job-supplied parameters.
DEFAULTS = {
    "ml_catalog": "bdp4_prod_ml_build",
    "ml_schema": "hi_customer_right_fit_cover_data",
    "batch_process_id": "",  # Explicit: don't silently generate a new dataset version
    "user_prefix": "NS",
    "source_table": "",  # Optional override of notebook's legacy source-table convention
    "dry_run": "true",
    "negative_sample_pct": "5",  # 0–100; retain all positive training observations
    "validate_outputs": "true",  # expensive Spark checks; runs only on write path
}
for key, default in DEFAULTS.items():
    dbutils.widgets.text(key, default)

params = {key: dbutils.widgets.get(key).strip() for key in DEFAULTS}


def parse_bool(value: str, name: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise ValueError(f"{name} must be 'true' or 'false', got {value!r}")
    return value.lower() == "true"


DRY_RUN = parse_bool(params["dry_run"], "dry_run")
VALIDATE_OUTPUTS = parse_bool(params["validate_outputs"], "validate_outputs")

if not params["batch_process_id"] or not params["batch_process_id"].isdigit():
    raise ValueError("Provide an explicit numeric batch_process_id")

# Restrict identifier components: never interpolate arbitrary SQL identifiers.
for key in ("ml_catalog", "ml_schema", "user_prefix"):
    value = params[key]
    if not value or not all(c.isalnum() or c == "_" for c in value):
        raise ValueError(f"Invalid identifier component {key}: {value!r}")

schema_path = f'{params["ml_catalog"]}.{params["ml_schema"]}'
batch_id = params["batch_process_id"]
source_table = params["source_table"] or (
    f'{schema_path}.rfc_training_{batch_id}_test_3_{params["user_prefix"]}'
)
# Only accept simple three-part table identifiers for the optional override.
if len(source_table.split(".")) != 3 or any(
    not part or not all(c.isalnum() or c == "_" for c in part)
    for part in source_table.split(".")
):
    raise ValueError(f"Invalid source_table: {source_table!r}")

base = f"{schema_path}.rfc_training_{batch_id}"
output_tables = {
    "train": f"{base}_train",
    "validation": f"{base}_validation",
    "test": f"{base}_test",
    "train_sampled": f"{base}_train_sampled",
    "validation_sampled": f"{base}_validation_sampled",
    "test_sampled": f"{base}_test_sampled",
}

# RFC baseline contract: keep unchanged for baseline reproducibility.
TRAIN_PCT = 81
VALIDATION_PCT = 9
TEST_PCT = 10
try:
    NEGATIVE_SAMPLE_PCT = float(params["negative_sample_pct"])
except ValueError as exc:
    raise ValueError("negative_sample_pct must be a number between 0 and 100") from exc
if not 0 <= NEGATIVE_SAMPLE_PCT <= 100:
    raise ValueError("negative_sample_pct must be between 0 and 100")
VALIDATION_SAMPLE_PCT = 100
TEST_SAMPLE_PCT = 100
SEED = 42
BUCKET_COUNT = 10_000

assert TRAIN_PCT + VALIDATION_PCT + TEST_PCT == 100
assert len(output_tables.values()) == len(set(output_tables.values()))

print("RFC preparation plan")
print("Source:", source_table)
print("Dry run:", DRY_RUN)
print("Training negative sample percentage:", NEGATIVE_SAMPLE_PCT)
for name, table in output_tables.items():
    print(f"  {name}: {table}")

# COMMAND ----------
# Fail fast for nonexistent source and existing output tables, including dry runs.
if not spark.catalog.tableExists(source_table):
    raise ValueError(f"Source Delta table not found: {source_table}")

existing = [table for table in output_tables.values() if spark.catalog.tableExists(table)]
if existing:
    raise RuntimeError(
        "RFC outputs already exist; preparation will not overwrite them. "
        "Use the existing dataset version for training, or request a new batch_process_id.\n"
        + "\n".join(existing)
    )

source_df = spark.table(source_table)
required_columns = {"policy_id", "effective_dttm", "target"}
missing = required_columns - set(source_df.columns)
if missing:
    raise ValueError(f"Source table missing required columns: {sorted(missing)}")

if DRY_RUN:
    print("DRY RUN PASSED: source exists, required columns exist, outputs do not exist.")
    print("No Delta tables were created. Set dry_run=false to execute preparation.")
    dbutils.notebook.exit("DRY_RUN_OK")

# COMMAND ----------
# Policy-level assignment; all snapshots of one policy remain in the same split.
policy_split = (
    source_df.select("policy_id")
    .where(F.col("policy_id").isNotNull())
    .distinct()
    .withColumn(
        "split_bucket",
        F.pmod(F.xxhash64(F.col("policy_id"), F.lit(SEED)), F.lit(BUCKET_COUNT)),
    )
    .withColumn(
        "dataset",
        F.when(F.col("split_bucket") < TRAIN_PCT * 100, "train")
        .when(F.col("split_bucket") < (TRAIN_PCT + VALIDATION_PCT) * 100, "validation")
        .otherwise("test"),
    )
    .select("policy_id", "dataset")
)

split_df = source_df.join(policy_split, on="policy_id", how="inner")
train_df = split_df.filter(F.col("dataset") == "train").drop("dataset")
validation_df = split_df.filter(F.col("dataset") == "validation").drop("dataset")
test_df = split_df.filter(F.col("dataset") == "test").drop("dataset")


def observation_bucket():
    return F.pmod(
        F.xxhash64(F.col("policy_id"), F.col("effective_dttm"), F.lit(SEED)),
        F.lit(BUCKET_COUNT),
    )


sampled_train_df = train_df.filter(
    (F.col("target") == 1)
    | ((F.col("target") == 0) & (observation_bucket() < NEGATIVE_SAMPLE_PCT * 100))
)
sampled_validation_df = validation_df.filter(
    observation_bucket() < VALIDATION_SAMPLE_PCT * 100
)
sampled_test_df = test_df.filter(observation_bucket() < TEST_SAMPLE_PCT * 100)

datasets = {
    "train": train_df,
    "validation": validation_df,
    "test": test_df,
    "train_sampled": sampled_train_df,
    "validation_sampled": sampled_validation_df,
    "test_sampled": sampled_test_df,
}

# COMMAND ----------
if VALIDATE_OUTPUTS:
    # Validation is deliberately run before writing any table.
    invalid = source_df.filter(
        F.col("policy_id").isNull()
        | F.col("effective_dttm").isNull()
        | (~F.col("target").isin(0, 1))
        | F.col("target").isNull()
    ).limit(1).count()
    if invalid:
        raise ValueError("Source contains null keys or a missing/nonbinary target")

    duplicate = (
        source_df.groupBy("policy_id", "effective_dttm")
        .count().filter(F.col("count") > 1).limit(1).count()
    )
    if duplicate:
        raise ValueError("Duplicate policy_id/effective_dttm observations found")

    full_positives = train_df.filter(F.col("target") == 1).count()
    sampled_positives = sampled_train_df.filter(F.col("target") == 1).count()
    if full_positives != sampled_positives:
        raise AssertionError("Training negative sampling unexpectedly dropped positives")

    # Full evaluation populations are retained with the current 100% configuration.
    if VALIDATION_SAMPLE_PCT == 100 and TEST_SAMPLE_PCT == 100:
        print("Validation and test samples retain all rows by construction")

print("Pre-write checks passed. Creating six RFC Delta tables...")

# COMMAND ----------
# No overwrite, no schema evolution. An error after a successful write can leave
# some outputs present; reruns intentionally fail instead of silently replacing them.
created = []
try:
    for name, df in datasets.items():
        table = output_tables[name]
        if spark.catalog.tableExists(table):
            raise RuntimeError(f"Output appeared during preparation: {table}")
        df.write.format("delta").mode("errorifexists").saveAsTable(table)
        created.append(table)
        print(f"Created {name}: {table}")
except Exception:
    print("Preparation stopped. Tables already created in this run (not deleted):")
    for table in created:
        print(" -", table)
    raise

print("RFC preparation complete. Six Delta tables created.")

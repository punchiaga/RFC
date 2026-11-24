# Databricks notebook source
# MAGIC %md
# MAGIC # RFC — Full training and Optuna HPO
# MAGIC Uses existing Delta tables; does not recreate datasets. Validation drives HPO; test is untouched.
# MAGIC Run on approved DBR 15.4 LTS driver with sufficient memory. This notebook trains on all sampled training rows.

# COMMAND ----------
# MAGIC %pip install -q xgboost==2.1.4 scikit-learn==1.3.0 optuna==4.5.0 shap==0.46.0

# COMMAND ----------
# Databricks notebook widgets
for key, default in {
    "ml_catalog": "bdp4_prod_ml_build",
    "ml_schema": "hi_customer_right_fit_cover_data",
    "batch_process_id": "20261119034822",
    "negative_sample_pct": "5",
    "n_trials": "20",
    "n_jobs": "16",
    "collection_batches": "10",
    "run_baseline": "true",
    "run_hpo": "true",
    "score_test_after_hpo": "false",
    "run_shap": "true",
    "shap_sample_size": "1000",
    "mlflow_experiment": "",  # optional existing experiment path
}.items():
    dbutils.widgets.text(key, default)

# COMMAND ----------
import gc
import math
import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib.pyplot as plt
from scipy import sparse
import mlflow
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

catalog = dbutils.widgets.get("ml_catalog")
schema = dbutils.widgets.get("ml_schema")
batch_id = dbutils.widgets.get("batch_process_id")
negative_sample_pct = float(dbutils.widgets.get("negative_sample_pct"))
n_trials = int(dbutils.widgets.get("n_trials"))
n_jobs = int(dbutils.widgets.get("n_jobs"))
collection_batches = int(dbutils.widgets.get("collection_batches"))
run_baseline = dbutils.widgets.get("run_baseline").lower() == "true"
run_hpo = dbutils.widgets.get("run_hpo").lower() == "true"
score_test = dbutils.widgets.get("score_test_after_hpo").lower() == "true"
run_shap = dbutils.widgets.get("run_shap").lower() == "true"
shap_sample_size = int(dbutils.widgets.get("shap_sample_size"))
assert shap_sample_size > 0
assert 0 < negative_sample_pct <= 100
assert collection_batches > 0 and n_trials > 0 and n_jobs > 0
prefix = f"{catalog}.{schema}.rfc_training_{batch_id}"
train_table = prefix + "_train_sampled"
validation_table = prefix + "_validation"
test_table = prefix + "_test"
for name in [train_table, validation_table] + ([test_table] if score_test else []):
    if not spark.catalog.tableExists(name):
        raise ValueError(f"Missing prepared RFC Delta table: {name}")
experiment = dbutils.widgets.get("mlflow_experiment").strip()
if experiment:
    mlflow.set_experiment(experiment)
mlflow.autolog(disable=True)
print("Train:", train_table, "Validation:", validation_table)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Feature contract — derived once from training data, then frozen for this run
# MAGIC Exclusions and high-cardinality deferrals are copied from the RFC source notebook.

# COMMAND ----------
RFC_EXCLUDE_COLUMNS = ['policy_id', 'effective_dttm', 'target', 'target_first_event_date', 'target_first_source', 'days_from_snapshot_to_first_event', 'policy_latest_coverchange_date', 'policy_package_code_prev', 'policy_package_code', 'policy_package_cd', 'policy_package_name_prev', 'policy_package_name', 'policy_package_dsc', 'blr_first_login_date', 'top3_lifetime_last_order_date', 'blr_recent_event_date', 'top1_lifetime_last_order_date', 'top2_lifetime_last_order_date', 'policy_date_paid_to', 'policy_latest_coverchange_dtls', 'policy_last_payment_date', 'top1_24m_last_order_date', 'top3_24m_last_order_date', 'top2_24m_last_order_date', 'policy_latest_call_date', 'policy_is_hi_suspension_fh_start_date', 'policy_is_hi_suspension_fh_end_date', 'policy_future_org_id', 'policy_future_org_name', 'policy_future_org_discount_per', 'policy_overseas_type', 'policy_hosp_prod_desc', 'policy_is_ancl_prod_desc', 'policy_member_financial_status', 'policy_portfolio_dsc', 'policy_ambulance_product_id', 'policy_ambulance_product_rank', 'policy_fund_no', 'state', 'MEMBERSHIP_STATUS_SUSPENDED', 'policy_ancillary_product_id', 'policy_extras_product_id', 'policy_hospital_product_id', 'policy_national_pharmacy_product_id', 'policy_pharmacy_product_id', 'policy_future_org_from_date', 'policy_future_org_to_date']
RFC_HIGH_CARDINALITY_COLUMNS = ['blr_recent_order_item', 'policy_latest_call_app_tag', 'top1_24m_order_item', 'top1_lifetime_order_item', 'top2_24m_order_item', 'top2_lifetime_order_item', 'top3_24m_order_item', 'top3_lifetime_order_item']
RFC_RAW_DATE_COLUMNS = ["policy_org_from_date", "policy_org_to_date"]
RFC_ADDITIONAL_BINARY_COLUMNS = ["Hosp_group_OverseasHospitalOnsale"]
RFC_EXCLUDE_COLUMNS = set(RFC_EXCLUDE_COLUMNS + RFC_RAW_DATE_COLUMNS)
train_spark = spark.table(train_table)
validation_spark = spark.table(validation_table)
feature_types = {f.name: f.dataType.simpleString() for f in train_spark.schema.fields}
string_features = sorted(c for c, t in feature_types.items() if t == "string" and c not in RFC_EXCLUDE_COLUMNS and c not in RFC_HIGH_CARDINALITY_COLUMNS)
# Detect Y/N indicators from the FULL training set (not a 10k prototype).
# A non-Y/N value disqualifies a feature; null/blank values are allowed.
checks = [F.max(F.when(F.col(c).isNotNull() & (F.trim(F.col(c)) != "") & (~F.upper(F.trim(F.col(c))).isin("Y", "N")), F.lit(1)).otherwise(F.lit(0))).alias(c) for c in string_features]
invalid = train_spark.agg(*checks).first().asDict() if checks else {}
binary_cols = sorted(set(c for c in string_features if invalid[c] == 0) | (set(RFC_ADDITIONAL_BINARY_COLUMNS) & set(string_features)))
# Avoid accidentally categorising all-null strings as Y/N.
# Approved source feature exclusions and grouping should be frozen in version control for DAB.
onehot_cols = sorted(set(string_features) - set(binary_cols))
numeric_cols = sorted(c for c, t in feature_types.items() if c not in RFC_EXCLUDE_COLUMNS and c not in RFC_HIGH_CARDINALITY_COLUMNS and c not in binary_cols and c not in onehot_cols and (t in ("int", "bigint", "double", "float", "smallint", "tinyint") or t.startswith("decimal")))
if "policy_org_discount_per" in feature_types and "policy_org_discount_per" not in RFC_EXCLUDE_COLUMNS:
    numeric_cols = sorted(set(numeric_cols) | {"policy_org_discount_per"})
    onehot_cols = [c for c in onehot_cols if c != "policy_org_discount_per"]
    binary_cols = [c for c in binary_cols if c != "policy_org_discount_per"]
if "policy_org_discount_per" not in numeric_cols:
    raise ValueError("Required policy_org_discount_per missing")
feature_cols = binary_cols + onehot_cols + numeric_cols + ["policy_org_discount_per_missing"]
assert len(feature_cols) == len(set(feature_cols)), "Duplicate feature names"
assert "target" not in feature_cols and "policy_id" not in feature_cols
print(f"Features: {len(feature_cols)}; binary={len(binary_cols)}, categorical={len(onehot_cols)}, numeric={len(numeric_cols)+1}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Prepare full training and full validation (no sampling here)

# COMMAND ----------
def prepare_features(df):
    for c in binary_cols:
        val = F.upper(F.trim(F.col(c)))
        df = df.withColumn(c, F.when(val == "Y", 1.0).when(val == "N", 0.0).otherwise(F.lit(None).cast("double")))
    df = df.withColumn("policy_org_discount_per", F.expr("try_cast(`policy_org_discount_per` AS DOUBLE)"))
    return df.withColumn("policy_org_discount_per_missing", F.when(F.col("policy_org_discount_per").isNull(), 1.0).otherwise(0.0))

def collect_batched(df, batches):
    # Use policy_id + effective_dttm, as in original RFC notebook, for bounded Spark->pandas conversion.
    tagged = df.withColumn("_batch", F.pmod(F.xxhash64("policy_id", "effective_dttm", F.lit(42)), F.lit(batches)))
    parts = []
    for i in range(batches):
        pdf = prepare_features(tagged.filter(F.col("_batch") == i)).select(*feature_cols, "target").toPandas()
        for c in binary_cols + numeric_cols + ["policy_org_discount_per_missing"]:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").astype("float32")
        parts.append(pdf)
        print(f"Collected batch {i+1}/{batches}: {len(pdf):,} rows")
    return pd.concat(parts, ignore_index=True)

train_pd = collect_batched(train_spark, collection_batches)
valid_pd = collect_batched(validation_spark, collection_batches)
X_train = train_pd[feature_cols]
y_train = train_pd["target"].astype("int8")
X_valid = valid_pd[feature_cols]
y_valid = valid_pd["target"].astype("int8")
del train_pd, valid_pd
gc.collect()
assert y_train.nunique() == 2 and y_valid.nunique() == 2
print("Training", X_train.shape, "Validation", X_valid.shape)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Preprocess once; use identical matrices for baseline and Optuna

# COMMAND ----------
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), onehot_cols),
    ("numeric", "passthrough", binary_cols + numeric_cols + ["policy_org_discount_per_missing"]),
], remainder="drop", sparse_threshold=0.3)
X_train_enc = preprocessor.fit_transform(X_train)
X_valid_enc = preprocessor.transform(X_valid)
# Dense or sparse: cast without forcing sparse matrices to dense.
X_train_enc = X_train_enc.astype(np.float32)
X_valid_enc = X_valid_enc.astype(np.float32)
del X_train, X_valid
gc.collect()
print("Encoded shapes", X_train_enc.shape, X_valid_enc.shape)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Baseline and Optuna (validation PR-AUC objective)

# COMMAND ----------
def correct_probability(raw):
    rate = negative_sample_pct / 100.0
    p = np.asarray(raw, dtype=np.float64)
    return rate * p / (1.0 - p + rate * p)

def validation_metrics(model):
    raw = model.predict_proba(X_valid_enc)[:, 1]
    corrected = correct_probability(raw)
    y = y_valid.to_numpy()
    result = {
        "val_average_precision": float(average_precision_score(y, raw)),
        "val_roc_auc": float(roc_auc_score(y, raw)),
        "val_brier_corrected": float(brier_score_loss(y, corrected)),
        "val_logloss_corrected": float(log_loss(y, corrected, labels=[0, 1])),
    }
    order = np.argsort(-raw)
    for pct in (1, 5, 10, 20):
        k = int(np.ceil(len(y) * pct / 100))
        result[f"val_lift_top_{pct}pct"] = float(y[order[:k]].mean() / y.mean())
    return result

baseline_params = dict(objective="binary:logistic", n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", eval_metric="aucpr", random_state=42, n_jobs=n_jobs)
parent_tags = {"model": "rfc", "dataset_batch_id": batch_id, "negative_sample_pct": str(negative_sample_pct)}
with mlflow.start_run(run_name=f"rfc_train_hpo_{batch_id}") as parent_run:
    mlflow.set_tags(parent_tags)
    mlflow.log_param("training_table", train_table)
    mlflow.log_param("validation_table", validation_table)
    mlflow.log_param("feature_count", len(feature_cols))
    mlflow.log_param("optuna_trials", n_trials if run_hpo else 0)
    baseline_result = None
    if run_baseline:
        with mlflow.start_run(run_name="baseline_xgb", nested=True):
            baseline = XGBClassifier(**baseline_params)
            baseline.fit(X_train_enc, y_train)
            baseline_result = validation_metrics(baseline)
            mlflow.log_params(baseline_params)
            mlflow.log_metrics(baseline_result)
            print("BASELINE", baseline_result)
    study = None
    if run_hpo:
        def objective(trial):
            params = dict(baseline_params)
            params.update({
                "n_estimators": trial.suggest_int("n_estimators", 150, 650, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.2, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 15),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            })
            with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
                model = XGBClassifier(**params)
                model.fit(X_train_enc, y_train)
                result = validation_metrics(model)
                mlflow.log_params(params)
                mlflow.log_metrics(result)
            return result["val_average_precision"]
        study = optuna.create_study(direction="maximize", study_name="rfc_xgb_validation_ap", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
        mlflow.log_metric("best_hpo_val_average_precision", study.best_value)
        mlflow.log_dict(study.best_params, "best_optuna_params.json")
        print("BEST HPO AP", study.best_value, "PARAMS", study.best_params)
    # Final selected model: HPO winner if HPO enabled, otherwise baseline.
    final_params = dict(baseline_params)
    if study is not None:
        final_params.update(study.best_params)
    with mlflow.start_run(run_name="selected_model", nested=True) as selected_run:
        selected_model = XGBClassifier(**final_params)
        selected_model.fit(X_train_enc, y_train)
        selected_metrics = validation_metrics(selected_model)
        mlflow.log_params(final_params)
        mlflow.log_metrics(selected_metrics)
        # Persist preprocessing AND model together, so inference uses identical feature encoding.
        # Preprocessor is already fitted above; no retraining of preprocessing here.
        full_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", selected_model)])
        mlflow.sklearn.log_model(full_pipeline, artifact_path="model")
        mlflow.log_dict({"binary_columns": binary_cols, "categorical_columns": onehot_cols,
                         "numeric_columns": numeric_cols, "feature_columns": feature_cols,
                         "negative_sample_pct": negative_sample_pct}, "feature_contract.json")
        print("SELECTED RUN", selected_run.info.run_id, "METRICS", selected_metrics)

        # Explain only the selected fitted model; never run SHAP inside Optuna trials.
        if run_shap:
            rng = np.random.default_rng(42)
            row_ids = rng.choice(
                X_valid_enc.shape[0],
                size=min(shap_sample_size, X_valid_enc.shape[0]),
                replace=False,
            )
            X_shap = X_valid_enc[row_ids]
            # Densify only the bounded validation sample for plotting.
            if sparse.issparse(X_shap):
                X_shap = X_shap.toarray()
            X_shap = np.asarray(X_shap, dtype=np.float32)
            encoded_names = preprocessor.get_feature_names_out()
            X_shap_df = pd.DataFrame(X_shap, columns=encoded_names)

            shap_explainer = shap.TreeExplainer(selected_model)
            shap_explanation = shap_explainer(X_shap_df)

            # TreeExplainer default output is the raw XGBoost margin (log-odds),
            # NOT the negative-sampling-corrected probability.
            mlflow.log_params({
                "shap_sample_size": len(row_ids),
                "shap_sample_source": validation_table,
                "shap_explanation_units": "raw_margin_log_odds",
            })
            mlflow.log_dict({
                "sampling": "fixed RNG seed 42; sample from encoded full validation matrix",
                "explanation": "SHAP contributions describe the fitted XGBoost raw margin (log-odds). They are not corrected probabilities.",
                "feature_names": "ColumnTransformer.get_feature_names_out()",
            }, "shap/explanation_metadata.json")

            shap.plots.bar(shap_explanation, max_display=20, show=False)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap/feature_importance_bar.png")
            plt.show()
            plt.close()

            shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap/beeswarm.png")
            plt.show()
            plt.close()

            shap.plots.waterfall(shap_explanation[0], max_display=15, show=False)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap/waterfall_validation_row.png")
            plt.show()
            plt.close()
            print("SHAP charts logged to selected model MLflow run:", selected_run.info.run_id)
            del X_shap, X_shap_df, shap_explanation, shap_explainer
            gc.collect()
    print("PARENT RUN", parent_run.info.run_id)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional one-time final test (disabled by default)
# MAGIC Enable only after final model selection. Never use test metrics to choose hyperparameters.

# COMMAND ----------
if score_test:
    test_pd = collect_batched(spark.table(test_table), collection_batches)
    X_test_enc = preprocessor.transform(test_pd[feature_cols]).astype(np.float32)
    raw_test = selected_model.predict_proba(X_test_enc)[:, 1]
    y_test = test_pd["target"].astype("int8").to_numpy()
    corrected_test = correct_probability(raw_test)
    test_metrics = {
        "test_average_precision": float(average_precision_score(y_test, raw_test)),
        "test_roc_auc": float(roc_auc_score(y_test, raw_test)),
        "test_brier_corrected": float(brier_score_loss(y_test, corrected_test)),
        "test_logloss_corrected": float(log_loss(y_test, corrected_test, labels=[0, 1])),
    }
    with mlflow.start_run(run_id=selected_run.info.run_id):
        mlflow.log_metrics(test_metrics)
    print("FINAL HOLDOUT TEST", test_metrics)


# COMMAND ----------
# MAGIC %md
# MAGIC ## SHAP explainability — selected model only

# COMMAND ----------
import shap
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

SHAP_SAMPLE_SIZE = 1000
SHAP_SEED = 42

# Reproducible sample from the full validation matrix
rng = np.random.default_rng(SHAP_SEED)
sample_indices = rng.choice(
    X_valid_enc.shape[0],
    size=min(SHAP_SAMPLE_SIZE, X_valid_enc.shape[0]),
    replace=False,
)

X_shap = X_valid_enc[sample_indices]

# SHAP plotting can require a dense matrix.
# Convert only the small explainability sample.
if sparse.issparse(X_shap):
    X_shap = X_shap.toarray()

feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(selected_model)
shap_values = explainer(X_shap)

# 1. Global feature importance
shap.plots.bar(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.show()

# 2. Feature impact and direction
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.show()

# 3. Explain one validation observation
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
plt.tight_layout()
plt.show()

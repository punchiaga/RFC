# Databricks notebook source
# MAGIC %md
# MAGIC # Train Switcher Model

# COMMAND ----------

# MAGIC %md
# MAGIC #Load packages and functions

# COMMAND ----------

# MAGIC  %pip install -q lightgbm==4.6.0 shap==0.49.1 cloudpickle==2.2.1 scikit-learn==1.6.0 typing-extensions==4.10.0 bupaml==2.1.3 numpy==1.24.4 category_encoders==2.6 --upgrade
# MAGIC # mlflow==2.11.4 
# MAGIC  %pip install fairlearn==0.9.0
# MAGIC  %pip install pygam

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./switcher_functions"

# COMMAND ----------

mlflow.autolog(disable=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# COMMAND ----------

# DBTITLE 1,comment out later
# dbutils.widgets.text("training_table_name", "drop_all_features_final_target_remain")
# dbutils.widgets.text("validation_table_name", "drop_all_features_final_target_val")
# dbutils.widgets.text("test_table_name", "drop_all_features_final_target_test")
dbutils.widgets.text("model_name","hi_switcher")
dbutils.widgets.text("batch_process_id", "9090901")
dbutils.widgets.text("batch_process_dttm", "")

# batch_process_id = dbutils.widgets.get("batch_process_id")
# batch_process_dttm = dbutils.widgets.get("batch_process_dttm")
# training_table_name = dbutils.widgets.get("training_table_name")
# validation_table_name = dbutils.widgets.get("validation_table_name")
# test_table_name = dbutils.widgets.get("test_table_name")

# COMMAND ----------

batch_process_id = dbutils.widgets.get("batch_process_id")
batch_process_dttm = dbutils.widgets.get("batch_process_dttm")
effective_dttm = generate_batch_process_dttm(batch_process_dttm)

print("batch_process_id", batch_process_id)
print("batch_process_dttm", batch_process_dttm)
print("effective_dttm: ", effective_dttm)

# COMMAND ----------

target = "Switch_Target"
# target = "churn_target"
model_name = dbutils.widgets.get("model_name")
logging_prefix = "T_" # training
seed = 42
model_meta = BupaModel(model_name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #Read input data

# COMMAND ----------

# input_table_name = f"{model_meta.catalog}.{model_meta.data_schema}.{model_name}_churn_train_input_{batch_process_id}"
# input_table_name = "bdp4_prod_ml_build.hi_customer_churn_data.hi_dropper_churn_train_input_9090901"
input_table_name = "bdp4_prod_ml_build.hi_customer_churn_data.hi_switcher_churn_train_input_345577223226526"
print("input_table_name", input_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #deterministic split

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train-test-val split with hashing

# COMMAND ----------

input_df = transform_data(spark.table(input_table_name))
# policy_id_unique = input_df.select("policy_id").distinct()

id_cols = ["policy_id"]

df = input_df.withColumn(
    "sample_key",
    F.abs(F.xxhash64(F.concat_ws("||", *[F.col(c) for c in id_cols])))
)

# Deterministic split bucket
df = df.withColumn("split_bucket", F.col("sample_key") % 10)

# 80/10/10 split based purely on hash
train_df = df.filter(F.col("split_bucket") < 8)
val_df   = df.filter((F.col("split_bucket") >= 8) & (F.col("split_bucket") < 9))
test_df  = df.filter(F.col("split_bucket") >= 9)

train_df = train_df.drop("sample_key", "split_bucket")
val_df   = val_df.drop("sample_key", "split_bucket")
test_df  = test_df.drop("sample_key", "split_bucket")

# COMMAND ----------

orig_train_df_cnt = train_df.count()
orig_validation_df_cnt = val_df.count()
orig_test_df_cnt = test_df.count()

log_metrics(
    model_name=model_name,
    metric_name=f"{logging_prefix.rstrip('_')}_original_train_df_count",
    metric_value=orig_train_df_cnt,
    batch_process_id=batch_process_id
)
log_metrics(
    model_name=model_name,
    metric_name=f"{logging_prefix.rstrip('_')}_original_validation_df_count",
    metric_value=orig_validation_df_cnt,
    batch_process_id=batch_process_id
)
log_metrics(
    model_name=model_name,
    metric_name=f"{logging_prefix.rstrip('_')}_original_test_df_count",
    metric_value=orig_test_df_cnt,
    batch_process_id=batch_process_id
)

print(f"Original train df count: {orig_train_df_cnt}")
print(f"Original validation df count:  {orig_validation_df_cnt}")
print(f"Original test df count: {orig_test_df_cnt}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Preparing dataframe and features for encoding

# COMMAND ----------

id_cols = ["policy_id", "effective_dttm"]  # Multiple stable keys          
# NEG_SAMPLE_MOD = 20      
TRAIN_NEG_MOD = 20     
VALTEST_MOD = 10            
VAL_KEEP = 6
TEST_KEEP = 3      

train_df = train_df.withColumn(
    "sample_key",
    F.abs(F.xxhash64(F.concat_ws("||", *[F.col(c) for c in id_cols])))
)
val_df = val_df.withColumn(
    "sample_key",
    F.abs(F.xxhash64(F.concat_ws("||", *[F.col(c) for c in id_cols])))
)
test_df = test_df.withColumn(
    "sample_key",
    F.abs(F.xxhash64(F.concat_ws("||", *[F.col(c) for c in id_cols])))
)

# COMMAND ----------

dt_train = train_df.filter(F.col("sample_key") % TRAIN_NEG_MOD==0) #RETAIN 5% OF RANDOM SAMPLED TRAIN DATA

dt_val = val_df.filter(F.col("sample_key") % VALTEST_MOD < VAL_KEEP) #RETAIN 60% OF RANDOM SAMPLED VAL DATA. Will be further split into cal and val later to train calibrator
dt_test = test_df.filter(F.col("sample_key") % VALTEST_MOD < TEST_KEEP)

dt_train = dt_train.drop("sample_key")
dt_val = dt_val.drop("sample_key")
dt_test = dt_test.drop("sample_key")

# COMMAND ----------

print(dt_train.count() / train_df.count())
print(dt_val.count() / val_df.count())
print(dt_test.count() / test_df.count())

# COMMAND ----------

# dt_train.write.mode("overwrite").saveAsTable("bdp4_prod_datalabs.datalab_hi_drp.nanda_sample_dt_train_20260303")
# dt_val.write.mode("overwrite").saveAsTable("bdp4_prod_datalabs.datalab_hi_drp.nanda_sample_dt_val_20260303")
# dt_test.write.mode("overwrite").saveAsTable("bdp4_prod_datalabs.datalab_hi_drp.nanda_sample_dt_test_20260303")

# COMMAND ----------

# DBTITLE 1,target class proportion
# Calculate proportions of target class in dt_train, dt_val, and dt_test
train_class_proportions = (
    dt_train.groupBy(target)
    .count()
    .withColumn("proportion", (F.col("count") / dt_train.count()) * 100)
    .select(target, "proportion")
)
val_class_proportions = (
    dt_val.groupBy(target)
    .count()
    .withColumn("proportion", (F.col("count") / dt_val.count()) * 100)
    .select(target, "proportion")
)

test_class_proportions = (
    dt_test.groupBy(target)
    .count()
    .withColumn("proportion", (F.col("count") / dt_test.count()) * 100)
    .select(target, "proportion")
)

print("Proportions of target class in Training Set:")
display(train_class_proportions)

print("Proportions of target class in Validation Set:")
display(val_class_proportions)

print("Proportions of target class in Test Set:")
display(test_class_proportions)

# COMMAND ----------

sampled_train_df_cnt = dt_train.count()
sampled_validation_df_cnt = dt_val.count()
sampled_test_df_cnt = dt_test.count()

log_metrics(
    model_name=model_name,
    metric_name=f"{logging_prefix.rstrip('_')}_sampled_train_df_count",
    metric_value=sampled_train_df_cnt,
    batch_process_id=batch_process_id
)
log_metrics(
    model_name=model_name,
    metric_name=f"{logging_prefix.rstrip('_')}_sampled_validation_df_count",
    metric_value=sampled_validation_df_cnt,
    batch_process_id=batch_process_id
)
log_metrics(
    model_name=model_name,
    metric_name=f"{logging_prefix.rstrip('_')}_sampled_test_df_count",
    metric_value=sampled_test_df_cnt,
    batch_process_id=batch_process_id
)

print(f"Sampled train df count: {sampled_train_df_cnt}")
print(f"Sampled validation df count:  {sampled_validation_df_cnt}")
print(f"Sampled test df count: {sampled_test_df_cnt}")

assert sampled_train_df_cnt > 0, "No rows returned, aborting"
print(f"Number of records for training (validated): {sampled_train_df_cnt}")

# COMMAND ----------

# dt_train = spark.table("bdp4_prod_datalabs.datalab_hi_drp.nanda_sample_dt_train_20260303")
# dt_val = spark.table("bdp4_prod_datalabs.datalab_hi_drp.nanda_sample_dt_val_20260303")
# dt_test = spark.table("bdp4_prod_datalabs.datalab_hi_drp.nanda_sample_dt_test_20260303")

# COMMAND ----------

dt_train = dt_train.drop(*["effective_dttm",'effective_date', "policy_id"]).toPandas()
dt_val = dt_val.drop(*["effective_dttm", 'effective_date',"policy_id"]).toPandas()
dt_test = dt_test.drop(*["effective_dttm", 'effective_date', "policy_id"]).toPandas()

# COMMAND ----------

for frame in (dt_train, dt_val, dt_test):
    frame.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)

# COMMAND ----------

# Check class distribution in each split
print("Training Set Balance:")
print(dt_train[target].value_counts(dropna=False))
print(dt_train[target].value_counts(normalize=True) * 100)

print("Validation Set Balance:")
print(dt_val[target].value_counts(dropna=False))
print(dt_val[target].value_counts(normalize=True) * 100)

print("Test Set Balance:")
print(dt_test[target].value_counts(dropna=False))
print(dt_test[target].value_counts(normalize=True) * 100)

# COMMAND ----------

X_train = dt_train.drop(columns=[target])
y_train = dt_train[target].astype(int)

X_val = dt_val.drop(columns=[target])
y_val = dt_val[target].astype(int)

X_test = dt_test.drop(columns=[target])
y_test = dt_test[target].astype(int)

print(X_train.shape, X_val.shape, X_test.shape)
del dt_train, dt_val, dt_test

# COMMAND ----------

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of X_val:", X_val.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model final train & evaluate

# COMMAND ----------

class CalibratedPipelineWrapper(BaseEstimator):
    """
    Combines a pre-fitted pipeline with a probability calibrator.
    Avoids CalibratedClassifierCV's check_is_fitted compatibility issues
    with custom estimator wrappers (e.g. TransformedTargetClassifierWithProba).
    """
    def __init__(self, base_model, calibrator, threshold=None, calibration_method: str = "sigmoid"):
        self.base_model = base_model
        self.calibrator = calibrator
        self.calibration_method = calibration_method
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _safe_log_odds(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Clip probabilities to (eps, 1-eps) then return log(p/(1-p))."""
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    
    def predict_proba(self, X):
        raw_proba = self.base_model.predict_proba(X)[:, 1]
        if self.calibration_method == 'isotonic':
            cal_proba = self.calibrator.predict(raw_proba)
        elif self.calibration_method == 'modified_sigmoid':
            log_odds = self._safe_log_odds(raw_proba)
            cal_proba = self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        elif self.calibration_method == 'gam':
            cal_proba = self.calibrator.predict_proba(raw_proba.reshape(-1, 1))
        else:  # sigmoid / logistic
            cal_proba = self.calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
        cal_proba = np.clip(cal_proba, 0, 1)
        return np.column_stack([1 - cal_proba, cal_proba])

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Threshold must be set for calibrator wrapper before calling predict.")
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

# COMMAND ----------

is_native_encoding = False

model_params: Dict[str, float] = {
    # "applied_weight": 10.251565067711809,
    "bagging_fraction": 0.7834558931942185,
    "bagging_freq": 2,
    "boosting_type": "gbdt",
    # "encoding_strategy": "native",
    "feature_fraction": 0.5441059053621013,
    # "force_col_wise": True,
    "force_row_wise": True,
    "deterministic": True,
    "learning_rate": 0.00860275402191551,
    "max_depth": 6,
    "metric": "average_precision",
    "min_child_samples": 197,
    "min_gain_to_split": 1.9928282303699665,
    # "model_type": "lightgbm",
    "n_estimators": 2519,
    "num_leaves": 42,
    "objective": "binary",
    "random_state": seed,
    "reg_alpha": 2.8996563101212436,
    "reg_lambda": 1.4143630848280593,
    "scale_pos_weight": 10.251565067711809,
    "verbosity": -1,
    # "weight_method": "sqrt",
    "early_stopping": 150
}

# COMMAND ----------

def train_and_evaluate(is_native_encoding, model_params, X_train, y_train, X_val, y_val, X_test, y_test, calibration_method = "sigmoid", feature_select = []):
    
    print("Squeezing y")
    y_train_s = y_train.squeeze()
    y_val_s   = y_val.squeeze()

    print("getting 1D y")
    y_train_1d = y_train_s.values
    y_val_1d   = y_val_s.values

    # Split validation set into calibration and evaluation sets
    X_val_cal, X_val, y_val_cal, y_val = train_test_split(X_val, y_val_1d, test_size=0.5, random_state=seed, stratify=y_val_1d
    )
    print(f"Validation set split: calibration={len(X_val_cal)}, evaluation={len(X_val)}")


    print("starting MLflow run")

    with mlflow.start_run() as run:
        # Capture run ID for returning and registering
        run_id = run.info.run_id
        print(f"Commencing run_id: {run_id}")
        
        ordinal_features, ordinal_categories, onehot_features, target_encode_features, boolean_features, numeric_ordinal_features, categorical_features = feature_check_in_train(X_train)
        
        model = build_pipeline(
            onehot_cols=onehot_features,
            ordinal_cols=ordinal_features,
            ordinal_categories=ordinal_categories,
            numeric_ordinal_cols=numeric_ordinal_features,
            target_cols=target_encode_features,
            # numeric_cols=numerical_features,
            bool_cols=boolean_features,
            is_native_encoding=is_native_encoding,
            model_params=model_params,
            feature_select=feature_select
        )

        print("pipeline built")
        
        # Log training sample size
        log_metrics(
            model_name=model_name,
            metric_name=logging_prefix+"training_X_train_rowcount",
            metric_value=X_train.shape[0],
            batch_process_id=batch_process_id
        )

        for col in X_train.columns[:10]:
            missing_pct = X_train[col].isnull().mean() * 100
            log_metrics(
                model_name=model_name,
                metric_name=f"{logging_prefix.rstrip('_')}_X_train_missing_pct_{col}",
                metric_value=missing_pct,
                batch_process_id=batch_process_id
            )
            print(f"X_train_missing_pct_{col}",missing_pct)

        # Log target distribution in y_train
        y_train_counts = pd.Series(y_train).value_counts()
        for val, count in y_train_counts.items():
            log_metrics(
                model_name=model_name,
                metric_name=f"{logging_prefix.rstrip('_')}_y_train_count_{val}",
                metric_value=int(count),
                batch_process_id=batch_process_id
            )
            print(f"_y_train_count_{val}",int(count))

        # Train the full model pipeline.
        # Notice that we pass y_val as-is in the eval_set; the custom wrapper will handle encoding.
        pre_processor = model[:-1]

        print("processing val set",print(pre_processor))
        X_train_processed = pre_processor.fit_transform(X_train, y_train_1d)        
        X_val_processed = pre_processor.transform(X_val)

        print("Sanity Check: Identifying other categorical features")
        other_categorical_features = [c for c in X_train_processed.select_dtypes(include='object').columns if c not in (onehot_features + ordinal_features + target_encode_features + boolean_features)]
        print("Other categorical features (should be empty list):", other_categorical_features)

        print("Sanity Check: Identifying remaining columns")
        numerical_features = [c for c in X_train_processed.select_dtypes(include=[np.number]).columns if c not in (categorical_features + numeric_ordinal_features)]
        
        remaining_columns = [col for col in X_train_processed.columns if col not in (onehot_features + ordinal_features + target_encode_features + boolean_features + numerical_features + numeric_ordinal_features)]
        print("Remaining columns:", remaining_columns)

        #fitting model on train and early stopping on val
        model.fit(
            X_train,
            y_train_1d,
            classifier__eval_set=[(X_val_processed, y_val)]
        )

        # Get predictions from uncalibrated models
        # Use evaluation set for final validation metrics
        print("Generating predictions for validation and test sets...")
        y_pred_val_proba_uncal = model.predict_proba(X_val)[:, 1]
        y_pred_test_proba_uncal = model.predict_proba(X_test)[:, 1]
        y_pred_train_proba_uncal = model.predict_proba(X_train)[:, 1]

        _,_,_,_, threshold_uncalibrated = get_best_threshold_from_scores(y_test,y_pred_test_proba_uncal)
        print("threshold_uncalibrated",threshold_uncalibrated)
        mlflow.log_metric("best_threshold_uncalibrated", threshold_uncalibrated)

        y_pred_train_class_uncal = (y_pred_train_proba_uncal >= threshold_uncalibrated).astype(int)
        y_pred_val_class_uncal = (y_pred_val_proba_uncal >= threshold_uncalibrated).astype(int)
        y_pred_test_class_uncal = (y_pred_test_proba_uncal >= threshold_uncalibrated).astype(int)
 
        # Calibrate using raw predicted probabilities from the fitted base model.We bypass CalibratedClassifierCV to avoid check_is_fitted incompatibility with the TransformedTargetClassifierWithProba custom wrapper.
        
        print(f"Calibrating using all methods on {len(X_val_cal)} samples...")

        calibration_succeeded = False
        calibrated_models: Dict[str, CalibratedPipelineWrapper] = {}
        calibration_brier_scores: Dict[str, float] = {}

        try:
            y_cal_proba_raw = model.predict_proba(X_val_cal)[:, 1]
            eps = 1e-8
            y_cal_log_odds = np.log(np.clip(y_cal_proba_raw, eps, 1 - eps) /
                                    (1 - np.clip(y_cal_proba_raw, eps, 1 - eps)))

            # Isotonic
            try:
                iso_calibrator = IsotonicRegression(out_of_bounds='clip')
                iso_calibrator.fit(y_cal_proba_raw, y_val_cal)
                iso_wrapper = CalibratedPipelineWrapper(model, iso_calibrator, calibration_method='isotonic')
            
                y_pred_test_proba = iso_wrapper.predict_proba(X_test)[:, 1]
                _,_,_,_, iso_threshold_calibrated = get_best_threshold_from_scores(y_test,y_pred_test_proba)
                print("Iso_threshold_calibrated",iso_threshold_calibrated)
                mlflow.log_metric("Iso_threshold_calibrated",iso_threshold_calibrated)
            
                iso_wrapper.threshold = iso_threshold_calibrated
                calibrated_models['isotonic'] = iso_wrapper
                iso_brier = brier_score_loss(y_test, y_pred_test_proba)
                

                calibration_brier_scores['isotonic'] = iso_brier
                print(f"  isotonic Brier = {iso_brier:.6f}")
            except Exception as e:
                print(f"  isotonic FAILED: {e}")

            # --- 2. Sigmoid (plain logistic on raw proba) ---
            try:
                sig_calibrator = LogisticRegression()
                sig_calibrator.fit(y_cal_proba_raw.reshape(-1, 1), y_val_cal)
                sig_wrapper = CalibratedPipelineWrapper(model, sig_calibrator, calibration_method='sigmoid')
            
                y_pred_test_proba = sig_wrapper.predict_proba(X_test)[:, 1]
                _,_,_,_, sig_threshold_calibrated = get_best_threshold_from_scores(y_test,y_pred_test_proba)
                print("sig_threshold_calibrated",sig_threshold_calibrated)
                mlflow.log_metric("sig_threshold_calibrated",sig_threshold_calibrated)
            
                sig_wrapper.threshold = sig_threshold_calibrated
                calibrated_models['sigmoid'] = sig_wrapper
                sig_brier = brier_score_loss(y_test, y_pred_test_proba)

                calibration_brier_scores['sigmoid'] = sig_brier
                print(f"  Sigmoid Brier = {sig_brier:.6f}")
                
            except Exception as e:
                print(f"  sigmoid FAILED: {e}")

            # --- 3. Modified sigmoid (logistic on log-odds) ---
            try:
                msig_calibrator = LogisticRegression()
                msig_calibrator.fit(y_cal_log_odds.reshape(-1, 1), y_val_cal)
                msig_wrapper = CalibratedPipelineWrapper(model, msig_calibrator, calibration_method='modified_sigmoid')

                y_pred_test_proba = msig_wrapper.predict_proba(X_test)[:, 1]
                _,_,_,_, msig_threshold_calibrated = get_best_threshold_from_scores(y_test,y_pred_test_proba)
                print("msig_threshold_calibrated",msig_threshold_calibrated)
                mlflow.log_metric("msig_threshold_calibrated",msig_threshold_calibrated)
            
                msig_wrapper.threshold = msig_threshold_calibrated
                calibrated_models['modified_sigmoid'] = msig_wrapper
                msig_brier = brier_score_loss(y_test, y_pred_test_proba)

                calibration_brier_scores['modified_sigmoid'] = msig_brier
                print(f"  modified_sigmoid  Brier = {msig_brier:.6f}")
            except Exception as e:
                print(f"  modified_sigmoid  FAILED: {e}")

            # --- 4. Logistic GAM ---
            try:
                gam_calibrator = LogisticGAM(s(0)).fit(
                    y_cal_proba_raw.reshape(-1, 1), y_val_cal
                )
                gam_wrapper = CalibratedPipelineWrapper(model, gam_calibrator, calibration_method='gam')
                y_pred_test_proba = gam_wrapper.predict_proba(X_test)[:, 1]
                _,_,_,_, gam_threshold_calibrated = get_best_threshold_from_scores(y_test,y_pred_test_proba)
                print("gam_threshold_calibrated",gam_threshold_calibrated)
                mlflow.log_metric("gam_threshold_calibrated",gam_threshold_calibrated)
            
                gam_wrapper.threshold = gam_threshold_calibrated
                calibrated_models['gam'] = gam_wrapper
                gam_brier = brier_score_loss(y_test, y_pred_test_proba)

                calibration_brier_scores['gam'] = gam_brier
                print(f"  gam Brier = {gam_brier:.6f}")
            except Exception as e:
                print(f"  gam FAILED: {e}")    

            calibration_succeeded = bool(calibrated_models)
             
        except Exception as e:
            print(f"Calibration failed: {e}")
            print("Falling back to uncalibrated model")
            mlflow.log_param("calibration_method", "none (all failed)")

        mlflow.log_param("calibration_set_size", len(X_val_cal))

        # Get predictions from calibrated models
        # Use evaluation set for final validation metrics
        print("Generating predictions for validation and test sets...")
 
        if calibration_succeeded:
            if calibration_method is None:
                calibration_method = min(calibration_brier_scores, key=calibration_brier_scores.get)
            
            calibrated_model = calibrated_models[calibration_method]

            print(f"\nBest calibration method: {calibration_method} "
                    f"(Brier = {calibration_brier_scores[calibration_method]:.6f})")
            mlflow.log_param("calibration_method", calibration_method)
            for method, brier in calibration_brier_scores.items():
                mlflow.log_metric(f"cal_brier_{method}", brier)

            y_pred_train_proba = calibrated_model.predict_proba(X_train)[:, 1]
            y_pred_val_proba = calibrated_model.predict_proba(X_val)[:, 1]
            y_pred_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
            
            _,_,_,_, threshold_calibrated = get_best_threshold_from_scores(y_test,y_pred_test_proba)
            print("threshold_calibrated",threshold_calibrated)
            mlflow.log_metric("threshold_calibrated",threshold_calibrated)
            
            y_pred_train_class = calibrated_model.predict(X_train)
            y_pred_val_class = calibrated_model.predict(X_val)
            y_pred_test_class = calibrated_model.predict(X_test)

            final_model = calibrated_model 

        else:
            # Calibration failed, use uncalibrated predictions
            print("All calibration methods failed.")
            mlflow.log_param("calibration_method", "none (all failed)")

            final_model = model

            y_pred_train_proba = y_pred_train_proba_uncal
            y_pred_val_proba = y_pred_val_proba_uncal
            y_pred_test_proba = y_pred_test_proba_uncal
            y_pred_train_class = y_pred_train_class_uncal
            y_pred_val_class = y_pred_val_class_uncal
            y_pred_test_class = y_pred_test_class_uncal
            

        #PR, ROC curve, lift, confusion matrix etc for calibrated model. If calibration fails, it uses uncalibrated scores
        plot_metrics(y_test,y_pred_test_proba)
        
        #METRICS TABLE - Train,val,test
        #if calibration fails, brier and logloss of uncal scores will be same as cal scores
        train_metrics_dict = get_dataset_metrics(y_train, y_pred_train_class, y_pred_train_proba, y_pred_train_proba_uncal)
        val_metrics_dict = get_dataset_metrics(y_val, y_pred_val_class, y_pred_val_proba, y_pred_val_proba_uncal)
        test_metrics_dict = get_dataset_metrics(y_test, y_pred_test_class, y_pred_test_proba, y_pred_test_proba_uncal)

        metrics_table = pd.DataFrame(
            [train_metrics_dict, val_metrics_dict, test_metrics_dict],
            index=["Train", "Validation", "Test"]
        )
        metrics_table = metrics_table.round(4)
        mlflow.log_table(data=metrics_table, artifact_file="artifacts/model_metrics.json")

        # CALIBRATION CURVE
        if calibration_succeeded:
            print("Generating calibration curves for all calibration methods...")
            
            plot_all_calibration_curves(X_test, y_test, y_pred_test_proba, y_pred_test_proba_uncal, test_metrics_dict['Brier Score'], test_metrics_dict['Brier Score Uncalibrated'], calibrated_models, calibration_method)
                
        print("Generating calibration curve for chosen calibration method...")
        generate_calibration_curve(y_test,y_pred_test_proba,y_pred_test_proba_uncal, test_metrics_dict['Brier Score'], test_metrics_dict['Brier Score Uncalibrated'], calibration_succeeded)
        
        #FAIRNESS EVALUATION - based on final model (calibrated if it was successful)
        thres = threshold_calibrated if calibration_succeeded else threshold_uncalibrated
        fairness_summary = evaluate_fairness(X_test, y_test, y_pred_test_proba, thres)

        #FEATURE IMPORTANCE - based on base model
        generate_feature_importance_plots(model, topn = 50)
        
        #SHAP PLOT - based on base model
        generate_shap_plots(X_train, X_val, model, sample_size = 1000, seed=seed)
    
        #PIPELINE ARCHITECTURE
        print("Pipeline Steps:")
        print(model.named_steps)
        set_config(display="diagram")
        display(model)

        # Needs to be inside a run otherwise will create a new one.  
        #  NB: Still logs as pyfunc model instead of correct flavor

        signature = infer_signature(X_train, y_train)

        # mlflow.sklearn.log_model(
        #     model,
        #     artifact_path="model",
        #     signature=signature,
        #     pyfunc_predict_fn='predict_proba',
        #     # registered_model_name=f"{model_meta.catalog}.{model_meta.model_output_schema}.{model_name}"
        # )

        print("Logging uncalibrated model...")
        mlflow.sklearn.log_model(
            model,
            artifact_path="uncalibrated_model",
            signature=signature,
            pyfunc_predict_fn='predict_proba',
        )

        # Log calibrated model as the primary artifact (or uncalibrated if calibration failed)
        print(f"Logging {'calibrated' if calibration_succeeded else 'uncalibrated'} model as primary artifact...")
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            signature=signature,
            pyfunc_predict_fn='predict_proba',
        )

        return model, final_model, run_id

# COMMAND ----------

# Train and evaluate the tuned model
model, final_model, run_id = train_and_evaluate(is_native_encoding = is_native_encoding, model_params = model_params, X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test, calibration_method='sigmoid') 
#calibration_method defaults to sigmoid
#include feature select if we wish to prune features post OHE.

# COMMAND ----------

# MAGIC %skip
# MAGIC # Register model
# MAGIC register_model_by_run_id(model_meta, run_id)
# MAGIC
# MAGIC # model_uri = f"runs:/{run_id}/model"
# MAGIC # registered_model_name = f"{catalog_name}.{schema_name}.{model_name}"
# MAGIC # mlflow.register_model(model_uri=model_uri, name=registered_model_name)

# COMMAND ----------



# COMMAND ----------

def register_model_by_run_id(
        bupamodel: BupaModel,
        run_id: str,
        artifact_path: str | list[str] = "model",
        user_prefix: str | None = None
    ) -> ModelVersion | list[ModelVersion]:
    """
    Registers a model in Unity Catalog via the provided MLFlow run_id.
    This wrapper function also registers the metadata for the model in UC.

    v2.1.2 update - added artifact_path as using mlarea they have changed the default.
    v2.1.3 update - added user_prefix to allow custom naming of model in registry.
    v2.1.4 update - artifact_path can now be a list to register multiple artifacts.
                    The first artifact uses the primary model name; each subsequent
                    artifact appends its path name as a suffix (e.g. _uncalibrated_model).
                    Returns a single ModelVersion when given a str, or a list when given a list.
    """

    artifact_paths = [artifact_path] if isinstance(artifact_path, str) else artifact_path

    if user_prefix is not None and user_prefix.strip() != "":
        primary_model_location = f"{bupamodel.catalog}.{bupamodel.model_output_schema}.{user_prefix}_{bupamodel.cleaned_model_name}"
        # Also update in the BupaModel instance
        bupamodel.model_location = primary_model_location
    else:
        primary_model_location = bupamodel.model_location

    versions = []
    for i, path in enumerate(artifact_paths):
        # Primary artifact uses the base model name; additional artifacts get a suffix
        resolved_model_location = primary_model_location if i == 0 else f"{primary_model_location}_{path}"

        version = mlflow.register_model(
            f"runs:/{run_id}/{path}",
            resolved_model_location
        )

        # Print some diagnostics
        print(f"Artifact  : {path}")
        print(f"Model name: {version.name}")
        print(f"Version   : {version.version}")

        try:
            meta_tags = update_model_metadata(bupamodel)
        except RestException as ex:
            if ex.message.startswith("PERMISSION_DENIED"):
                _logger.warning(f"Error updating model metadata for model.  Metadata will NOT be set.")
            else:
                raise ex

        versions.append(version)

    return versions[0] if isinstance(artifact_path, str) else versions

# COMMAND ----------

# Register model
# register_model_by_run_id(model_meta, run_id)
register_model_by_run_id(model_meta, run_id, artifact_path=["model", "uncalibrated_model"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Score range check

# COMMAND ----------

# Generate predictions on the test set
y_test_pred_proba_uncal = model.predict_proba(X_test)[:, 1]

# Calculate the score range
score_min = y_test_pred_proba_uncal.min()
score_max = y_test_pred_proba_uncal.max()
print(f"Score range: Min = {score_min}, Max = {score_max}")

# Plot histogram of prediction scores
plt.figure(figsize=(10, 6))
plt.hist(y_test_pred_proba_uncal, bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Prediction Scores on Test Set")
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

# COMMAND ----------

# Generate predictions on the test set
y_test_pred_proba_cal = final_model.predict_proba(X_test)[:, 1]

# Calculate the score range
score_min = y_test_pred_proba_cal.min()
score_max = y_test_pred_proba_cal.max()
print(f"Score range: Min = {score_min}, Max = {score_max}")

# Plot histogram of prediction scores
plt.figure(figsize=(10, 6))
plt.hist(y_test_pred_proba_cal, bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Prediction Scores on Test Set")
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

# COMMAND ----------



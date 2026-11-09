# Databricks notebook source
# MAGIC %md
# MAGIC # Train Switcher Model

# COMMAND ----------

# %pip install mlflow==2.11.4 cloudpickle==2.2.1 hyperopt==0.2.7 scikit-learn==1.6.0 typing-extensions==4.10.0 xgboost==1.7.6 lightgbm==3.3.5 shap==0.42.0 bupaml==2.1.3 numpy==1.23.5 category_encoders==2.6.0 catboost==1.2 optuna

# COMMAND ----------

# MAGIC  %pip install -q hyperopt==0.2.7 xgboost==1.7.6 lightgbm==4.6.0 catboost==1.2.8 shap==0.49.1 optuna==4.5.0 mlflow==2.11.4 cloudpickle==2.2.1 scikit-learn==1.3.0 typing-extensions==4.10.0 bupaml==2.1.3 numpy==1.23.5 category_encoders==2.6 --upgrade

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

import sys
sys.path.append('/Workspace/Users/alakananda.giridhar@bupa.com.au/ml-hi-customer-churn/src/train')
 
from switcher_functions import *

# COMMAND ----------


mlflow.autolog(disable=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# COMMAND ----------

# TABLES: Dict[str, str] = {
#     "train": "bdp4_prod_datalabs.datalab_hi_drp.drop_all_features_final_target_remain",
#     "val": "bdp4_prod_datalabs.datalab_hi_drp.drop_all_features_final_target_val",
#     "test": "bdp4_prod_datalabs.datalab_hi_drp.drop_all_features_final_target_test",
# }
# TABLES = globals().get("TABLES", TABLES)

# COMMAND ----------

dbutils.widgets.text("training_table_name", "drop_all_features_final_target_remain")
dbutils.widgets.text("validation_table_name", "drop_all_features_final_target_val")
dbutils.widgets.text("test_table_name", "drop_all_features_final_target_test")
dbutils.widgets.text("batch_process_id", "999")
dbutils.widgets.text("batch_process_dttm", "")

batch_process_id = dbutils.widgets.get("batch_process_id")
batch_process_dttm = dbutils.widgets.get("batch_process_dttm")
training_table_name = dbutils.widgets.get("training_table_name")
validation_table_name = dbutils.widgets.get("validation_table_name")
test_table_name = dbutils.widgets.get("test_table_name")

# COMMAND ----------

print("batch_process_id", batch_process_id)
print("batch_process_dttm", batch_process_dttm)
print("training_table_name: ", training_table_name)
print("validation_table_name: ", validation_table_name)
print("test_table_name: ", test_table_name)

# COMMAND ----------

target = "Switch_Target"
model_name = "hi_switcher"
catalog_name = "bdp4_prod_datalabs"
schema_name = "datalab_hi_drp"
logging_prefix = "T_" # training
seed = 42
model_meta = BupaModel(model_name=model_name)

# COMMAND ----------

full_training_table_name = f"{model_meta.catalog}.{model_meta.data_schema}.{training_table_name}"
print("full_training_table_name", full_training_table_name)

full_validation_table_name = f"{model_meta.catalog}.{model_meta.data_schema}.{validation_table_name}"
print("full_validation_table_name", full_validation_table_name)

full_test_table_name = f"{model_meta.catalog}.{model_meta.data_schema}.{test_table_name}"
print("full_test_table_name", full_test_table_name)

# COMMAND ----------

effective_dttm = generate_batch_process_dttm(batch_process_dttm)
effective_dttm

# COMMAND ----------

shortlisted_columns = ['policy_payment_arrangement', 'policy_payment_frequency', 'policy_org_sub_type', 'policy_org_type', 'policy_is_hosp_prod_excess_amt', 'Cover_Type', 'days_to_max_lhc_person_expiry_date', 'policy_is_ancl_prod_tier', 'policy_count_of_adults', 'sales_channel_name', 'days_to_min_lhc_person_expiry_date', 'TENURE_MONTHS', 'TENURE_YEARS', 'policy_count_of_persons_covered', 'mybupa_login_cnt_12_month', 'max_prev_anc_outofpocket_12_month', 'policy_lifestage_segment', 'lhc_persons_cnt', 'avg_prev_anc_outofpocket_24_month', 'call_retention_cnt_12_month', 'total_prev_anc_outofpocket_6_month', 'max_age', 'web_cnt_12_month', 'TENURE_DAYS', 'num_prev_anc_claims_12_month', 'call_retention_cnt_9_month', 'avg_prev_anc_outofpocket_6_month', 'retail_visit_cnt_9_month', 'avg_prev_anc_outofpocket_3_month', 'policy_is_hosp_prod_copay', 'contributor_age', 'retail_visit_cnt_12_month', 'housing_lending_rate', 'policy_is_hosp_prod_tier', 'avg_prev_anc_outofpocket_12_month', 'adult_age_max', 'policy_lifestage_segment_group', 'total_prev_anc_outofpocket_12_month', 'Annual_Base_Rate_Amt', 'call_cover_review_cnt_1_month', 'policy_is_hosp_prod_rank', 'web_cnt_1_month', 'Ancl_group_CorporateAncillaryOnsale', 'Family_Type_Code', 'Ancl_group_RetailAncillaryOnsale', 'fund_no_state', 'call_retention_cnt_6_month', 'Hosp_group_CorporateHospitalOffsale', 'adult_age_min', 'avg_age', 'mybupa_login_cnt_9_month', 'total_prev_anc_outofpocket_24_month', 'web_cnt_3_month', 'min_prev_med_service_24_month', 'total_prev_hos_days_12_month', 'total_prev_anc_outofpocket_3_month', 'total_prev_hos_days_24_month', 'policy_org_bupa_staff', 'NUM_DAYS_Since_Latest_FinSusp', 'retail_visit_cnt_6_month', 'ieo_decile_2021', 'female_adult_age_max', 'total_prev_anc_service_12_month', 'policy_is_hosp_prod_off_sale', 'policy_is_ancl_prod_rank', 'policy_is_ancl_prod_off_sale', 'avg_lhc_person_age', 'min_prev_hos_days_12_month', 'max_prev_anc_outofpocket_3_month', 'messaging_cnt_12_month', 'ieo_score_2021', 'female_adult_age_min', 'Ancl_group_RetailAncillaryOffsale', 'avg_prev_anc_benefit_12_month', 'call_sales_cnt_6_month', 'policy_org_corp_voluntary', 'call_cover_review_cnt_12_month', 'mybupa_login_cnt_3_month', 'mybupa_login_cnt_1_month', 'call_other_cnt_6_month', 'avg_prev_anc_benefit_24_month', 'min_prev_hos_outofpocket_24_month', 'call_other_cnt_9_month', 'call_customer_service_cnt_1_month', 'policy_org_corp_subsidised', 'mybupa_login_cnt_6_month', 'Hosp_group_CorporateHospitalOnsale', 'call_customer_service_cnt_12_month', 'max_lhc_person_age', 'call_cover_review_cnt_3_month', 'min_lhc_person_age', 'DAYS_FinSusp', 'Hosp_group_RetailHospitalOnsale', 'avg_prev_hos_days_24_month', 
 f"{target}",
 "policy_id",
 "effective_dttm"
]

# COMMAND ----------

# # List of raw features with small volume of negative values (converted to 0 in training & scoring)
# small_volume_neg_features = [
#     'total_prev_anc_benefit_3_month',
#     'min_prev_anc_benefit_3_month',
#     'NUM_DAYS_Since_Latest_FinSusp',
#     'contributor_age',
#     'days_to_min_lhc_person_expiry_date',
#     'days_to_max_lhc_person_expiry_date',
#     'days_to_lhc_membership_expiry_date'
# ]

# # List of raw features with large volume of negative values (removed from training & scoring)
# large_volume_neg_features = [
#     'min_age',
#     'without_contributor_adult_age_max',
#     'without_contributor_adult_age_min',
#     'female_child_age_max',
#     'female_child_age_min',
#     'male_child_age_max',
#     'male_child_age_min',
#     'child_age_min',
#     'child_age_max'
# ]

# small_volume_neg_features_shortlisted = [c for c in shortlisted_columns if c in small_volume_neg_features]
# large_volume_neg_features_shortlisted = [c for c in shortlisted_columns if c in large_volume_neg_features]
# print("small volume negative features:", small_volume_neg_features_shortlisted)
# print("large volume negative features:", large_volume_neg_features_shortlisted)

# COMMAND ----------

print(len(shortlisted_columns))

# COMMAND ----------

# train_df = data_subset(transform_data(spark.table(TABLES["train"])),column_subset = shortlisted_columns)
# val_df = data_subset(transform_data(spark.table(TABLES["val"])),column_subset = shortlisted_columns)
# test_df = data_subset(transform_data(spark.table(TABLES["test"])),column_subset = shortlisted_columns)

train_df = data_subset(transform_data(spark.table(full_training_table_name)), column_subset=shortlisted_columns)
val_df = data_subset(transform_data(spark.table(full_validation_table_name)), column_subset=shortlisted_columns)
test_df = data_subset(transform_data(spark.table(full_test_table_name)), column_subset=shortlisted_columns)

# COMMAND ----------

display(train_df.limit(5))

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
# MAGIC
# MAGIC ###check deterministic sampling
# MAGIC

# COMMAND ----------

# fixed_column_order = ['policy_id', 'effective_dttm', 'Ancl_group_CorporateAncillaryOnsale', 'Ancl_group_RetailAncillaryOffsale', 'Ancl_group_RetailAncillaryOnsale', 'Annual_Base_Rate_Amt', 'Cover_Type', 'DAYS_FinSusp', 'Family_Type_Code', 'Hosp_group_CorporateHospitalOffsale', 'Hosp_group_CorporateHospitalOnsale', 'Hosp_group_RetailHospitalOnsale', 'NUM_DAYS_Since_Latest_FinSusp', 'Switch_Target', 'TENURE_DAYS', 'TENURE_MONTHS', 'TENURE_YEARS', 'adult_age_max', 'adult_age_min', 'avg_age', 'avg_lhc_person_age', 'avg_prev_anc_benefit_12_month', 'avg_prev_anc_benefit_24_month', 'avg_prev_anc_outofpocket_12_month', 'avg_prev_anc_outofpocket_24_month', 'avg_prev_anc_outofpocket_3_month', 'avg_prev_anc_outofpocket_6_month', 'avg_prev_hos_days_24_month', 'call_cover_review_cnt_12_month', 'call_cover_review_cnt_1_month', 'call_cover_review_cnt_3_month', 'call_customer_service_cnt_12_month', 'call_customer_service_cnt_1_month', 'call_other_cnt_6_month', 'call_other_cnt_9_month', 'call_retention_cnt_12_month', 'call_retention_cnt_6_month', 'call_retention_cnt_9_month', 'call_sales_cnt_6_month', 'child_age_max', 'contributor_age', 'days_to_max_lhc_person_expiry_date', 'days_to_min_lhc_person_expiry_date', 'female_adult_age_max', 'female_adult_age_min', 'fund_no_state', 'housing_lending_rate', 'ieo_decile_2021', 'ieo_score_2021', 'lhc_persons_cnt', 'max_age', 'max_lhc_person_age', 'max_prev_anc_outofpocket_12_month', 'max_prev_anc_outofpocket_3_month', 'messaging_cnt_12_month', 'min_age', 'min_lhc_person_age', 'min_prev_hos_days_12_month', 'min_prev_hos_outofpocket_24_month', 'min_prev_med_service_24_month', 'mybupa_login_cnt_12_month', 'mybupa_login_cnt_1_month', 'mybupa_login_cnt_3_month', 'mybupa_login_cnt_6_month', 'mybupa_login_cnt_9_month', 'num_prev_anc_claims_12_month', 'policy_count_of_adults', 'policy_count_of_persons_covered', 'policy_is_ancl_prod_off_sale', 'policy_is_ancl_prod_rank', 'policy_is_ancl_prod_tier', 'policy_is_hosp_prod_copay', 'policy_is_hosp_prod_excess_amt', 'policy_is_hosp_prod_off_sale', 'policy_is_hosp_prod_rank', 'policy_is_hosp_prod_tier', 'policy_lifestage_segment', 'policy_lifestage_segment_group', 'policy_org_bupa_staff', 'policy_org_corp_subsidised', 'policy_org_corp_voluntary', 'policy_org_sub_type', 'policy_org_type', 'policy_payment_arrangement', 'policy_payment_frequency', 'retail_visit_cnt_12_month', 'retail_visit_cnt_6_month', 'retail_visit_cnt_9_month', 'sales_channel_name', 'total_prev_anc_outofpocket_12_month', 'total_prev_anc_outofpocket_24_month', 'total_prev_anc_outofpocket_3_month', 'total_prev_anc_outofpocket_6_month', 'total_prev_anc_service_12_month', 'total_prev_hos_days_12_month', 'total_prev_hos_days_24_month', 'web_cnt_12_month', 'web_cnt_1_month', 'web_cnt_3_month', 'without_contributor_adult_age_min']


# COMMAND ----------

# from pyspark.sql import functions as F
# from pyspark.sql.functions import col, sha2, concat_ws, collect_list, array_sort

# # ---------- hashing utility (order-independent) ----------
# def hash_df_ignore_order(df, fixed_column_order):
#     """
#     Returns a single deterministic hash string for the DataFrame,
#     ignoring row order (and ignoring '_tiebreaker' if present).
#     """
#     if "_tiebreaker" in df.columns:
#         df = df.drop("_tiebreaker")

#     # Project to fixed column order
#     proj_df = df.select(*fixed_column_order)

#     # Hash each row
#     row_hashes = proj_df.select(
#         sha2(concat_ws("||", *fixed_column_order), 256).alias("h")
#     )

#     # Aggregate hashes in an order-independent way
#     final_hash = row_hashes.agg(
#         sha2(
#             concat_ws("||", array_sort(collect_list("h"))),
#             256
#         ).alias("df_hash")
#     ).collect()[0]["df_hash"]

#     return final_hash

# # ---------- sampling + hashing loop ----------
# num_runs = 5

# sampled_train_dfs = {}
# sampled_val_dfs = {}
# sampled_test_dfs = {}

# train_hashes = {}
# val_hashes = {}
# test_hashes = {}

# for i in range(num_runs):
#     # Read & transform
#     train_df = data_subset(
#         transform_data(spark.table(full_training_table_name)),
#         column_subset=shortlisted_columns
#     )
#     val_df = data_subset(
#         transform_data(spark.table(full_validation_table_name)),
#         column_subset=shortlisted_columns
#     )
#     test_df = data_subset(
#         transform_data(spark.table(full_test_table_name)),
#         column_subset=shortlisted_columns
#     )

#     # Optional: stable cosmetic ordering before sampling
#     sorted_train_df = train_df#.orderBy(*[F.col(c) for c in fixed_column_order])
#     sorted_val_df   = val_df#.orderBy(*[F.col(c) for c in fixed_column_order])
#     sorted_test_df  = test_df#.orderBy(*[F.col(c) for c in fixed_column_order])

#     # Train: keep all positives, sample 5% of negatives
#     dt_train_switch = sorted_train_df.filter(col(target) == 1)
#     dt_train_noswitch = sorted_train_df.filter(col(target) == 0)

#     dt_train_noswitch_sample = dt_train_noswitch.sample(
#         withReplacement=False,
#         fraction=0.05,
#         seed=seed,
#     )

#     dt_train = dt_train_switch.unionByName(dt_train_noswitch_sample)

#     # Val & Test: 30% random sample
#     dt_val = sorted_val_df.sample(
#         withReplacement=False,
#         fraction=0.3,
#         seed=seed,
#     )
#     dt_test = sorted_test_df.sample(
#         withReplacement=False,
#         fraction=0.3,
#         seed=seed,
#     )

#     # Cache (optional, avoids recompute when hashing)
#     dt_train = dt_train.cache()
#     dt_val = dt_val.cache()
#     dt_test = dt_test.cache()

#     sampled_train_dfs[i] = dt_train
#     sampled_val_dfs[i] = dt_val
#     sampled_test_dfs[i] = dt_test

#     # Hash each sampled DF
#     train_hashes[i] = hash_df_ignore_order(dt_train, fixed_column_order)
#     val_hashes[i] = hash_df_ignore_order(dt_val, fixed_column_order)
#     test_hashes[i] = hash_df_ignore_order(dt_test, fixed_column_order)

# # ---------- summarise determinism ----------
# def all_equal(vals):
#     vals = list(vals)
#     return len(set(vals)) <= 1

# print("Train hashes:")
# for i, h in train_hashes.items():
#     print(f"  run {i}: {h}")

# print("\nVal hashes:")
# for i, h in val_hashes.items():
#     print(f"  run {i}: {h}")

# print("\nTest hashes:")
# for i, h in test_hashes.items():
#     print(f"  run {i}: {h}")

# print("\nDeterminism check (ignoring row order):")
# print(f"  All train samples identical? {all_equal(train_hashes.values())}")
# print(f"  All val samples identical?   {all_equal(val_hashes.values())}")
# print(f"  All test samples identical?  {all_equal(test_hashes.values())}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### end of deterministic sampling test

# COMMAND ----------

fixed_column_order = ['policy_id', 'effective_dttm']+[c for c in shortlisted_columns if c not in ['policy_id', 'effective_dttm']]

# COMMAND ----------

print(len(fixed_column_order))
fixed_column_order

# COMMAND ----------

# Optional: stable cosmetic ordering of columns before sampling
sorted_train_df = train_df.orderBy(*[F.col(c) for c in fixed_column_order])
sorted_val_df   = val_df.orderBy(*[F.col(c) for c in fixed_column_order])
sorted_test_df  = test_df.orderBy(*[F.col(c) for c in fixed_column_order])

# COMMAND ----------

# Train: keep all positives, sample 5% of negatives
dt_train_switch = sorted_train_df.filter(col(target) == 1)
dt_train_noswitch = sorted_train_df.filter(col(target) == 0)
dt_train_noswitch_sample = dt_train_noswitch.sample(withReplacement=False, fraction=0.05, seed=seed)
dt_train = dt_train_switch.unionByName(dt_train_noswitch_sample)

# Val & Test: 30% random sample
dt_val = sorted_val_df.sample(withReplacement=False, fraction=0.3, seed=seed)
dt_test = sorted_test_df.sample(withReplacement=False, fraction=0.3, seed=seed)

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

# display(dt_train.groupBy(F.to_date(F.col('effective_dttm')),F.col(target)).count())

# COMMAND ----------

dt_train = dt_train.to_pandas_on_spark().drop(columns=["effective_dttm", "policy_id"])
dt_val = dt_val.to_pandas_on_spark().drop(columns=["effective_dttm", "policy_id"])
dt_test = dt_test.to_pandas_on_spark().drop(columns=["effective_dttm", "policy_id"])

# COMMAND ----------


dt_train = dt_train.to_pandas()
dt_val = dt_val.to_pandas()
dt_test = dt_test.to_pandas()

# COMMAND ----------

for frame in (dt_train, dt_val, dt_test):
    frame.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)

# COMMAND ----------

# dt_train = dt_train.sample(frac=0.1, random_state=seed)
# dt_val = dt_val.sample(frac=0.1, random_state=seed)
# dt_test = dt_test.sample(frac=0.1, random_state=seed)

# for df in [train_df, val_df, test_df, sorted_train_df, sorted_val_df, sorted_test_df]:
#     if 'unpersist' in dir(df):
#         df.unpersist()
# del train_df, val_df, test_df, sorted_train_df, sorted_val_df, sorted_test_df

# COMMAND ----------

# Check class distribution in each split
print("Training Set Balance:")
print(dt_train[target].value_counts(dropna=False))

print("Validation Set Balance:")
print(dt_val[target].value_counts(dropna=False))

print("Test Set Balance:")
print(dt_test[target].value_counts(dropna=False))

# COMMAND ----------

X_train = dt_train.drop(columns=[target])
y_train = dt_train[target].astype(int)

X_val = dt_val.drop(columns=[target])
y_val = dt_val[target].astype(int)

X_test = dt_test.drop(columns=[target])
y_test = dt_test[target].astype(int)

print(X_train.shape, X_val.shape, X_test.shape)
# del dt_train, dt_val, dt_test

# COMMAND ----------

BOOLEAN_COLS: List[str] =  [
    'policy_org_bupa_staff',
    'policy_org_corp_voluntary',
    'policy_org_corp_subsidised',
    'policy_org_corp_split_bill',
    'policy_org_google',
    'policy_org_defence',
    'policy_is_hosp_prod_copay',
    'policy_is_hosp_prod_corp',
    'policy_is_hosp_prod_off_sale',
    'policy_is_ancl_prod_corp',
    'policy_is_ancl_prod_off_sale',
    'policy_is_ancl_prod_srv_sel',
    'policy_is_ancl_prod_gap_free',
    'Ancl_group_CorporateAncillaryOffsale',
    'Ancl_group_CorporateAncillaryOnsale',
    'Ancl_group_RetailAncillaryOffsale',
    'Ancl_group_RetailAncillaryOnsale',
    'Hosp_group_CorporateHospitalOffsale',
    'Hosp_group_CorporateHospitalOnsale',
    'Hosp_group_OverseasHospitalOnsale',
    'Hosp_group_RetailHospitalOffsale',
    'Hosp_group_RetailHospitalOnsale'
    ]


ONEHOT_COLS: List[str] =  [
    "Cover_Type",
    "Family_Type_Code",
    "policy_org_type",
    "policy_org_sub_type",
    "policy_is_hosp_prod_tier",
    "policy_is_ancl_prod_tier",
    "policy_family_type",
    "policy_payment_frequency",
    "policy_lifestage_segment",
    "policy_lifestage_segment_group",
    "fund_no_state",
    "policy_payment_arrangement"
]


# Columns for Target Encoding (non-business-critical)
TARGET_ENCODE_COLS: List[str] = [
    "sales_channel_name" 
]

# Ordinal columns
NUMERIC_ORDINAL_COLS: List[str] = [
    "irsd_decile_2021",
    "irsad_decile_2021",
    "ier_decile_2021",
    "ieo_decile_2021",
    "months_to_rate_rise",
    "months_to_end_of_fy"
]

ORDINAL_COLS: List[str] = []

# COMMAND ----------

boolean_cols = list(globals().get("BOOLEAN_COLS", globals().get("cols_Y_N_to_1_0", [])))
onehot_cols = list(globals().get("ONEHOT_COLS", globals().get("onehot_cols", [])))
target_encode_cols = list(globals().get("TARGET_ENCODE_COLS", globals().get("target_encode_cols", [])))
ordinal_cols = list(globals().get("ORDINAL_COLS", globals().get("ordinal_columns", [])))
numeric_ordinal_cols = list(globals().get("NUMERIC_ORDINAL_COLS", globals().get("numeric_ordinal_columns", [])))
categorical_cols = list(globals().get("cols_str", []))

# COMMAND ----------

ordinal_features = [col for col in ordinal_cols if col in X_train.columns]
onehot_features = [col for col in onehot_cols if col in X_train.columns]
target_encode_features = [col for col in target_encode_cols if col in X_train.columns]
boolean_features = [col for col in boolean_cols if col in X_train.columns]
numeric_ordinal_features = [col for col in numeric_ordinal_cols if col in X_train.columns]
categorical_features = [col for col in categorical_cols if col in X_train.columns]+onehot_features+ordinal_features+target_encode_features+boolean_features


# COMMAND ----------

#numeric columns
X_train = convert_object_to_numeric(X_train, categorical_features)
X_test = convert_object_to_numeric(X_test, categorical_features)
X_val = convert_object_to_numeric(X_val, categorical_features)

numerical_features = [c for c in X_train.select_dtypes(include=[np.number]).columns if c not in (categorical_features+numeric_ordinal_features)]


# COMMAND ----------

other_categorical_features = [c for c in X_train.select_dtypes(include='object').columns if c not in (onehot_features+ordinal_features+target_encode_features+boolean_features)] #should be empty list
print(other_categorical_features)

# COMMAND ----------

for col_name in numerical_features:
    X_train[col_name] = X_train[col_name].mask(X_train[col_name] < 0, 0)
    X_val[col_name] = X_val[col_name].mask(X_val[col_name] < 0, 0)
    X_test[col_name] = X_test[col_name].mask(X_test[col_name] < 0, 0)

# COMMAND ----------

# for col_name in small_volume_neg_features_shortlisted:
#     print("\n",col_name)
#     display(X_train[col_name].apply(lambda x: x<0).value_counts())


# COMMAND ----------

# MAGIC %md
# MAGIC ### hyperparameter tuning

# COMMAND ----------


experiment_path = "/Workspace/Users/alakananda.giridhar@bupa.com.au/switchermodel_hyperparametertuning_20251126"

mlflow.set_experiment(experiment_path)
mlflow.xgboost.autolog()
mlflow.lightgbm.autolog()

if mlflow.active_run():
    mlflow.end_run()

#Pre-processing val and test sets before passing them as args to the model

pre_processor = build_pipeline(
    onehot_features,
    ordinal_features,
    numeric_ordinal_features,
    target_encode_features,
    numerical_features,
    boolean_features,
    model_type,
    model_params,
    label_encoder=LabelEncoder(),
    feature_select=[]
)[:-1]

pre_processor.fit(X_train, y_train)
X_train_processed = pre_processor.transform(X_train)
print(f"Preprocessing produced {X_train_processed.shape[1]} features.")
print("Training columns", X_train_processed.columns)
print("training_X_train_rowcount",X_train_processed.shape[0])

X_val_processed = pre_processor.transform(X_val)
print("shape of X_val_processed",X_val_processed.shape)

X_test_processed = pre_processor.transform(X_test)
print("shape of X_test_processed",X_test_processed.shape)




# COMMAND ----------

# from sklearn.model_selection import train_test_split

# X_tune, _, y_tune, _ = train_test_split(
#     X_train_processed,
#     y_train,
#     train_size=0.1,
#     stratify=y_train,
#     random_state=seed
# )
# X_val_tune, _, y_val_tune, _ = train_test_split(
#     X_val_processed,
#     y_val,
#     train_size=0.1,
#     stratify=y_val,
#     random_state=seed
# )
# X_test_tune, _, y_test_tune, _ = train_test_split(
#     X_test_processed,
#     y_test,
#     train_size=0.1,
#     stratify=y_test,
#     random_state=seed
# )


# COMMAND ----------

# print("Train processed class counts and proportions:")
# print(y_train.value_counts())
# print(y_train.value_counts(normalize=True))

# print("Train tune class counts and proportions:")
# print(y_tune.value_counts())
# print(y_tune.value_counts(normalize=True))

# print("Validation processed class counts and proportions:")
# print(y_val.value_counts())
# print(y_val.value_counts(normalize=True))

# print("Validation tune class counts and proportions:")
# print(y_val_tune.value_counts())
# print(y_val_tune.value_counts(normalize=True))

# print("Test processed class counts and proportions:")
# print(y_test.value_counts())
# print(y_test.value_counts(normalize=True))

# print("Test tune class counts and proportions:")
# print(y_test_tune.value_counts())
# print(y_test_tune.value_counts(normalize=True))

# COMMAND ----------

# print([c for c in X_train_processed.columns if c in large_volume_neg_features])

# COMMAND ----------



# COMMAND ----------

import gc
import lightgbm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score
)
import optuna
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog(disable=True)

OPTUNA_OBJECTIVE = "average_precision"


def objective(trial):

    print("Starting new trial")

    model_type = trial.suggest_categorical(
        "model_type",
        ["lightgbm", "xgboost", "catboost", "hist_gbdt"]
    )
    print(f"Selected model type: {model_type}")

    fit_kwargs = {}
    logged_params = {}

    if model_type == "lightgbm":

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "max_depth": -1,
            "num_leaves": trial.suggest_int("num_leaves", 50, 100),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "reg_alpha": trial.suggest_float("reg_alpha", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.5, 1.0)
        }
        print(f"LightGBM params: {params}")

        clf = LGBMClassifier(**params, n_jobs=-1, force_col_wise=True)

        fit_kwargs["classifier__callbacks"] = [
            lightgbm.early_stopping(20),
            lightgbm.log_evaluation(0)
        ]
        fit_kwargs["classifier__eval_set"] = [(X_val_processed, y_val)]

        logged_params = params.copy()

    elif model_type == "xgboost":

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-4, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),

        }
        print(f"XGBoost params: {params}")

        clf = XGBClassifier(**params, use_label_encoder=False, random_state=42, n_jobs=-1)

        fit_kwargs["classifier__eval_set"] = [(X_val_processed, y_val)]
        fit_kwargs["classifier__early_stopping_rounds"] = 30
        fit_kwargs["classifier__verbose"] = False

        logged_params = params.copy()

    elif model_type == "catboost":

        params = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("cb_learning_rate", 0.02, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0), 
        }
        print(f"CatBoost params: {params}")

        clf = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            allow_writing_files=False,
            thread_count=-1,
            verbose=False,
        )

        fit_kwargs["classifier__eval_set"] = [(X_val_processed, y_val)]
        fit_kwargs["classifier__early_stopping_rounds"] = 20
        fit_kwargs["classifier__use_best_model"] = True

        logged_params = params.copy()

    else:  # hist_gbdt

        params = {
            "learning_rate": trial.suggest_float("hgb_learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 15, 255),
            "min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 20, 200),
            "max_iter": trial.suggest_int("hgb_max_iter", 100, 400),
            "l2_regularization": trial.suggest_float("hgb_l2_regularization", 1e-4, 10.0, log=True),
        }
        print(f"HistGradientBoosting params: {params}")

        clf = HistGradientBoostingClassifier(
            **params,
            early_stopping=True,
            validation_fraction=0.1,      # <-- REQUIRED for early stopping
            n_iter_no_change=20,
            random_state=42,
        )

        logged_params = params.copy()

    model = Pipeline([
        ("classifier", clf)
    ])

    print("Starting MLflow run")
    with mlflow.start_run(nested=True):

        print("Fitting model")
        model.fit(X_train_processed, y_train, **fit_kwargs)
        print("Model fit complete")

        # Predict validation
        y_prob_val = model.predict_proba(X_val_processed)[:, 1]
        y_pred_val = (y_prob_val >= 0.5).astype(int)

        # Predict test
        y_prob_test = model.predict_proba(X_test_processed)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)

        # Metrics
        metrics = {
            "val_roc_auc": roc_auc_score(y_val, y_prob_val),
            "val_average_precision": average_precision_score(y_val, y_prob_val),
            "val_log_loss": log_loss(y_val, y_prob_val),
            "val_accuracy": accuracy_score(y_val, y_pred_val),
            "val_precision": precision_score(y_val, y_pred_val, zero_division=0),
            "val_recall": recall_score(y_val, y_pred_val, zero_division=0),
            "val_f1_score": f1_score(y_val, y_pred_val, zero_division=0),

            "test_roc_auc": roc_auc_score(y_test, y_prob_test),
            "test_average_precision": average_precision_score(y_test, y_prob_test),
            "test_log_loss": log_loss(y_test, y_prob_test),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
            "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
            "test_f1_score": f1_score(y_test, y_pred_test, zero_division=0),
        }
        print(f"Trial metrics: {metrics}")

        # Log parameters + metrics
        mlflow.log_params({f"{model_type}_{k}": v for k, v in logged_params.items()})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=f"{model_type}_pipeline")

    print("Trial complete, cleaning up")
    del model, clf
    gc.collect()

    return metrics["val_" + OPTUNA_OBJECTIVE]

# COMMAND ----------

study = optuna.create_study(direction="maximize", study_name=f"optimize_{OPTUNA_OBJECTIVE}")
study.optimize(objective, n_trials=20)

print(f"Optimized metric: {OPTUNA_OBJECTIVE}")
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")

trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# COMMAND ----------

# MAGIC %md
# MAGIC ## model final train & evaluate

# COMMAND ----------

# model_type = 'lightgbm'
# lightgbm_params: Dict[str, float] = {
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'boosting_type': 'gbdt',
#     'num_leaves': 78,
#     'min_data_in_leaf': 92,
#     'max_depth': -1,
#     'learning_rate': 0.0811489066428134,
#     'reg_alpha': 7.370707749622083,
#     'reg_lambda': 4.330201728265486,
#     'n_estimators': 120,
#     'random_state': 42,
#     'verbose': -1,
#     'early_stopping_round': 20,
# }
 
model_type = 'hist_gbdt'
model_params: Dict[str, float] = {
    'early_stopping': False,
    'learning_rate': 0.022954004519141206,
    'max_leaf_nodes': 123,
    'min_samples_leaf': 58,
    'max_iter': 376,
    'l2_regularization': 5.734164736624926,
}

# COMMAND ----------

def train_and_evaluate(onehot_cols,
                   ordinal_cols,
                   numeric_ordinal_cols,
                   target_cols,
                   numeric_cols,
                   bool_cols, 
                   model_type, model_params, X_train, y_train, X_val, y_val, X_test, y_test, feature_select = []):
    
    print("Squeezing y")
    y_train_s = y_train.squeeze()
    y_val_s   = y_val.squeeze()

    print("getting 1D y")
    y_train_1d = y_train_s.values
    y_val_1d   = y_val_s.values

    print("starting MLflow run")

    with mlflow.start_run() as run:
        # Capture run ID for returning and registering
        run_id = run.info.run_id
        print(f"Commencing run_id: {run_id}")

        # Create a single LabelEncoder instance for target transformation.
        # Do not pre-fit it; let the custom wrapper do that internally.
        label_encoder = LabelEncoder()
        print("label_encoder")

        model =  build_pipeline(onehot_cols,
                   ordinal_cols,
                   numeric_ordinal_cols,
                   target_cols,
                   numeric_cols,
                   bool_cols,
                   model_type, model_params, label_encoder, feature_select)

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
        
        model.fit(
            X_train,
            y_train_1d
        )

        print("classifier fit check:")
        print(model.named_steps["classifier"])

        y_pred_val_proba = model.predict_proba(X_val)[:, 1]
        y_pred_test_proba = model.predict_proba(X_test)[:, 1]

        y_pred_val_class = model.predict(X_val)
        y_pred_test_class = model.predict(X_test)

        val_f1_score = f1_score(y_val, y_pred_val_class)
        val_roc_auc = roc_auc_score(y_val, y_pred_val_proba)
        val_log_loss = log_loss(y_val, y_pred_val_proba)
        val_avg_precision = average_precision_score(y_val, y_pred_val_proba)
        val_recall = recall_score(y_val, y_pred_val_class)
        val_precision = precision_score(y_val, y_pred_val_class)

        test_f1_score = f1_score(y_test, y_pred_test_class)
        test_roc_auc = roc_auc_score(y_test, y_pred_test_proba)
        test_log_loss = log_loss(y_test, y_pred_test_proba)
        test_avg_precision = average_precision_score(y_test, y_pred_test_proba)
        test_recall = recall_score(y_test, y_pred_test_class)
        test_precision = precision_score(y_test, y_pred_test_class)

        cm_val = confusion_matrix(y_val, y_pred_val_class)
        tn_val, fp_val, fn_val, tp_val = cm_val.ravel()
        example_count_val = len(y_val)
        accuracy_val = (tn_val + tp_val) / example_count_val

        cm_test = confusion_matrix(y_test, y_pred_test_class)
        tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
        example_count_test = len(y_test)
        accuracy_test = (tn_test + tp_test) / example_count_test

        val_metrics_dict = {
            "true_negatives": tn_val,
            "false_positives": fp_val,
            "false_negatives": fn_val,
            "true_positives": tp_val,
            "example_count": example_count_val,
            "accuracy_score": accuracy_val,
            "recall_score": val_recall,
            "precision_score": val_precision,
            "f1_score": val_f1_score,
            "ROC AUC": val_roc_auc,
            "Log Loss": val_log_loss,
            "Avg Precision": val_avg_precision
        }

        test_metrics_dict = {
            "true_negatives": tn_test,
            "false_positives": fp_test,
            "false_negatives": fn_test,
            "true_positives": tp_test,
            "example_count": example_count_test,
            "accuracy_score": accuracy_test,
            "recall_score": test_recall,
            "precision_score": test_precision,
            "f1_score": test_f1_score,
            "ROC AUC": test_roc_auc,
            "Log Loss": test_log_loss,
            "Avg Precision": test_avg_precision
        }

        metrics_table = pd.DataFrame(
            [val_metrics_dict, test_metrics_dict],
            index=["Validation", "Test"]
        )
        metrics_table = metrics_table.round(4)
        display(metrics_table)

        for k, v in val_metrics_dict.items():
            mlflow.log_metric(f"val_{k}", v)

        for k, v in test_metrics_dict.items():
            mlflow.log_metric(f"test_{k}", v)

        plot_metrics(y_test, X_test, model)

        print("Setting deterministic seed")
        random.seed(seed)
        np.random.seed(seed)

        print("Preparing train and validation samples")
        mode = X_train.mode().iloc[0]
        train_sample = X_train.sample(n=min(100, len(X_train)), random_state=seed).fillna(mode)
        example = X_val.sample(n=min(100, len(X_val)), random_state=seed).fillna(mode)

        def predict_fn(data):
            # Ensure the input data matches the schema of the training data
            if isinstance(data, pd.DataFrame):
                # Reorder columns to match the training data schema
                data = data[X_train.columns]
            else:
                # Convert to DataFrame and ensure column order matches
                data = pd.DataFrame(data, columns=X_train.columns)
            # Ensure data types match the training data
            data = data.astype(X_train.dtypes.to_dict())
            return model.predict(data).ravel()

        underlying_preprocessor = model.named_steps["preprocessor"]
        underlying_model = model.named_steps["classifier"]

        is_tree_model = hasattr(underlying_model, "booster_") or hasattr(underlying_model, "n_estimators_")  
        if is_tree_model:
            print("Using TreeExplainer (deterministic)")
            example = underlying_preprocessor.transform(example)
            explainer = TreeExplainer(underlying_model)
            shap_values = explainer.shap_values(example)
        else:
            print("Using KernelExplainer with 1000 samples per instance")
            explainer = KernelExplainer(predict_fn, train_sample, link="identity")
            shap_values = explainer.shap_values(example, nsamples="auto", l1_reg="num_features(10)")

        print("Generating SHAP summary plot")
        shap.summary_plot(shap_values, example, feature_names=example.columns, show=False)

        print("Saving SHAP plot")
        shap_plot_path = "plots/shap_travel_summary.png"
        os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
        plt.savefig(shap_plot_path)

        # Log artifact with full path
        mlflow.log_artifact(shap_plot_path)
        plt.show()
        plt.close()

        print("Pipeline Steps:")
        print(model.named_steps)
        set_config(display="diagram")
        display(model)

        # Needs to be inside a run otherwise will create a new one.  
        #  NB: Still logs as pyfunc model instead of correct flavor

        signature = infer_signature(X_train, y_train)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            pyfunc_predict_fn='predict_proba',
            registered_model_name=f"{catalog_name}.{schema_name}.{model_name}"
        )

        return model, run_id

# COMMAND ----------

feature_select = []
print(len(feature_select))

# COMMAND ----------

# Train and evaluate the tuned model
model, run_id = train_and_evaluate(onehot_cols = onehot_features,
                   ordinal_cols = ordinal_features,
                   numeric_ordinal_cols = numeric_ordinal_features,
                   target_cols = target_encode_features,
                   numeric_cols = numerical_features,
                   bool_cols = boolean_features,
                   model_type = model_type, model_params = model_params, X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test, feature_select = feature_select)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Invoke model artifact to make predictions

# COMMAND ----------

# import mlflow.pyfunc

# # Set the registry URI to Unity Catalog
# mlflow.set_registry_uri("databricks-uc")

# # Correctly specify the model URI for Unity Catalog
# model_uri = "models:/bdp4_prod_datalabs.datalab_hi_drp.hi_switcher/5"

# # Load the model from Unity Catalog
# loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# y_pred_test_proba = loaded_model.predict(X_test)[:, 1]
# # y_pred_test_class = model.predict(X_test)

# COMMAND ----------

# y_pred_test_proba = model.predict(X_test)[:, 1]
# # y_pred_test_class = model.predict(X_test)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Plotting model performance

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Ensure both are 1D
y_true = np.ravel(y_test)  # or y_test.values.ravel() if it's a pandas Series
y_scores = np.ravel(model.predict_proba(X_test)[:, 1])  # Probability of class 1

# Now safely create the DataFrame
test_results = pd.DataFrame({
    "actual": y_true,
    "score": y_scores
})

test_results = test_results.sort_values("score", ascending=False).reset_index(drop=True)

n_samples = len(test_results)
if n_samples == 0:
    raise ValueError("test_results is empty.")

total_positives = test_results["actual"].sum()
if total_positives == 0:
    raise ValueError("test_results must contain at least one positive outcome.")

test_results["population_pct"] = (np.arange(n_samples) + 1) / n_samples
test_results["cumulative_positives"] = test_results["actual"].cumsum()
test_results["gain_pct"] = test_results["cumulative_positives"] / total_positives
test_results["decile"] = pd.qcut(test_results["population_pct"], 10, labels=False, duplicates="drop")

lift_table = (
    test_results
    .groupby("decile")
    .agg(
        count=("actual", "count"),
        positives=("actual", "sum"),
        population_pct=("population_pct", "max"),
        gain_pct=("gain_pct", "max"),
    )
    .sort_index()
)

lift_table["response_rate"] = lift_table["positives"] / lift_table["count"]
lift_table["cumulative_positives"] = lift_table["positives"].cumsum()
lift_table["cumulative_percentage"] = lift_table["cumulative_positives"] / total_positives

plt.figure(figsize=(8, 6))
plt.plot(test_results["population_pct"], test_results["gain_pct"], label="Model", color="C0")
plt.plot([0, 1], [0, 1], label="Baseline", color="C1", linestyle="--")
plt.xlabel("Cumulative population")
plt.ylabel("Cumulative gain")
plt.title("Cumulative Gain Chart")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

display(lift_table)

# COMMAND ----------

# Calculate precision, recall and thresholds for test set
precisions_test, recalls_test, thresholds_test = precision_recall_curve(y_test, y_scores)

# Calculate F1 scores for all thresholds (thresholds size is one less than precisions and recalls)
f1_scores_test = 2 * (precisions_test[:-1] * recalls_test[:-1]) / (precisions_test[:-1] + recalls_test[:-1])
f1_scores_test = np.nan_to_num(f1_scores_test, nan=0.0)
best_idx_test = f1_scores_test.argmax()
best_threshold_test = thresholds_test[best_idx_test]

precision_recall_diff_test = abs(precisions_test[:-1] - recalls_test[:-1])
min_diff_index_test = precision_recall_diff_test.argmin()
precision_equals_recall_threshold_test = thresholds_test[min_diff_index_test]

best_precision_test = precisions_test[best_idx_test]
best_recall_test = recalls_test[best_idx_test]
best_f1_test = f1_scores_test[best_idx_test]

print(f"Test set - Threshold where precision equals recall: {precision_equals_recall_threshold_test}")
print(f"Test set - Best threshold for max F1 score: {best_threshold_test}")
print(f"Test set - Precision at best threshold: {best_precision_test}")
print(f"Test set - Recall at best threshold: {best_recall_test}")
print(f"Test set - F1 at best threshold: {best_f1_test}")

# Calculate precision, recall and thresholds for validation set
y_val_scores = np.ravel(model.predict_proba(X_val)[:, 1])
precisions_val, recalls_val, thresholds_val = precision_recall_curve(y_val, y_val_scores)

f1_scores_val = 2 * (precisions_val[:-1] * recalls_val[:-1]) / (precisions_val[:-1] + recalls_val[:-1])
f1_scores_val = np.nan_to_num(f1_scores_val, nan=0.0)
best_idx_val = f1_scores_val.argmax()
best_threshold_val = thresholds_val[best_idx_val]

precision_recall_diff_val = abs(precisions_val[:-1] - recalls_val[:-1])
min_diff_index_val = precision_recall_diff_val.argmin()
precision_equals_recall_threshold_val = thresholds_val[min_diff_index_val]

best_precision_val = precisions_val[best_idx_val]
best_recall_val = recalls_val[best_idx_val]
best_f1_val = f1_scores_val[best_idx_val]

print(f"Validation set - Threshold where precision equals recall: {precision_equals_recall_threshold_val}")
print(f"Validation set - Best threshold for max F1 score: {best_threshold_val}")
print(f"Validation set - Precision at best threshold: {best_precision_val}")
print(f"Validation set - Recall at best threshold: {best_recall_val}")
print(f"Validation set - F1 at best threshold: {best_f1_val}")

# Calculate test metrics at validation best threshold
y_pred_test_at_val_threshold = (y_scores >= best_threshold_val).astype(int)
precision_test_at_val = precision_score(y_test, y_pred_test_at_val_threshold)
recall_test_at_val = recall_score(y_test, y_pred_test_at_val_threshold)
f1_test_at_val = f1_score(y_test, y_pred_test_at_val_threshold)

print(f"Test set - Precision at validation best threshold: {precision_test_at_val}")
print(f"Test set - Recall at validation best threshold: {recall_test_at_val}")
print(f"Test set - F1 at validation best threshold: {f1_test_at_val}")

# Plot results for test set
plt.figure(figsize=(10, 6))
plt.plot(thresholds_test, precisions_test[:-1], label='Precision (Test)', color='blue')
plt.plot(thresholds_test, recalls_test[:-1], label='Recall (Test)', color='green')
plt.plot(thresholds_test, f1_scores_test, label='F1 Score (Test)', color='red')
plt.axvline(best_threshold_test, color='gray', linestyle='--', label=f'Test Best Threshold: {best_threshold_test:.2f}')
plt.axvline(precision_equals_recall_threshold_test, color='gray', linestyle=':', label=f'Test Precision=Recall Threshold: {precision_equals_recall_threshold_test:.2f}')

# Plot results for validation set
plt.plot(thresholds_val, precisions_val[:-1], label='Precision (Val)', color='cyan')
plt.plot(thresholds_val, recalls_val[:-1], label='Recall (Val)', color='lime')
plt.plot(thresholds_val, f1_scores_val, label='F1 Score (Val)', color='orange')
plt.axvline(best_threshold_val, color='black', linestyle='--', label=f'Val Best Threshold: {best_threshold_val:.2f}')
plt.axvline(precision_equals_recall_threshold_val, color='black', linestyle=':', label=f'Val Precision=Recall Threshold: {precision_equals_recall_threshold_val:.2f}')

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1 Score vs Classification Threshold (Validation and Test)")
plt.legend()
plt.show()

# COMMAND ----------

# Register model
register_model_by_run_id(model_meta, run_id)

# model_uri = f"runs:/{run_id}/model"
# registered_model_name = f"{catalog_name}.{schema_name}.{model_name}"
# mlflow.register_model(model_uri=model_uri, name=registered_model_name)

# COMMAND ----------


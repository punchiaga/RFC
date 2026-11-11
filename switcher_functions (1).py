from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, roc_auc_score, log_loss,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    recall_score, precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn import set_config
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders import TargetEncoder

from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import optuna
import hyperopt

from pyspark.sql import functions as F, Window as W, DataFrame, SparkSession
from pyspark.sql.functions import col, when, month
from pyspark.sql.types import TimestampType
from pyspark import StorageLevel

from typing import Dict, List, Tuple

from BupaML import BupaModel
from BupaML.ml.utils import register_model_by_run_id
from BupaML.ml.metrics import log_metrics
from BupaML.utils import generate_batch_process_dttm

from databricks.feature_engineering import FeatureLookup
from databricks.feature_engineering.client import FeatureEngineeringClient

import mlflow
import mlflow.sklearn
from mlflow.models import Model, infer_signature
from mlflow.pyfunc import PyFuncModel

import shap
from shap import KernelExplainer, summary_plot, TreeExplainer
from joblib import Parallel, delayed


def transform_data(input_data, excluded_columns=[]):
    
    output_data = (
        input_data
        # 1. Handle DAYS_FinSusp and NUM_DAYS_Since_Latest_FinSusp
        .withColumn("DAYS_FinSusp", when(col("NUM_DAYS_Since_Latest_FinSusp") > 0, col("DAYS_FinSusp")).otherwise(F.lit(0)))
        .withColumn("NUM_DAYS_Since_Latest_FinSusp", when(col("NUM_DAYS_Since_Latest_FinSusp") > 0, col("NUM_DAYS_Since_Latest_FinSusp")).otherwise(F.lit(0)))

        # 2. Map Fund_No to State
        .withColumn(
            "fund_no_state",
            when(col("Fund_No").cast("int").isin([1, 43]), "VIC")
            .when(col("Fund_No").cast("int").isin([2, 42]), "NSW")
            .when(col("Fund_No").cast("int").isin([4, 44]), "QLD")
            .when(col("Fund_No").cast("int").isin([5]), "SA")
            .when(col("Fund_No").cast("int").isin([46]), "WA")
            .when(col("Fund_No").cast("int").isin([47]), "TAS")
            .when(col("Fund_No").cast("int").isin([48]), "NT")
            .otherwise(F.lit(None))
        )

        # 3. Calculate months_to_rate_rise (April)
        .withColumn(
            "months_to_rate_rise",
            when(F.month("policy_start_dt") == 4, F.lit(0))  # April
            .when(F.month("policy_start_dt") < 4, 4 - F.month("policy_start_dt"))  # Before April
            .otherwise(12 - F.month("policy_start_dt") + 4)  # After April
        )

        # 4. Calculate months_to_end_of_fy (June)
        .withColumn(
            "months_to_end_of_fy",
            when(F.month("policy_start_dt") == 6, F.lit(0))  # June
            .when(F.month("policy_start_dt") < 6, 6 - F.month("policy_start_dt"))  # Before June
            .otherwise(12 - F.month("policy_start_dt") + 6)  # After June
        )

        # 5. Add tenure_le_1y column
        .withColumn("tenure_le_1y", when(col("TENURE_MONTHS") <= 12, F.lit(1)).otherwise(F.lit(0)))
        .filter(F.col("Dropper_Target") != 1).drop(*excluded_columns)
    )
    return output_data

def data_subset(df,column_subset = []):
    if len(column_subset)==0:
        return df
    else:
        column_subset = [c for c in column_subset if c in df.columns]
        return df.select(*column_subset)


def convert_object_to_numeric(df, categorical_features):
    for col in df.select_dtypes(include=['object']).columns:
        # Try converting to numeric, ignore errors to keep non-numeric columns unchanged
        if col not in (categorical_features):
            df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

class TransformedTargetClassifierWithProba(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier, transformer):
        self.classifier = classifier
        self.transformer = transformer

    def fit(self, X, y, **kwargs):
        # Fit the transformer on y and transform the training targets.
        self.transformer.fit(y)
        y_encoded = self.transformer.transform(y)

        # If an eval_set is provided, encode its targets using the fitted transformer.
        if 'eval_set' in kwargs:
            new_eval_set = []
            for (X_eval, y_eval) in kwargs['eval_set']:
                new_eval_set.append((X_eval, self.transformer.transform(y_eval)))
            kwargs['eval_set'] = new_eval_set

        # Fit the underlying classifier.
        self.classifier.fit(X, y_encoded, **kwargs)

        # Set the regressor_ attribute for compatibility with evaluation functions.
        self.regressor_ = self.classifier
        return self

    def predict(self, X):
        # Predict encoded labels and inverse transform them to original labels.
        y_encoded = self.classifier.predict(X)
        return self.transformer.inverse_transform(y_encoded)

    def predict_proba(self, X):
        # Delegate predict_proba directly to the underlying classifier.
        return self.classifier.predict_proba(X)


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, subset):
        self.subset = subset

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if len(self.subset)==0:
            return X.copy()
        else:
            self.subset = [c for c in self.subset if c in X.columns]
            return X[self.subset].copy()
        



class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.target_means_ = {}
        
    def fit(self, X, y):
        # Ensure y is a Pandas Series
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        self.target_means_ = {}
        for col in X.columns:
            mean_map = y.groupby(X[col]).mean()
            self.target_means_[col] = mean_map
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            mean_map = self.target_means_.get(col, pd.Series())
            X_encoded[col] = X_encoded[col].map(mean_map).fillna(mean_map.mean())
        return X_encoded.values  # Return numpy array for sklearn pipeline compatibility
    
class BooleanMapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map({'Y':1, 'N':0}).fillna(0).astype(int)
        return X

# Fit on train with y passed to TargetEncoder specifically
# Custom wrapper to pass y to only TargetEncoder in column transformer
class TargetEncoderWrapper(ColumnTransformer):
    def fit(self, X, y=None):
        # First fit all transformers to initialize self.transformers_
        super().fit(X, y)
        # Then fit TargetEncoder manually with its subset and y
        for name, transformer, cols in self.transformers_:
            if isinstance(transformer, Pipeline):
                for step_name, step_transformer in transformer.steps:
                    if isinstance(step_transformer, TargetEncoder):
                        step_transformer.fit(X[cols], y)
        return self

class NumpyToPandas(BaseEstimator, TransformerMixin):

    def __init__(self,
                 onehot_cols,
                 ordinal_cols,
                 numeric_ordinal_cols,
                 target_cols,
                 numeric_cols,
                 bool_cols):
        self.onehot_cols = onehot_cols
        self.ordinal_cols = ordinal_cols
        self.numeric_ordinal_cols = numeric_ordinal_cols
        self.target_cols = target_cols
        self.numeric_cols = numeric_cols
        self.bool_cols = bool_cols

    def fit(self, X, y=None):
        return self

    def get_onehot_feature_names(self, preprocessor):
        onehot_encoder = (
            preprocessor.named_transformers_['onehot']
                       .named_steps['onehot']
        )
        return list(onehot_encoder.get_feature_names_out(self.onehot_cols))

    def transform(self, X):
        preprocessor = self.preprocessor_ref

        onehot_names = self.get_onehot_feature_names(preprocessor)
        columns = (
            onehot_names +
            self.ordinal_cols +
            self.numeric_ordinal_cols+
            self.target_cols +
            self.numeric_cols +
            self.bool_cols
        )

        # Convert to dense if needed
        if hasattr(X, "todense"):
            X = X.todense()

        return pd.DataFrame(X, columns=columns)

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.columns = X.columns  # Save column names
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=self.columns, index=X.index)
    


def build_pipeline(onehot_cols,
                   ordinal_cols,
                   numeric_ordinal_cols,
                   target_cols,
                   numeric_cols,
                   bool_cols, 
                   model_type, model_params, label_encoder, feature_select):

    onehot_pipeline = Pipeline([
    ('imputer', DataFrameImputer(strategy='constant', fill_value="Unknown")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_ordinal_pipeline = Pipeline([
        ('imputer', DataFrameImputer(strategy='constant', fill_value = -1))
    ])

    target_encode_pipeline = Pipeline([
        ('imputer', DataFrameImputer(strategy='constant', fill_value="Unknown")),
        ('target_enc', TargetEncoder())
    ])

    numeric_pipeline = Pipeline([
        ('imputer', DataFrameImputer(strategy='constant', fill_value = 0)),
        ('scaler', StandardScaler())
    ])

    boolean_pipeline = Pipeline([
        ('bool_mapper', BooleanMapper(columns=bool_cols)),
        ('imputer', DataFrameImputer(strategy='constant', fill_value=0))
    ])

    preprocessor = TargetEncoderWrapper(transformers=[
        ('onehot', onehot_pipeline, onehot_cols),
        ('numeric_ordinal', numeric_ordinal_pipeline, numeric_ordinal_cols),
        ('target', target_encode_pipeline, target_cols),
        ('numeric', numeric_pipeline, numeric_cols),
        ('bool', boolean_pipeline, bool_cols)
    ])

    np_to_df = NumpyToPandas(
        onehot_cols=onehot_cols,
        ordinal_cols = ordinal_cols,
        numeric_ordinal_cols=numeric_ordinal_cols,
        target_cols=target_cols,
        numeric_cols=numeric_cols,
        bool_cols=bool_cols
    )

    column_subset_step = ColumnSelector(subset=feature_select)


    if model_type == 'lightgbm':
        classifier = TransformedTargetClassifierWithProba(
            classifier=LGBMClassifier(**model_params),
            transformer=label_encoder
            )
    elif model_type == 'hist_gbdt':
        classifier = TransformedTargetClassifierWithProba(
            classifier=HistGradientBoostingClassifier(**model_params),
            transformer=label_encoder
            )
    elif model_type == 'catboost':
        classifier = TransformedTargetClassifierWithProba(
            classifier=CatBoostClassifier(**model_params),
            transformer=label_encoder
            )
    elif model_type == 'xgboost':
        classifier = TransformedTargetClassifierWithProba(
            classifier=XGBClassifier(**model_params),
            transformer=label_encoder
            )
        


    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('np_to_df', np_to_df),
        ('column_subset', column_subset_step),
        ('classifier', classifier)
    ])

    # inject preprocessor reference into the np_to_df step
    np_to_df.preprocessor_ref = full_pipeline.named_steps['preprocessor']

    return full_pipeline

def plot_metrics(y_true, X_test, model, model_name="manual_notebook"):
    try:
        X_test_processed = model.named_steps["preprocessor"].transform(X_test)
        classifier = model.named_steps["classifier"]
        y_proba = classifier.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_proba >= 0.8).astype(int)  # threshold can be tuned
        # y_pred = classifier.predict(X_test_processed)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Test Set)')
        plt.legend(loc='lower right')
        plt.grid(True)
        roc_plot_path = "plots/roc_curve.png"
        os.makedirs(os.path.dirname(roc_plot_path), exist_ok=True)
        plt.savefig(roc_plot_path)
        mlflow.log_artifact(roc_plot_path)
        plt.show()

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f"PR AUC = {pr_auc:.2f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Test Set)')
        plt.legend(loc='lower left')
        plt.grid(True)
        pr_plot_path = "plots/pr_curve.png"
        os.makedirs(os.path.dirname(pr_plot_path), exist_ok=True)
        plt.savefig(pr_plot_path)
        mlflow.log_artifact(pr_plot_path)
        plt.show()

        # Lift Curve
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_y_true = np.array(y_true)[sorted_indices]
        cumulative_gains = np.cumsum(sorted_y_true) / np.sum(sorted_y_true)
        plt.figure(figsize=(8, 6))
        plt.plot(np.linspace(0, 1, len(cumulative_gains)), cumulative_gains, color='green', lw=2, label='Lift Curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Model')
        plt.xlabel('Proportion of Population')
        plt.ylabel('Proportion of Positives Captured')
        plt.title('Lift Curve (Test Set)')
        plt.legend(loc='lower right')
        plt.grid(True)
        lift_plot_path = "plots/lift_curve.png"
        os.makedirs(os.path.dirname(lift_plot_path), exist_ok=True)
        plt.savefig(lift_plot_path)
        mlflow.log_artifact(lift_plot_path)
        plt.show()

        # Confusion Matrix (normalized)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        print("\nNormalized Confusion Matrix:")
        print(cm)

        # Plot the normalized confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_ if hasattr(model, "classes_") else [0, 1])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=ax)
        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()

        # Save and log plot to MLflow
        conf_matrix_path = "plots/confusion_matrix.png"
        os.makedirs(os.path.dirname(conf_matrix_path), exist_ok=True)
        plt.savefig(conf_matrix_path)
        mlflow.log_artifact(conf_matrix_path)
        plt.show()

    except Exception as e:
        print(f"Error generating test set evaluation plots: {e}")
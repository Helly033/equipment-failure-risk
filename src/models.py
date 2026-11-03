"""Machine Learning Modeling Pipeline for Equipment Failure Prediction.

Trains baseline models (Logistic Regression, Random Forest) and cost-sensitive
gradient boosted decision trees (XGBoost, LightGBM) with class imbalance compensation.
"""

from __future__ import annotations

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def get_feature_lists() -> tuple[list[str], list[str]]:
    """Return statistically screened numeric features and categorical metadata."""
    # Screened by ANOVA (p < 0.05)
    numeric_features = [
        "power_to_thermal_ratio",
        "torque_nm",
        "overstrain_index",
        "mechanical_power_w",
        "torque_rolling_std",
        "torque_rate_of_change",
        "temp_difference_k",
        "tool_wear_min",
        "speed_rolling_std",
        "air_temperature_k",
        "temp_diff_rolling_mean",
        "speed_rate_of_change",
        "process_temperature_k",
        "rotational_speed_rpm",
    ]
    categorical_features = ["product_type"]
    return numeric_features, categorical_features


def create_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build preprocessing pipeline fitted strictly on training data."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def train_models() -> dict[str, object]:
    """Train all benchmark and champion models, saving artifacts to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train_engineered.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "Engineered train data missing! Run src.features first."
        )

    train_df = pd.read_csv(train_path)
    numeric_cols, categorical_cols = get_feature_lists()
    all_feature_cols = numeric_cols + categorical_cols

    X_train = train_df[all_feature_cols]
    y_train = train_df["machine_failure"]

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_ratio = neg_count / pos_count
    logger.info(
        f"Training set imbalance: {neg_count:,} normal vs {pos_count:,} failures "
        f"(scale_pos_weight = {imbalance_ratio:.2f})"
    )

    # 1. Build Preprocessor
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    logger.info("Saved preprocessor to models/preprocessor.joblib")

    X_train_transformed = preprocessor.transform(X_train)

    # 2. Train Model 1: Logistic Regression (Linear Baseline)
    logger.info("Training Baseline 1: Logistic Regression (Balanced)...")
    log_reg = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    log_reg.fit(X_train_transformed, y_train)
    joblib.dump(log_reg, MODELS_DIR / "logistic_regression.joblib")

    # 3. Train Model 2: Random Forest (Bagging Baseline)
    logger.info("Training Baseline 2: Random Forest (Balanced Subsample)...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_transformed, y_train)
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    # 4. Train Model 3: XGBoost (Champion Gradient Booster)
    logger.info("Training Champion Model: XGBoost (Cost-Sensitive Weighted)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=imbalance_ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_model.fit(X_train_transformed, y_train)
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.joblib")

    # 5. Train Model 4: LightGBM
    logger.info("Training Alternative Model: LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_model.fit(X_train_transformed, y_train)
    joblib.dump(lgb_model, MODELS_DIR / "lightgbm_model.joblib")

    # Save feature metadata for inference and SHAP
    encoded_cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    )
    final_feature_names = numeric_cols + encoded_cat_names
    joblib.dump(final_feature_names, MODELS_DIR / "feature_names.joblib")

    logger.info(
        f"All 4 models successfully trained and serialized to: {MODELS_DIR}"
    )

    return {
        "preprocessor": preprocessor,
        "logistic_regression": log_reg,
        "random_forest": rf,
        "xgboost": xgb_model,
        "lightgbm": lgb_model,
        "feature_names": final_feature_names,
    }


if __name__ == "__main__":
    logger.info("--- Starting Model Training Pipeline ---")
    train_models()
    logger.info("--- Model Training Pipeline Completed Successfully ---")

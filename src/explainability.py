"""Explainable AI (SHAP) Module for Equipment Failure Diagnostics.

Computes TreeSHAP feature attributions at both the global fleet level
and local per-machine alert level to diagnose the root physical causes of impending breakdown.
"""

from __future__ import annotations

import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.features import engineer_features
from src.models import get_feature_lists

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
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def generate_shap_explanations() -> tuple[shap.TreeExplainer, np.ndarray]:
    """Calculate global and local SHAP attributions using TreeSHAP on test data."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test_path = PROCESSED_DATA_DIR / "test_engineered.csv"
    if not test_path.exists():
        raise FileNotFoundError("Engineered test data missing! Run src.features first.")

    test_df = pd.read_csv(test_path)
    numeric_cols, categorical_cols = get_feature_lists()
    all_feature_cols = numeric_cols + categorical_cols

    X_test = test_df[all_feature_cols]
    y_test = test_df["machine_failure"].values

    # Load artifacts
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    X_test_trans = preprocessor.transform(X_test)

    # Initialize TreeSHAP
    logger.info("Initializing TreeSHAP Explainer for XGBoost...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_trans)

    # Assign clean feature names to SHAP Explanation
    shap_values.feature_names = feature_names

    # 1. Global Fleet Summary Plot
    logger.info("Generating Global Fleet SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test_trans,
        feature_names=feature_names,
        show=False,
        max_display=12,
    )
    plt.title("Fleet-Wide Root Cause Predictors (TreeSHAP Summary)", fontsize=13, pad=12)
    plt.tight_layout()
    summary_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot to: {summary_path}")

    # 2. Local Waterfall Plot for a Critical Breakdown Sample
    # Find a true breakdown sample with high predicted probability
    probs = model.predict_proba(X_test_trans)[:, 1]
    failing_indices = np.where((y_test == 1) & (probs >= 0.70))[0]
    sample_idx = failing_indices[0] if len(failing_indices) > 0 else 0

    logger.info(f"Generating Local SHAP Waterfall Plot for Machine Sample #{sample_idx}...")
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[sample_idx], show=False, max_display=10)
    plt.title(
        f"Local Root Cause Attribution: Machine #{sample_idx} (Risk = {probs[sample_idx]*100:.1f}%)",
        fontsize=12,
        pad=10,
    )
    plt.tight_layout()
    waterfall_path = FIGURES_DIR / "shap_waterfall_sample.png"
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved local waterfall plot to: {waterfall_path}")

    # Save explainer artifact for live serving
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
    logger.info(f"Serialized SHAP explainer to: {MODELS_DIR / 'shap_explainer.joblib'}")

    return explainer, shap_values


def diagnose_telemetry_reading(sensor_dict: dict) -> dict:
    """Diagnose a single live telemetry reading, outputting probability and top root causes."""
    # Convert single reading to DataFrame and engineer features
    raw_df = pd.DataFrame([sensor_dict])
    eng_df = engineer_features(raw_df)

    numeric_cols, categorical_cols = get_feature_lists()
    all_feature_cols = numeric_cols + categorical_cols
    X_input = eng_df[all_feature_cols]

    # Load artifacts
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    X_trans = preprocessor.transform(X_input)
    prob_failure = float(model.predict_proba(X_trans)[0, 1])

    # Compute SHAP values for this instance
    shap_vals = explainer(X_trans)
    vals = shap_vals.values[0]

    # Rank features by contribution towards failure (positive SHAP push)
    feature_impacts = sorted(
        zip(feature_names, vals, X_trans[0]),
        key=lambda x: x[1],
        reverse=True,
    )

    top_drivers = [
        {
            "sensor": feat,
            "shap_attribution": round(float(impact), 4),
            "normalized_value": round(float(val), 3),
        }
        for feat, impact, val in feature_impacts[:4]
        if impact > 0
    ]

    # Assign Operational Risk Tier
    if prob_failure < 0.35:
        risk_tier = "GREEN"
        status = "Healthy / Normal Operation"
        prescription = "No action required. Telemetry within nominal limits."
    elif prob_failure < 0.70:
        risk_tier = "AMBER"
        status = "Warning / Degradation Detected"
        prescription = "Schedule preventive tool/lubricant inspection during upcoming shift change."
    else:
        risk_tier = "RED"
        status = "CRITICAL / Breakdown Imminent"
        prescription = "Emergency interlock: Throttle spindle RPM and replace cutting insert immediately."

    return {
        "failure_probability": round(prob_failure, 4),
        "risk_tier": risk_tier,
        "status": status,
        "prescription": prescription,
        "top_drivers": top_drivers,
    }


if __name__ == "__main__":
    logger.info("--- Starting SHAP Explainability Pipeline ---")
    generate_shap_explanations()

    # Test single diagnosis
    sample_telemetry = {
        "udi": 1,
        "product_id": "L47181",
        "product_type": "L",
        "air_temperature_k": 301.0,
        "process_temperature_k": 310.5,
        "rotational_speed_rpm": 1400.0,
        "torque_nm": 65.0,
        "tool_wear_min": 210.0,
        "machine_failure": 0,
        "twf": 0,
        "hdf": 0,
        "pwf": 0,
        "osf": 0,
        "rnf": 0,
    }
    logger.info("Testing sample diagnosis on high-stress telemetry...")
    diag = diagnose_telemetry_reading(sample_telemetry)
    logger.info(f"Diagnosis Result: Risk = {diag['failure_probability']*100:.1f}%, Tier = {diag['risk_tier']}")
    logger.info(f"Prescription: {diag['prescription']}")
    logger.info(f"Top Root Causes: {diag['top_drivers']}")
    logger.info("--- SHAP Explainability Pipeline Completed Successfully ---")

"""Physics-Informed Signal Processing and Feature Engineering Module.

Synthesizes thermodynamic, kinematic, and dynamics features from raw sensor feeds
to capture early mechanical degradation before catastrophic breakdown.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def calculate_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inject domain-specific thermodynamic and mechanical stress equations."""
    df_feat = df.copy()

    # 1. Mechanical Power Draw [Watts]
    # Power = Torque [Nm] * Angular Velocity [rad/s]
    # Angular Velocity = RPM * 2*pi / 60
    angular_velocity = df_feat["rotational_speed_rpm"] * (2 * np.pi / 60.0)
    df_feat["mechanical_power_w"] = df_feat["torque_nm"] * angular_velocity

    # 2. Thermal Gradient [Kelvin]
    # Delta T = Process Interface Temp - Ambient Air Temp
    # Critical boundary: Delta T < 8.6 K combined with low RPM (<1380) signals cooling breakdown (HDF)
    df_feat["temp_difference_k"] = (
        df_feat["process_temperature_k"] - df_feat["air_temperature_k"]
    )

    # 3. Overstrain Interaction Index
    # Overstrain = Tool Wear [min] * Torque [Nm]
    # Heavy torque on a worn cutter leads to catastrophic shear fracture (OSF)
    df_feat["overstrain_index"] = df_feat["tool_wear_min"] * df_feat["torque_nm"]

    # 4. Thermal Dissipation Ratio
    # Quantifies heat dissipation capability per revolution
    df_feat["thermal_dissipation_ratio"] = df_feat["temp_difference_k"] / (
        df_feat["rotational_speed_rpm"] + 1e-5
    )

    # 5. Power-to-Thermal Ratio
    # Measures how much electrical power is converted to retained interface heat
    df_feat["power_to_thermal_ratio"] = df_feat["mechanical_power_w"] / (
        df_feat["temp_difference_k"] + 1e-5
    )

    # 6. Torque-to-Wear Strain Ratio
    # High torque on high wear increases strain exponentially
    df_feat["torque_wear_ratio"] = df_feat["torque_nm"] / (
        df_feat["tool_wear_min"] + 1.0
    )

    return df_feat


def calculate_dynamics_features(
    df: pd.DataFrame, window_size: int = 5
) -> pd.DataFrame:
    """Calculate rolling window volatility (std dev) and rate-of-change dynamics."""
    df_feat = df.copy()

    # 1. Rolling Volatility (Proxy for mechanical vibration and spindle chatter)
    df_feat["torque_rolling_std"] = (
        df_feat["torque_nm"]
        .rolling(window=window_size, min_periods=1)
        .std()
        .fillna(0.0)
    )
    df_feat["speed_rolling_std"] = (
        df_feat["rotational_speed_rpm"]
        .rolling(window=window_size, min_periods=1)
        .std()
        .fillna(0.0)
    )
    df_feat["temp_diff_rolling_mean"] = (
        df_feat["temp_difference_k"]
        .rolling(window=window_size, min_periods=1)
        .mean()
        .fillna(df_feat["temp_difference_k"])
    )

    # 2. First-order Rate of Change (Spike detection)
    df_feat["torque_rate_of_change"] = df_feat["torque_nm"].diff().fillna(0.0)
    df_feat["speed_rate_of_change"] = (
        df_feat["rotational_speed_rpm"].diff().fillna(0.0)
    )

    return df_feat


def engineer_features(
    df: pd.DataFrame, window_size: int = 5
) -> pd.DataFrame:
    """Full feature engineering pipeline combining physics and time-series dynamics."""
    logger.info("Computing physics-informed features (Mechanical Power, Delta T, Overstrain)...")
    df_physics = calculate_physics_features(df)

    logger.info(f"Computing rolling volatility and rate-of-change (window={window_size})...")
    df_engineered = calculate_dynamics_features(df_physics, window_size=window_size)

    new_features = [c for c in df_engineered.columns if c not in df.columns]
    logger.info(f"Successfully engineered {len(new_features)} domain features: {new_features}")
    return df_engineered


def run_feature_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute feature engineering on raw train and test splits."""
    train_raw_path = PROCESSED_DATA_DIR / "train_raw.csv"
    test_raw_path = PROCESSED_DATA_DIR / "test_raw.csv"

    if not train_raw_path.exists() or not test_raw_path.exists():
        raise FileNotFoundError(
            "Raw split files not found in data/processed/. Run src.data_loader first!"
        )

    train_raw = pd.read_csv(train_raw_path)
    test_raw = pd.read_csv(test_raw_path)

    logger.info(f"Transforming train dataset ({len(train_raw):,} records)...")
    train_eng = engineer_features(train_raw)

    logger.info(f"Transforming test dataset ({len(test_raw):,} records)...")
    test_eng = engineer_features(test_raw)

    train_eng_path = PROCESSED_DATA_DIR / "train_engineered.csv"
    test_eng_path = PROCESSED_DATA_DIR / "test_engineered.csv"

    train_eng.to_csv(train_eng_path, index=False)
    test_eng.to_csv(test_eng_path, index=False)

    logger.info(f"Saved engineered datasets to:\n  {train_eng_path}\n  {test_eng_path}")
    return train_eng, test_eng


if __name__ == "__main__":
    logger.info("--- Starting Feature Engineering Pipeline ---")
    run_feature_pipeline()
    logger.info("--- Feature Engineering Pipeline Completed Successfully ---")

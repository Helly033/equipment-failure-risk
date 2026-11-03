"""Telemetry Data Ingestion and Validation Module.

Fetches the benchmark AI4I 2020 Predictive Maintenance Dataset,
validates physical sensor limits, cleans schema, and generates
leakage-free stratified train/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Repository Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"

# Primary & Fallback Ingestion Endpoints
DATA_URLS = [
    "https://raw.githubusercontent.com/marshmallow77/ai4i2020-predictive-maintenance-dataset/master/ai4i2020.csv",
    "https://raw.githubusercontent.com/yohanunsw/AI4I-2020-Predictive-Maintenance-Dataset/main/ai4i2020.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
]

# Standardized Column Schema
COLUMN_MAPPING = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "product_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

# Physical boundary constraints for data validation
PHYSICAL_BOUNDS = {
    "air_temperature_k": (250.0, 350.0),      # Ambient factory temperature
    "process_temperature_k": (280.0, 380.0),  # Tool interface temperature
    "rotational_speed_rpm": (500.0, 5000.0),  # CNC Spindle RPM
    "torque_nm": (0.0, 150.0),                # Rotational mechanical torque
    "tool_wear_min": (0.0, 400.0),            # Cumulative cutter wear time
}


def download_dataset(target_path: Path = RAW_DATA_FILE) -> Path:
    """Download the raw AI4I 2020 dataset if not already present."""
    if target_path.exists():
        logger.info(f"Dataset already exists at: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching raw telemetry dataset from remote mirror...")

    for url in DATA_URLS:
        try:
            logger.info(f"Attempting download from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f:
                f.write(resp.read())
            logger.info(f"Successfully downloaded {target_path.stat().st_size:,} bytes.")
            return target_path
        except Exception as err:
            logger.warning(f"Failed to download from {url}: {err}")

    raise RuntimeError("All remote mirrors failed to download AI4I 2020 dataset.")


def load_and_clean_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw dataset, standardize schema, and validate physical ranges."""
    if not file_path.exists():
        download_dataset(file_path)

    df = pd.read_csv(file_path)
    logger.info(f"Loaded raw records: {len(df):,} rows, {len(df.columns)} columns.")

    # Rename columns to snake_case standard
    df = df.rename(columns=COLUMN_MAPPING)

    # Missing value validation
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Missing values detected:\n{null_counts[null_counts > 0]}")
        df = df.dropna()
    else:
        logger.info("Data Hygiene Check: 0 missing or NaN values detected.")

    # Validate physical sensor limits
    for col, (lower, upper) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if len(outliers) > 0:
                logger.warning(
                    f"Sensor '{col}' has {len(outliers)} readings outside [{lower}, {upper}]."
                )

    # Class imbalance audit
    failure_count = df["machine_failure"].sum()
    failure_pct = (failure_count / len(df)) * 100
    logger.info(
        f"Target Distribution: Normal={len(df) - failure_count:,} ({100 - failure_pct:.2f}%), "
        f"Breakdown={failure_count:,} ({failure_pct:.2f}%)"
    )

    return df


def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/test split to preserve failure ratio."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratify by machine_failure to ensure test set contains representative breakdowns
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["machine_failure"],
    )

    train_path = output_dir / "train_raw.csv"
    test_path = output_dir / "test_raw.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        f"Train set: {len(train_df):,} rows (Breakdowns: {train_df['machine_failure'].sum():,})"
    )
    logger.info(
        f"Test set:  {len(test_df):,} rows (Breakdowns: {test_df['machine_failure'].sum():,})"
    )
    logger.info(f"Saved splits to: {output_dir}")

    return train_df, test_df


if __name__ == "__main__":
    logger.info("--- Starting Telemetry Data Loader Pipeline ---")
    data_path = download_dataset()
    df_clean = load_and_clean_data(data_path)
    train_split, test_split = split_and_save_data(df_clean)
    logger.info("--- Data Loader Pipeline Completed Successfully ---")

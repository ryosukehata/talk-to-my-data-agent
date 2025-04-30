"""
Module for loading, validating, and normalizing the raw log data.

Reads the raw CSV log file, performs initial validation, parses timestamps,
normalizes data types, and consolidates user identity.
Outputs a cleaned pandas DataFrame.
"""

import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_normalize_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Loads raw log data from a CSV file, validates essential columns,
    parses timestamps, normalizes data types, and consolidates user identity.

    Args:
        raw_data_path: Path to the raw input CSV file.

    Returns:
        A pandas DataFrame containing the normalized log data ('norm_logs').

    Raises:
        FileNotFoundError: If the raw_data_path does not exist.
        ValueError: If essential columns contain unexpected null values.
    """
    logging.info(f"Starting data loading and normalization from: {raw_data_path}")

    if not raw_data_path.exists():
        logging.error(f"Raw data file not found at {raw_data_path}")
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

    # --- 3.1. Data Ingestion & Raw Storage ---
    try:
        df = pd.read_csv(raw_data_path)
        logging.info(f"Successfully loaded {len(df)} records from raw data.")
    except Exception as e:
        logging.error(f"Failed to load raw data CSV: {e}")
        raise

    # --- Initial Data Validation ---
    key_columns = ["association_id", "timestamp", "chat_id", "visitTimestamp"]
    null_counts = df[key_columns].isnull().sum()
    if null_counts.sum() > 0:
        logging.warning(
            f"Found null values in key columns:\n{null_counts[null_counts > 0]}"
        )
        # Depending on strictness, could raise ValueError here.
        # For now, we log a warning and proceed, but downstream steps might fail.
        # Example: raise ValueError(f"Null values found in key columns: {null_counts[null_counts > 0]}")

    # Drop rows where essential timestamps are missing for session/chat analysis
    initial_len = len(df)
    df.dropna(subset=["timestamp", "visitTimestamp"], inplace=True)
    if len(df) < initial_len:
        logging.warning(
            f"Dropped {initial_len - len(df)} rows due to missing timestamps."
        )

    # --- 3.2. Parsing and Normalization ---

    # --- Timestamp Parsing ---
    # Assume timestamps are already in local timezone (GMT+9) as per docs
    try:
        df["parsed_timestamp"] = pd.to_datetime(df["timestamp"])
        df["parsed_visitTimestamp"] = pd.to_datetime(df["visitTimestamp"])
        df["date"] = df["parsed_timestamp"].dt.date
        logging.info(
            "Successfully parsed 'timestamp' and 'visitTimestamp'. Created 'date'."
        )
    except Exception as e:
        logging.error(f"Error parsing timestamps: {e}")
        # Consider raising an error if parsing fails critically
        raise ValueError("Failed to parse timestamp columns.") from e

    # --- Data Type Normalization ---
    # Booleans: Ensure True/False. Fill NA with default True as per doc.
    bool_cols = ["enable_chart_generation", "enable_business_insights"]
    for col in bool_cols:
        if col in df.columns:
            # Convert potential string representations ('True', 'False') and handle NAs
            df[col] = df[col].fillna(True).astype(str).str.lower() == "true"
            logging.debug(f"Normalized boolean column: {col}")
        else:
            logging.warning(
                f"Boolean column '{col}' not found, creating with default True."
            )
            df[col] = True  # Default True if column missing

    # Strings: Trim whitespace
    str_cols = [
        "user_email",
        "data_source",
        "query_type",
        "user_msg",
        "error_message",
        "username",
        "domain",
        "userType",
    ]
    for col in str_cols:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            logging.debug(f"Normalized string column: {col}")

    # --- User Identity Consolidation ---
    # Use 'user_email' as the primary identifier. 'username' is a duplicate.
    df["user_id"] = df[
        "user_email"
    ]  # Keep user_id consistent with requirement doc reference
    # Check consistency if needed: assert (df['user_email'] == df['username']).all()

    # Set userType: "creator" (logged in) or "guest."
    # Based on docs: "userId - Account id (if missing, the user is in guest mode)."
    # Ensure userId is treated as string for comparison, handle NaN/None
    df["userId_str"] = df["userId"].fillna("").astype(str).str.strip()
    df["userType_inferred"] = df.apply(
        lambda row: "creator" if row["userId_str"] != "" else "guest", axis=1
    )

    # Optional: Compare inferred type with existing 'userType' if needed for validation
    # mismatches = df[df['userType'] != df['userType_inferred']]
    # if not mismatches.empty:
    #     logging.warning(f"Found {len(mismatches)} mismatches between provided 'userType' and inferred 'userType'. Using inferred type.")
    df["userType"] = df["userType_inferred"]  # Overwrite with inferred type

    # Drop intermediate columns
    df.drop(columns=["userId_str", "userType_inferred"], inplace=True)

    logging.info("User identity consolidated. 'user_id' created, 'userType' updated.")
    logging.info(f"Normalization complete. Output DataFrame shape: {df.shape}")

    # Rename DataFrame conceptually to 'norm_logs' for clarity in pipeline
    norm_logs = df
    return norm_logs


if __name__ == "__main__":
    # Example Usage (can be run directly for testing)
    INPUT_DIR = Path("notebooks/input")
    OUTPUT_DIR = Path(
        "notebooks/output/intermediate"
    )  # Define an intermediate output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RAW_FILE = INPUT_DIR / "merged_dataset.csv"
    NORMALIZED_FILE = OUTPUT_DIR / "norm_logs.parquet"  # Save as parquet for efficiency

    try:
        normalized_df = load_and_normalize_data(RAW_FILE)
        logging.info(
            f"Successfully created normalized DataFrame with columns: {normalized_df.columns.tolist()}"
        )

        # Save the intermediate normalized data (optional, but good practice)
        normalized_df.to_parquet(NORMALIZED_FILE, index=False)
        logging.info(f"Saved normalized data to {NORMALIZED_FILE}")

        # Display sample data
        logging.info("Sample of normalized data:")
        print(normalized_df.head().to_markdown(index=False))
        logging.info("\nData types of normalized data:")
        print(normalized_df.info())

    except FileNotFoundError:
        logging.error(
            "Input file not found. Please ensure 'merged_dataset.csv' exists in 'notebooks/input'."
        )
    except Exception as e:
        logging.error(
            f"An error occurred during the load and normalize process: {e}",
            exc_info=True,
        )

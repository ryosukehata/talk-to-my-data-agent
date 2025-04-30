"""
Module for reconstructing chat flows and calculating retry counts.

Groups normalized log data by chat_id and chat_seq, calculates metrics
like event counts, retry counts for code/chart generation, and sequences
of query types within each chat interaction step.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import the loading function
try:
    from load_normalize import load_and_normalize_data
except ImportError:
    logging.error(
        "Could not import load_and_normalize_data. Ensure load_normalize.py is accessible."
    )

    def load_and_normalize_data(path):  # type: ignore
        raise ImportError("load_and_normalize_data unavailable.")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def reconstruct_chat_flows(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs chat flows from normalized log data.

    Args:
        norm_logs_df: DataFrame containing normalized log data including
                      'chat_id', 'chat_seq', 'user_id', 'parsed_timestamp',
                      'query_type', and 'error_message'.

    Returns:
        A pandas DataFrame containing chat flow information ('chat_flows'):
        chat_id, chat_seq, chat_no, user_id, number_of_events,
        retry_count_code_gen, retry_count_chart_gen, first_timestamp,
        last_timestamp, query_types_sequence.
    """
    logging.info(
        f"Starting chat flow reconstruction for {len(norm_logs_df)} log entries."
    )

    if norm_logs_df.empty:
        logging.warning(
            "Input DataFrame is empty. Returning empty chat_flows DataFrame."
        )
        return pd.DataFrame(
            columns=[
                "chat_id",
                "chat_seq",
                "chat_no",
                "user_id",
                "number_of_events",
                "retry_count_code_gen",
                "retry_count_chart_gen",
                "first_timestamp",
                "last_timestamp",
                "query_types_sequence",
            ]
        )

    # Ensure required columns are present and chat_seq is numeric
    required_cols = [
        "chat_id",
        "chat_seq",
        "user_id",
        "parsed_timestamp",
        "query_type",
        "error_message",
    ]
    if not all(col in norm_logs_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in norm_logs_df.columns]
        logging.error(
            f"Missing required columns for chat flow reconstruction: {missing}"
        )
        raise ValueError(f"Missing required columns: {missing}")

    # Convert chat_seq to numeric, coercing errors to NaN, then drop NaN rows
    df = norm_logs_df.copy()
    # Convert chat_seq to numeric, coercing errors to NaN
    df["chat_seq"] = pd.to_numeric(df["chat_seq"], errors="coerce")
    initial_len = len(df)
    # Drop rows where chat_id or chat_seq is NaN
    df.dropna(subset=["chat_id", "chat_seq"], inplace=True)
    # Cast chat_seq to integer AFTER dropping NaNs
    if not df.empty:  # Avoid error if df becomes empty after dropna
        df["chat_seq"] = df["chat_seq"].astype(int)

    if len(df) < initial_len:
        logging.warning(
            f"Dropped {initial_len - len(df)} rows due to non-numeric or missing chat_id/chat_seq."
        )

    if df.empty:
        logging.warning(
            "DataFrame became empty after dropping rows with invalid chat_id/chat_seq. Cannot proceed."
        )
        return pd.DataFrame(
            columns=[
                "chat_id",
                "chat_seq",
                "chat_no",
                "user_id",
                "number_of_events",
                "retry_count_code_gen",
                "retry_count_chart_gen",
                "first_timestamp",
                "last_timestamp",
                "query_types_sequence",
            ]
        )

    # Sort data for grouping and sequence generation
    df_sorted = df.sort_values(by=["chat_id", "chat_seq", "parsed_timestamp"])

    # Define aggregation functions
    def count_code_gen_retries(x):
        # Counts occurrences of '03_generate_code_database' or '03_generate_code_file'
        # Retry means count > 1, so subtract 1 and ensure >= 0
        count = x.str.startswith("03_generate_code").sum()
        return max(0, count - 1)

    def count_chart_gen_retries(x):
        # Counts occurrences of '04_generate_run_charts_python_code'
        # Retry means count > 1, so subtract 1 and ensure >= 0
        count = (x == "04_generate_run_charts_python_code").sum()
        return max(0, count - 1)

    def get_query_sequence(x):
        # Concatenates query types in chronological order
        return " -> ".join(x.astype(str))

    # Group by chat_id and chat_seq and aggregate
    logging.info("Grouping by chat_id and chat_seq to aggregate chat flow metrics...")
    chat_flows = (
        df_sorted.groupby(["chat_id", "chat_seq"])
        .agg(
            user_id=("user_id", "first"),
            number_of_events=("association_id", "size"),  # Count events in the group
            first_timestamp=("parsed_timestamp", "min"),
            last_timestamp=("parsed_timestamp", "max"),
            # Apply custom aggregations on the 'query_type' column
            retry_count_code_gen=("query_type", count_code_gen_retries),
            retry_count_chart_gen=("query_type", count_chart_gen_retries),
            query_types_sequence=("query_type", get_query_sequence),
        )
        .reset_index()
    )

    # Calculate chat_no (ensure chat_seq is int first if needed)
    # Requirement: chat_no = chat_seq / 2
    chat_flows["chat_no"] = (chat_flows["chat_seq"] / 2).astype(int)

    # Reorder columns as specified
    chat_flows = chat_flows[
        [
            "chat_id",
            "chat_seq",
            "chat_no",
            "user_id",
            "number_of_events",
            "retry_count_code_gen",
            "retry_count_chart_gen",
            "first_timestamp",
            "last_timestamp",
            "query_types_sequence",
        ]
    ]

    logging.info(
        f"Chat flow reconstruction complete. Generated {len(chat_flows)} chat flow records."
    )
    return chat_flows


if __name__ == "__main__":
    INPUT_DIR = Path("notebooks/input")
    INTERMEDIATE_DIR = Path("notebooks/output/intermediate")
    OUTPUT_DIR = Path("notebooks/output")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RAW_FILE = INPUT_DIR / "merged_dataset.csv"
    NORMALIZED_FILE = INTERMEDIATE_DIR / "norm_logs.parquet"
    CHAT_FLOWS_FILE = OUTPUT_DIR / "chat_flows.csv"

    try:
        # Load normalized data
        if NORMALIZED_FILE.exists():
            logging.info(f"Loading normalized data from {NORMALIZED_FILE}")
            norm_data = pd.read_parquet(NORMALIZED_FILE)
        else:
            logging.warning(
                f"Normalized data file {NORMALIZED_FILE} not found. "
                f"Loading and normalizing from raw file {RAW_FILE} instead."
            )
            norm_data = load_and_normalize_data(RAW_FILE)
            norm_data.to_parquet(NORMALIZED_FILE, index=False)
            logging.info(f"Saved newly normalized data to {NORMALIZED_FILE}")

        # Reconstruct chat flows
        chat_flows_df = reconstruct_chat_flows(norm_data)
        logging.info(f"Successfully reconstructed {len(chat_flows_df)} chat flows.")

        # Save chat flows data
        chat_flows_df.to_csv(CHAT_FLOWS_FILE, index=False)
        logging.info(f"Saved chat flows data to {CHAT_FLOWS_FILE}")

        # Display sample data
        logging.info("Sample of chat flows data:")
        print(chat_flows_df.head().to_markdown(index=False))
        logging.info("\nData types of chat flows data:")
        print(chat_flows_df.info())

    except FileNotFoundError:
        logging.error(
            f"Input file {RAW_FILE} not found and intermediate file {NORMALIZED_FILE} also missing."
        )
    except ImportError as e:
        logging.error(f"Import error: {e}. Cannot proceed.")
    except Exception as e:
        logging.error(
            f"An error occurred during the chat flow reconstruction process: {e}",
            exc_info=True,
        )

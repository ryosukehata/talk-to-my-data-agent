"""
Module for inferring user sessions from normalized log data.

Groups log events by user, identifies session boundaries based on visit timestamps
and time gaps between events, and aggregates session-level information.
"""

import logging
from pathlib import Path

import pandas as pd

# Import the loading function from the previous step
# Assumes load_normalize.py is in the same directory or Python path is set up
try:
    from load_normalize import load_and_normalize_data
except ImportError:
    logging.error(
        "Could not import load_and_normalize_data. Ensure load_normalize.py is accessible."
    )

    # Define a placeholder if import fails, though execution will likely fail later
    def load_and_normalize_data(path):  # type: ignore
        raise ImportError("load_and_normalize_data unavailable.")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SESSION_TIMEOUT_MINUTES = 30


def infer_sessions(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infers user sessions from normalized log data.

    Args:
        norm_logs_df: DataFrame containing normalized log data with 'user_id',
                      'parsed_timestamp', and 'parsed_visitTimestamp'.

    Returns:
        A pandas DataFrame containing session-level information ('sessions'):
        session_id, user_id, visitTimestamp (unique per session),
        session_start, session_end, session_duration.
    """
    logging.info(f"Starting session inference for {len(norm_logs_df)} log entries.")

    if norm_logs_df.empty:
        logging.warning("Input DataFrame is empty. Returning empty sessions DataFrame.")
        return pd.DataFrame(
            columns=[
                "session_id",
                "user_id",
                "visitTimestamp",
                "session_start",
                "session_end",
                "session_duration",
            ]
        )

    # Ensure required columns are present
    required_cols = ["user_id", "parsed_timestamp", "parsed_visitTimestamp"]
    if not all(col in norm_logs_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in norm_logs_df.columns]
        logging.error(f"Missing required columns for session inference: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Sort data for session calculation
    df_sorted = norm_logs_df.sort_values(by=["user_id", "parsed_timestamp"]).copy()

    # Calculate time difference between consecutive events *within each user group*
    df_sorted["time_diff"] = df_sorted.groupby("user_id")["parsed_timestamp"].diff()

    # Identify session starts:
    # A new session starts if:
    # 1. It's the first event for the user (time_diff is NaT).
    # 2. The visitTimestamp changes compared to the previous event for the user.
    # 3. The time gap exceeds the timeout threshold.
    df_sorted["prev_visitTimestamp"] = df_sorted.groupby("user_id")[
        "parsed_visitTimestamp"
    ].shift(1)

    is_new_session = (
        df_sorted["time_diff"].isna()
        | (df_sorted["parsed_visitTimestamp"] != df_sorted["prev_visitTimestamp"])
        | (df_sorted["time_diff"] > pd.Timedelta(minutes=SESSION_TIMEOUT_MINUTES))
    )

    # Assign a unique session group number for each user
    df_sorted["session_group"] = is_new_session.cumsum()

    # Generate session_id: user_id + session start timestamp (using first event's timestamp)
    # First, get the start timestamp for each session group
    session_starts = df_sorted.groupby("session_group")["parsed_timestamp"].transform(
        "min"
    )

    # Format session_id as required: user_id + session_start timestamp string
    # Ensure user_id is string and handle potential formatting issues
    df_sorted["session_id"] = (
        df_sorted["user_id"].astype(str)
        + "_"
        + session_starts.dt.strftime("%Y%m%d%H%M%S")
    )

    logging.info(
        f"Identified {df_sorted['session_group'].nunique()} potential session groups."
    )

    # Aggregate session information
    sessions = (
        df_sorted.groupby("session_id")
        .agg(
            user_id=("user_id", "first"),
            # Assuming visitTimestamp is constant within a detected session
            # based on the logic (new session starts on visitTimestamp change)
            visitTimestamp=("parsed_visitTimestamp", "first"),
            session_start=("parsed_timestamp", "min"),
            session_end=("parsed_timestamp", "max"),
        )
        .reset_index()
    )

    # Calculate session duration
    sessions["session_duration"] = sessions["session_end"] - sessions["session_start"]

    # Reorder columns as specified
    sessions = sessions[
        [
            "session_id",
            "user_id",
            "visitTimestamp",
            "session_start",
            "session_end",
            "session_duration",
        ]
    ]

    logging.info(f"Session inference complete. Generated {len(sessions)} sessions.")
    return sessions


if __name__ == "__main__":
    INPUT_DIR = Path("notebooks/input")
    INTERMEDIATE_DIR = Path("notebooks/output/intermediate")
    OUTPUT_DIR = Path("notebooks/output")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RAW_FILE = INPUT_DIR / "merged_dataset.csv"
    NORMALIZED_FILE = INTERMEDIATE_DIR / "norm_logs.parquet"
    SESSIONS_FILE = OUTPUT_DIR / "sessions.csv"

    try:
        # Try loading from the intermediate parquet file first
        if NORMALIZED_FILE.exists():
            logging.info(f"Loading normalized data from {NORMALIZED_FILE}")
            norm_data = pd.read_parquet(NORMALIZED_FILE)
        else:
            logging.warning(
                f"Normalized data file {NORMALIZED_FILE} not found. "
                f"Loading and normalizing from raw file {RAW_FILE} instead."
            )
            norm_data = load_and_normalize_data(RAW_FILE)
            # Optionally save it now if it wasn't saved before
            norm_data.to_parquet(NORMALIZED_FILE, index=False)
            logging.info(f"Saved newly normalized data to {NORMALIZED_FILE}")

        # Infer sessions
        sessions_df = infer_sessions(norm_data)
        logging.info(f"Successfully inferred {len(sessions_df)} sessions.")

        # Save sessions data
        sessions_df.to_csv(SESSIONS_FILE, index=False)
        logging.info(f"Saved sessions data to {SESSIONS_FILE}")

        # Display sample data
        logging.info("Sample of sessions data:")
        print(sessions_df.head().to_markdown(index=False))
        logging.info("\nData types of sessions data:")
        print(sessions_df.info())

    except FileNotFoundError:
        logging.error(
            f"Input file {RAW_FILE} not found and intermediate file {NORMALIZED_FILE} also missing."
        )
    except ImportError as e:
        logging.error(f"Import error: {e}. Cannot proceed.")
    except Exception as e:
        logging.error(
            f"An error occurred during the sessionize process: {e}", exc_info=True
        )

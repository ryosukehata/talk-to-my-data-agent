"""
Main orchestration script for the data pipeline.

Executes the pipeline steps sequentially:
1. Load and Normalize Raw Data
2. Infer Sessions
3. Reconstruct Chat Flows
4. Aggregate KPIs
5. Process NLP for Word Cloud

Handles data dependencies between steps, preferring intermediate files if available.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Add the pipeline directory to sys.path to ensure imports work
PIPELINE_DIR = Path(__file__).parent
# Check if the path is already in sys.path to avoid duplicates
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))
    logging.info(f"Added {PIPELINE_DIR} to sys.path")


# Import pipeline step functions
try:
    from chat_flow import reconstruct_chat_flows
    from kpi_aggregation import calculate_and_save_kpis
    from load_normalize import load_and_normalize_data
    from nlp import generate_word_cloud_data_daily
    from sessionize import infer_sessions
except ImportError as e:
    logging.error(
        f"Failed to import pipeline modules: {e}. Ensure all .py files are in {PIPELINE_DIR} and dependencies are installed."
    )
    sys.exit(1)  # Exit if core modules are missing

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define File Paths
# Assuming run_pipeline.py is in notebooks/pipeline/
# Parent is notebooks/, Parent.parent is the project root
PROJECT_ROOT = PIPELINE_DIR.parent.parent
INPUT_DIR = PROJECT_ROOT / "notebooks" / "input"
INTERMEDIATE_DIR = PROJECT_ROOT / "notebooks" / "output" / "intermediate"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "output"

RAW_FILE = INPUT_DIR / "merged_dataset.csv"
NORMALIZED_FILE = INTERMEDIATE_DIR / "norm_logs.parquet"
SESSIONS_FILE = OUTPUT_DIR / "sessions.csv"
CHAT_FLOWS_FILE = OUTPUT_DIR / "chat_flows.csv"
WORD_CLOUD_FILE = OUTPUT_DIR / "daily_word_cloud_data.csv"

# Ensure output directories exist
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    """Executes the full data pipeline."""
    start_time = time.time()
    logging.info("--- Starting Data Pipeline Execution ---")

    norm_data = None
    sessions_data = None
    chat_flows_data = None

    # --- Step 1: Load and Normalize Data ---
    try:
        step_start = time.time()
        logging.info("Step 1: Loading and Normalizing Data...")
        if NORMALIZED_FILE.exists():
            logging.info(
                f"Found existing normalized file: {NORMALIZED_FILE}. Loading..."
            )
            norm_data = pd.read_parquet(NORMALIZED_FILE)
        else:
            logging.info(f"Normalized file not found. Processing raw file: {RAW_FILE}")
            if not RAW_FILE.exists():
                raise FileNotFoundError(f"Raw input file {RAW_FILE} does not exist.")
            norm_data = load_and_normalize_data(RAW_FILE)
            logging.info(f"Saving normalized data to {NORMALIZED_FILE}...")
            norm_data.to_parquet(NORMALIZED_FILE, index=False)
        logging.info(
            f"Step 1 completed in {time.time() - step_start:.2f} seconds. "
            f"Normalized data shape: {norm_data.shape if isinstance(norm_data, pd.DataFrame) else 'N/A'}"
        )
    except Exception as e:
        logging.error(f"Error in Step 1 (Load/Normalize): {e}", exc_info=True)
        return  # Stop pipeline if this critical step fails

    # --- Step 2: Infer Sessions ---
    try:
        step_start = time.time()
        logging.info("Step 2: Inferring Sessions...")
        if SESSIONS_FILE.exists():
            logging.info(f"Found existing sessions file: {SESSIONS_FILE}. Loading...")
            # Ensure correct parsing when loading CSV
            sessions_data = pd.read_csv(
                SESSIONS_FILE,
                parse_dates=["session_start", "session_end", "visitTimestamp"],
            )
        else:
            logging.info("Sessions file not found. Inferring from normalized data...")
            if isinstance(norm_data, pd.DataFrame):
                sessions_data = infer_sessions(norm_data)
                logging.info(f"Saving sessions data to {SESSIONS_FILE}...")
                sessions_data.to_csv(SESSIONS_FILE, index=False)
            else:
                logging.error(
                    "Cannot infer sessions because normalized data is missing or invalid."
                )
                sessions_data = pd.DataFrame()  # Assign empty DF
        logging.info(
            f"Step 2 completed in {time.time() - step_start:.2f} seconds. "
            f"Sessions data shape: {sessions_data.shape if isinstance(sessions_data, pd.DataFrame) else 'N/A'}"
        )
    except Exception as e:
        logging.error(f"Error in Step 2 (Sessionize): {e}", exc_info=True)
        # Log error and assign empty DF, but continue
        sessions_data = pd.DataFrame()

    # --- Step 3: Reconstruct Chat Flows ---
    try:
        step_start = time.time()
        logging.info("Step 3: Reconstructing Chat Flows...")
        if CHAT_FLOWS_FILE.exists():
            logging.info(
                f"Found existing chat flows file: {CHAT_FLOWS_FILE}. Loading..."
            )
            chat_flows_data = pd.read_csv(
                CHAT_FLOWS_FILE, parse_dates=["first_timestamp", "last_timestamp"]
            )
        else:
            logging.info(
                "Chat flows file not found. Reconstructing from normalized data..."
            )
            if isinstance(norm_data, pd.DataFrame):
                chat_flows_data = reconstruct_chat_flows(norm_data)
                logging.info(f"Saving chat flows data to {CHAT_FLOWS_FILE}...")
                chat_flows_data.to_csv(CHAT_FLOWS_FILE, index=False)
            else:
                logging.error(
                    "Cannot reconstruct chat flows because normalized data is missing or invalid."
                )
                chat_flows_data = pd.DataFrame()  # Assign empty DF
        logging.info(
            f"Step 3 completed in {time.time() - step_start:.2f} seconds. "
            f"Chat flows data shape: {chat_flows_data.shape if isinstance(chat_flows_data, pd.DataFrame) else 'N/A'}"
        )
    except Exception as e:
        logging.error(f"Error in Step 3 (Chat Flow): {e}", exc_info=True)
        # Log error and assign empty DF, but continue
        chat_flows_data = pd.DataFrame()

    # --- Step 4: Aggregate KPIs ---
    try:
        step_start = time.time()
        logging.info("Step 4: Aggregating KPIs...")
        # Check if required inputs are valid DataFrames before calling
        if (
            isinstance(norm_data, pd.DataFrame)
            and not norm_data.empty
            and isinstance(
                sessions_data, pd.DataFrame
            )  # Allow empty sessions/chat_flows
            and isinstance(chat_flows_data, pd.DataFrame)
        ):
            calculate_and_save_kpis(norm_data, sessions_data, chat_flows_data)
        else:
            logging.error(
                "Cannot aggregate KPIs because one or more required input dataframes (norm_logs) are invalid or missing."
            )
        logging.info(f"Step 4 completed in {time.time() - step_start:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error in Step 4 (KPI Aggregation): {e}", exc_info=True)
        # Log error and continue to NLP step

    # --- Step 5: Process NLP for Word Cloud ---
    try:
        step_start = time.time()
        logging.info("Step 5: Processing NLP for Word Cloud...")
        # Skip if file exists OR if norm_data is invalid
        if WORD_CLOUD_FILE.exists():
            logging.info(
                f"Found existing word cloud file: {WORD_CLOUD_FILE}. Skipping generation."
            )
        elif isinstance(norm_data, pd.DataFrame) and not norm_data.empty:
            word_cloud_data = generate_word_cloud_data_daily(norm_data)
            if not word_cloud_data.empty:
                logging.info(f"Saving word cloud data to {WORD_CLOUD_FILE}...")
                word_cloud_data.to_csv(WORD_CLOUD_FILE, index=False)
            else:
                logging.warning(
                    "Word cloud data generation resulted in an empty DataFrame. File not saved."
                )
        else:
            logging.error(
                "Cannot generate word cloud data because normalized data is invalid or missing."
            )
        logging.info(f"Step 5 completed in {time.time() - step_start:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error in Step 5 (NLP): {e}", exc_info=True)

    end_time = time.time()
    logging.info(
        f"--- Pipeline Execution Finished in {end_time - start_time:.2f} seconds ---"
    )


if __name__ == "__main__":
    run_pipeline()

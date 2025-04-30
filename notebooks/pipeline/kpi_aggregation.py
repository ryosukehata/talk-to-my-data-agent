"""
Module for aggregating Key Performance Indicators (KPIs) daily.

Calculates various metrics based on normalized logs, sessions, and chat flows,
aggregating them by date and saving them to separate CSV files.
"""

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import functions from previous steps
try:
    from chat_flow import reconstruct_chat_flows
    from load_normalize import load_and_normalize_data
    from sessionize import infer_sessions
except ImportError:
    logging.error("Could not import functions from previous pipeline steps.")

    # Define placeholders if import fails
    def load_and_normalize_data(path):
        raise ImportError("load_normalize unavailable.")  # type: ignore

    def infer_sessions(df):
        raise ImportError("infer_sessions unavailable.")  # type: ignore

    def reconstruct_chat_flows(df):
        raise ImportError("reconstruct_chat_flows unavailable.")  # type: ignore


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_DIR = Path("notebooks/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Individual KPI Aggregation Functions ---


def aggregate_unique_sessions_daily(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily unique sessions."""
    if "session_start" not in sessions_df.columns:
        logging.error("Missing 'session_start' in sessions_df for daily aggregation.")
        return pd.DataFrame(columns=["date", "unique_sessions"])
    sessions_df["date"] = pd.to_datetime(sessions_df["session_start"]).dt.date
    daily_sessions = sessions_df.groupby("date")["session_id"].nunique().reset_index()
    daily_sessions.rename(columns={"session_id": "unique_sessions"}, inplace=True)
    logging.info(f"Aggregated daily unique sessions: {len(daily_sessions)} days found.")
    return daily_sessions


def aggregate_active_users_daily(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily active users."""
    if "date" not in norm_logs_df.columns or "user_id" not in norm_logs_df.columns:
        logging.error("Missing 'date' or 'user_id' in norm_logs_df for active users.")
        return pd.DataFrame(columns=["date", "active_users"])
    daily_users = norm_logs_df.groupby("date")["user_id"].nunique().reset_index()
    daily_users.rename(columns={"user_id": "active_users"}, inplace=True)
    logging.info(f"Aggregated daily active users: {len(daily_users)} days found.")
    return daily_users


def aggregate_llm_calls_daily(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates total daily LLM calls."""
    if "date" not in norm_logs_df.columns:
        logging.error("Missing 'date' in norm_logs_df for LLM calls.")
        return pd.DataFrame(columns=["date", "total_llm_calls"])
    daily_calls = (
        norm_logs_df.groupby("date").size().reset_index(name="total_llm_calls")
    )
    logging.info(f"Aggregated total daily LLM calls: {len(daily_calls)} days found.")
    return daily_calls


def aggregate_chat_metrics_daily(chat_flows_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily chat threads and user messages."""
    if "first_timestamp" not in chat_flows_df.columns:
        logging.error("Missing 'first_timestamp' in chat_flows_df for chat metrics.")
        return pd.DataFrame(columns=["date", "chat_threads", "chat_user_messages"])
    chat_flows_df["date"] = pd.to_datetime(chat_flows_df["first_timestamp"]).dt.date
    daily_chats = (
        chat_flows_df.groupby("date")
        .agg(
            chat_threads=("chat_id", "nunique"),
            # Count unique user messages (chat_id + chat_seq combination)
            chat_user_messages=(
                "chat_seq",
                lambda x: chat_flows_df.loc[x.index, ["chat_id", "chat_seq"]]
                .drop_duplicates()
                .shape[0],
            ),
        )
        .reset_index()
    )
    logging.info(f"Aggregated daily chat metrics: {len(daily_chats)} days found.")
    return daily_chats


def aggregate_query_type_metrics_daily(
    norm_logs_df: pd.DataFrame, chat_flows_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculates daily query type distribution and average retries."""
    if not all(
        c in norm_logs_df.columns for c in ["date", "query_type", "chat_id", "chat_seq"]
    ):
        logging.error(
            "Missing required columns in norm_logs_df for query type metrics."
        )
        return pd.DataFrame()
    if not all(
        c in chat_flows_df.columns
        for c in [
            "chat_id",
            "chat_seq",
            "retry_count_code_gen",
            "retry_count_chart_gen",
        ]
    ):
        logging.error(
            "Missing required columns in chat_flows_df for query type metrics."
        )
        return pd.DataFrame()

    # Query counts per day/type
    query_counts = (
        norm_logs_df.groupby(["date", "query_type"]).size().reset_index(name="count")
    )

    # Prepare chat_flows for merge - keep only relevant retry info per chat/seq
    # Use first() as retry counts are constant within the group
    retry_info = (
        chat_flows_df.groupby(["chat_id", "chat_seq"])
        .agg(
            retry_count_code_gen=("retry_count_code_gen", "first"),
            retry_count_chart_gen=("retry_count_chart_gen", "first"),
        )
        .reset_index()
    )

    # Merge counts with retry info based on the log entry's chat/seq
    # Need to merge norm_logs with retry_info first, then aggregate average retries
    logs_with_retries = pd.merge(
        norm_logs_df[["date", "query_type", "chat_id", "chat_seq"]],
        retry_info,
        on=["chat_id", "chat_seq"],
        how="left",
    )

    # Calculate average retries ONLY for relevant query types
    code_gen_mask = logs_with_retries["query_type"].str.startswith("03_generate_code")
    chart_gen_mask = (
        logs_with_retries["query_type"] == "04_generate_run_charts_python_code"
    )

    # Calculate means separately for relevant groups
    avg_code_retries = (
        logs_with_retries[code_gen_mask]
        .groupby(["date", "query_type"])["retry_count_code_gen"]
        .mean()
        .reset_index(name="avg_retry_code_gen")
    )
    avg_chart_retries = (
        logs_with_retries[chart_gen_mask]
        .groupby(["date", "query_type"])["retry_count_chart_gen"]
        .mean()
        .reset_index(name="avg_retry_chart_gen")
    )

    # Merge counts with specific average retries
    query_metrics = pd.merge(
        query_counts, avg_code_retries, on=["date", "query_type"], how="left"
    )
    query_metrics = pd.merge(
        query_metrics, avg_chart_retries, on=["date", "query_type"], how="left"
    )

    # Fill NaN values with 0 AFTER merging, ensuring only relevant types might have non-zero averages
    query_metrics["avg_retry_code_gen"] = query_metrics["avg_retry_code_gen"].fillna(0)
    query_metrics["avg_retry_chart_gen"] = query_metrics["avg_retry_chart_gen"].fillna(
        0
    )

    # Ensure correct types after fillna
    query_metrics["avg_retry_code_gen"] = query_metrics["avg_retry_code_gen"].astype(
        float
    )  # Or appropriate type
    query_metrics["avg_retry_chart_gen"] = query_metrics["avg_retry_chart_gen"].astype(
        float
    )

    logging.info(
        f"Aggregated daily query type metrics: {len(query_metrics)} records found."
    )
    return query_metrics


def aggregate_feature_usage_daily(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily feature toggle usage."""
    if not all(
        c in norm_logs_df.columns
        for c in ["date", "enable_chart_generation", "enable_business_insights"]
    ):
        logging.error("Missing required columns in norm_logs_df for feature usage.")
        return pd.DataFrame()

    feature_usage = (
        norm_logs_df.groupby("date")
        .agg(
            chart_gen_true=(
                "enable_chart_generation",
                lambda x: x.sum(),
            ),  # Summing True values (True=1)
            chart_gen_false=(
                "enable_chart_generation",
                lambda x: (1 - x).sum(),
            ),  # Summing False values (False=1)
            business_insights_true=("enable_business_insights", lambda x: x.sum()),
            business_insights_false=(
                "enable_business_insights",
                lambda x: (1 - x).sum(),
            ),
        )
        .reset_index()
    )

    # Optional: Convert to long format as specified in docs (period, feature_name, true_count, false_count)
    charts = feature_usage[["date", "chart_gen_true", "chart_gen_false"]].copy()
    charts.rename(
        columns={"chart_gen_true": "true_count", "chart_gen_false": "false_count"},
        inplace=True,
    )
    charts["feature_name"] = "enable_chart_generation"

    insights = feature_usage[
        ["date", "business_insights_true", "business_insights_false"]
    ].copy()
    insights.rename(
        columns={
            "business_insights_true": "true_count",
            "business_insights_false": "false_count",
        },
        inplace=True,
    )
    insights["feature_name"] = "enable_business_insights"

    long_format_usage = pd.concat([charts, insights], ignore_index=True)
    long_format_usage = long_format_usage[
        ["date", "feature_name", "true_count", "false_count"]
    ]

    logging.info(
        f"Aggregated daily feature usage: {len(long_format_usage)} records found."
    )
    return long_format_usage


def aggregate_error_metrics_daily(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily error counts by message."""
    if not all(
        c in norm_logs_df.columns for c in ["date", "error_message", "association_id"]
    ):
        logging.error("Missing required columns in norm_logs_df for error metrics.")
        return pd.DataFrame()

    errors_df = norm_logs_df[
        norm_logs_df["error_message"].notna() & (norm_logs_df["error_message"] != "")
    ].copy()
    if errors_df.empty:
        logging.info("No errors found in the logs.")
        return pd.DataFrame(
            columns=["date", "error_category", "count", "sample_association_ids"]
        )

    # Simplify error messages (optional, group similar errors)
    # For now, using the raw message as the category
    errors_df["error_category"] = errors_df["error_message"]

    error_metrics = (
        errors_df.groupby(["date", "error_category"])
        .agg(
            count=("association_id", "size"),
            sample_association_ids=(
                "association_id",
                lambda x: list(x.unique()[:3]),
            ),  # Get up to 3 samples
        )
        .reset_index()
    )

    logging.info(f"Aggregated daily error metrics: {len(error_metrics)} records found.")
    return error_metrics


def parse_dataset_names(name_str):
    """Safely parse the string representation of a list of dataset names."""
    if (
        pd.isna(name_str)
        or not isinstance(name_str, str)
        or not name_str.startswith("[")
    ):
        return []
    try:
        # Use literal_eval for safe evaluation of list string
        return ast.literal_eval(name_str)
    except (ValueError, SyntaxError):
        logging.warning(f"Could not parse dataset_names string: {name_str}")
        return []  # Return empty list on error


def aggregate_dataset_usage_daily(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily dataset usage counts."""
    if not all(c in norm_logs_df.columns for c in ["date", "datasets_names"]):
        logging.error("Missing required columns in norm_logs_df for dataset usage.")
        return pd.DataFrame()

    df = norm_logs_df[["date", "datasets_names"]].copy()
    # Apply parsing row by row
    df["parsed_datasets"] = df["datasets_names"].apply(parse_dataset_names)

    # Explode the DataFrame to have one row per dataset used in an event
    exploded_df = df.explode("parsed_datasets")
    exploded_df.dropna(
        subset=["parsed_datasets"], inplace=True
    )  # Drop rows where parsing failed or list was empty

    if exploded_df.empty:
        logging.info("No valid dataset usage found after parsing.")
        return pd.DataFrame(columns=["date", "dataset_name", "selection_count"])

    # Count selections per dataset per day
    dataset_usage = (
        exploded_df.groupby(["date", "parsed_datasets"])
        .size()
        .reset_index(name="selection_count")
    )
    dataset_usage.rename(columns={"parsed_datasets": "dataset_name"}, inplace=True)

    # TODO: Add counting combinations (max 2) if required. This adds complexity.
    # Example for pairs: Generate combinations within each original row's list, then count.

    logging.info(f"Aggregated daily dataset usage: {len(dataset_usage)} records found.")
    return dataset_usage


def aggregate_user_governance_daily(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily user counts by domain and userType."""
    if not all(
        c in norm_logs_df.columns for c in ["date", "domain", "userType", "user_id"]
    ):
        logging.error("Missing required columns in norm_logs_df for user governance.")
        return pd.DataFrame()

    # Count unique users per group to avoid overcounting due to multiple events
    user_governance = (
        norm_logs_df.groupby(["date", "domain", "userType"])["user_id"]
        .nunique()
        .reset_index(name="count")
    )

    logging.info(
        f"Aggregated daily user governance metrics: {len(user_governance)} records found."
    )
    return user_governance


# --- Main Orchestration Function ---


def calculate_and_save_kpis(
    norm_logs_df: pd.DataFrame, sessions_df: pd.DataFrame, chat_flows_df: pd.DataFrame
):
    """
    Calculates all daily KPIs and saves them to CSV files.

    Args:
        norm_logs_df: DataFrame of normalized log data.
        sessions_df: DataFrame of inferred session data.
        chat_flows_df: DataFrame of reconstructed chat flow data.
    """
    logging.info("Starting calculation and saving of all daily KPIs...")

    # --- Calculate individual metrics first ---
    daily_sessions_df = aggregate_unique_sessions_daily(sessions_df)
    daily_users_df = aggregate_active_users_daily(norm_logs_df)
    daily_calls_df = aggregate_llm_calls_daily(norm_logs_df)
    daily_chats_df = aggregate_chat_metrics_daily(chat_flows_df)
    query_metrics_df = aggregate_query_type_metrics_daily(norm_logs_df, chat_flows_df)
    feature_usage_df = aggregate_feature_usage_daily(norm_logs_df)
    error_metrics_df = aggregate_error_metrics_daily(norm_logs_df)
    dataset_usage_df = aggregate_dataset_usage_daily(norm_logs_df)
    user_governance_df = aggregate_user_governance_daily(norm_logs_df)

    # --- Merge date-based metrics ---
    logging.info("Merging daily summary metrics (sessions, users, calls, chats)...")
    # Start with a base of all possible dates from the inputs if possible, or merge iteratively
    merged_summary_df = daily_sessions_df
    for df_to_merge in [daily_users_df, daily_calls_df, daily_chats_df]:
        if not df_to_merge.empty:
            merged_summary_df = pd.merge(
                merged_summary_df, df_to_merge, on="date", how="outer"
            )
        else:
            logging.warning(f"Skipping merge for an empty DataFrame.")

    # Fill NaN values that might result from outer merge (e.g., days with users but no sessions)
    # Decide on fill value - 0 seems appropriate for counts
    merged_summary_df.fillna(0, inplace=True)
    # Ensure date column is first if needed after merges
    if "date" in merged_summary_df.columns:
        merged_summary_df = merged_summary_df[
            ["date"] + [col for col in merged_summary_df.columns if col != "date"]
        ]
        # Ensure integer counts where appropriate after fillna
        for col in [
            "unique_sessions",
            "active_users",
            "total_llm_calls",
            "chat_threads",
            "chat_user_messages",
        ]:
            if col in merged_summary_df.columns:
                merged_summary_df[col] = merged_summary_df[col].astype(int)

    # --- Save merged and other individual files ---
    metrics_to_save = {
        "daily_summary_metrics.csv": merged_summary_df,
        "daily_query_type_metrics.csv": query_metrics_df,
        "daily_feature_usage.csv": feature_usage_df,
        "daily_error_metrics.csv": error_metrics_df,
        "daily_dataset_usage.csv": dataset_usage_df,
        "daily_user_governance.csv": user_governance_df,
    }

    for filename, result_df in metrics_to_save.items():
        output_path = OUTPUT_DIR / filename
        try:
            logging.info(f"Processing save for {filename}...")

            if not result_df.empty:
                result_df.to_csv(output_path, index=False)
                logging.info(f"Successfully saved {filename} to {output_path}")
            else:
                logging.warning(f"Result for {filename} was empty, file not saved.")
        except Exception as e:
            logging.error(
                f"Failed to process or save KPI for {filename}: {e}", exc_info=True
            )

    logging.info("Finished calculating and saving all daily KPIs.")


if __name__ == "__main__":
    INPUT_DIR = Path("notebooks/input")
    INTERMEDIATE_DIR = Path("notebooks/output/intermediate")
    # OUTPUT_DIR is defined globally

    RAW_FILE = INPUT_DIR / "merged_dataset.csv"
    NORMALIZED_FILE = INTERMEDIATE_DIR / "norm_logs.parquet"
    SESSIONS_FILE = OUTPUT_DIR / "sessions.csv"  # Previous step's output
    CHAT_FLOWS_FILE = OUTPUT_DIR / "chat_flows.csv"  # Previous step's output

    try:
        # --- Load Data ---
        # Load norm_logs
        if NORMALIZED_FILE.exists():
            logging.info(f"Loading normalized data from {NORMALIZED_FILE}")
            norm_data = pd.read_parquet(NORMALIZED_FILE)
        else:
            logging.warning(
                f"{NORMALIZED_FILE} not found. Loading and normalizing from {RAW_FILE}."
            )
            norm_data = load_and_normalize_data(RAW_FILE)
            norm_data.to_parquet(NORMALIZED_FILE, index=False)  # Save if generated

        # Load sessions
        if SESSIONS_FILE.exists():
            logging.info(f"Loading sessions data from {SESSIONS_FILE}")
            sessions_data = pd.read_csv(
                SESSIONS_FILE,
                parse_dates=["session_start", "session_end", "visitTimestamp"],
            )
        else:
            logging.warning(
                f"{SESSIONS_FILE} not found. Inferring sessions from normalized data."
            )
            sessions_data = infer_sessions(norm_data)
            sessions_data.to_csv(SESSIONS_FILE, index=False)  # Save if generated

        # Load chat_flows
        if CHAT_FLOWS_FILE.exists():
            logging.info(f"Loading chat flows data from {CHAT_FLOWS_FILE}")
            chat_flows_data = pd.read_csv(
                CHAT_FLOWS_FILE, parse_dates=["first_timestamp", "last_timestamp"]
            )
        else:
            logging.warning(
                f"{CHAT_FLOWS_FILE} not found. Reconstructing chat flows from normalized data."
            )
            chat_flows_data = reconstruct_chat_flows(norm_data)
            chat_flows_data.to_csv(CHAT_FLOWS_FILE, index=False)  # Save if generated

        # --- Calculate and Save KPIs ---
        if (
            not norm_data.empty
            and not sessions_data.empty
            and not chat_flows_data.empty
        ):
            calculate_and_save_kpis(norm_data, sessions_data, chat_flows_data)
        else:
            logging.error(
                "One or more input DataFrames are empty. Cannot calculate KPIs."
            )

    except FileNotFoundError:
        logging.error(
            f"Raw input file {RAW_FILE} not found and intermediate files also missing."
        )
    except ImportError as e:
        logging.error(f"Import error: {e}. Cannot proceed.")
    except Exception as e:
        logging.error(
            f"An error occurred during the KPI aggregation process: {e}", exc_info=True
        )

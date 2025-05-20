# --- Utility functions ported from notebooks ---
import json
import logging
import os
from pathlib import Path

import pandas as pd
from pydantic import AliasChoices, Field

from .config import DynamicSettings
from .dataset_helper import download_dataset


def parse_json(x):
    try:
        return json.loads(x) if pd.notnull(x) else {}
    except Exception:
        return {}


def extract_error_message_regex(text):
    # Placeholder: implement your regex extraction logic here
    return text if pd.notnull(text) else None


# --- Data Normalization (from load_and_normalize_data) ---
def load_and_normalize_data(df_trace: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes raw log data DataFrame, parses timestamps, normalizes data types, and consolidates user identity.
    """
    logging.info(f"Starting data normalization for {len(df_trace)} records.")

    # --- Error Message Extraction ---
    df_trace["error_message"] = df_trace["promptText"].apply(
        extract_error_message_regex
    )
    df_trace["error_type"] = df_trace["error_message"].str.split(":").str[0]
    actual_values_dicts = df_trace["actual_value"].apply(parse_json)
    actual_values_normalized = pd.json_normalize(actual_values_dicts.tolist())
    df_trace = pd.concat(
        [
            df_trace.reset_index(drop=True),
            actual_values_normalized.reset_index(drop=True),
        ],
        axis=1,
    )
    df_trace["query_no"] = df_trace["query_type"].str[:2]
    df_trace["timestamp"] = pd.to_datetime(
        df_trace["timestamp"].str[:19]
    ) + pd.Timedelta(hours=9)
    df_trace["startTimestamp"] = pd.to_datetime(
        df_trace["startTimestamp"].str[:19]
    ) + pd.Timedelta(hours=9)
    df_trace["endTimestamp"] = pd.to_datetime(
        df_trace["endTimestamp"].str[:19]
    ) + pd.Timedelta(hours=9)
    df_trace["date"] = df_trace["startTimestamp"].dt.date
    df_trace["chat_seq"] = df_trace["chat_seq"].astype("Int64")
    df_trace.sort_values("startTimestamp", inplace=True)
    df_trace.sort_values(
        ["user_email", "chat_id", "chat_seq", "startTimestamp"], inplace=True
    )
    col_list = [
        "user_email",
        "date",
        "startTimestamp",
        "endTimestamp",
        "association_id",
        "enable_chart_generation",
        "enable_business_insights",
        "chat_id",
        "chat_seq",
        "query_type",
        "query_no",
        "data_source",
        "datasets_names",
        "user_msg",
        "error_message",
        "error_type",
        "promptText",
        "DR_RESERVED_PREDICTION_VALUE",
    ]
    return df_trace[col_list].reset_index(drop=True).copy()


# --- Session Inference (from infer_sessions) ---
def infer_chat_sessions(df_trace: pd.DataFrame) -> pd.DataFrame:
    """
    Infers user sessions from normalized log data.
    """
    import hashlib

    logging.info(f"Starting session inference for {len(df_trace)} log entries.")
    df_trace_chat = (
        df_trace.groupby(by=["user_email", "chat_id", "chat_seq"])
        .agg(
            date=("date", "first"),
            startTimestamp=("startTimestamp", "first"),
            endTimestamp=("endTimestamp", "last"),
            userMsg=("user_msg", "first"),
            datasetCount=(
                "datasets_names",
                lambda x: len(list(set([item for sublist in x for item in sublist]))),
            ),
            datasetNames=("datasets_names", "first"),
            dataSource=("data_source", "first"),
            chartGen=("enable_chart_generation", "first"),
            businessInsights=("enable_business_insights", "first"),
            errorCount=("error_message", lambda x: x.notna().sum()),
            callCount=("association_id", "count"),
        )
        .reset_index()
    )
    df_trace_chat["idealCount"] = (
        (df_trace_chat["chartGen"].astype(str).str.lower() == "true").astype(int)
        + (df_trace_chat["businessInsights"].astype(str).str.lower() == "true").astype(
            int
        )
        + 2
    )
    df_trace_chat["stopUnexpected"] = (
        df_trace_chat["callCount"] - df_trace_chat["errorCount"]
    ) < df_trace_chat["idealCount"]
    df_trace_chat["time"] = (
        (df_trace_chat["endTimestamp"] - df_trace_chat["startTimestamp"])
        .dt.total_seconds()
        .astype("Int64")
    )
    df_trace_chat.insert(
        0,
        "id",
        df_trace_chat["user_email"].map(
            lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:8]
        )
        + "-"
        + df_trace_chat["chat_id"]
        + "-"
        + df_trace_chat["chat_seq"].astype(str),
    )
    df_trace_req = (
        df_trace.groupby(by=["user_email", "chat_id", "chat_seq", "query_no"])
        .agg(
            count=("association_id", "count"),
            startTimestamp=("startTimestamp", "first"),
            endTimestamp=("endTimestamp", "last"),
            error=("error_message", lambda x: x.notna().sum()),
        )
        .reset_index()
    )
    df_trace_req["time"] = (
        (df_trace_req["endTimestamp"] - df_trace_req["startTimestamp"])
        .dt.total_seconds()
        .astype("Int64")
    )
    df_trace_req = df_trace_req.pivot(
        index=["user_email", "chat_id", "chat_seq"],
        columns="query_no",
        values=["count", "time", "error"],
    ).reset_index()
    df_trace_req.columns = [
        "_".join(col[::-1]).strip() if col[1] else col[0]
        for col in df_trace_req.columns.values
    ]
    df_trace_chat = df_trace_chat.merge(
        df_trace_req,
        on=["user_email", "chat_id", "chat_seq"],
        how="left",
    )
    logging.info(
        f"Session inference complete. Generated {len(df_trace_chat)} sessions."
    )
    return df_trace_chat


# Helper for loading DATASET_TRACE_ID like LLMDeployment
class TraceDatasetId(DynamicSettings):
    id: str = Field(
        validation_alias=AliasChoices(
            "MLOPS_RUNTIME_PARAM_DATASET_TRACE_ID",
            "DATASET_TRACE_ID",
        )
    )


# --- Main Pipeline ---
def run_pipeline():
    """
    Downloads, normalizes, sessionizes, and saves trace data as Parquet files.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    # Download raw CSV from DataRobot
    trace_dataset_id = TraceDatasetId().id
    df_raw_csv = download_dataset(
        os.getenv("DATAROBOT_ENDPOINT"),
        os.getenv("DATAROBOT_API_TOKEN"),
        trace_dataset_id,
    )
    # Normalize data
    df_trace = load_and_normalize_data(df_raw_csv)
    df_trace.to_parquet(data_dir / "trace_raw.parquet", index=False)
    # Infer sessions
    df_chat = infer_chat_sessions(df_trace)
    df_chat.to_parquet(data_dir / "trace_chat.parquet", index=False)
    logging.info("Pipeline complete: data saved as Parquet.")

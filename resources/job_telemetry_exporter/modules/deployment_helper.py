import logging
from typing import Dict

import requests
from modules import config
from tenacity import retry, stop_after_delay, wait_fixed

logger = logging.getLogger(__name__)
API_V2_BASE = config.DATAROBOT_ENDPOINT


def initiate_prediction_data_export(
    deployment_id: str,
    start_date: str = config.DEFAULT_EXPORT_START_DATE,
    end_date: str = config.DEFAULT_EXPORT_END_DATE,
) -> str:
    """
    Initiates a prediction data export job via DataRobot API v2.
    """
    export_endpoint = (
        f"{API_V2_BASE}/deployments/{deployment_id}/predictionDataExports/"
    )
    payload = {
        # "modelId": model_id,
        "start": start_date,
        "end": end_date,
        "augmentationType": config.AUGMENTATION_TYPE,
    }
    logger.info(
        f"Initiating prediction data export for deployment {deployment_id} "
        f"from {start_date} to {end_date}..."
    )
    response = requests.post(
        export_endpoint, headers=config.JSON_API_HEADERS, json=payload, timeout=60
    )
    response.raise_for_status()
    response_data = response.json()
    export_id = response_data.get("id")
    if not export_id:
        logger.error("API response missing 'id' field: %s", response_data)
        raise ValueError("Could not extract export job ID from API response.")
    logger.info(f"Successfully initiated export job with ID: {export_id}")
    return export_id


@retry(
    stop=stop_after_delay(config.MAX_WAIT_SECONDS),
    wait=wait_fixed(config.POLL_INTERVAL_SECONDS),
    reraise=True,
)
def poll_export_status(deployment_id: str, export_id: str) -> Dict[str, Dict]:
    """
    Polls the status of a specific prediction data export job until it succeeds or fails.
    """
    export_endpoint = (
        f"{API_V2_BASE}/deployments/{deployment_id}/predictionDataExports/"
    )
    logger.info(f"Polling status for export job ID: {export_id}...")
    params = {"limit": 100}
    response = requests.get(
        export_endpoint, headers=config.API_HEADERS, params=params, timeout=30
    )
    response.raise_for_status()
    exports_data = response.json()
    target_export = None
    for export in exports_data.get("data", []):
        if export.get("id") == export_id:
            target_export = export
            break
    if not target_export:
        logger.error(f"Export job with ID {export_id} not found in the list response.")
        raise ValueError(f"Export job {export_id} not found.")
    status = target_export.get("status")
    if status == "SUCCEEDED":
        logger.info(f"Export job {export_id} succeeded.")
        return target_export
    elif status == "FAILED":
        error_details = target_export.get("error", "No error details provided.")
        logger.error(f"Export job {export_id} failed: {error_details}")
        raise ValueError(f"Export job {export_id} failed: {error_details}")
    elif status in ["CREATED", "RUNNING", "SCHEDULED"]:
        logger.info("Job still in progress, retrying...")
        raise Exception("Job not yet complete.")
    else:
        logger.warning(f"Unknown status encountered for job {export_id}: {status}")
        raise ValueError(f"Unknown status for job {export_id}: {status}")

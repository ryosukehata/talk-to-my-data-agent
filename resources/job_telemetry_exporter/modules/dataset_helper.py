import logging
import tempfile
import time
from io import StringIO
from typing import Optional

import pandas as pd
import requests
from modules import config
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)

logger = logging.getLogger(__name__)
API_V2_BASE = config.DATAROBOT_ENDPOINT


def get_mime_type(filepath: str) -> str:
    """Get MIME type based on file extension."""
    if filepath.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif filepath.endswith(".csv"):
        return "text/csv"
    elif filepath.endswith(".parquet"):
        return "application/octet-stream"
    else:
        raise ValueError("Unsupported file format")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(3) + wait_random(0, 10),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def download_dataset(endpoint: str, token: str, dataset_id: str) -> pd.DataFrame:
    """Download a dataset from DataRobot."""
    url = f"{endpoint}/datasets/{dataset_id}/file/"
    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "*/*",
    }
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    csv_content = response.content.decode("utf-8")
    df = pd.read_csv(StringIO(csv_content))
    return df


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(3) + wait_random(0, 10),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def update_dataset(
    # endpoint: str,
    # token: str,
    dataset_id: str,
    filepath: str,
    wait_for_completion: bool = False,
) -> str:
    """Update a dataset with a new file."""
    # headers = {
    #     "Authorization": f"Bearer {token}",
    # }
    url = f"{config.DATAROBOT_ENDPOINT}/datasets/{dataset_id}/versions/fromFile/"
    payload = {}
    mime_type = get_mime_type(filepath)
    with open(filepath, "rb") as f:
        files = [
            (
                "file",
                (
                    filepath,
                    f,
                    mime_type,
                ),
            )
        ]
        response = requests.post(
            url, headers=config.API_HEADERS, data=payload, files=files
        )
    logger.debug(response.json())
    assert response.status_code // 100 == 2
    version_id = response.json()["catalogVersionId"]
    logger.info(f"{dataset_id}: adding version: {version_id}")
    if wait_for_completion:
        _poll_dataset_version_status(dataset_id, version_id)
    return version_id


def _poll_dataset_version_status(dataset_id: str, version_id: str) -> dict:
    """
    Polls the status of a specific dataset version until its processingState is COMPLETED or ERROR.
    Returns the final version_info dict.
    """
    status_url = f"{API_V2_BASE}/datasets/{dataset_id}/versions/{version_id}"
    logger.info(
        f"Polling status for dataset version {version_id} (dataset {dataset_id}) from {status_url}"
    )
    for _ in range(config.MAX_WAIT_SECONDS // config.POLL_INTERVAL_SECONDS):
        response = requests.get(status_url, headers=config.API_HEADERS)
        response.raise_for_status()
        version_info = response.json()
        processing_state = version_info.get("processingState")
        if processing_state == "COMPLETED":
            logger.info(f"Dataset version {version_id} processing completed.")
            return version_info
        elif processing_state == "ERROR":
            error_details = version_info.get("error", "No error details provided.")
            logger.error(
                f"Dataset version {version_id} processing failed: {error_details}"
            )
            raise ValueError(
                f"Dataset version {version_id} processing failed: {error_details}"
            )
        elif processing_state == "RUNNING":
            logger.info("Dataset version processing still running, retrying...")
            time.sleep(config.POLL_INTERVAL_SECONDS)
        else:
            logger.warning(
                f"Unknown processingState for dataset version {version_id}: {processing_state}"
            )
            time.sleep(config.POLL_INTERVAL_SECONDS)
    logger.error(
        f"Timeout waiting for dataset version {version_id} to complete processing."
    )
    raise ValueError(
        f"Timeout waiting for dataset version {version_id} to complete processing."
    )


def download_exported_data(
    dataset_id: str, output_filename: Optional[str] = None
) -> str:
    """
    Downloads the CSV data for a given dataset ID.
    """
    download_url = f"{API_V2_BASE}/datasets/{dataset_id}/file/"
    logger.info(
        f"Attempting to download data for dataset ID: {dataset_id} from {download_url}"
    )
    if output_filename is not None:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix="_" + output_filename, mode="wb"
        )
    else:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_dataset_{dataset_id}.csv", mode="wb"
        )
    output_path = temp_file.name
    try:
        with requests.get(
            download_url, headers=config.API_HEADERS, stream=True, timeout=300
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                temp_file.write(chunk)
        temp_file.close()
        logger.info(f"Successfully downloaded data to: {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download data for dataset {dataset_id}: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to write downloaded data to {output_path}: {e}")
        raise


def _get_latest_dataset_version_id(dataset_id: str) -> str:
    """
    Fetches dataset versions and returns the ID of the latest version.
    """
    versions_url = f"{API_V2_BASE}/datasets/{dataset_id}/versions/"
    logger.info(f"Fetching versions for dataset ID: {dataset_id} from {versions_url}")
    response = requests.get(versions_url, headers=config.API_HEADERS)
    response.raise_for_status()
    versions_data = response.json()
    if not versions_data.get("data"):
        logger.error(f"No versions found for dataset {dataset_id}.")
        raise ValueError(f"No versions found for dataset {dataset_id}.")
    latest_version_id = None
    for version in versions_data["data"]:
        if version.get("isLatestVersion"):
            latest_version_id = version.get("versionId")
            break
    if not latest_version_id:
        logger.warning(
            f"Could not find 'isLatestVersion=true' for dataset {dataset_id}. "
            f"Assuming first version returned by orderBy=-created is latest."
        )
        latest_version_id = versions_data["data"][0].get("versionId")
    if not latest_version_id:
        logger.error(f"Could not determine latest version ID for dataset {dataset_id}.")
        raise ValueError(
            f"Could not determine latest version ID for dataset {dataset_id}."
        )
    logger.info(
        f"Found latest version ID for dataset {dataset_id}: {latest_version_id}"
    )
    return latest_version_id


def delete_dataset(dataset_id: str) -> None:
    """Marks the dataset with the given ID as deleted using the DataRobot API v2.

    Args:
        dataset_id: The ID of the dataset to delete.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        ValueError: If the dataset cannot be deleted (e.g., 409 Conflict due to refresh jobs).
    """
    url = f"{API_V2_BASE}/datasets/{dataset_id}/"
    logger.info(f"Attempting to delete dataset ID: {dataset_id} via {url}")
    try:
        response = requests.delete(url, headers=config.API_HEADERS)
        if response.status_code == 409:
            logger.error(
                f"Failed to delete dataset {dataset_id}: {response.status_code}. "
                f"Reason: Cannot delete a dataset that has refresh jobs or other dependencies. "
                f"Details: {response.text}"
            )
            raise ValueError(
                f"Cannot delete dataset {dataset_id} due to existing refresh jobs or dependencies."
            )
        response.raise_for_status()
        # Check for expected 204 No Content on success
        if response.status_code == 204:
            logger.info(f"Successfully marked dataset {dataset_id} as deleted.")
        else:
            logger.warning(
                f"Dataset deletion request for {dataset_id} returned an unexpected status code: {response.status_code}. "
                f"Content: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed during deletion of dataset {dataset_id}: {e}")
        raise

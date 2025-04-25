# nb/demo_api/data_exporter.py
import logging
import os
from typing import Any, Dict, Optional

# Import configuration values using absolute path from project root
import config
import requests
from tenacity import RetryError, retry, stop_after_delay, wait_fixed

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
API_V2_BASE = f"{config.DATAROBOT_ENDPOINT}"
assert API_V2_BASE.endswith("/api/v2")
# EXPORT_ENDPOINT is now constructed dynamically within functions


# --- API Interaction Functions ---
def initiate_prediction_data_export(
    deployment_id: str,
    start_date: str = config.DEFAULT_EXPORT_START_DATE,
    end_date: str = config.DEFAULT_EXPORT_END_DATE,
) -> str:
    """
    Initiates a prediction data export job via DataRobot API v2.

    Args:
        deployment_id: The ID of the deployment.
        model_id: The ID of the model associated with the deployment.
        start_date: The start date for the export period (ISO 8601 format).
        end_date: The end date for the export period (ISO 8601 format).

    Returns:
        The ID of the created export job.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        ValueError: If the API response does not contain the expected 'id'.
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
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
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
def poll_export_status(deployment_id: str, export_id: str) -> Dict[str, Any]:
    """
    Polls the status of a specific prediction data export job until it succeeds or fails.

    Uses tenacity for retrying with fixed waits and a maximum duration.

    Args:
        deployment_id: The ID of the deployment.
        export_id: The ID of the export job to poll.

    Returns:
        The dictionary representing the export job details once it's completed.

    Raises:
        requests.exceptions.RequestException: If API requests fail persistently.
        ValueError: If the target export job is not found or status indicates failure.
        tenacity.RetryError: If the job doesn't complete within MAX_WAIT_SECONDS.
    """
    export_endpoint = (
        f"{API_V2_BASE}/deployments/{deployment_id}/predictionDataExports/"
    )
    logger.info(f"Polling status for export job ID: {export_id}...")
    # We need to list exports and find the one with our ID, as there isn't a direct GET by ID.
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
        # Consider if this should be a permanent failure or retry
        raise ValueError(f"Export job {export_id} not found.")

    status = target_export.get("status")
    logger.info(f"Current status for job {export_id}: {status}")

    if status == "SUCCEEDED":
        logger.info(f"Export job {export_id} completed successfully.")
        return target_export
    elif status == "FAILED":
        error_details = target_export.get("error", "No error details provided.")
        logger.error(f"Export job {export_id} failed: {error_details}")
        raise ValueError(f"Export job {export_id} failed: {error_details}")
    elif status in ["CREATED", "RUNNING", "SCHEDULED"]:
        logger.info("Job still in progress, retrying...")
        raise Exception("Job not yet complete.")  # Generic exception for tenacity retry
    else:
        logger.warning(f"Unknown status encountered for job {export_id}: {status}")
        raise ValueError(f"Unknown status for job {export_id}: {status}")


def _get_latest_dataset_version_id(dataset_id: str) -> str:
    """
    Fetches dataset versions and returns the ID of the latest version.

    Args:
        dataset_id: The ID of the dataset.

    Returns:
        The version ID of the latest dataset version.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        ValueError: If no versions are found or the latest version cannot be identified.
    """
    versions_url = f"{API_V2_BASE}/datasets/{dataset_id}/versions/"
    logger.info(f"Fetching versions for dataset ID: {dataset_id} from {versions_url}")
    # Add orderBy=-created to be reasonably sure the first one is latest if flag is missing
    params = {"limit": 10, "orderBy": "-created"}
    response = requests.get(
        versions_url, headers=config.API_HEADERS, params=params, timeout=30
    )
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

    # Fallback if isLatestVersion flag isn't present (shouldn't happen based on docs)
    if not latest_version_id and versions_data["data"]:
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


# Add a new polling function specifically for dataset version status
@retry(
    stop=stop_after_delay(config.MAX_WAIT_SECONDS // 2),  # Use shorter timeout maybe?
    wait=wait_fixed(config.POLL_INTERVAL_SECONDS // 2),
    reraise=True,
)
def _poll_dataset_version_status(dataset_id: str, version_id: str) -> None:
    """
    Polls the status of a specific dataset version until its processingState is COMPLETED.

    Args:
        dataset_id: The ID of the dataset.
        version_id: The ID of the dataset version to poll.

    Raises:
        requests.exceptions.RequestException: If API requests fail persistently.
        ValueError: If status indicates failure (ERROR).
        tenacity.RetryError: If the version doesn't complete within the timeout.
    """
    version_url = f"{API_V2_BASE}/datasets/{dataset_id}/versions/{version_id}/"
    logger.info(
        f"Polling status for dataset {dataset_id} version {version_id} from {version_url}"
    )
    response = requests.get(version_url, headers=config.API_HEADERS, timeout=30)
    response.raise_for_status()
    version_details = response.json()

    processing_state = version_details.get("processingState")
    logger.info(
        f"Current processingState for dataset {dataset_id} version {version_id}: {processing_state}"
    )

    if processing_state == "COMPLETED":
        logger.info(f"Dataset {dataset_id} version {version_id} processing completed.")
        if not version_details.get("dataPersisted"):
            logger.warning(
                f"Dataset version {version_id} completed but dataPersisted is False. Download might fail."
            )
        return  # Success
    elif processing_state == "ERROR":
        error_details = version_details.get("error", "No error details provided.")
        logger.error(
            f"Dataset {dataset_id} version {version_id} processing failed: {error_details}"
        )
        raise ValueError(
            f"Dataset version {version_id} processing failed: {error_details}"
        )
    elif processing_state == "RUNNING":
        logger.info("Dataset version processing still running, retrying...")
        raise Exception("Dataset version not yet complete.")  # For tenacity
    else:
        logger.warning(
            f"Unknown processingState for dataset version {version_id}: {processing_state}"
        )
        # Treat unknown state as retryable for now
        raise Exception("Dataset version has unknown status.")


def download_exported_data(
    dataset_id: str, output_filename: Optional[str] = None
) -> str:
    """
    Downloads the CSV data for a given dataset ID.

    Args:
        dataset_id: The ID of the dataset to download.
        output_filename: Optional desired filename for the downloaded CSV.

    Returns:
        The full path to the saved CSV file.

    Raises:
        requests.exceptions.RequestException: If the download request fails.
        IOError: If the file cannot be written.
    """
    download_url = f"{API_V2_BASE}/datasets/{dataset_id}/file/"
    logger.info(
        f"Attempting to download data for dataset ID: {dataset_id} from {download_url}"
    )

    # Determine output path and filename
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # Create a default filename if none is provided
    if output_filename is None:
        output_filename = f"dataset_{dataset_id}.csv"
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)

    try:
        # Use stream=True for potentially large files
        with requests.get(
            download_url, headers=config.CSV_API_HEADERS, stream=True, timeout=300
        ) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Successfully downloaded data to: {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download data for dataset {dataset_id}: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to write downloaded data to {output_path}: {e}")
        raise


# --- Main Orchestration ---
def run_export_workflow(
    deployment_id: str = config.DEPLOYMENT_ID,
    start_date: str = config.DEFAULT_EXPORT_START_DATE,
    end_date: str = config.DEFAULT_EXPORT_END_DATE,
    output_filename: Optional[str] = config.DEFAULT_OUTPUT_FILENAME,
):
    """
    Runs the complete prediction data export and download workflow.

    Args:
        deployment_id: The ID of the deployment (defaults to config).
        start_date: Start date for the export (defaults to config).
        end_date: End date for the export (defaults to config).
        output_filename: Desired name for the output CSV file (defaults to config).
    """
    # Validate required IDs are provided or set in config
    if deployment_id == "YOUR_DEPLOYMENT_ID_HERE":
        logger.error(
            "Deployment ID is missing. Set via argument or environment variable."
        )
        return

    try:
        # Step 1: Initiate Export
        export_id = initiate_prediction_data_export(deployment_id, start_date, end_date)

        # Step 2: Poll for Status
        successful_export_details = poll_export_status(deployment_id, export_id)

        # Step 3: Get Dataset ID from successful export
        if successful_export_details:
            data_list = successful_export_details.get("data")
            if (
                not data_list
                or not isinstance(data_list, list)
                or not data_list[0].get("id")
            ):
                logger.error(
                    "Export details missing data ID: %s", successful_export_details
                )
                raise ValueError("Could not find data ID in successful export details.")
            dataset_id = data_list[0]["id"]
            logger.info(f"Export job succeeded. Extracted dataset ID: {dataset_id}")

            # Step 4: Get Latest Dataset Version ID
            version_id = _get_latest_dataset_version_id(dataset_id)

            # Step 5: Poll for Dataset Version Completion
            _poll_dataset_version_status(dataset_id, version_id)

            # Step 6: Download Data
            saved_filepath = download_exported_data(dataset_id, output_filename)
            logger.info(f"Workflow completed. Data saved to {saved_filepath}")
        else:
            # This case should ideally not be reached due to error handling in poll
            logger.error("Polling finished without success or specific error.")

    except requests.exceptions.RequestException as e:
        logger.error(f"API Request failed: {e}")
    except ValueError as e:
        logger.error(f"Data validation or processing error: {e}")
    except RetryError as e:
        logger.error(
            f"Export job did not complete within the maximum wait time "
            f"({config.MAX_WAIT_SECONDS} seconds): {e}"
        )
    except IOError as e:
        logger.error(f"File system error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    logger.info("Starting DataRobot Prediction Data Export Workflow...")

    # --- Example Usage ---
    # Option 1: Use defaults from config.py (requires environment variables or .env file)
    # print("Running with default configuration...")
    # run_export_workflow()

    # Option 2: Specify deployment ID, model ID, dates, and filename explicitly
    # Make sure to replace placeholders with actual IDs if using this option directly.
    # print("\nRunning with specified arguments...")
    # run_export_workflow(
    #     deployment_id="YOUR_SPECIFIC_DEPLOYMENT_ID",
    #     model_id="YOUR_SPECIFIC_MODEL_ID",
    #     start_date="2025-04-01T00:00:00.000Z",
    #     end_date="2025-04-20T00:00:00.000Z",
    #     output_filename="specific_export_run.csv"
    # )

    # Default behavior: Run using configuration (environment variables recommended)
    # Ensure DATAROBOT_API_TOKEN, DEPLOYMENT_ID, MODEL_ID are set in your environment or .env file
    run_export_workflow()

    logger.info("Workflow finished.")

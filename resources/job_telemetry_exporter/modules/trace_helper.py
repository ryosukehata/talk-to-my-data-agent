"""
Trace Helper Module

This module provides functionality for exporting prediction data and updating trace datasets.
It contains two main workflows:
1. Export Flow: Exports prediction data from a deployment
2. Update Flow: Updates an existing trace dataset with new data

Typical usage:
    temp_filepath = run_export_flow()
    run_update_flow(temp_filepath)
"""

import logging
import os
import tempfile
from typing import Optional

from modules import config
from modules.dataframe_helper import clean_trace_data, combine_dataframes
from modules.dataset_helper import (
    _get_latest_dataset_version_id,
    _poll_dataset_version_status,
    delete_dataset,
    download_exported_data,
    update_dataset,
)
from modules.deployment_helper import (
    initiate_prediction_data_export,
    poll_export_status,
)
from modules.file_helper import safe_remove_file

logger = logging.getLogger(__name__)

# --- Constants ---
DATASET_TRACE_ID = config.DATASET_TRACE_ID


# --- Trace Update Workflow ---
def run_update_flow(new_trace_filepath: str) -> Optional[str]:
    """
    Downloads the existing trace dataset, concatenates it with new trace data,
    cleans the combined data, and uploads it as a new version.

    Args:
        new_trace_filepath: Path to the file containing newly exported trace data.

    Returns:
        Version ID of the new dataset version or None if the operation failed.
    """
    logger.info("Starting trace dataset update workflow")
    logger.debug(f"Using new data from: {new_trace_filepath}")

    existing_trace_filepath = None
    processed_temp_filepath = None

    try:
        # 1. Download existing dataset (already creates a temp file)
        logger.info(f"Downloading existing trace dataset: {DATASET_TRACE_ID}")
        existing_trace_filepath = download_exported_data(DATASET_TRACE_ID)
        logger.info(f"Existing trace data downloaded to: {existing_trace_filepath}")

        # 2. Combine dataframes
        combined_df = combine_dataframes(existing_trace_filepath, new_trace_filepath)

        if combined_df.empty:
            logger.error("Combined DataFrame is empty. Cannot proceed.")
            return None

        # 3. Clean data
        cleaned_df = clean_trace_data(combined_df)

        if cleaned_df.empty:
            logger.info("DataFrame is empty after cleaning. Nothing to upload.")
            return None

        # 4. Save processed data to a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="_processed_trace.csv", mode="w", encoding="utf-8"
        ) as temp_file:
            processed_temp_filepath = temp_file.name

        logger.info(f"Saving cleaned data to temporary file: {processed_temp_filepath}")
        cleaned_df.to_csv(processed_temp_filepath, index=False)

        # 5. Upload cleaned data as new version
        logger.info(
            f"Uploading processed data as new version for dataset: {DATASET_TRACE_ID}"
        )
        version_id = update_dataset(
            dataset_id=DATASET_TRACE_ID,
            filepath=processed_temp_filepath,
            wait_for_completion=True,
        )
        logger.info(
            f"Successfully updated dataset {DATASET_TRACE_ID} with new version: {version_id}"
        )
        return version_id

    except Exception as e:
        logger.exception(f"An error occurred during the trace update workflow: {e}")
        return None
    finally:
        # Clean up temporary files
        safe_remove_file(existing_trace_filepath)
        safe_remove_file(processed_temp_filepath)


# --- Trace Export Workflow ---
def run_export_flow(
    deployment_id: str = config.LLM_DEPLOYMENT_ID,
    start_date: str = config.DEFAULT_EXPORT_START_DATE,
    end_date: str = config.DEFAULT_EXPORT_END_DATE,
    output_filename: Optional[str] = None,
    cleanup: bool = True,
) -> Optional[str]:
    """
    Runs the complete prediction data export and download workflow.

    Args:
        deployment_id: The ID of the deployment (defaults to config).
        start_date: Start date for the export (defaults to config).
        end_date: End date for the export (defaults to config).
        output_filename: Desired name for the output CSV file (optional).
        cleanup: Whether to delete the dataset after downloading (defaults to True).

    Returns:
        The full path to the saved CSV file, or None if the operation failed.
    """
    # Validate required IDs are provided or set in config
    if deployment_id == "YOUR_DEPLOYMENT_ID_HERE":
        logger.error(
            "Deployment ID is missing. Set via argument or environment variable."
        )
        return None

    logger.info(
        f"Starting export workflow for deployment {deployment_id} "
        f"from {start_date} to {end_date}"
    )

    try:
        # Step 1: Initiate Export
        logger.info("Initiating prediction data export...")
        export_id = initiate_prediction_data_export(deployment_id, start_date, end_date)
        logger.info(f"Export initiated with ID: {export_id}")

        # Step 2: Poll for Status
        logger.info(f"Polling export status for job ID: {export_id}")
        export_details = poll_export_status(deployment_id, export_id)

        # Step 3: Get Dataset ID from successful export
        if not export_details:
            logger.error("Polling finished without success or specific error.")
            return None

        data_list = export_details.get("data")
        if (
            not data_list
            or not isinstance(data_list, list)
            or not data_list[0].get("id")
        ):
            logger.error("Export details missing data ID: %s", export_details)
            return None

        dataset_id = data_list[0]["id"]
        logger.info(f"Export job succeeded. Extracted dataset ID: {dataset_id}")

        # Step 4: Get Latest Dataset Version ID
        logger.info(f"Getting latest version ID for dataset: {dataset_id}")
        version_id = _get_latest_dataset_version_id(dataset_id)
        logger.info(f"Latest version ID: {version_id}")

        # Step 5: Poll for Dataset Version Completion
        logger.info("Polling dataset version status...")
        _poll_dataset_version_status(dataset_id, version_id)

        # Step 6: Download Data
        logger.info("Downloading exported data...")
        saved_filepath = download_exported_data(dataset_id, output_filename)
        logger.info(f"Data successfully downloaded to: {saved_filepath}")

        # Step 7: Cleanup
        if cleanup:
            logger.info(f"Cleaning up dataset ID: {dataset_id}")
            delete_dataset(dataset_id)
            logger.info(f"Dataset {dataset_id} deleted successfully.")

        return saved_filepath

    except Exception as e:
        logger.exception(f"Export workflow failed: {e}")
        return None


def run_trace_update_workflow(deployment_id=config.LLM_DEPLOYMENT_ID) -> str:
    """
    Run the complete workflow: export data and update trace dataset.
    This function orchestrates both the export and update workflows.
    Args:
        deployment_id: The ID of the deployment (defaults to config).
    Returns:
        The full path to the saved CSV file, or None if the operation failed.
    """
    temp_filepath = None

    try:
        # Run export workflow
        logger.info("Running prediction data export workflow...")
        temp_filepath = run_export_flow(deployment_id=deployment_id)

        # Run trace update workflow if export was successful
        if temp_filepath and os.path.exists(temp_filepath):
            logger.info(
                "Export workflow finished. Proceeding to trace dataset update workflow."
            )
            version_id = run_update_flow(temp_filepath)
            if version_id:
                logger.info(
                    f"Trace dataset successfully updated with version: {version_id}"
                )
            else:
                logger.warning("Trace dataset update did not complete successfully.")
        elif temp_filepath:
            logger.error(
                f"Export workflow finished but the output file does not exist: {temp_filepath}. Skipping trace update."
            )
        else:
            logger.error(
                "Export workflow did not return a valid filepath. Skipping trace update."
            )

    except Exception as e:
        logger.exception(f"An error occurred in the main workflow: {e}")

    logger.info("Overall workflow finished.")
    return temp_filepath


if __name__ == "__main__":
    logger.info(
        "Starting DataRobot Prediction Data Export and Trace Update Workflow..."
    )
    run_trace_update_workflow()

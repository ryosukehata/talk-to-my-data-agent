import asyncio
import logging
import os
import tempfile

from modules import application_helper, config
from modules.dataset_helper import update_dataset

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
DATASET_ACCESS_LOG_ID = config.DATASET_ACCESS_LOG_ID


# --- Usage Update Workflow ---
def run_usage_update_flow():
    """
    Fetches application usage data, processes it, and uploads it as a new
    version to the specified DataRobot dataset.
    """
    logger.info("Starting application usage update workflow...")
    processed_temp_file = None
    processed_temp_filepath = None

    try:
        # 1. Fetch Usage Data
        logger.info("Fetching application usage data...")
        # Run the async function synchronously
        usage_df = asyncio.run(application_helper.fetch_usage_data())

        if usage_df.empty:
            logger.info("Fetched usage data is empty. Nothing to upload.")
            return  # Exit early

        logger.info(f"Successfully fetched {len(usage_df)} usage records.")

        # 2. Save processed data to a temporary file
        # Use 'with' statement for better file handling
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="_processed_usage.csv", mode="w", encoding="utf-8"
        ) as processed_temp_file:
            processed_temp_filepath = processed_temp_file.name
            logger.info(
                f"Saving processed usage data to temporary file: {processed_temp_filepath}"
            )
            # Ensure consistent datetime format if needed before saving
            # Example: usage_df['visitTimestamp'] = usage_df['visitTimestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            usage_df.to_csv(processed_temp_filepath, index=False)
        # File is closed automatically by 'with' statement

        # 3. Upload as new version
        logger.info(
            f"Uploading processed usage data as new version for dataset: {DATASET_ACCESS_LOG_ID}"
        )
        version_id = update_dataset(
            dataset_id=DATASET_ACCESS_LOG_ID,
            filepath=processed_temp_filepath,
            wait_for_completion=True,  # Wait for upload and processing
        )
        logger.info(
            f"Successfully updated dataset {DATASET_ACCESS_LOG_ID} with new version: {version_id}"
        )

    except FileNotFoundError as e:
        logger.error(f"Temporary file issue during processing: {e}")
    except Exception as e:
        logger.exception(
            f"An error occurred during the application usage update workflow: {e}"
        )
        # Re-raise the exception if needed, or handle appropriately
        # raise
    finally:
        # 4. Cleanup temporary file
        if processed_temp_filepath and os.path.exists(processed_temp_filepath):
            logger.debug(
                f"Deleting temporary processed usage file: {processed_temp_filepath}"
            )
            try:
                os.remove(processed_temp_filepath)
            except OSError as e:
                logger.error(
                    f"Error removing temporary file {processed_temp_filepath}: {e}"
                )


# Example of how to run this flow (optional, can be called from job.py)
if __name__ == "__main__":
    logger.info("Running Usage Helper standalone for testing...")
    # Ensure environment variables are set if running directly (e.g., via config loading)
    # config.load_config() # Assuming config module has a function to load vars if needed
    if DATASET_ACCESS_LOG_ID == "YOUR_DATASET_ID_HERE":
        logger.error(
            "DATASET_ACCESS_LOG_ID is not set in config. Please set the environment variable."
        )
    else:
        run_usage_update_flow()
    logger.info("Usage Helper standalone run finished.")

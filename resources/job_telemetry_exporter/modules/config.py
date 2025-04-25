# nb/demo_api/config.py
import logging
import os
from datetime import datetime, timedelta, timezone

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
current_dir = os.path.dirname(os.path.abspath(__file__))
env_file_path = os.path.join(current_dir, ".env")

# Determine if we should load .env (default: True for local/dev, False for production)
USE_DOTENV = os.getenv("USE_DOTENV", "true").lower() == "true"

if USE_DOTENV and os.path.exists(env_file_path):
    load_dotenv(env_file_path, override=False)


# --- API Configuration ---
# Replace with your DataRobot API endpoint
# e.g., "https://app.datarobot.com" or your dedicated instance URL
DATAROBOT_ENDPOINT = os.getenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")
# Fetch API token from environment variables for security
# Ensure you have DATAROBOT_API_TOKEN set in your environment or a .env file
API_TOKEN = os.getenv("DATAROBOT_API_TOKEN", "YOUR_API_TOKEN_HERE")

# Parse metadata.yaml to get required fieldNames dynamically
metadata_path = os.path.join(current_dir, "../metadata.yaml")
with open(metadata_path, "r") as f:
    metadata = yaml.safe_load(f)

field_names = [
    item["fieldName"] for item in metadata.get("runtimeParameterDefinitions", [])
]

# Dynamically create a Pydantic V2 Settings class
fields = {name: (str, ...) for name in field_names}

DynamicSettings = create_model(
    "DynamicSettings",
    __base__=BaseSettings,
    **fields,
)

# Conditionally set env_file in model_config only if using dotenv
if USE_DOTENV:
    DynamicSettings.model_config = SettingsConfigDict(
        env_file=os.path.join(current_dir, ".env"),
        case_sensitive=True,
    )
else:
    DynamicSettings.model_config = SettingsConfigDict(
        case_sensitive=True,
    )

try:
    settings = DynamicSettings()
except ValidationError as e:
    raise RuntimeError(f"Missing required environment variable(s): {e}")

# Expose as variables for compatibility
for name in field_names:
    globals()[name] = getattr(settings, name)
    logger.info(f"{name}: {getattr(settings, name)}")

# --- Date Calculation ---
# Calculate end date: tomorrow at 00:00:00 UTC
now = datetime.now(timezone.utc)
tomorrow = now.date() + timedelta(days=1)
end_date_dt = datetime(
    tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0, 0, tzinfo=timezone.utc
)

# Calculate start date: 7 days before the end date
start_date_dt = end_date_dt - timedelta(days=7)

# Format dates as ISO 8601 strings required by the API
# e.g., "2025-04-25T00:00:00.000Z"
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

# --- Export Parameters ---
# Default start and end dates for the prediction data export, calculated dynamically
DEFAULT_EXPORT_START_DATE = start_date_dt.strftime(ISO_FORMAT)[:-4] + "Z"
DEFAULT_EXPORT_END_DATE = end_date_dt.strftime(ISO_FORMAT)[:-4] + "Z"
# Type of data to augment predictions with
AUGMENTATION_TYPE = "ACTUALS_AND_METRICS"

# --- Polling Configuration ---
# Time to wait between status checks (in seconds)
POLL_INTERVAL_SECONDS = 10
# Maximum time to wait for the export to complete (in seconds)
MAX_WAIT_SECONDS = 600  # 10 minutes

# --- Output Configuration ---
# Directory to save the exported CSV file
OUTPUT_DIR = "."
# Default filename for the exported CSV
DEFAULT_OUTPUT_FILENAME = "prediction_export.csv"

# --- API Headers ---
API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "Authorization": f"Bearer {API_TOKEN}",
}

JSON_API_HEADERS = {
    **API_HEADERS,
    "content-type": "application/json",
}

CSV_API_HEADERS = {
    "accept": "*/*",
    "Authorization": f"Bearer {API_TOKEN}",
}

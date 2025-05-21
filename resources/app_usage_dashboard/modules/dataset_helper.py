import logging
from io import StringIO

import pandas as pd
import requests
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(3) + wait_random(0, 10),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def download_dataset(endpoint: str, token: str, dataset_id: str) -> pd.DataFrame:
    """Download a dataset from DataRobot."""
    logger.info(f"Downloading dataset {dataset_id} from {endpoint}")
    url = f"{endpoint}/datasets/{dataset_id}/file/"
    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "*/*",
    }
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    csv_content = response.content.decode("utf-8")
    df = pd.read_csv(StringIO(csv_content))
    logger.info(f"Downloaded dataset {dataset_id} with {len(df)} rows.")
    return df

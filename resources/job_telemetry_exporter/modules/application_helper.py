"""Module to handle application state and usage telemetry."""

import logging

import httpx
import pandas as pd
from modules import config

logger = logging.getLogger(__name__)


async def _fetch_page(client: httpx.AsyncClient, url: str):
    """Fetch a single page of data from the API using settings for headers."""
    response = await client.get(url, headers=config.API_HEADERS, timeout=60.0)
    response.raise_for_status()
    return response.json()


async def _fetch_all_data(start_url: str) -> list:
    """Fetch all data from the API, handling pagination."""
    all_rows = []
    next_url = start_url

    async with httpx.AsyncClient() as client:
        while next_url:
            data = await _fetch_page(client, next_url)
            fetched_rows = data.get("data", [])
            all_rows.extend(fetched_rows)
            next_url = data.get("next")
            if next_url:
                logger.info(f"Fetched {len(fetched_rows)} rows, next URL: {next_url}")

    return all_rows


def _process_data(rows: list) -> pd.DataFrame:
    """Process raw data into a DataFrame with transformations."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Perform data transformations
    # timestamp format is mixed take until the seconds from '2025-04-23 23:54:43.183000'
    df["visitTimestamp"] = pd.to_datetime(df["visitTimestamp"].str[:19]) + pd.Timedelta(
        hours=9
    )
    df["domain"] = df["username"].str.split("@").str[1].fillna("N/A")
    df = df.sort_values("visitTimestamp")
    logger.info(f"Processed {len(df)} rows into DataFrame")

    return df


async def fetch_usage_data(start_url=config.APP_LOG_URL) -> pd.DataFrame:
    """Fetch data from the application usage and return as dataframe."""

    all_rows = await _fetch_all_data(start_url=start_url)
    return _process_data(all_rows)

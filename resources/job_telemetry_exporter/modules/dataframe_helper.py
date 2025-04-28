import logging

import pandas as pd

logger = logging.getLogger(__name__)


# load_dataframe function removed as requested


def combine_dataframes(filepath1: str, filepath2: str) -> pd.DataFrame:
    """
    Loads data from two CSV file paths and concatenates them.
    Handles cases where one or both files might be empty or not found.

    Args:
        filepath1: Path to the first CSV file.
        filepath2: Path to the second CSV file.


    Returns:
        Combined DataFrame, or an empty DataFrame if both files are empty/not found.
    """
    logger.info(f"Attempting to load and combine data from {filepath1} and {filepath2}")

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    # Load first dataframe
    try:
        df1 = pd.read_csv(filepath1)
        logger.info(f"Loaded {len(df1)} rows from {filepath1}.")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        logger.warning(f"File not found or empty: {filepath1}.")
    except Exception as e:
        logger.exception(f"Error loading dataframe from {filepath1}: {e}")
        # Treat unexpected loading errors as if the file was empty/missing for robustness
        df1 = pd.DataFrame()

    # Load second dataframe
    try:
        df2 = pd.read_csv(filepath2)
        logger.info(f"Loaded {len(df2)} rows from {filepath2}.")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        logger.warning(f"File not found or empty: {filepath2}.")
    except Exception as e:
        logger.exception(f"Error loading dataframe from {filepath2}: {e}")
        df2 = pd.DataFrame()

    # existing df, df1, won't be empty
    if df2.empty:
        logger.info("Second dataframe is empty. Returning first dataframe.")
        return df1
    else:
        logger.info("Concatenating non-empty dataframes...")
        combined_df = pd.concat([df1, df2], ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} rows.")
        return combined_df


def clean_trace_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the trace DataFrame by removing placeholder prompts and duplicates.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Starting data cleaning process...")

    cleaned_df = (
        df.loc[df["promptText"] != "PLACEHOLDER"]
        .drop_duplicates(keep="last")
        .reset_index(drop=True)
        .copy()
    )

    logger.info(f"Initial rows: {len(df)}, Cleaned rows: {len(cleaned_df)}")

    return cleaned_df

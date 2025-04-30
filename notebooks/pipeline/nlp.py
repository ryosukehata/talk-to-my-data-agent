"""
Module for Natural Language Processing (NLP) on user messages.

Performs text normalization, tokenization using Fugashi, stop word removal,
part-of-speech filtering, and aggregates token frequencies daily for word cloud data.
"""

import logging
import unicodedata
from collections import Counter
from pathlib import Path

import fugashi
import pandas as pd

# Import the loading function
try:
    from load_normalize import load_and_normalize_data
except ImportError:
    logging.error(
        "Could not import load_and_normalize_data. Ensure load_normalize.py is accessible."
    )

    def load_and_normalize_data(path):
        raise ImportError("load_normalize unavailable.")  # type: ignore


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_DIR = Path("notebooks/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Basic Japanese Stop Words List (consider using a more comprehensive external list)
# Source: Inspired by SlothLib, but simplified for example
STOP_WORDS = set(
    [
        "の",
        "に",
        "は",
        "を",
        "た",
        "が",
        "で",
        "て",
        "と",
        "し",
        "れ",
        "さ",
        "ある",
        "いる",
        "も",
        "する",
        "から",
        "な",
        "こと",
        "として",
        "い",
        "や",
        "れる",
        "など",
        "なっ",
        "ない",
        "この",
        "ため",
        "その",
        "あっ",
        "よう",
        "また",
        "もの",
        "という",
        "あり",
        "まで",
        "られ",
        "なる",
        "へ",
        "か",
        "だ",
        "これ",
        "によって",
        "により",
        "おり",
        "より",
        "による",
        "ず",
        "なり",
        "られる",
        "において",
        "ただし",
        "あるい",
        "というのは",
        "それで",
        "しかし",
        "そう",
        "ので",
        "そして",
        "それ",
        "ここ",
        "どこ",
        "そこ",
        "あれ",
        "どれ",
        "どの",
        "そして",
        "なら",
        "どういう",
        "どうして",
        "まあ",
        "みたい",
        "なく",
        "特に",
        "せる",
        "及び",
        "及びこれら",
        "および",
        "ならびに",
        "または",
        "あるいは",
        "しかも",
        "しかつ",
        "かつ又",
        "且つ",
        "且つ又",
        "および",
        "それから",
        "そうして",
        "もっとも",
        "但し",
        "ただし",
        "しかしながら",
        "すなわち",
        "ですから",
        "だから",
        "それで",
        "では",
        "さて",
        "ところで",
        "ときに",
        "ところで",
        "そういえば",
        "さて",
        "それはさておき",
        "思う",
        "思います",
        "ください",
        "お願いします",
        "教えて",
        "表示",
        "表示して",
        "グラフ",
        "チャート",
        "作成",
        "表示する",  # Domain specific additions
    ]
)

# Target Parts of Speech (POS) for Word Cloud (using UniDic tags)
# Example: Noun (名詞), Verb (動詞), Adjective (形容詞)
TARGET_POS = ["名詞", "動詞", "形容詞"]

# Initialize Fugashi Tagger with unidic-lite dictionary
try:
    tagger = fugashi.Tagger(
        "-Owakati"
    )  # Using Wakati mode for faster tokenization first
    tagger_features = fugashi.Tagger()  # Separate tagger for feature extraction
except RuntimeError as e:
    logging.error(
        f"Failed to initialize Fugashi Tagger: {e}. Is unidic-lite installed? "
        "Try: pip install fugashi[unidic-lite]"
    )
    # Define placeholder tagger to avoid crashing immediately, but processing will fail
    tagger = None
    tagger_features = None


def normalize_text(text: str) -> str:
    """Normalizes text using NFKC and converts full-width to half-width."""
    if not isinstance(text, str):
        return ""
    # NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)
    # Convert full-width characters to half-width (primarily for numbers, alphabet)
    # This is a basic conversion, might need mojimoji for more complex cases
    return normalized.translate(
        str.maketrans(
            "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ　",
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
        )
    ).lower()  # Convert to lowercase


def tokenize_and_filter(text: str) -> list[str]:
    """
    Tokenizes Japanese text, filters by POS and stop words.
    Returns the base form (lemma) of the word.
    """
    if (
        not tagger
        or not tagger_features
        or not isinstance(text, str)
        or not text.strip()
    ):
        return []

    tokens = []
    # Use feature tagger to get details for filtering
    words = tagger_features(text)
    for word in words:
        pos = word.feature.pos1  # Using pos1 (major category) from UniDic features
        lemma = (
            word.feature.lemma if word.feature.lemma else word.surface
        )  # Use lemma (base form) if available
        lemma_normalized = normalize_text(lemma)  # Normalize the lemma itself

        # Filter by POS and stop words (check normalized lemma)
        if (
            pos in TARGET_POS
            and lemma_normalized not in STOP_WORDS
            and len(lemma_normalized) > 1
        ):
            tokens.append(lemma_normalized)
    return tokens


def generate_word_cloud_data_daily(norm_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates daily token frequency data for word cloud.

    Args:
        norm_logs_df: DataFrame with normalized logs, including 'date' and 'user_msg'.

    Returns:
        DataFrame with columns ['date', 'token', 'frequency'].
    """
    logging.info("Starting daily word cloud data generation.")

    if tagger is None or tagger_features is None:
        logging.error("Fugashi tagger not initialized. Cannot perform NLP.")
        return pd.DataFrame(columns=["date", "token", "frequency"])

    required_cols = ["date", "user_msg"]
    if not all(col in norm_logs_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in norm_logs_df.columns]
        logging.error(f"Missing required columns for NLP processing: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    df = norm_logs_df[["date", "user_msg"]].copy()
    df.dropna(subset=["user_msg"], inplace=True)
    if df.empty:
        logging.warning(
            "No user messages found after dropping NA. Returning empty DataFrame."
        )
        return pd.DataFrame(columns=["date", "token", "frequency"])

    logging.info(f"Processing {len(df)} user messages for tokenization...")

    # Apply normalization and tokenization
    # Note: This can be slow for large datasets. Consider parallelization (e.g., Dask, multiprocessing).
    df["tokens"] = df["user_msg"].apply(tokenize_and_filter)

    # Explode the DataFrame to have one row per token
    token_df = df.explode("tokens")
    token_df.dropna(
        subset=["tokens"], inplace=True
    )  # Remove rows where tokenization yielded nothing
    token_df.rename(columns={"tokens": "token"}, inplace=True)

    if token_df.empty:
        logging.warning(
            "No valid tokens found after filtering. Returning empty DataFrame."
        )
        return pd.DataFrame(columns=["date", "token", "frequency"])

    # Calculate daily frequency
    logging.info("Calculating daily token frequencies...")
    daily_freq = (
        token_df.groupby(["date", "token"]).size().reset_index(name="frequency")
    )

    logging.info(f"Generated {len(daily_freq)} daily token frequency records.")
    return daily_freq


if __name__ == "__main__":
    INPUT_DIR = Path("notebooks/input")
    INTERMEDIATE_DIR = Path("notebooks/output/intermediate")
    OUTPUT_DIR = Path("notebooks/output")  # Defined globally

    RAW_FILE = INPUT_DIR / "merged_dataset.csv"
    NORMALIZED_FILE = INTERMEDIATE_DIR / "norm_logs.parquet"
    WORD_CLOUD_FILE = OUTPUT_DIR / "daily_word_cloud_data.csv"

    try:
        # Load normalized data
        if NORMALIZED_FILE.exists():
            logging.info(f"Loading normalized data from {NORMALIZED_FILE}")
            norm_data = pd.read_parquet(NORMALIZED_FILE)
        else:
            logging.warning(
                f"{NORMALIZED_FILE} not found. Loading and normalizing from {RAW_FILE}."
            )
            norm_data = load_and_normalize_data(RAW_FILE)
            norm_data.to_parquet(NORMALIZED_FILE, index=False)

        # Generate word cloud data
        word_cloud_df = generate_word_cloud_data_daily(norm_data)
        logging.info(
            f"Successfully generated {len(word_cloud_df)} word cloud frequency records."
        )

        # Save the data
        if not word_cloud_df.empty:
            word_cloud_df.to_csv(WORD_CLOUD_FILE, index=False)
            logging.info(f"Saved daily word cloud data to {WORD_CLOUD_FILE}")

            # Display sample data
            logging.info("Sample of word cloud data:")
            print(
                word_cloud_df.sort_values(
                    by=["date", "frequency"], ascending=[True, False]
                )
                .head()
                .to_markdown(index=False)
            )
            logging.info("\nData types of word cloud data:")
            print(word_cloud_df.info())
        else:
            logging.warning("Word cloud data frame is empty. Nothing saved.")

    except FileNotFoundError:
        logging.error(
            f"Raw input file {RAW_FILE} not found and intermediate file {NORMALIZED_FILE} also missing."
        )
    except ImportError as e:
        logging.error(f"Import error: {e}. Cannot proceed.")
    except Exception as e:
        logging.error(
            f"An error occurred during the NLP processing: {e}", exc_info=True
        )

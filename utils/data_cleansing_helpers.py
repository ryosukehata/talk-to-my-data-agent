# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd

from utils.schema import CleansedColumnReport


@contextmanager
def suppress_datetime_warnings() -> Generator[None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Parsing dates in %d/%m/%Y format when dayfirst=False \(the default\) was specified.",
        )
        warnings.filterwarnings(
            "ignore",
            message="Parsing dates in %m/%d/%Y format when dayfirst=True was specified.",
        )
        yield


def try_simple_numeric_conversion(
    series: pd.Series[str],
    sample_series: pd.Series[str],
    original_nulls: pd.Series[Any],
) -> tuple[bool, pd.Series[Any], list[str]]:
    simple_cleaned = sample_series.str.strip().str.replace(r"['\s]+", "", regex=True)
    numeric_simple = pd.to_numeric(simple_cleaned, errors="coerce")
    new_nulls = numeric_simple.isna()
    simple_success_rate = 1 - (new_nulls.sum() - original_nulls.sum()) / len(
        sample_series
    )

    warnings = []

    if simple_success_rate > 0.8:
        warnings.append(
            f"Converted to numeric after removing spaces/quotes. Success rate: {simple_success_rate:.1%}"
        )
        series = series.str.strip().str.replace(r"['\s]+", "", regex=True)
        series = pd.to_numeric(series, errors="coerce")
        return True, series, warnings
    else:
        # If simple cleaning failed but had some success, add warning
        if simple_success_rate > 0.2:
            warnings.append(
                f"Simple numeric conversion partially successful ({simple_success_rate:.1%}) but below threshold"
            )
    return False, series, warnings


def _convert_units(
    series: pd.Series[str], patterns: dict[str, bool] | None = None
) -> tuple[pd.Series[Any], dict[str, bool]]:
    """
    Convert a series containing various unit formats to numeric values.
    Returns the converted series and pattern detection info.

    Args:
        series: Input series to convert
        patterns: Optional pre-detected patterns. If None, patterns will be detected.
    """
    if patterns is None:
        # Only detect patterns if not provided
        patterns = {
            "has_currency": series.str.contains(r"[$€£¥]").any(),
            "has_commas": series.str.contains(r",").any(),
            "has_percent": series.str.contains(r"%").any(),
            "has_magnitude": series.str.contains(r"[KMB]$", case=False).any(),
        }

    # Create a working copy
    result = series.copy()

    # Apply transformations based on detected patterns
    if patterns["has_currency"]:
        result = result.str.replace(r"[$€£¥]", "", regex=True)

    if patterns["has_commas"]:
        result = result.str.replace(",", "")

    if patterns["has_magnitude"]:
        k_mask = result.str.contains(r"K$", case=False)
        m_mask = result.str.contains(r"M$", case=False)
        b_mask = result.str.contains(r"B$", case=False)

        result = result.str.replace(r"[KMB]$", "", case=False, regex=True)
        numeric_result = pd.to_numeric(result, errors="coerce")

        # Only apply magnitude multipliers to valid numbers
        valid_mask = numeric_result.notna()
        if k_mask.any():
            numeric_result.loc[valid_mask & k_mask] *= 1000
        if m_mask.any():
            numeric_result.loc[valid_mask & m_mask] *= 1000000
        if b_mask.any():
            numeric_result.loc[valid_mask & b_mask] *= 1000000000

        result = numeric_result.astype(str)

    if patterns["has_percent"]:
        result = result.str.replace("%", "")
        result = pd.to_numeric(result, errors="coerce") / 100
    else:
        result = pd.to_numeric(result, errors="coerce")

    return result, patterns


def try_unit_conversion(
    series: pd.Series[str],
    sample_series: pd.Series[str],
    original_nulls: pd.Series[Any],
) -> tuple[bool, pd.Series[Any], list[str]]:
    """
    Try to convert a series with units to numeric values, first testing on a sample.
    Returns success status, converted series, and warnings.
    """
    warnings: list[str] = []

    # Check if series potentially contains convertible numeric data
    if not sample_series.astype(str).str.contains(r"[$€£¥%KMB,\d.]").mean() > 0.5:
        return False, series, warnings

    # Try conversion on sample first and detect patterns
    sample_result, patterns = _convert_units(sample_series)

    # Generate warning messages for detected patterns
    pattern_names = {
        "has_currency": "currency symbols",
        "has_commas": "thousand separators",
        "has_magnitude": "magnitude suffixes (K/M/B)",
        "has_percent": "percentages",
    }

    detected = [pattern_names[k] for k, v in patterns.items() if v]
    if detected:
        warnings.append(f"Detected patterns in data: {', '.join(detected)}")

    # Calculate conversion success rate
    new_nulls = sample_result.isna()
    conversion_success_rate = 1 - (new_nulls.sum() - original_nulls.sum()) / len(
        sample_result
    )

    if conversion_success_rate > 0.8:
        # If sample conversion was successful, convert full dataset using detected patterns
        warnings.append(
            f"Converted to numeric with pattern handling. Success rate: {conversion_success_rate:.1%}"
        )
        result, _ = _convert_units(series, patterns)  # Reuse patterns from sample
        return True, result, warnings

    elif conversion_success_rate > 0.2:
        warnings.append(
            f"Complex numeric conversion partially successful ({conversion_success_rate:.1%}) but below threshold"
        )

    return False, series, warnings


def try_datetime_conversion(
    series: pd.Series[str],
    sample_series: pd.Series[str],
    original_nulls: pd.Series[Any],
) -> tuple[bool, pd.Series[Any], list[str]]:
    # try to convert to date

    warnings = []
    with suppress_datetime_warnings():
        candidate_1 = pd.to_datetime(sample_series, errors="coerce", cache=True)
        candidate_2 = pd.to_datetime(
            sample_series, dayfirst=True, errors="coerce", cache=True
        )

    if candidate_1.notna().mean() > 0.8 or candidate_2.notna().mean() > 0.8:
        success_rate = max(candidate_1.notna().mean(), candidate_2.notna().mean())
        warnings.append(f"Converted to datetime. Success rate: {success_rate:.1%}")
        if candidate_1.notna().mean() > candidate_2.notna().mean():
            warnings.append("Used month-first date parsing")
            with suppress_datetime_warnings():
                ts_series = pd.to_datetime(series, errors="coerce")
        else:
            warnings.append("Used day-first date parsing")
            with suppress_datetime_warnings():
                ts_series = pd.to_datetime(series, dayfirst=True, errors="coerce")
        return True, ts_series, warnings
    return False, series, warnings


def add_summary_statistics(
    df: pd.DataFrame, report: list[CleansedColumnReport]
) -> None:
    """Add summary statistics to the report."""
    for column_report in report:
        col = column_report.new_column_name
        if column_report.new_dtype:
            null_count = df[col].isna().sum()
            total_count = len(df)
            if null_count > 0:
                column_report.warnings.append(
                    f"Contains {null_count} null values ({null_count / total_count:.1%} of data)"
                )

            if column_report.new_dtype == "float64":
                unique_count = df[col].nunique()
                if unique_count == 1:
                    column_report.warnings.append("Contains only one unique value")
                elif unique_count == 2:
                    column_report.warnings.append(
                        "Contains only two unique values - consider boolean conversion"
                    )

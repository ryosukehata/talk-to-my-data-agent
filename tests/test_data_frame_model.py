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

import logging
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from utils.schema import AnalystDataset


@pytest.fixture
def data() -> dict[str, Any]:
    """Test data generator to keep tests DRY"""
    return {
        "df": pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
        "records": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}],
    }


class TestDatasetInput:
    def test_accepts_dataframe(self, data: dict[str, Any]) -> None:
        model = AnalystDataset(data=data["df"], name="test")
        # assert isinstance(model.data, pd.DataFrame)
        assert_frame_equal(model.to_df(), data["df"])

    def test_accepts_records(self, data: dict[str, Any]) -> None:
        model = AnalystDataset(data=data["records"], name="test")
        # assert isinstance(model.data, pd.DataFrame)
        assert_frame_equal(model.to_df(), data["df"])

    def test_serialization(self, data: dict[str, Any]) -> None:
        model = AnalystDataset(data=data["df"], name="test")
        serialized = model.model_dump_json()
        logging.info(serialized)
        deserialized = AnalystDataset.model_validate_json(serialized)

        assert_frame_equal(deserialized.to_df(), data["df"])

    def test_empty_dataframe(self) -> None:
        empty_df = pd.DataFrame(columns=["a", "b"])
        model = AnalystDataset(data=empty_df, name="test")
        # assert model.data == pd.DataFrame
        assert_frame_equal(model.to_df(), empty_df)

    def test_single_row(self) -> None:
        single_row_df = pd.DataFrame({"a": [1], "b": ["x"]})
        model = AnalystDataset(data=single_row_df, name="test")
        assert len(model.to_df()) == 1
        assert_frame_equal(model.to_df(), single_row_df)

    def test_invalid_types(self) -> None:
        invalid_inputs = [42, "not a json string", {"not": "a list"}, [1, 2, 3], None]
        for invalid_input in invalid_inputs:
            with pytest.raises((ValidationError, ValueError)):
                AnalystDataset(data=invalid_input, name="test")

    def test_preserves_dtypes(self) -> None:
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "datetime_col": pd.date_range("2024-01-01", periods=3),
            }
        )
        model = AnalystDataset(data=df, name="test")
        result_df = model.to_df()

        # Check each column's dtype is preserved
        assert result_df["int_col"].dtype == "int64"
        assert result_df["float_col"].dtype == "float64"
        assert result_df["str_col"].dtype == "object"
        assert result_df["bool_col"].dtype == "bool"
        assert result_df["datetime_col"].dtype == "datetime64[ns]"

    def test_handles_null_values(self) -> None:
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
        model = AnalystDataset(data=df, name="test")
        result_df = model.to_df()
        assert result_df["a"].isna().sum() == 1
        assert result_df["b"].isna().sum() == 1

    def test_handles_index(self) -> None:
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": ["x", "y", "z"]}, index=["row1", "row2", "row3"]
        )
        model = AnalystDataset(data=df, name="test")
        result_df = model.to_df()
        # Check if index is preserved when converting back to DataFrame
        assert_frame_equal(result_df, df)

    @pytest.mark.parametrize("input_type", ["df", "records"])
    def test_different_input_types(self, input_type: str, data: dict[str, Any]) -> None:
        model = AnalystDataset(data=data[input_type], name="test")
        assert_frame_equal(model.to_df(), data["df"])

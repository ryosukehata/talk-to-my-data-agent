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

from typing import Any

import plotly.graph_objects as go
import pytest
import pytest_asyncio

from utils.schema import (
    AnalystDataset,
    CleansedDataset,
    DataDictionary,
    DataDictionaryColumn,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunBusinessAnalysisRequest,
    RunBusinessAnalysisResult,
    RunChartsRequest,
    RunChartsResult,
)


@pytest_asyncio.fixture(scope="module")
async def dataset_cleansed(
    pulumi_up: Any, dataset_loaded: AnalystDataset
) -> list[CleansedDataset]:
    from utils.api import (
        cleanse_dataframes,
    )

    result = await cleanse_dataframes([dataset_loaded])
    return result


def test_dataset_is_cleansed(dataset_cleansed: list[CleansedDataset]) -> None:
    assert len(dataset_cleansed) == 1


@pytest_asyncio.fixture(scope="module")
async def data_dictionary(
    pulumi_up: Any, dataset_loaded: AnalystDataset
) -> list[DataDictionary]:
    from utils.api import (
        get_dictionaries,
    )

    dictionary_result = await get_dictionaries([dataset_loaded])
    return dictionary_result


@pytest.fixture
def question() -> str:
    return "What are some interesting insights about the medication?"


@pytest.fixture
def run_analysis_request(
    pulumi_up: Any,
    dataset_cleansed: list[CleansedDataset],
    data_dictionary: list[DataDictionary],
    question: str,
) -> RunAnalysisRequest:
    analysis_request = RunAnalysisRequest(
        datasets=[ds.dataset for ds in dataset_cleansed],
        dictionaries=data_dictionary,
        question=question,
    )
    return analysis_request


@pytest.fixture
def run_analysis_result_canned() -> RunAnalysisResult:
    with open("tests/models/run_analysis_result.json") as f:
        return RunAnalysisResult.model_validate_json(f.read())


@pytest.fixture
def run_charts_result_canned() -> RunChartsResult:
    with open("tests/models/run_charts_result.json") as f:
        return RunChartsResult.model_validate_json(f.read())


@pytest.fixture
def run_business_result_canned() -> RunBusinessAnalysisResult:
    with open("tests/models/run_business_result.json") as f:
        return RunBusinessAnalysisResult.model_validate_json(f.read())


@pytest.fixture
def chart_request(
    pulumi_up: Any, run_analysis_result_canned: RunAnalysisResult, question: str
) -> RunChartsRequest:
    # Prepare requests
    chart_request = RunChartsRequest(
        dataset=run_analysis_result_canned.dataset,
        question=question,
    )
    return chart_request


@pytest.fixture
def business_request(
    pulumi_up: Any, run_analysis_result_canned: RunAnalysisResult, question: str
) -> RunBusinessAnalysisRequest:
    assert run_analysis_result_canned.dataset is not None
    business_request = RunBusinessAnalysisRequest(
        dataset=run_analysis_result_canned.dataset,
        dictionary=DataDictionary(
            name="analysis_result",
            column_descriptions=[
                DataDictionaryColumn(
                    column=col,
                    description="Analysis result column",
                    data_type=str(
                        run_analysis_result_canned.dataset.to_df()[col].dtype
                    ),
                )
                for col in run_analysis_result_canned.dataset.to_df().columns
            ],
        ),
        question=question,
    )
    return business_request


@pytest.mark.asyncio
async def test_run_analysis(
    pulumi_up: Any, run_analysis_request: RunAnalysisRequest
) -> None:
    from utils.api import (
        run_analysis,
    )

    run_analysis_result = await run_analysis(run_analysis_request)

    assert run_analysis_result.code is not None
    assert len(run_analysis_result.code) > 1
    assert run_analysis_result.dataset is not None
    df = run_analysis_result.dataset.to_df()
    assert df.shape[0] > 0
    assert run_analysis_result.status == "success"


@pytest.mark.asyncio
async def test_run_charts_analysis(
    pulumi_up: Any, chart_request: RunChartsRequest
) -> None:
    from utils.api import (
        run_charts,
    )

    run_charts_result = await run_charts(chart_request)
    assert isinstance(run_charts_result.fig1, go.Figure)
    assert isinstance(run_charts_result.fig2, go.Figure)
    assert run_charts_result.code is not None
    assert len(run_charts_result.code) > 1


@pytest.mark.asyncio
async def test_run_business_analysis(
    pulumi_up: Any,
    business_request: RunBusinessAnalysisRequest,
) -> None:
    from utils.api import (
        get_business_analysis,
    )

    run_business_result = await get_business_analysis(business_request)
    assert len(run_business_result.bottom_line) > 1
    assert len(run_business_result.additional_insights) > 1
    assert len(run_business_result.follow_up_questions) > 0


# TODO: add tests of reflection in run_analysis once test_api refactored/cleaned up

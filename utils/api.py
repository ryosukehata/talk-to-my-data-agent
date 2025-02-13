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

import ast
import asyncio
import functools
import inspect
import json
import logging
import re
import sys
import tempfile
from datetime import datetime
from typing import (
    Any,
    TypeVar,
    cast,
)

import datarobot as dr
import instructor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import sklearn
import statsmodels as sm
from joblib import Memory
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pandas import DataFrame
from plotly.subplots import make_subplots
from pydantic import ValidationError

sys.path.append("..")
from utils import prompts, tools
from utils.code_execution import (
    InvalidGeneratedCode,
    MaxReflectionAttempts,
    execute_python,
    reflect_code_generation_errors,
)
from utils.data_cleansing_helpers import (
    add_summary_statistics,
    try_datetime_conversion,
    try_simple_numeric_conversion,
    try_unit_conversion,
)
from utils.database_helpers import Database
from utils.resources import LLMDeployment
from utils.schema import (
    AiCatalogDataset,
    AnalysisError,
    AnalystDataset,
    BusinessAnalysisGeneration,
    ChartGenerationExecutionResult,
    ChatRequest,
    CleansedColumnReport,
    CleansedDataset,
    CodeGeneration,
    DatabaseAnalysisCodeGeneration,
    DataDictionary,
    DataDictionaryColumn,
    DictionaryGeneration,
    EnhancedQuestionGeneration,
    GetBusinessAnalysisMetadata,
    GetBusinessAnalysisRequest,
    GetBusinessAnalysisResult,
    QuestionListGeneration,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunAnalysisResultMetadata,
    RunChartsRequest,
    RunChartsResult,
    RunDatabaseAnalysisRequest,
    RunDatabaseAnalysisResult,
    RunDatabaseAnalysisResultMetadata,
    Tool,
    ValidatedQuestion,
)

logger = logging.getLogger("DataAnalystFrontend")

try:
    dr_client = dr.Client()  # type: ignore[attr-defined]
    chat_agent_deployment_id = LLMDeployment().id
    deployment_chat_base_url = (
        dr_client.endpoint + f"/deployments/{chat_agent_deployment_id}/"
    )

    openai_client = AsyncOpenAI(
        api_key=dr_client.token,
        base_url=deployment_chat_base_url,
        timeout=90,
        max_retries=2,
    )

    client = instructor.from_openai(openai_client, mode=instructor.Mode.MD_JSON)


except ValidationError as e:
    raise ValueError(
        "Unable to load Deployment ID."
        "If running locally, verify you have selected the correct "
        "stack and that it is active using `pulumi stack output`. "
        "If running in DataRobot, verify your runtime parameters have been set correctly."
    ) from e

ALTERNATIVE_LLM_BIG = "gpt-4o"
ALTERNATIVE_LLM_SMALL = "gpt-4o-mini"
DICTIONARY_BATCH_SIZE = 10
MAX_AI_CATALOG_DATASET_SIZE = 400e6  # aligns to 400MB set in streamlit config.toml
DISK_CACHE_LIMIT_BYTES = 512e6

_memory = Memory(tempfile.gettempdir(), verbose=0)
_memory.clear(warn=False)  # clear cache on startup

T = TypeVar("T")


def cache(f: T) -> T:
    """Cache function and coroutine results to disk using joblib."""
    cached_f = _memory.cache(f)

    if asyncio.iscoroutinefunction(f):

        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            in_cache = cached_f.check_call_in_cache(*args, **kwargs)
            result = await cached_f(*args, **kwargs)
            if not in_cache:
                _memory.reduce_size(DISK_CACHE_LIMIT_BYTES)
            else:
                logger.info(
                    f"Using previously cached result for function `{f.__name__}`"
                )
            return result

        return cast(T, awrapper)
    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            in_cache = cached_f.check_call_in_cache(*args, **kwargs)
            result = cached_f(*args, **kwargs)
            if not in_cache:
                _memory.reduce_size(DISK_CACHE_LIMIT_BYTES)
            else:
                logger.info(
                    f"Using previously cached result for function `{f.__name__}`"  # type: ignore
                )
            return result

        return cast(T, wrapper)


# This can be large as we are not storing the actual datasets in memory, just metadata
@functools.lru_cache(maxsize=32)
def list_catalog_datasets(limit: int = 100) -> list[AiCatalogDataset]:
    """
    Fetch datasets from AI Catalog with specified limit

    Args:
        limit: int
        Datasets to retrieve. Max value: 100
    """

    url = f"datasets?limit={limit}"

    # Get all datasets and manually limit the results
    datasets = dr.client.get_client().get(url).json()["data"]

    return [
        AiCatalogDataset(
            id=ds["datasetId"],
            name=ds["name"],
            created=(
                ds["creationDate"][:10] if "creationDate" in ds else "N/A"  # %Y-%m-%d
            ),
            size=(
                f"{ds['datasetSize'] / (1024 * 1024):.1f} MB"
                if "datasetSize" in ds
                else "N/A"
            ),
        )
        for ds in datasets
    ]


@cache
def download_catalog_datasets(*args: Any) -> list[AnalystDataset]:
    """Load selected datasets as pandas DataFrames

    Args:
        *args: list of dataset IDs to download

    Returns:
        list[AnalystDataset]: Dictionary of dataset names and data
    """
    dataset_ids = list(args)
    datasets = [dr.Dataset.get(id_) for id_ in dataset_ids]  # type: ignore
    if (
        sum([ds.size for ds in datasets if ds.size is not None])
        > MAX_AI_CATALOG_DATASET_SIZE
    ):
        raise ValueError(
            f"The requested AI Catalog datasets must total <= {int(MAX_AI_CATALOG_DATASET_SIZE)} bytes"
        )

    result_datasets: list[AnalystDataset] = []
    for dataset in datasets:
        try:
            df_records = cast(
                list[dict[str, Any]],
                dataset.get_as_dataframe().to_dict(orient="records"),
            )
            result_datasets.append(AnalystDataset(name=dataset.name, data=df_records))
            logger.info(f"Successfully downloaded {dataset.name}")
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset.name}: {str(e)}")
            continue
    return result_datasets


@cache
async def _get_dictionary_batch(
    columns: list[str], df: pd.DataFrame, batch_size: int = 5
) -> list[DataDictionaryColumn]:
    """Process a batch of columns to get their descriptions"""

    # Get sample data and stats for just these columns
    # Convert timestamps to ISO format strings for JSON serialization
    try:
        sample_data = {}
        for col in columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Convert timestamps to ISO format strings
                sample_data[col] = (
                    df[col]
                    .head(10)
                    .apply(lambda x: x.isoformat() if pd.notnull(x) else None)
                    .to_dict()
                )
            else:
                sample_data[col] = df[col].head(10).to_dict()

        # Handle numeric summary
        numeric_summary = {}
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                numeric_summary[col] = {
                    k: float(v) if pd.notnull(v) else None
                    for k, v in desc.to_dict().items()
                }

        # Get categories for non-numeric columns
        categories = []
        for column in columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                try:
                    value_counts = df[column].value_counts().head(10)
                    # Convert any timestamp values to strings
                    if pd.api.types.is_datetime64_any_dtype(df[column]):
                        value_counts.index = value_counts.index.map(
                            lambda x: x.isoformat() if pd.notnull(x) else None
                        )
                    categories.append({column: list(value_counts.keys())})
                except Exception:
                    continue

        # Create messages for OpenAI
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system", content=prompts.SYSTEM_PROMPT_GET_DICTIONARY
            ),
            ChatCompletionUserMessageParam(
                role="user", content=f"Data:\n{sample_data}\n"
            ),
            ChatCompletionUserMessageParam(
                role="user", content=f"Statistical Summary:\n{numeric_summary}\n"
            ),
        ]

        if categories:
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user", content=f"Categorical Values:\n{categories}\n"
                )
            )

        # Get descriptions from OpenAI
        completion: DictionaryGeneration = await client.chat.completions.create(
            response_model=DictionaryGeneration,
            model=ALTERNATIVE_LLM_SMALL,
            messages=messages,
        )

        # Convert to dictionary format
        descriptions = completion.to_dict()

        # Only return descriptions for requested columns
        return [
            DataDictionaryColumn(
                column=col,
                description=descriptions.get(col, "No description available"),
                data_type=str(df[col].dtype),
            )
            for col in columns
        ]

    except ValueError as e:
        logger.error(f"Invalid dictionary response: {str(e)}")
        return [
            DataDictionaryColumn(
                column=col,
                description="No valid description available",
                data_type=str(df[col].dtype),
            )
            for col in columns
        ]


async def _get_dictionaries(dataset: AnalystDataset) -> DataDictionary:
    """Process a single dataset with parallel column batch processing"""

    try:
        # Convert JSON to DataFrame
        df = dataset.to_df()

        # Add debug logging
        logger.info(f"Processing dataset {dataset.name} with shape {df.shape}")

        # Handle empty dataset
        if df.empty:
            logger.warning(f"Dataset {dataset.name} is empty")
            return DataDictionary(
                name=dataset.name,
                column_descriptions=[],
            )

        # Split columns into batches
        column_batches = [
            list(df.columns[i : i + DICTIONARY_BATCH_SIZE])
            for i in range(0, len(df.columns), DICTIONARY_BATCH_SIZE)
        ]
        logger.info(
            f"Created {len(column_batches)} batches for {len(df.columns)} columns"
        )

        tasks = [
            _get_dictionary_batch(batch, df, DICTIONARY_BATCH_SIZE)
            for batch in column_batches
        ]

        results = await asyncio.gather(*tasks)
        dictionary = sum(results, [])

        logger.info(
            f"Created dictionary with {len(dictionary)} entries for dataset {dataset.name}"
        )

        return DataDictionary(
            name=dataset.name,
            column_descriptions=dictionary,
        )

    except Exception as e:
        raise Exception(f"Error processing dataset {dataset.name}: {str(e)}")


def _validate_question_feasibility(
    question: str, available_columns: list[str]
) -> ValidatedQuestion | None:
    """Validate if a question can be answered with available data

    Checks if common data elements mentioned in the question exist in columns
    """
    # Convert question and columns to lowercase for matching
    question_lower = question.lower()
    columns_lower = [col.lower() for col in available_columns]

    # Extract potential column references from question
    words = set(re.findall(r"\b\w+\b", question_lower))

    # Find matches and missing terms
    found_columns = [col for col in columns_lower if any(word in col for word in words)]

    is_valid = len(found_columns) > 0
    if is_valid:
        return ValidatedQuestion(
            question=question,
        )
    return None


async def suggest_questions(
    datasets: list[AnalystDataset], max_columns: int = 40
) -> list[ValidatedQuestion]:
    """Generate and validate suggested analysis questions

    Args:
        dictionary: DataFrame containing data dictionary
        max_columns: Maximum number of columns to include in prompt

    Returns:
        Dict containing:
            - questions: List of validated question objects
            - metadata: Dictionary of processing information
    """
    # Validate input
    dictionary = sum(
        [
            DataDictionary.from_analyst_df(
                ds.to_df(),
                column_descriptions=f"Column from dataset {ds.name}",
            ).column_descriptions
            for ds in datasets
        ],
        [],
    )

    if len(dictionary) < 1:
        raise ValueError("Dictionary DataFrame cannot be empty")

    # Limit columns for OpenAI prompt
    total_columns = len(dictionary)
    if total_columns > max_columns:
        # Take first and last 20 columns
        half_max = max_columns // 2
        first_half = dictionary[:half_max]
        last_half = dictionary[-half_max:]

        # Remove any duplicates
        dictionary = first_half + last_half

        # deduplicate
        dictionary = list({item.column: item for item in dictionary}.values())

    # Convert dictionary to format expected by OpenAI
    dict_data = {
        "columns": [d.column for d in dictionary],
        "descriptions": [d.description for d in dictionary],
        "data_types": [d.data_type for d in dictionary],
    }

    # Create OpenAI messages
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=prompts.SYSTEM_PROMPT_SUGGEST_A_QUESTION
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data Dictionary:\n{json.dumps(dict_data)}"
        ),
    ]

    completion: QuestionListGeneration = await client.chat.completions.create(
        response_model=QuestionListGeneration,
        model=ALTERNATIVE_LLM_SMALL,
        messages=messages,
    )

    available_columns = dict_data["columns"]
    validated_questions: list[ValidatedQuestion] = []

    for question in completion.questions:
        validated_question = _validate_question_feasibility(question, available_columns)
        if validated_question is not None:
            validated_questions.append(validated_question)

    return validated_questions


def find_imports(module: Any) -> list[str]:
    """
    Get top-level third-party imports from a Python module.

    Args:
        module: Python module object to analyze

    Returns:
        List of third-party package names

    Example:
        >>> import my_module
        >>> imports = find_third_party_imports(my_module)
        >>> print(imports)  # ['pandas', 'numpy', 'requests']
    """
    # Get the source code of the module
    source = inspect.getsource(module)
    tree = ast.parse(source)

    stdlib_modules = set(sys.stdlib_module_names)
    third_party = set()

    # Only look at top-level imports
    for node in tree.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                module_name = name.name.split(".")[0]
                if module_name not in stdlib_modules:
                    third_party.add(module_name)

        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            module_name = node.module.split(".")[0]
            if module_name not in stdlib_modules:
                third_party.add(module_name)

    return sorted(third_party)


def get_tools() -> list[Tool]:
    # find all functions defined in the tools module
    tool_functions = [func for func in dir(tools) if callable(getattr(tools, func))]

    # find the function signatures and doc strings
    tools_list = []
    for func_name in tool_functions:
        func = getattr(tools, func_name)
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)
        tools_list.append(
            Tool(
                name=func_name,
                signature=str(signature),
                docstring=docstring,
                function=func,
            )
        )
    return tools_list


async def _generate_run_charts_python_code(
    request: RunChartsRequest, validation_error: InvalidGeneratedCode | None = None
) -> str:
    df = request.dataset.to_df()
    question = request.question
    dataframe_metadata = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "statistics": df.describe(include="all").to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=prompts.SYSTEM_PROMPT_PLOTLY_CHART,
        ),
        ChatCompletionUserMessageParam(role="user", content=f"Question: {question}"),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data Metadata:\n{dataframe_metadata}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data top 25 rows:\n{df.head(25).to_string()}"
        ),
    ]
    if validation_error:
        msg = type(validation_error).__name__ + f": {str(validation_error)}"
        messages.extend(
            [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Previous attempt failed with error: {msg}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Failed code: {validation_error.code}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Please generate new code that avoids this error.",
                ),
            ]
        )

    # Get response based on model mode
    response: CodeGeneration = await client.chat.completions.create(
        response_model=CodeGeneration,
        model=ALTERNATIVE_LLM_BIG,
        temperature=0,
        messages=messages,
    )
    return response.code


async def _generate_run_analysis_python_code(
    request: RunAnalysisRequest,
    validation_error: InvalidGeneratedCode | None = None,
    use_tools: bool = False,
) -> str:
    """
    Generate Python analysis code based on JSON data and question.

    Parameters:
    - request: RunAnalysisRequest containing data and question
    - validation_errors: Past validation errors to include in prompt

    Returns:
    - Generated code
    """
    # Convert dictionary data structure to list of columns for all datasets
    all_columns = []
    all_descriptions = []
    all_data_types = []

    for dictionary in request.dictionaries:
        for entry in dictionary.column_descriptions:
            all_columns.append(f"{dictionary.name}.{entry.column}")
            all_descriptions.append(entry.description)
            all_data_types.append(entry.data_type)

    # Create dictionary format for prompt
    dictionary_data = {
        "columns": all_columns,
        "descriptions": all_descriptions,
        "data_types": all_data_types,
    }

    # Get sample data and shape info for all datasets
    all_samples = []
    all_shapes = []

    for dataset in request.datasets:
        df = dataset.to_df()
        all_shapes.append(f"{dataset.name}: {df.shape[0]} rows x {df.shape[1]} columns")
        # Limit sample to 10 rows
        sample_df = df.head(10)
        all_samples.append(f"{dataset.name}:\n{sample_df.to_string()}")

    shape_info = "\n".join(all_shapes)
    sample_data = "\n\n".join(all_samples)

    # Create messages for OpenAI
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=prompts.SYSTEM_PROMPT_PYTHON_ANALYST
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Business Question: {request.question}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data Shapes:\n{shape_info}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Sample Data:\n{sample_data}"
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Data Dictionary:\n{json.dumps(dictionary_data)}",
        ),
    ]
    if use_tools:
        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content="If it helps the analysis, you can optionally use following functions:\n"
                + "\n".join([str(t) for t in get_tools()]),
            )
        )

    # Add error context if available
    if validation_error:
        msg = type(validation_error).__name__ + f": {str(validation_error)}"
        messages.extend(
            [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Previous attempt failed with error: {msg}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Failed code: {validation_error.code}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Please generate new code that avoids this error.",
                ),
            ]
        )

    completion: CodeGeneration = await client.chat.completions.create(
        response_model=CodeGeneration,
        model=ALTERNATIVE_LLM_BIG,
        temperature=0.1,
        messages=messages,
        max_retries=10,
    )

    return completion.code


async def cleanse_dataframes(
    datasets: list[AnalystDataset],
) -> list[CleansedDataset]:
    """Clean and standardize multiple pandas DataFrames.

    Args:
        datasets: List of AnalystDataset objects to clean

    Returns:
        List of CleansedDataset objects containing cleaned data and reports

    Raises:
        ValueError: If a dataset is empty
    """
    cleaned_datasets: list[CleansedDataset] = []

    for dataset in datasets:
        report: list[CleansedColumnReport] = []
        cleaned_df: DataFrame = dataset.to_df()

        sample_df = cleaned_df.sample(min(100, len(cleaned_df)))
        if cleaned_df.empty:
            raise ValueError(f"Dataset {dataset.name} is empty")

        dtypes = cleaned_df.dtypes.to_dict()
        for column_name in cleaned_df.columns:
            # rename column:
            cleaned_column_name = re.sub(r"\s+", " ", str(column_name).strip())
            original_nulls = sample_df[column_name].isna()

            column_report = CleansedColumnReport(new_column_name=cleaned_column_name)
            if cleaned_column_name != column_name:
                column_report.original_column_name = column_name
                column_report.warnings.append(
                    f"Column renamed from '{column_name}' to '{cleaned_column_name}'"
                )

            if dtypes[column_name] == "object":
                column_report.original_dtype = "object"

                conversions = [
                    ("simple_clean", try_simple_numeric_conversion),
                    ("unit_conversion", try_unit_conversion),
                    ("datetime", try_datetime_conversion),
                ]
                try:
                    for conversion_type, conversion_func in conversions:
                        success, cleaned_series, warnings = conversion_func(
                            cleaned_df[column_name],
                            sample_df[column_name],
                            original_nulls,
                        )

                        # Add any warnings from the attempt
                        column_report.warnings.extend(warnings)

                        if success:
                            cleaned_df[column_name] = cleaned_series
                            column_report.new_dtype = str(cleaned_series.dtype)
                            column_report.conversion_type = conversion_type
                            break
                except Exception as e:
                    column_report.errors.append(str(e))

            cleaned_df = cleaned_df.rename(columns={column_name: cleaned_column_name})
            report.append(column_report)

        add_summary_statistics(cleaned_df, report)

        cleaned_datasets.append(
            CleansedDataset(
                dataset=AnalystDataset(
                    name=dataset.name,
                    data=cleaned_df,
                ),
                cleaning_report=report,
            )
        )
    return cleaned_datasets


async def get_dictionaries(datasets: list[AnalystDataset]) -> list[DataDictionary]:
    """
    Generate data dictionary for multiple datasets.

    Parameters:
    - datasets: list[AnalystDataset] containing datasets

    Returns:
    - Dictionary containing column descriptions and metadata
    """

    try:
        # Add debug logging
        logger.info(f"Received dictionary request with {len(datasets)} datasets")

        tasks = [_get_dictionaries(dataset) for dataset in datasets]

        results = await asyncio.gather(*tasks)

        logger.info(f"Returning dictionary response with {len(results)} results")
        return results

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in get_dictionary: {msg}")
        return [
            DataDictionary(
                name=dataset.name,
                column_descriptions=[
                    DataDictionaryColumn(
                        column=c,
                        data_type=str(dataset.to_df()[c].dtype),
                        description="No Description Available",
                    )
                    for c in dataset.columns
                ],
            )
            for dataset in datasets
        ]


async def rephrase_message(messages: ChatRequest) -> str:
    """Process chat messages history and return a new question

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields

    Returns:
        Dict[str, str]: Dictionary containing response content
    """
    # Convert messages to string format for prompt
    messages_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in messages.messages]
    )

    prompt_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            content=prompts.SYSTEM_PROMPT_REPHRASE_MESSAGE,
            role="system",
        ),
        ChatCompletionUserMessageParam(
            content=f"Message History:\n{messages_str}",
            role="user",
        ),
    ]

    completion: EnhancedQuestionGeneration = await client.chat.completions.create(
        response_model=EnhancedQuestionGeneration,
        model=ALTERNATIVE_LLM_BIG,
        messages=prompt_messages,
    )

    return completion.enhanced_user_message


@reflect_code_generation_errors(max_attempts=7)
async def _run_charts(
    request: RunChartsRequest,
    exception_history: list[InvalidGeneratedCode] | None = None,
) -> RunChartsResult:
    """Generate and validate chart code with retry logic"""
    # Create messages for OpenAI
    start_time = datetime.now()

    if not request.dataset:
        raise ValueError("Input data cannot be empty")

    df = request.dataset.to_df()
    if exception_history is None:
        exception_history = []

    code = await _generate_run_charts_python_code(
        request, next(iter(exception_history), None)
    )
    try:
        result = execute_python(
            modules={
                "pd": pd,
                "np": np,
                "go": go,
                "scipy": scipy,
            },
            functions={
                "make_subplots": make_subplots,
            },
            expected_function="create_charts",
            code=code,
            input_data=df,
            output_type=ChartGenerationExecutionResult,
            allowed_modules={"pandas", "numpy", "plotly", "scipy", "datetime"},
        )
    except InvalidGeneratedCode:
        raise
    except Exception as e:
        raise InvalidGeneratedCode(code=code, exception=e)

    duration = datetime.now() - start_time

    return RunChartsResult(
        status="success",
        code=code,
        fig1_json=result.fig1.to_json(),
        fig2_json=result.fig2.to_json(),
        metadata=RunAnalysisResultMetadata(
            duration=duration.total_seconds(),
            attempts=len(exception_history) + 1,
        ),
    )


async def run_charts(request: RunChartsRequest) -> RunChartsResult:
    """Execute analysis workflow on datasets."""
    try:
        chart_result = await _run_charts(request)
        return chart_result
    except ValidationError:
        return RunChartsResult(
            status="error", metadata=RunAnalysisResultMetadata(duration=0, attempts=1)
        )
    except MaxReflectionAttempts as e:
        return RunChartsResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )


async def get_business_analysis(
    request: GetBusinessAnalysisRequest,
) -> GetBusinessAnalysisResult:
    """
    Generate business analysis based on data and question.

    Parameters:
    - request: BusinessAnalysisRequest containing data and question

    Returns:
    - Dictionary containing analysis components
    """
    try:
        # Convert JSON data to DataFrame for analysis
        start = datetime.now()

        df = request.dataset.to_df()

        # Get first 1000 rows as CSV with quoted values for context
        df_csv = df.head(750).to_csv(index=False, quoting=1)

        # Create messages for OpenAI
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system", content=prompts.SYSTEM_PROMPT_BUSINESS_ANALYSIS
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Business Question: {request.question}",
            ),
            ChatCompletionUserMessageParam(
                role="user", content=f"Analyzed Data:\n{df_csv}"
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Data Dictionary:\n{request.dictionary.model_dump_json()}",
            ),
        ]

        completion: BusinessAnalysisGeneration = await client.chat.completions.create(
            response_model=BusinessAnalysisGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0.1,
            messages=messages,
        )
        duration = (datetime.now() - start).total_seconds()
        # Ensure all response fields are present
        metadata = GetBusinessAnalysisMetadata(
            duration=duration,
            question=request.question,
            rows_analyzed=len(df),
            columns_analyzed=len(df.columns),
        )
        return GetBusinessAnalysisResult(
            status="success",
            **completion.model_dump(),
            metadata=metadata,
        )

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in get_business_analysis: {msg}")
        return GetBusinessAnalysisResult(
            status="error",
            metadata=GetBusinessAnalysisMetadata(exception_str=msg),
            additional_insights="",
            follow_up_questions=[],
            bottom_line="",
        )


@reflect_code_generation_errors(max_attempts=7)
async def _run_analysis(
    request: RunAnalysisRequest,
    exception_history: list[InvalidGeneratedCode] | None = None,
    use_tools: bool = True,
) -> RunAnalysisResult:
    start_time = datetime.now()
    if not request.datasets:
        raise ValueError("Input data cannot be empty")

    if exception_history is None:
        exception_history = []

    code = await _generate_run_analysis_python_code(
        request, next(iter(exception_history), None), use_tools
    )

    dataframes: dict[str, pd.DataFrame] = {}
    for dataset in request.datasets:
        dataframes[dataset.name] = dataset.to_df()
    functions = {}
    if use_tools:
        for tool in get_tools():
            functions[tool.name] = tool.function
    try:
        result = execute_python(
            modules={
                "pd": pd,
                "np": np,
                "sm": sm,
                "scipy": scipy,
                "sklearn": sklearn,
            },
            functions=functions,
            expected_function="analyze_data",
            code=code,
            input_data=dataframes,
            output_type=AnalystDataset,
            allowed_modules={
                "pandas",
                "numpy",
                "scipy",
                "sklearn",
                "statsmodels",
                "datetime",
                *find_imports(tools),
            },
        )
    except InvalidGeneratedCode:
        raise
    except Exception as e:
        raise InvalidGeneratedCode(code=code, exception=e)

    duration = datetime.now() - start_time
    return RunAnalysisResult(
        status="success",
        code=code,
        dataset=result,
        metadata=RunAnalysisResultMetadata(
            duration=duration.total_seconds(),
            attempts=len(exception_history) + 1,
            datasets_analyzed=len(dataframes),
            total_rows_analyzed=sum(
                len(df) for df in dataframes.values() if not df.empty
            ),
            total_columns_analyzed=sum(
                len(df.columns) for df in dataframes.values() if not df.empty
            ),
        ),
    )


async def run_analysis(request: RunAnalysisRequest) -> RunAnalysisResult:
    """Execute analysis workflow on datasets."""
    try:
        return await _run_analysis(request)
    except MaxReflectionAttempts as e:
        return RunAnalysisResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )


async def _generate_database_analysis_code(
    request: RunDatabaseAnalysisRequest,
    validation_error: InvalidGeneratedCode | None = None,
) -> str:
    """
    Generate Snowflake SQL analysis code based on data samples and question.

    Parameters:
    - request: DatabaseAnalysisRequest containing data samples and question

    Returns:
    - Dictionary containing generated code and description
    """

    # Convert dictionary data structure to list of columns for all tables
    all_tables_info = [d.model_dump(mode="json") for d in request.dictionaries]

    # Get sample data for all tables
    all_samples = []
    for table in request.datasets:
        df = table.to_df()
        sample_str = f"Table: {table.name}\n{df.head(10).to_string()}"
        all_samples.append(sample_str)

    # Create messages for OpenAI
    messages: list[ChatCompletionMessageParam] = [
        Database.get_system_prompt(),
        ChatCompletionUserMessageParam(
            content=f"Business Question: {request.question}",
            role="user",
        ),
        ChatCompletionUserMessageParam(
            content=f"Sample Data:\n{chr(10).join(all_samples)}", role="user"
        ),
        ChatCompletionUserMessageParam(
            content=f"Data Dictionary:\n{json.dumps(all_tables_info)}", role="user"
        ),
    ]
    if validation_error:
        msg = type(validation_error).__name__ + f": {str(validation_error)}"
        messages.extend(
            [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Previous attempt failed with error: {msg}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Failed code: {validation_error.code}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Please generate new code that avoids this error.",
                ),
            ]
        )

    # Get response from OpenAI
    completion = await client.chat.completions.create(
        response_model=DatabaseAnalysisCodeGeneration,
        model=ALTERNATIVE_LLM_BIG,
        temperature=0.1,
        messages=messages,
    )

    return completion.code


@reflect_code_generation_errors(max_attempts=7)
async def _run_database_analysis(
    request: RunDatabaseAnalysisRequest,
    exception_history: list[InvalidGeneratedCode] | None = None,
) -> RunDatabaseAnalysisResult:
    start_time = datetime.now()
    if not request.datasets:
        raise ValueError("Input data cannot be empty")

    if exception_history is None:
        exception_history = []

    sql_code = await _generate_database_analysis_code(
        request, next(iter(exception_history), None)
    )
    try:
        results = Database.execute_query(query=sql_code)
        results = cast(list[dict[str, Any]], results)
        duration = datetime.now() - start_time

    except InvalidGeneratedCode:
        raise
    except Exception as e:
        raise InvalidGeneratedCode(code=sql_code, exception=e)
    return RunDatabaseAnalysisResult(
        status="success",
        code=sql_code,
        dataset=AnalystDataset(
            data=results,
        ),
        metadata=RunDatabaseAnalysisResultMetadata(
            duration=duration.total_seconds(),
            attempts=len(exception_history),
            datasets_analyzed=len(request.datasets),
            total_columns_analyzed=sum(len(ds.columns) for ds in request.datasets),
        ),
    )


async def run_database_analysis(
    request: RunDatabaseAnalysisRequest,
) -> RunDatabaseAnalysisResult:
    """Execute analysis workflow on datasets."""
    try:
        return await _run_database_analysis(request)
    except MaxReflectionAttempts as e:
        return RunDatabaseAnalysisResult(
            status="error",
            metadata=RunDatabaseAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )

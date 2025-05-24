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
import inspect
import json
import logging
import re
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from types import ModuleType, TracebackType
from typing import Any, AsyncGenerator, Type, TypeVar, cast

import datarobot as dr
import instructor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import requests
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
from plotly.subplots import make_subplots
from pydantic import ValidationError

sys.path.append("..")
from utils import prompts, tools
from utils.analyst_db import AnalystDB, DataSourceType
from utils.code_execution import (
    InvalidGeneratedCode,
    MaxReflectionAttempts,
    execute_python,
    reflect_code_generation_errors,
)
from utils.data_cleansing_helpers import add_summary_statistics, process_column
from utils.database_helpers import get_external_database
from utils.dr_helper import async_submit_actuals_to_datarobot, initialize_deployment
from utils.logging_helper import get_logger, log_api_call
from utils.schema import (
    AnalysisError,
    AnalystChatMessage,
    AnalystDataset,
    BusinessAnalysisGeneration,
    ChartGenerationExecutionResult,
    ChatRequest,
    CleansedDataset,
    CodeGeneration,
    Component,
    DatabaseAnalysisCodeGeneration,
    DataDictionary,
    DataDictionaryColumn,
    DataRegistryDataset,
    DictionaryGeneration,
    DownloadedRegistryDataset,
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

logger = get_logger()
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.http_client").setLevel(logging.WARNING)


def log_memory() -> None:
    process = psutil.Process()
    memory = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"Memory usage: {memory:.2f} MB")


class AsyncLLMClient:
    async def __aenter__(self) -> instructor.AsyncInstructor:
        dr_client, deployment_chat_base_url = initialize_deployment()
        self.openai_client = AsyncOpenAI(
            api_key=dr_client.token,
            base_url=deployment_chat_base_url,
            timeout=90,
            max_retries=2,
        )
        self.client = instructor.from_openai(
            self.openai_client, mode=instructor.Mode.MD_JSON
        )
        logger.info(f"Initialized witn deployment url: {deployment_chat_base_url}")
        return self.client

    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.openai_client.close()  # Properly close the client


ALTERNATIVE_LLM_BIG = "datarobot-deployed-llm"
ALTERNATIVE_LLM_SMALL = "datarobot-deployed-llm"
DICTIONARY_BATCH_SIZE = 10
MAX_REGISTRY_DATASET_SIZE = 400e6  # aligns to 400MB set in streamlit config.toml
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
                    f"Using previously cached result for function `{f.__name__}`"  # type: ignore[attr-defined]
                )
            return result

        return cast(T, wrapper)


# This can be large as we are not storing the actual datasets in memory, just metadata
def list_registry_datasets(token: str, limit: int = 100) -> list[DataRegistryDataset]:
    """
    Fetch datasets from Data Registry with specified limit

    Args:
        limit: int
        Datasets to retrieve. Max value: 100
    """

    url = f"{dr.client.get_client().endpoint}/datasets?limit={limit}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Get all datasets and manually limit the results
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    datasets = response.json()["data"]

    return [
        DataRegistryDataset(
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


def get_registry_dataset_info(dataset_id: str, token: str) -> tuple[str, int]:
    """
    Fetch a dataset's name and size from DataRobot API using requests.

    Args:
        dataset_id: The ID of the dataset to fetch.
        token: The DataRobot API token.
    Returns:
        Tuple of (name, size in bytes)
    Raises:
        Exception if the request fails or the response is invalid.
    """
    base_url = dr.client.get_client().endpoint
    url = f"{base_url}/datasets/{dataset_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    name = data.get("name", "")
    size = data.get("datasetSize", 0)
    return name, size


def download_registry_dataset_as_dataframe(token: str, dataset_id: str) -> pd.DataFrame:
    """
    Download a dataset from DataRobot as a pandas DataFrame using requests.

    Args:
        token: The DataRobot API token.
        dataset_id: The ID of the dataset to download.
    Returns:
        DataFrame containing the dataset.
    Raises:
        Exception if the request fails or the response is invalid.
    """
    base_url = dr.client.get_client().endpoint
    url = f"{base_url}/datasets/{dataset_id}/file/"
    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "*/*",
    }
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    csv_content = response.content.decode("utf-8")
    df = pd.read_csv(StringIO(csv_content))
    return df


async def download_registry_datasets(
    dataset_ids: list[str], analyst_db: AnalystDB, token: str
) -> list[DownloadedRegistryDataset]:
    """Load selected datasets as pandas DataFrames

    Args:
        *args: list of dataset IDs to download

    Returns:
        list[AnalystDataset]: Dictionary of dataset names and data
    """
    downloaded_datasets = []

    # Use requests to get dataset info instead of dr.Dataset
    total_size = 0
    dataset_infos = []
    for id_ in dataset_ids:
        try:
            name, size = get_registry_dataset_info(id_, token)
            total_size += size if size is not None else 0
            dataset_infos.append((id_, name, size))
        except Exception as e:
            logger.error(f"Failed to fetch dataset info for {id_}: {str(e)}")
            downloaded_datasets.append(
                DownloadedRegistryDataset(name=id_, error=str(e))
            )
    if total_size > MAX_REGISTRY_DATASET_SIZE:
        raise ValueError(
            f"The requested Data Registry datasets must total <= {int(MAX_REGISTRY_DATASET_SIZE)} bytes"
        )

    result_datasets: list[AnalystDataset] = []
    for id_, name, size in dataset_infos:
        try:
            # Use requests to download the dataset as DataFrame
            df = download_registry_dataset_as_dataframe(token, id_)
            df_records = cast(
                list[dict[str, Any]],
                df.to_dict(orient="records"),
            )
            result_datasets.append(AnalystDataset(name=name, data=df_records))
            logger.info(f"Successfully downloaded {name}")
        except Exception as e:
            logger.error(f"Failed to read dataset {name}: {str(e)}")
            downloaded_datasets.append(
                DownloadedRegistryDataset(name=name, error=str(e))
            )
            continue
    for result_dataset in result_datasets:
        await analyst_db.register_dataset(
            result_dataset, DataSourceType.REGISTRY, size or 0
        )
        downloaded_datasets.append(DownloadedRegistryDataset(name=result_dataset.name))
    return downloaded_datasets


async def _get_dictionary_batch(
    columns: list[str],
    df: pd.DataFrame,
    batch_size: int = 5,
    telemetry_json: dict[str, Any] | None = None,
) -> list[DataDictionaryColumn]:
    """Process a batch of columns to get their descriptions"""

    # Get sample data and stats for just these columns
    # Convert timestamps to ISO format strings for JSON serialization
    if telemetry_json is not None:
        telemetry_send = deepcopy(telemetry_json)
        telemetry_send["startTimestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        telemetry_send = None
    try:
        logger.debug(f"Processing batch of {len(columns)} columns")
        sample_data = {}
        logger.debug("Converting datetime columns to ISO format")
        num_samples = 10
        for col in columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Convert timestamps to ISO format strings
                sample_values = df[col].head(num_samples).copy()
                # NaT値を処理する
                sample_values = sample_values.apply(
                    lambda x: x.isoformat() if pd.notna(x) else None
                )
                sample_data[col] = sample_values.to_list()
            else:
                # For non-datetime columns, just take the samples as is
                sample_data[col] = df[col].head(num_samples).to_list()

        # Handle numeric summary
        numeric_summary = {}
        logger.debug("Calculating numeric summaries")
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                desc = (
                    df[col]
                    .describe()
                    .reset_index()
                    .rename(columns={"index": "statistic", col: "value"})
                )
                numeric_summary[col] = desc.to_dict("list")

        # Get categories for non-numeric columns
        categories = []
        logger.debug("Getting categories for non-numeric columns")
        for column in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    # サンプルを取得してvalue_countsを計算
                    if len(df) > 1000:
                        sample = df[column].sample(n=1000, random_state=42)
                    else:
                        sample = df[column]

                    value_counts = sample.value_counts().head(10).reset_index()

                    # Convert any timestamp values to strings
                    if pd.api.types.is_datetime64_any_dtype(df[column]):
                        value_counts["index"] = value_counts["index"].map(
                            lambda x: x.isoformat() if pd.notna(x) else None
                        )
                    categories.append({column: value_counts["index"].tolist()})
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
        logger.debug(
            f"total_characters: {len(''.join([str(msg) for msg in messages]))}"
        )
        # Get descriptions from OpenAI
        async with AsyncLLMClient() as client:
            (
                completion,
                completion_org,
            ) = await client.chat.completions.create_with_completion(
                response_model=DictionaryGeneration,
                model=ALTERNATIVE_LLM_SMALL,
                messages=messages,
            )

        # Convert to dictionary format
        descriptions = completion.to_dict()
        association_id = completion_org.datarobot_moderations["association_id"]
        logger.info(f"Association ID: {association_id}")

        if telemetry_send is not None:
            # query type added in parent function
            # submit telemetry
            asyncio.create_task(
                async_submit_actuals_to_datarobot(
                    association_id=association_id, telemetry_json=telemetry_send
                )
            )

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


@log_api_call
async def get_dictionary(
    dataset: AnalystDataset, telemetry_json: dict[str, Any] | None = None
) -> DataDictionary:
    """Process a single dataset with parallel column batch processing"""

    try:
        logger.info(f"Processing dataset {dataset.name} init")
        # Convert JSON to DataFrame
        df_full = dataset.to_df()
        df = df_full.sample(n=min(10000, len(df_full)), random_state=42)

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
            _get_dictionary_batch(batch, df, DICTIONARY_BATCH_SIZE, telemetry_json)
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

    except Exception:
        return DataDictionary(
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


@log_api_call
async def suggest_questions(
    datasets: list[AnalystDataset],
    max_columns: int = 40,
    telemetry_json: dict[str, Any] | None = None,
) -> list[ValidatedQuestion]:
    """Generate and validate suggested analysis questions

    Args:
        dictionary: DataFrame containing data dictionary
        max_columns: Maximum number of columns to include in prompt

    Returns:
        Dict containing:
            - questions: list of validated question objects
            - metadata: Dictionary of processing information
    """
    if telemetry_json is not None:
        telemetry_send = deepcopy(telemetry_json)
        telemetry_send["startTimestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        telemetry_send = None
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
            role="user",
            content=f"Data Dictionary:\n{json.dumps(dict_data, ensure_ascii=False)}",
        ),
    ]
    async with AsyncLLMClient() as client:
        (
            completion,
            completion_org,
        ) = await client.chat.completions.create_with_completion(
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

    association_id = completion_org.datarobot_moderations["association_id"]
    logger.info(f"Association ID: {association_id}")

    if telemetry_send is not None:
        # add query type to telemetry
        telemetry_send["query_type"] = "01_generate_suggested_questions"
        # submit telemetry
        asyncio.create_task(
            async_submit_actuals_to_datarobot(
                association_id=association_id, telemetry_json=telemetry_send
            )
        )
    return validated_questions


def find_imports(module: ModuleType) -> list[str]:
    """
    Get top-level third-party imports from a Python module.

    Args:
        module: Python module object to analyze

    Returns:
        list of third-party package names

    Example:
        >>> import my_module
        >>> imports = find_third_party_imports(my_module)
        >>> print(imports)  # ['pandas', 'numpy', 'requests']
    """
    try:
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
    except Exception:
        return []


def get_tools() -> list[Tool]:
    try:
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
    except Exception:
        return []


async def _generate_run_charts_python_code(
    request: RunChartsRequest,
    validation_error: InvalidGeneratedCode | None = None,
    telemetry_json: dict[str, Any] | None = None,
) -> str:
    if telemetry_json is not None:
        telemetry_send = deepcopy(telemetry_json)
        telemetry_send["startTimestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        telemetry_send = None
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
    async with AsyncLLMClient() as client:
        response, response_org = await client.chat.completions.create_with_completion(
            response_model=CodeGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0,
            messages=messages,
        )
    association_id = response_org.datarobot_moderations["association_id"]
    logger.info(f"Association ID: {association_id}")
    if telemetry_send is not None:
        # add query type to telemetry
        telemetry_send["query_type"] = "04_generate_run_charts_python_code"
        # submit telemetry
        asyncio.create_task(
            async_submit_actuals_to_datarobot(
                association_id=association_id, telemetry_json=telemetry_send
            )
        )
    return response.code


async def _generate_run_analysis_python_code(
    request: RunAnalysisRequest,
    analyst_db: AnalystDB,
    validation_error: InvalidGeneratedCode | None = None,
    attempt: int = 0,
    telemetry_json: dict[str, Any] | None = None,
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
    logger.info("Starting code gen")
    if telemetry_json is not None:
        telemetry_send = deepcopy(telemetry_json)
        telemetry_send["startTimestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        telemetry_send = None

    all_columns = []
    all_descriptions = []
    all_data_types = []

    dictionaries = [
        await analyst_db.get_data_dictionary(name) for name in request.dataset_names
    ]
    for dictionary in dictionaries:
        if dictionary is None:
            continue
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

    logger.debug(f"datasets: {request.dataset_names}")
    for dataset_name in request.dataset_names:
        try:
            dataset = (await analyst_db.get_cleansed_dataset(dataset_name)).to_df()
        except Exception:
            dataset = (await analyst_db.get_dataset(dataset_name)).to_df()
        all_shapes.append(
            f"{dataset_name}: {dataset.shape[0]} rows x {dataset.shape[1]} columns"
        )
        # Limit sample to 10 rows
        sample_df = dataset.head(10)
        all_samples.append(f"{dataset_name}:\n{sample_df}")

    shape_info = "\n".join(all_shapes)
    sample_data = "\n\n".join(all_samples)
    logger.debug("Assembling messages")
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
            content=f"Data Dictionary:\n{json.dumps(dictionary_data, ensure_ascii=False)}",
        ),
    ]

    tools_list = get_tools()
    if len(tools_list) > 0:
        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content="If it helps the analysis, you can optionally use following functions:\n"
                + "\n".join([str(t) for t in tools_list]),
            )
        )

    logger.debug(f"total_characters: {len(''.join([str(msg) for msg in messages]))}")
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
        if attempt > 2:
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Convert the dataframe to pandas!",
                )
            )
    logger.info("Running Code Gen")
    logger.debug(messages)
    async with AsyncLLMClient() as client:
        (
            completion,
            completion_org,
        ) = await client.chat.completions.create_with_completion(
            response_model=CodeGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0.1,
            messages=messages,
            max_retries=10,
        )
    association_id = completion_org.datarobot_moderations["association_id"]
    logger.info(f"Association ID: {association_id}")

    if telemetry_send is not None:
        # add query type to telemetry
        telemetry_send["query_type"] = "03_generate_code_file"
        # submit telemetry
        asyncio.create_task(
            async_submit_actuals_to_datarobot(
                association_id=association_id, telemetry_json=telemetry_send
            )
        )
    logger.info("Code Gen complete")
    return completion.code


async def cleanse_dataframe(dataset: AnalystDataset) -> CleansedDataset:
    """Clean and standardize multiple pandas DataFrames in parallel.

    Args:
        datasets: List of AnalystDataset objects to clean
    Returns:
        List of CleansedDataset objects containing cleaned data and reports
    Raises:
        ValueError: If a dataset is empty
    """

    if dataset.to_df().empty:
        raise ValueError(f"Dataset {dataset.name} is empty")

    df = dataset.to_df()
    sample_df = df.sample(n=min(100, len(df)), random_state=42)

    results = []
    for col in df.columns:
        results.append(process_column(df, col, sample_df))

    # Create new DataFrame from processed columns
    new_columns = {}
    reports = []

    for new_name, series, report in results:
        new_columns[new_name] = series
        reports.append(report)

    cleaned_df = pd.DataFrame(new_columns)
    add_summary_statistics(cleaned_df, reports)

    return CleansedDataset(
        dataset=AnalystDataset(
            name=dataset.name,
            data=cleaned_df,
        ),
        cleaning_report=reports,
    )


@log_api_call
async def rephrase_message(
    messages: ChatRequest, telemetry_json: dict[str, Any] | None = None
) -> str:
    """Process chat messages history and return a new question

    Args:
        messages: list of message dictionaries with 'role' and 'content' fields

    Returns:
        Dict[str, str]: Dictionary containing response content
    """
    if telemetry_json is not None:
        telemetry_send = deepcopy(telemetry_json)
        telemetry_send["startTimestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        telemetry_send = None


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
    async with AsyncLLMClient() as client:
        (
            completion,
            completion_org,
        ) = await client.chat.completions.create_with_completion(
            response_model=EnhancedQuestionGeneration,
            model=ALTERNATIVE_LLM_BIG,
            messages=prompt_messages,
        )
    association_id = completion_org.datarobot_moderations["association_id"]
    logger.info(f"Association ID: {association_id}")

    if telemetry_send is not None:
        # add query type to telemetry
        telemetry_send["query_type"] = "02_rephrase"
        # submit telemetry
        asyncio.create_task(
            async_submit_actuals_to_datarobot(
                association_id=association_id, telemetry_json=telemetry_send
            )
        )
    return completion.enhanced_user_message


@reflect_code_generation_errors(max_attempts=7)
async def _run_charts(
    request: RunChartsRequest,
    exception_history: list[InvalidGeneratedCode] | None = None,
    telemetry_json: dict[str, Any] | None = None,
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
        request,
        next(iter(exception_history[::-1]), None),
        telemetry_json=telemetry_json,
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
            allowed_modules={
                "pandas",
                "numpy",
                "plotly",
                "scipy",
                "datetime",
            },
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


@log_api_call
async def run_charts(
    request: RunChartsRequest, telemetry_json: dict[str, Any] | None = None
) -> RunChartsResult:
    """Execute analysis workflow on datasets."""
    try:
        chart_result = await _run_charts(request, telemetry_json=telemetry_json)
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


@log_api_call
async def get_business_analysis(
    request: GetBusinessAnalysisRequest,
    telemetry_json: dict[str, Any] | None = None,
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
        if telemetry_json is not None:
            telemetry_send = deepcopy(telemetry_json)
            telemetry_send["startTimestamp"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            telemetry_send = None
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
        async with AsyncLLMClient() as client:
            (
                completion,
                completion_org,
            ) = await client.chat.completions.create_with_completion(
                response_model=BusinessAnalysisGeneration,
                model=ALTERNATIVE_LLM_BIG,
                temperature=0.1,
                messages=messages,
            )
        association_id = completion_org.datarobot_moderations["association_id"]
        logger.info(f"Association ID: {association_id}")

        if telemetry_send is not None:
            # add query type to telemetry
            # although it's called the same time as 04, change number for clarity
            telemetry_send["query_type"] = "05_generate_business_analysis"
            # submit telemetry
            asyncio.create_task(
                async_submit_actuals_to_datarobot(
                    association_id=association_id, telemetry_json=telemetry_send
                )
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
    analyst_db: AnalystDB,
    exception_history: list[InvalidGeneratedCode] | None = None,
    telemetry_json: dict[str, Any] | None = None,
) -> RunAnalysisResult:
    start_time = datetime.now()

    if not request.dataset_names:
        raise ValueError("Input data cannot be empty")

    if exception_history is None:
        exception_history = []
    logger.info(f"Running analysis (attempt {len(exception_history)})")
    code = await _generate_run_analysis_python_code(
        request,
        analyst_db,
        next(iter(exception_history[::-1]), None),
        attempt=len(exception_history),
        telemetry_json=telemetry_json,
    )
    logger.info("Code generated, preparing execution")
    dataframes: dict[str, pd.DataFrame] = {}

    for dataset_name in request.dataset_names:
        try:
            dataset = (
                await analyst_db.get_cleansed_dataset(dataset_name, max_rows=None)
            ).to_df()
        except Exception:
            dataset = (
                await analyst_db.get_dataset(dataset_name, max_rows=None)
            ).to_df()
        dataframes[dataset_name] = dataset
    functions = {}
    tool_functions = get_tools()
    for tool in tool_functions:
        functions[tool.name] = tool.function
    try:
        logger.info("Executing")
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
    logger.info("Execution done")
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


@log_api_call
async def run_analysis(
    request: RunAnalysisRequest,
    analyst_db: AnalystDB,
    telemetry_json: dict[str, Any],
) -> RunAnalysisResult:
    """Execute analysis workflow on datasets."""
    logger.debug("Entering run_analysis")
    log_memory()
    try:
        return await _run_analysis(
            request, analyst_db=analyst_db, telemetry_json=telemetry_json
        )
    except MaxReflectionAttempts as e:
        return RunAnalysisResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )
    except ValueError as e:
        return RunAnalysisResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=0,
                attempts=1,
                exception=AnalysisError.from_value_error(e),
            ),
        )


async def _generate_database_analysis_code(
    request: RunDatabaseAnalysisRequest,
    analyst_db: AnalystDB,
    validation_error: InvalidGeneratedCode | None = None,
    telemetry_json: dict[str, Any] | None = None,
) -> str:
    """
    Generate Snowflake SQL analysis code based on data samples and question.

    Parameters:
    - request: DatabaseAnalysisRequest containing data samples and question

    Returns:
    - Dictionary containing generated code and description
    """
    if telemetry_json is not None:
        telemetry_send = deepcopy(telemetry_json)
        telemetry_send["startTimestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        telemetry_send = None

    # Convert dictionary data structure to list of columns for all tables
    dictionaries = [
        await analyst_db.get_data_dictionary(name) for name in request.dataset_names
    ]
    all_tables_info = [d.model_dump(mode="json") for d in dictionaries if d is not None]

    # Get sample data for all tables
    all_samples = []

    for table in request.dataset_names:
        df = (await analyst_db.get_dataset(table)).to_df()

        sample_str = f"Table: {table}\n{df.head(10).to_string()}"
        all_samples.append(sample_str)

    # Create messages for OpenAI
    messages: list[ChatCompletionMessageParam] = [
        get_external_database().get_system_prompt(),
        ChatCompletionUserMessageParam(
            content=f"Business Question: {request.question}",
            role="user",
        ),
        ChatCompletionUserMessageParam(
            content=f"Sample Data:\n{chr(10).join(all_samples)}", role="user"
        ),
        ChatCompletionUserMessageParam(
            content=f"Data Dictionary:\n{json.dumps(all_tables_info, ensure_ascii=False)}",
            role="user",
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
    async with AsyncLLMClient() as client:
        (
            completion,
            completion_org,
        ) = await client.chat.completions.create_with_completion(
            response_model=DatabaseAnalysisCodeGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0.1,
            messages=messages,
        )
    association_id = completion_org.datarobot_moderations["association_id"]
    logger.info(f"Association ID: {association_id}")

    if telemetry_send is not None:
        # add query type to telemetry
        telemetry_send["query_type"] = "03_generate_code_database"
        # submit telemetry
        asyncio.create_task(
            async_submit_actuals_to_datarobot(
                association_id=association_id, telemetry_json=telemetry_send
            )
        )
    return completion.code


@reflect_code_generation_errors(max_attempts=7)
async def _run_database_analysis(
    request: RunDatabaseAnalysisRequest,
    analyst_db: AnalystDB,
    exception_history: list[InvalidGeneratedCode] | None = None,
    telemetry_json: dict[str, Any] | None = None,
) -> RunDatabaseAnalysisResult:
    start_time = datetime.now()
    if not request.dataset_names:
        raise ValueError("Input data cannot be empty")

    if exception_history is None:
        exception_history = []

    sql_code = await _generate_database_analysis_code(
        request,
        analyst_db,
        next(iter(exception_history[::-1]), None),
        telemetry_json=telemetry_json,
    )
    try:
        results = get_external_database().execute_query(query=sql_code)
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
            datasets_analyzed=len(request.dataset_names),
            # total_columns_analyzed=sum(len(ds.columns) for ds in request.datasets),
        ),
    )


@log_api_call
async def run_database_analysis(
    request: RunDatabaseAnalysisRequest,
    analyst_db: AnalystDB,
    telemetry_json: dict[str, Any],
) -> RunDatabaseAnalysisResult:
    """Execute analysis workflow on datasets."""
    try:
        return await _run_database_analysis(
            request, analyst_db, telemetry_json=telemetry_json
        )
    except MaxReflectionAttempts as e:
        return RunDatabaseAnalysisResult(
            status="error",
            metadata=RunDatabaseAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )


# Type definitions
@dataclass
class AnalysisGenerationError:
    message: str
    original_error: BaseException | None = None


async def execute_business_analysis_and_charts(
    analysis_result: RunAnalysisResult | RunDatabaseAnalysisResult,
    enhanced_message: str,
    enable_chart_generation: bool = True,
    enable_business_insights: bool = True,
    telemetry_json: dict[str, Any] | None = None,
) -> tuple[
    RunChartsResult | BaseException | None,
    GetBusinessAnalysisResult | BaseException | None,
]:
    analysis_result.dataset = cast(AnalystDataset, analysis_result.dataset)
    # Prepare both requests
    chart_request = RunChartsRequest(
        dataset=analysis_result.dataset,
        question=enhanced_message,
    )

    business_request = GetBusinessAnalysisRequest(
        dataset=analysis_result.dataset,
        dictionary=DataDictionary.from_analyst_df(analysis_result.dataset.to_df()),
        question=enhanced_message,
    )

    if enable_chart_generation and enable_business_insights:
        # Run both analyses concurrently
        result = await asyncio.gather(
            run_charts(chart_request, telemetry_json=telemetry_json),
            get_business_analysis(business_request, telemetry_json=telemetry_json),
            return_exceptions=True,
        )

        return (result[0], result[1])
    elif enable_chart_generation:
        charts_result = await run_charts(chart_request, telemetry_json=telemetry_json)
        return charts_result, None
    else:
        business_result = await get_business_analysis(
            business_request, telemetry_json=telemetry_json
        )
        return None, business_result


async def run_complete_analysis(
    chat_request: ChatRequest,
    data_source: DataSourceType,
    datasets_names: list[str],
    analyst_db: AnalystDB,
    chat_id: str,
    message_id: str,
    enable_chart_generation: bool = True,
    enable_business_insights: bool = True,
    telemetry_json: dict[str, Any] | None = None,
) -> AsyncGenerator[Component | AnalysisGenerationError, None]:
    # Get enhanced message
    if telemetry_json is not None:
        telemetry_json["chat_id"] = chat_id
        telemetry_json["chat_seq"] = len(chat_request.messages)
        telemetry_json["data_source"] = data_source.value
        telemetry_json["datasets_names"] = datasets_names
        telemetry_json["enable_chart_generation"] = enable_chart_generation
        telemetry_json["enable_business_insights"] = enable_business_insights
    try:
        logger.info("Getting rephrased question...")
        enhanced_message = await rephrase_message(chat_request, telemetry_json)
        logger.info("Getting rephrased question done")
        yield enhanced_message
    except ValidationError:
        yield AnalysisGenerationError("LLM Error, please retry")
        return
    assistant_message = AnalystChatMessage(
        role="assistant",
        content=enhanced_message,
        components=[EnhancedQuestionGeneration(enhanced_user_message=enhanced_message)],
    )
    user_message = await analyst_db.get_chat_message(message_id=message_id)
    if user_message:
        if user_message.role == "user":
            user_message.in_progress = False
            await analyst_db.update_chat_message(
                message_id=message_id,
                message=user_message,
            )
            await analyst_db.add_chat_message(
                chat_id=chat_id, message=assistant_message
            )
    # Run main analysis
    logger.info("Start main analysis")
    try:
        is_database = data_source == DataSourceType.DATABASE
        logger.info("Getting analysis result...")
        log_memory()

        if is_database:
            analysis_result: (
                RunAnalysisResult | RunDatabaseAnalysisResult
            ) = await run_database_analysis(
                RunDatabaseAnalysisRequest(
                    dataset_names=datasets_names,
                    question=enhanced_message,
                ),
                analyst_db,
                telemetry_json=telemetry_json,
            )
        else:
            analysis_result = await run_analysis(
                RunAnalysisRequest(
                    dataset_names=datasets_names,
                    question=enhanced_message,
                ),
                analyst_db,
                telemetry_json=telemetry_json,
            )

        log_memory()
        logger.info("Getting analysis result done")

        if isinstance(analysis_result, BaseException):
            error_message = f"Error running initial analysis. Try rephrasing: {str(analysis_result)}"

            yield AnalysisGenerationError(error_message)

            assistant_message.in_progress = False
            await analyst_db.update_chat_message(
                message_id=assistant_message.id, message=assistant_message
            )
            return

        yield analysis_result

        assistant_message.components.append(analysis_result)
        await analyst_db.update_chat_message(
            message_id=assistant_message.id, message=assistant_message
        )

    except Exception as e:
        error_message = f"Error running initial analysis. Try rephrasing: {str(e)}"

        yield AnalysisGenerationError(error_message)

        assistant_message.in_progress = False
        await analyst_db.update_chat_message(
            message_id=assistant_message.id, message=assistant_message
        )
        return

    # Only proceed with additional analysis if we have valid initial results
    if not (
        analysis_result
        and analysis_result.dataset
        and (enable_chart_generation or enable_business_insights)
    ):
        assistant_message.in_progress = False
        await analyst_db.update_chat_message(
            message_id=assistant_message.id, message=assistant_message
        )
        return

    # Run concurrent analyses
    try:
        charts_result, business_result = await execute_business_analysis_and_charts(
            analysis_result,
            enhanced_message,
            enable_business_insights=enable_business_insights,
            enable_chart_generation=enable_chart_generation,
            telemetry_json=telemetry_json,
        )

        # Handle chart results
        if isinstance(charts_result, BaseException):
            error_message = "Error generating charts"

            yield AnalysisGenerationError(error_message)

            await analyst_db.update_chat_message(
                message_id=assistant_message.id, message=assistant_message
            )

        elif charts_result is not None:
            yield charts_result
            assistant_message.components.append(charts_result)
            await analyst_db.update_chat_message(
                message_id=assistant_message.id, message=assistant_message
            )

        # Handle business analysis results
        if isinstance(business_result, BaseException):
            error_message = "Error generating business insights"

            yield AnalysisGenerationError("Error generating business insights")

            await analyst_db.update_chat_message(
                message_id=assistant_message.id, message=assistant_message
            )
        elif business_result is not None:
            yield business_result
            assistant_message.components.append(business_result)
            assistant_message.in_progress = False

            await analyst_db.update_chat_message(
                message_id=assistant_message.id, message=assistant_message
            )

    except Exception as e:
        error_message = f"Error setting up additional analysis: {str(e)}"

        yield AnalysisGenerationError(error_message)

        assistant_message.in_progress = False
        await analyst_db.update_chat_message(
            message_id=assistant_message.id, message=assistant_message
        )


async def process_data_and_update_state(
    new_dataset_names: list[str],
    analyst_db: AnalystDB,
    data_source: str | DataSourceType,
    telemetry_json: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """Process datasets and yield progress updates asynchronously."""
    # Start processing and yield initial message
    logger.info("Starting data processing")
    log_memory()
    yield "Starting data processing"

    # Handle data cleansing based on the source
    # Convert string data_source to DataSourceType if needed
    data_source_type = (
        data_source
        if isinstance(data_source, DataSourceType)
        else DataSourceType(data_source)
    )
    if data_source_type != DataSourceType.DATABASE:
        try:
            logger.info("Cleansing datasets")
            yield "Cleansing datasets"
            for analysis_dataset_name in new_dataset_names:
                analysis_dataset = await analyst_db.get_dataset(
                    analysis_dataset_name, max_rows=None
                )
                cleansed_dataset = await cleanse_dataframe(analysis_dataset)
                await analyst_db.register_dataset(
                    cleansed_dataset, data_source=DataSourceType.GENERATED
                )
                yield f"Cleansed dataset: {analysis_dataset_name}"
                del cleansed_dataset
                del analysis_dataset
                log_memory()

            logger.info("Cleansing datasets complete")
            yield "Cleansing datasets complete"
            log_memory()
        except Exception:
            logger.error("Data processing failed", exc_info=True)
            yield "Data processing failed"
            raise
    else:
        pass

    # Generate data dictionaries
    logger.info("Data processing successful, generating dictionaries")
    yield "Data processing successful, generating dictionaries"
    log_memory()
    try:
        for analysis_dataset_name in new_dataset_names:
            try:
                existing_dictionary = await analyst_db.get_data_dictionary(
                    analysis_dataset_name
                )
                logger.info(
                    f"Found existing dictionary for dataset: {analysis_dataset_name}"
                )
                if existing_dictionary is not None:
                    continue

            except Exception:
                pass
            logger.info(f"Creating dictionary for dataset: {analysis_dataset_name}")
            analysis_dataset = await analyst_db.get_dataset(analysis_dataset_name)
            new_dictionary = await get_dictionary(analysis_dataset, telemetry_json)
            logger.info(new_dictionary.to_application_df())
            del analysis_dataset
            await analyst_db.register_data_dictionary(new_dictionary)
            logger.info(f"Registered dictionary for dataset: {analysis_dataset_name}")
            yield f"Registered data dictionary: {analysis_dataset_name}"
            log_memory()
            continue
    except Exception:
        logger.error("Failed to generate data dictionaries", exc_info=True)
        yield "Failed to generate data dictionaries"
        raise
    log_memory()
    # Final completion message
    yield "Processing complete"

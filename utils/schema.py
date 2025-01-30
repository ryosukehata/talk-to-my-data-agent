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

import json
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

from utils.code_execution import MaxReflectionAttempts


class LLMDeploymentSettings(BaseModel):
    target_feature_name: str = "resultText"
    prompt_feature_name: str = "promptText"


class AiCatalogDataset(BaseModel):
    id: str
    name: str
    created: str
    size: str


class AnalystDataset(BaseModel):
    name: str
    _data: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    data: list[dict[str, Any]] = Field(
        description="List of records with column names as keys",
        json_schema_extra={
            "example": [
                {"column1": "value1", "column2": "value2"},
                {"column1": "value3", "column2": "value4"},
            ]
        },
    )

    def __init__(self, **data: Any):
        df = None
        if "data" in data and isinstance(data["data"], pd.DataFrame):
            df = data["data"]
            records = df.to_dict("records")
            data["data"] = records
        if "name" not in data:
            data["name"] = "analyst_dataset"
        super().__init__(**data)
        if df is not None:
            self._data = df
        else:
            self._data = pd.DataFrame.from_records(self.data)

    @property
    def columns(self) -> list[str]:
        return self._data.columns.tolist()

    def to_df(self) -> pd.DataFrame:
        return self._data


class CleansingReport(BaseModel):
    columns_cleaned: list[str]
    errors: list[str]
    warnings: list[str]


class CleansedDataset(BaseModel):
    dataset: AnalystDataset
    cleaning_report: CleansingReport

    @property
    def name(self) -> str:
        return self.dataset.name

    def to_df(self) -> pd.DataFrame:
        return self.dataset.to_df()


class DataDictionaryColumn(BaseModel):
    data_type: str
    column: str
    description: str


class DataDictionary(BaseModel):
    name: str
    column_descriptions: list[DataDictionaryColumn]

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        name: str = "analysis_result",
        column_descriptions: str = "Analysis result column",
    ) -> "DataDictionary":
        return DataDictionary(
            name=name,
            column_descriptions=[
                DataDictionaryColumn(
                    column=col,
                    description=column_descriptions,
                    data_type=str(df[col].dtype),
                )
                for col in df.columns
            ],
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "column": [c.column for c in self.column_descriptions],
                "description": [c.description for c in self.column_descriptions],
                "data_type": [c.data_type for c in self.column_descriptions],
            }
        )


class DictionaryGeneration(BaseModel):
    """Validates LLM responses for data dictionary generation

    Attributes:
        columns: List of column names
        descriptions: List of column descriptions

    Raises:
        ValueError: If validation fails
    """

    columns: list[str]
    descriptions: list[str]

    @field_validator("descriptions")
    @classmethod
    def validate_descriptions(cls, v: Any, values: Any) -> Any:
        # Check if columns exists in values
        if "columns" not in values.data:
            raise ValueError("Columns must be provided before descriptions")

        # Check if lengths match
        if len(v) != len(values.data["columns"]):
            raise ValueError(
                f"Number of descriptions ({len(v)}) must match number of columns ({len(values['columns'])})"
            )

        # Validate each description
        for desc in v:
            if not desc or not isinstance(desc, str):
                raise ValueError("Each description must be a non-empty string")
            if len(desc.strip()) < 10:
                raise ValueError("Descriptions must be at least 10 characters long")

        return v

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v: Any) -> Any:
        if not v:
            raise ValueError("Columns list cannot be empty")

        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate column names are not allowed")

        # Validate each column name
        for col in v:
            if not col or not isinstance(col, str):
                raise ValueError("Each column name must be a non-empty string")

        return v

    def to_dict(self) -> dict[str, str]:
        """Convert columns and descriptions to dictionary format

        Returns:
            Dict mapping column names to their descriptions
        """
        return dict(zip(self.columns, self.descriptions))


class RunAnalysisRequest(BaseModel):
    datasets: list[AnalystDataset]
    dictionaries: list[DataDictionary]
    question: str


class RunAnalysisResultMetadata(BaseModel):
    duration: float
    attempts: int
    datasets_analyzed: int | None = None
    total_rows_analyzed: int | None = None
    total_columns_analyzed: int | None = None
    exception: AnalysisError | None = None


class RunAnalysisResult(BaseModel):
    status: Literal["success", "error"]
    metadata: RunAnalysisResultMetadata
    dataset: AnalystDataset | None = None
    code: str | None = None


class CodeExecutionError(BaseModel):
    code: str | None = None
    exception_str: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    traceback_str: str | None = None


class AnalysisError(BaseModel):
    exception_history: list[CodeExecutionError] | None = None

    @classmethod
    def from_max_reflection_exception(
        cls,
        exception: MaxReflectionAttempts,
    ) -> "AnalysisError":
        return AnalysisError(
            exception_history=[
                CodeExecutionError(
                    exception_str=str(exception.exception),
                    traceback_str=exception.traceback_str,
                    code=exception.code,
                    stdout=exception.stdout,
                    stderr=exception.stderr,
                )
                for exception in exception.exception_history
                if exception is not None
            ]
            if exception.exception_history is not None
            else None,
        )


class RunDatabaseAnalysisResultMetadata(BaseModel):
    duration: float
    attempts: int
    datasets_analyzed: int | None = None
    total_columns_analyzed: int | None = None
    exception: AnalysisError | None = None


class RunDatabaseAnalysisResult(BaseModel):
    status: Literal["success", "error"]
    metadata: RunDatabaseAnalysisResultMetadata
    dataset: AnalystDataset | None = None
    code: str | None = None


class ChartGenerationExecutionResult(BaseModel):
    fig1: go.Figure
    fig2: go.Figure

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RunChartsRequest(BaseModel):
    dataset: AnalystDataset
    question: str


class RunChartsResult(BaseModel):
    status: Literal["success", "error"]
    fig1_json: str | None = None
    fig2_json: str | None = None
    code: str | None = None
    metadata: RunAnalysisResultMetadata

    @property
    def fig1(self) -> go.Figure | None:
        return go.Figure(json.loads(self.fig1_json)) if self.fig1_json else None

    @property
    def fig2(self) -> go.Figure | None:
        return go.Figure(json.loads(self.fig2_json)) if self.fig2_json else None


class GetBusinessAnalysisMetadata(BaseModel):
    duration: float | None = None
    question: str | None = None
    rows_analyzed: int | None = None
    columns_analyzed: int | None = None
    exception_str: str | None = None


class BusinessAnalysisGeneration(BaseModel):
    bottom_line: str
    additional_insights: str
    follow_up_questions: list[str]


class GetBusinessAnalysisResult(BaseModel):
    status: Literal["success", "error"]
    bottom_line: str
    additional_insights: str
    follow_up_questions: list[str]
    metadata: GetBusinessAnalysisMetadata | None = None


class GetBusinessAnalysisRequest(BaseModel):
    dataset: AnalystDataset
    dictionary: DataDictionary
    question: str


class ChatRequest(BaseModel):
    """Request model for chat history processing

    Attributes:
        messages: list of dictionaries containing chat messages
                 Each message must have 'role' and 'content' fields
                 Role must be one of: 'user', 'assistant', 'system'
    """

    messages: list[ChatCompletionMessageParam] = Field(min_length=1)


class QuestionListGeneration(BaseModel):
    questions: list[str]


class ValidatedQuestion(BaseModel):
    """Stores validation results for suggested questions"""

    question: str


class RunDatabaseAnalysisRequest(BaseModel):
    datasets: list[AnalystDataset]
    dictionaries: list[DataDictionary]
    question: str = Field(min_length=1)


class DatabaseAnalysisCodeGeneration(BaseModel):
    code: str
    description: str


class EnhancedQuestionGeneration(BaseModel):
    enhanced_user_message: str


class CodeGeneration(BaseModel):
    code: str
    description: str


RuntimeCredentialType = Literal["llm", "db"]


DatabaseConnectionType = Literal["bigquery", "snowflake", "no_database"]


class AppInfra(BaseModel):
    llm: str
    database: DatabaseConnectionType


UserRoleType = Literal["assistant", "user", "system"]


class AnalystChatMessage(BaseModel):
    role: UserRoleType
    content: str
    components: list[
        RunAnalysisResult
        | RunChartsResult
        | GetBusinessAnalysisResult
        | EnhancedQuestionGeneration
        | RunDatabaseAnalysisResult
    ]

    def to_openai_message_param(self) -> ChatCompletionMessageParam:
        if self.role == "user":
            return ChatCompletionUserMessageParam(role=self.role, content=self.content)
        elif self.role == "assistant":
            return ChatCompletionAssistantMessageParam(
                role=self.role, content=self.content
            )
        elif self.role == "system":
            return ChatCompletionSystemMessageParam(
                role=self.role, content=self.content
            )

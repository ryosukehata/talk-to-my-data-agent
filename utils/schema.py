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
from typing import Annotated, Any, Literal, cast

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
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)


class LLMDeploymentSettings(BaseModel):
    target_feature_name: str = "resultText"
    prompt_feature_name: str = "promptText"


class AiCatalogDataset(BaseModel):
    id: str
    name: str
    created: str
    size: str


def _convert_to_records(v: Any) -> list[dict[str, Any]]:
    if isinstance(v, pd.DataFrame):
        return cast(list[dict[str, Any]], v.to_dict("records"))
    if isinstance(v, list):
        return v
    raise ValueError(f"Expected DataFrame or list of dicts, got {type(v)}")


class AnalystDataset(BaseModel):
    name: str
    data: Annotated[list[dict[str, Any]], BeforeValidator(_convert_to_records)]
    columns: list[str] = Field(default_factory=list)

    def __init__(
        self,
        data: list[dict[str, Any]] | pd.DataFrame,
        name: str = "analysis_data",
        **kwargs: Any,
    ):
        if "columns" in kwargs:
            columns = kwargs.pop("columns")
        elif isinstance(data, pd.DataFrame):
            columns = list(data.columns)
        elif isinstance(data, list):
            try:
                columns = list(data[0].keys())
            except Exception:
                columns = []
        else:
            raise ValueError("data has to be either list of records or pd.DataFrame")
        super().__init__(name=name, data=data, columns=columns, **kwargs)

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame.from_records(self.data)
        if not len(df) and self.columns:
            # If DataFrame is empty and we have stored columns, create empty DF with those columns
            return pd.DataFrame(columns=self.columns)
        return df


class CleansingReport(BaseModel):
    columns_cleaned: list[str]
    errors: list[str]
    warnings: list[str]


class CleansedDataset(AnalystDataset):
    cleaning_report: CleansingReport


class DataDictionaryColumn(BaseModel):
    data_type: str
    column: str
    description: str


class DataDictionary(BaseModel):
    name: str
    dictionary: list[DataDictionaryColumn]

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        name: str = "analysis_result",
        column_descriptions: str = "Analysis result column",
    ) -> "DataDictionary":
        return DataDictionary(
            name=name,
            dictionary=[
                DataDictionaryColumn(
                    column=col,
                    description=column_descriptions,
                    data_type=str(df[col].dtype),
                )
                for col in df.columns
            ],
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
    """Request model for analysis endpoint

    Attributes:
        data: Dictionary of datasets, where each dataset is a list of dictionaries
        dictionary: Dictionary of data dictionaries, where each dictionary describes a dataset's columns
        question: Business question to analyze
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: list[CleansedDataset]
    dictionary: list[DataDictionary]
    question: str
    error_message: str | None = None
    failed_code: str | None = None


class RunAnalysisResultMetadata(BaseModel):
    duration: float
    attempts: int
    datasets_analyzed: int | None = None
    total_rows_analyzed: int | None = None
    total_columns_analyzed: int | None = None


class RunAnalysisResult(BaseModel):
    status: str
    metadata: DatabaseExecutionMetadata | RunAnalysisResultMetadata
    data: AnalystDataset | None = None
    code: str | None = None
    suggestions: str | None = None


class ChartGenerationExecutionResult(BaseModel):
    fig1: go.Figure
    fig2: go.Figure

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RunChartsRequest(BaseModel):
    """Request model for charts endpoint

    Attributes:
        data: list of dictionaries representing a single dataset
        question: Business question to visualize
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: AnalystDataset  # Allow both string and integer keys
    question: str
    error_message: str | None = None
    failed_code: str | None = None


class RunChartsResult(BaseModel):
    status: str
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


class RunBusinessAnalysisMetadata(BaseModel):
    timestamp: str
    question: str
    rows_analyzed: int
    columns_analyzed: int


class BusinessAnalysisGeneration(BaseModel):
    bottom_line: str
    additional_insights: str
    follow_up_questions: list[str]


class RunBusinessAnalysisResult(BusinessAnalysisGeneration):
    metadata: RunBusinessAnalysisMetadata


class RunBusinessAnalysisRequest(BaseModel):
    """Request model for business analysis endpoint

    Attributes:
        data: list of dictionaries representing a single dataset
        dictionary: list of dictionary entries describing columns
        question: Business question to analyze
    """

    data: AnalystDataset  # Allow both string and integer keys
    dictionary: DataDictionary  # Allow both string and integer keys
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
    """Container for list of questions"""

    questions: list[str]


class ValidatedQuestion(BaseModel):
    """Stores validation results for suggested questions"""

    question: str
    is_valid: bool
    available_columns: list[str]
    missing_columns: list[str]
    validation_message: str


class RunDatabaseAnalysisRequest(BaseModel):
    """Request model for Database analysis endpoint

    Attributes:
        data: dictionary of sample data from each table
        dictionary: dictionary of data dictionaries for each table
        question: Business question to analyze
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: list[AnalystDataset]  # Sample data from each table
    dictionary: list[DataDictionary]  # Pre-generated data dictionary
    question: str = Field(min_length=1)
    error_message: str | None = None
    failed_code: str | None = None


class DatabaseAnalysisCodeGeneration(BaseModel):
    code: str
    description: str


class DatabaseExecutionMetadata(BaseModel):
    query_id: str
    row_count: int
    execution_time: float
    db_schema: str
    database: str | None = None
    warehouse: str | None = None


class EnhancedQuestionGeneration(BaseModel):
    enhanced_user_message: str


class CodeGeneration(BaseModel):
    code: str
    description: str


class AppInfra(BaseModel):
    llm: str
    database: Literal["bigquery", "snowflake", "no_database"]


class AnalystChatMessage(BaseModel):
    role: Literal["assistant", "user", "system"]
    content: str
    components: list[RunAnalysisResult | RunChartsResult | RunBusinessAnalysisResult]

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

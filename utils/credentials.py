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

from typing import Any, Dict, Optional

from pydantic import AliasChoices, AliasPath, Field
from pydantic_settings import BaseSettings


class DRCredentials(BaseSettings): ...


class AzureOpenAICredentials(DRCredentials):
    """LLM credentials auto-constructed using environment variables."""

    api_key: str = Field(
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            AliasPath("MLOPS_RUNTIME_PARAM_OPENAI_API_KEY", "payload", "apiToken"),
        ),
    )
    azure_endpoint: str = Field(
        validation_alias=AliasChoices(
            "OPENAI_API_BASE",
            AliasPath("MLOPS_RUNTIME_PARAM_OPENAI_API_BASE", "payload"),
        )
    )
    api_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_VERSION",
            AliasPath("MLOPS_RUNTIME_PARAM_OPENAI_API_VERSION", "payload"),
        ),
    )
    azure_deployment: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_DEPLOYMENT_ID",
            AliasPath("MLOPS_RUNTIME_PARAM_OPENAI_API_DEPLOYMENT_ID", "payload"),
        ),
    )


class GoogleCredentials(DRCredentials):
    service_account_key: Dict[str, Any] = Field(
        validation_alias=AliasChoices(
            "GOOGLE_SERVICE_ACCOUNT",
            AliasPath(
                "MLOPS_RUNTIME_PARAM_GOOGLE_SERVICE_ACCOUNT", "payload", "gcpKey"
            ),
        )
    )
    region: Optional[str] = Field(default="us-west1", validation_alias="GOOGLE_REGION")
    db_schema: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_DB_SCHEMA", AliasPath("MLOPS_RUNTIME_PARAM_GOOGLE_DB_SCHEMA")
        ),
    )


class AWSBedrockCredentials(DRCredentials):
    aws_access_key_id: str = Field(
        validation_alias=AliasChoices(
            "AWS_ACCESS_KEY_ID",
            AliasPath("MLOPS_RUNTIME_PARAM_AWS_ACCOUNT", "payload", "awsAccessKeyId"),
        )
    )
    aws_secret_access_key: str = Field(
        validation_alias=AliasChoices(
            "AWS_SECRET_ACCESS_KEY",
            AliasPath(
                "MLOPS_RUNTIME_PARAM_AWS_ACCOUNT", "payload", "awsSecretAccessKey"
            ),
        )
    )
    aws_session_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "AWS_SESSION_TOKEN",
            AliasPath("MLOPS_RUNTIME_PARAM_AWS_ACCOUNT", "payload", "awsSessionToken"),
        ),
    )
    region_name: Optional[str] = Field(default=None, validation_alias="AWS_REGION")


class SnowflakeCredentials(DRCredentials):
    """Snowflake Connection credentials auto-constructed using environment variables."""

    user: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "username"),
            "SNOWFLAKE_USER",
        )
    )
    password: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "password"),
            "SNOWFLAKE_PASSWORD",
        )
    )
    account: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_ACCOUNT"),
            "SNOWFLAKE_ACCOUNT",
        ),
    )
    database: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_DATABASE"),
            "SNOWFLAKE_DATABASE",
        )
    )
    warehouse: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_WAREHOUSE"),
            "SNOWFLAKE_WAREHOUSE",
        )
    )
    db_schema: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_SCHEMA"),
            "SNOWFLAKE_SCHEMA",
        )
    )
    role: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_ROLE"),
            "SNOWFLAKE_ROLE",
        )
    )
    snowflake_key_path: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_KEY_PATH"), "SNOWFLAKE_KEY_PATH"
        )
    )


class NoDatabaseCredentials(DRCredentials):
    pass

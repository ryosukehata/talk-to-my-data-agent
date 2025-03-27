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

import os

import datarobot as dr
import pulumi

from infra.common.globals import GlobalRuntimeEnvironment
from infra.common.stack import project_name
from utils.schema import LLMDeploymentSettings

from .common.schema import (
    CustomModelArgs,
    DeploymentArgs,
    RegisteredModelArgs,
)

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")

custom_model_args = CustomModelArgs(
    resource_name=f"Data Analyst Proxy LLM Custom Model [{project_name}]",
    name=f"Data Analyst Proxy LLM Custom Model [{project_name}]",
    target_name=LLMDeploymentSettings().target_feature_name,
    target_type=dr.enums.TARGET_TYPE.TEXT_GENERATION,
    replicas=2,
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_312_MODERATIONS.value.id,
    opts=pulumi.ResourceOptions(delete_before_replace=True),
)

registered_model_args = RegisteredModelArgs(
    resource_name=f"Data Analyst Proxy LLM Registered Model [{project_name}]",
)

deployment_args = DeploymentArgs(
    resource_name=f"Data Analyst Proxy LLM Deployment [{project_name}]",
    label=f"Data Analyst Proxy LLM Deployment [{project_name}]",
)

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

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot

from utils.schema import LLMDeploymentSettings

from .common.globals import GlobalLLM
from .common.schema import (
    CustomModelArgs,
    DeploymentArgs,
    LLMBlueprintArgs,
    PlaygroundArgs,
    RegisteredModelArgs,
)
from .common.stack import project_name

LLM = GlobalLLM.AZURE_OPENAI_GPT_4_O

custom_model_args = CustomModelArgs(
    resource_name=f"Generative Analyst Custom Model [{project_name}]",
    name="Generative Analyst Assistant",  # built-in QA app uses this as the AI's name
    target_name=LLMDeploymentSettings().target_feature_name,
    target_type=dr.enums.TARGET_TYPE.TEXT_GENERATION,
    replicas=2,
    opts=pulumi.ResourceOptions(delete_before_replace=True),
)

registered_model_args = RegisteredModelArgs(
    resource_name=f"Generative Analyst Registered Model [{project_name}]",
)


deployment_args = DeploymentArgs(
    resource_name=f"Generative Analyst Deployment [{project_name}]",
    label=f"Generative Analyst Deployment [{project_name}]",
    association_id_settings=datarobot.DeploymentAssociationIdSettingsArgs(
        column_names=["association_id"],
        auto_generate_id=False,
        required_in_prediction_requests=True,
    ),
    predictions_data_collection_settings=datarobot.DeploymentPredictionsDataCollectionSettingsArgs(
        enabled=True,
    ),
    predictions_settings=(
        datarobot.DeploymentPredictionsSettingsArgs(min_computes=0, max_computes=2)
    ),
)

playground_args = PlaygroundArgs(
    resource_name=f"Generative Analyst Playground [{project_name}]",
)

llm_blueprint_args = LLMBlueprintArgs(
    resource_name=f"Generative Analyst LLM Blueprint [{project_name}]",
    llm_id=LLM.name,
    llm_settings=datarobot.LlmBlueprintLlmSettingsArgs(
        max_completion_length=2048,
        temperature=0.1,
    ),
)

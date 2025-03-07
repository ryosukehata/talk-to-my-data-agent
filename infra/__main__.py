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

import os
import pathlib
import sys

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot

sys.path.append("..")
from infra import (
    settings_app_infra,
    settings_generative,
)
from infra.common.feature_flags import check_feature_flags
from infra.common.globals import GlobalRuntimeEnvironment
from infra.common.stack import PROJECT_ROOT, project_name
from infra.common.urls import get_deployment_url
from infra.components.custom_model_deployment import CustomModelDeployment
from infra.components.dr_credential import (
    get_credential_runtime_parameter_values,
    get_database_credentials,
    get_llm_credentials,
)
from infra.components.playground_custom_model import PlaygroundCustomModel
from infra.settings_database import DATABASE_CONNECTION_TYPE
from utils.resources import (
    app_env_name,
    llm_deployment_env_name,
)
from utils.schema import (
    AppInfra,
)

check_feature_flags(
    pathlib.Path(PROJECT_ROOT / "infra" / "feature_flag_requirements.yaml")
)

with open(PROJECT_ROOT / "frontend/app_infra.json", "w") as infra_selection:
    infra_selection.write(
        AppInfra(
            database=DATABASE_CONNECTION_TYPE,
            llm=settings_generative.LLM.name,
        ).model_dump_json()
    )

if "DATAROBOT_DEFAULT_USE_CASE" in os.environ:
    use_case_id = os.environ["DATAROBOT_DEFAULT_USE_CASE"]
    pulumi.info(f"Using existing use case '{use_case_id}'")
    use_case = datarobot.UseCase.get(
        id=use_case_id,
        resource_name="Data Analyst Use Case [PRE-EXISTING]",
    )
else:
    use_case = datarobot.UseCase(
        resource_name=f"Data Analyst Use Case [{project_name}]",
        description="Use case for Data Analyst application",
    )

prediction_environment = datarobot.PredictionEnvironment(
    resource_name=f"Data Analyst Prediction Environment [{project_name}]",
    platform=dr.enums.PredictionEnvironmentPlatform.DATAROBOT_SERVERLESS,
)

llm_credential = get_llm_credentials(settings_generative.LLM)

llm_runtime_parameter_values = get_credential_runtime_parameter_values(
    llm_credential, "llm"
)

llm_custom_model = PlaygroundCustomModel(
    resource_name=f"Chat Agent Buzok Deployment [{project_name}]",
    use_case=use_case,
    playground_args=settings_generative.playground_args,
    llm_blueprint_args=settings_generative.llm_blueprint_args,
    runtime_parameter_values=llm_runtime_parameter_values,
    custom_model_args=settings_generative.custom_model_args,
)


llm_deployment = CustomModelDeployment(
    resource_name=f"Chat Agent Deployment [{project_name}]",
    use_case_ids=[use_case.id],
    custom_model_version_id=llm_custom_model.version_id,
    registered_model_args=settings_generative.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_generative.deployment_args,
)


app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=llm_deployment_env_name,
        type="deployment",
        value=llm_deployment.id,
    ),
]


db_credential = get_database_credentials(DATABASE_CONNECTION_TYPE)

db_runtime_parameter_values = get_credential_runtime_parameter_values(
    db_credential, "db"
)
app_runtime_parameters += db_runtime_parameter_values  # type: ignore[arg-type]


app_source = datarobot.ApplicationSource(
    files=settings_app_infra.get_app_files(
        runtime_parameter_values=app_runtime_parameters
    ),
    runtime_parameter_values=app_runtime_parameters,
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_312_APPLICATION_BASE.value.id,
    resources=datarobot.ApplicationSourceResourcesArgs(
        replicas=2,
        resource_label="cpu.xlarge",
        session_affinity=True,
    ),
    **settings_app_infra.app_source_args,
)


app = datarobot.CustomApplication(
    resource_name=settings_app_infra.app_resource_name,
    source_version_id=app_source.version_id,
    use_case_ids=[use_case.id],
    allow_auto_stopping=True,
)


pulumi.export(llm_deployment_env_name, llm_deployment.id)
pulumi.export(
    settings_generative.deployment_args.resource_name,
    llm_deployment.id.apply(get_deployment_url),
)

# App output
pulumi.export(app_env_name, app.id)
pulumi.export(
    settings_app_infra.app_resource_name,
    app.application_url,
)

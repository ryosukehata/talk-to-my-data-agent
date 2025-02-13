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

import textwrap
from typing import List, Sequence, Tuple

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot
from settings_database import DATABASE_CONNECTION_TYPE

from infra.common.schema import ApplicationSourceArgs
from infra.common.stack import PROJECT_ROOT, project_name
from utils.credentials import SnowflakeCredentials

application_path = PROJECT_ROOT / "frontend"

app_source_args = ApplicationSourceArgs(
    resource_name=f"Data Analyst App Source [{project_name}]",
    replicas=2,
).model_dump(mode="json", exclude_none=True)


def ensure_app_settings(app_id: str) -> None:
    try:
        dr.client.get_client().patch(
            f"customApplications/{app_id}/",
            json={"allowAutoStopping": True},
        )
    except Exception:
        pulumi.warn("Patching app unsuccessful.")
    return


def ensure_app_source_settings(source_id: str, version_id: str) -> str:
    try:
        dr.client.get_client().patch(
            url=f"customApplicationSources/{source_id}/versions/{version_id}/",
            json={
                "resources": {
                    "sessionAffinity": True,
                    "resourceLabel": "cpu.xlarge",
                    "replicas": 2,
                }
            },
        )
    except dr.errors.ClientError:
        pulumi.warn("Patching app source unsuccessful.")
    return version_id


app_resource_name: str = f"Data Analyst Application [{project_name}]"


def _prep_metadata_yaml(
    runtime_parameter_values: Sequence[
        datarobot.ApplicationSourceRuntimeParameterValueArgs
        | datarobot.CustomModelRuntimeParameterValueArgs
    ],
) -> None:
    from jinja2 import BaseLoader, Environment

    llm_runtime_parameter_specs = "\n".join(
        [
            textwrap.dedent(
                f"""\
            - fieldName: {param.key}
              type: {param.type}
        """
            )
            for param in runtime_parameter_values
        ]
    )
    with open(application_path / "metadata.yaml.jinja") as f:
        template = Environment(loader=BaseLoader()).from_string(f.read())
    (application_path / "metadata.yaml").write_text(
        template.render(
            additional_params=llm_runtime_parameter_specs,
        )
    )


def get_app_files(
    runtime_parameter_values: Sequence[
        datarobot.ApplicationSourceRuntimeParameterValueArgs
        | datarobot.CustomModelRuntimeParameterValueArgs,
    ],
) -> List[Tuple[str, str]]:
    _prep_metadata_yaml(runtime_parameter_values)
    # Get all files from application path
    source_files = [
        (f.as_posix(), f.relative_to(application_path).as_posix())
        for f in application_path.glob("**/*")
        if f.is_file() and ".yaml" not in f.name
    ]

    # Get all .py files from utils directory
    utils_files = [
        (str(PROJECT_ROOT / f"utils/{f.name}"), f"utils/{f.name}")
        for f in (PROJECT_ROOT / "utils").glob("*.py")
        if f.is_file()
    ]

    # Add the metadata.yaml file
    source_files.extend(utils_files)
    source_files.append(
        ((application_path / "metadata.yaml").as_posix(), "metadata.yaml")
    )

    if DATABASE_CONNECTION_TYPE == "snowflake":
        credentials = SnowflakeCredentials()
        if credentials.snowflake_key_path:
            # Add the snowflake connection file if it exists
            snowflake_file = PROJECT_ROOT / credentials.snowflake_key_path
            if snowflake_file.is_file():
                source_files.append(
                    (str(snowflake_file), credentials.snowflake_key_path)
                )

    return source_files

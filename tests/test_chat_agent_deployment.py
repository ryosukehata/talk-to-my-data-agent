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

import pytest
from datarobot.rest import RESTClientObject
from openai import OpenAI


@pytest.fixture
def chat_client(
    pulumi_up: Any, llm_deployment_id: str, dr_client: RESTClientObject
) -> OpenAI:
    deployment_chat_base_url = dr_client.endpoint + f"/deployments/{llm_deployment_id}/"

    client = OpenAI(api_key=dr_client.token, base_url=deployment_chat_base_url)
    return client


def test_can_chat(chat_client: OpenAI) -> None:
    model = "gpt-4o"
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    response = chat_client.chat.completions.create(model=model, messages=messages)  # type: ignore[arg-type]
    assert response.choices[0].message.content is not None
    assert "paris" in response.choices[0].message.content.lower()


# @pytest.fixture(scope="class")
# def diabetes_dataset_url():
#     return "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv"

# @pytest.fixture(scope="class")
# def dataset(diabetes_dataset_url):
#     df = pd.read_csv(diabetes_dataset_url)
#     # Replace non-JSON compliant values
#     df = df.replace([float("inf"), -float("inf")], None)  # Replace infinity with None
#     df = df.where(pd.notnull(df), None)  # Replace NaN with None

#     # Create dataset dictionary
#     dataset = {
#         "name": os.path.splitext(os.path.basename(diabetes_dataset_url))[0],
#         "data": df.to_dict("records"),
#     }

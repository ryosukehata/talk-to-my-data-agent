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

# mypy: ignore-errors

import logging
import os
import subprocess
import uuid
from typing import Callable

import datarobot as dr
import pandas as pd
import pytest
import pytest_asyncio
from dotenv import dotenv_values

from utils.analyst_db import AnalystDB, DataSourceType
from utils.resources import LLMDeployment
from utils.schema import AnalystDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--pulumi_up",
        action="store_true",
        default=False,
        help="Run pulumi up before conducting test. Otherwise use existing stack.",
    )
    parser.addoption(
        "--always_delete_stack",
        action="store_true",
        default=False,
        help="Delete the stack even in case of failure (Only used in case `pulumi_up` is True)",
    )


@pytest.fixture(scope="session")
def stack_name():
    short_uuid = str(uuid.uuid4())[:5]
    return f"test-stack-{short_uuid}"


@pytest.fixture(scope="session")
def session_env_vars(request, stack_name):
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    env_vars = dotenv_values(env_file)
    session_vars = {
        "PROJECT_NAME": stack_name,
    }
    env_vars.update(session_vars)
    os.environ.update(env_vars)
    return session_vars


@pytest.fixture(scope="session")
def subprocess_runner():
    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
        cmd = " ".join(command)
        if proc.returncode:
            msg = f"'{cmd}' exited {proc.returncode}"
            logger.warning(msg)
            msg = f"'{cmd}' STDOUT:\n{proc.stdout}"
            logger.warning(msg)
            msg = f"'{cmd}' STDERR:\n{proc.stderr}"
            logger.warning(msg)
            logger.info(proc)
        return proc

    return run_command


@pytest.fixture(scope="session")
def pulumi_up(
    stack_name,
    session_env_vars,
    pytestconfig,
    request: pytest.FixtureRequest,
    subprocess_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
):
    if pytestconfig.getoption("pulumi_up"):
        logger.info(f"Running {stack_name} with {session_env_vars}")
        subprocess_runner(["pulumi", "stack", "init", stack_name, "--non-interactive"])
        # ensure stack is deleted - stack init can fail if the name is the same as currently selected
        subprocess_runner(
            ["pulumi", "stack", "select", stack_name, "--non-interactive"]
        )
        proc = subprocess_runner(["pulumi", "up", "-y", "--non-interactive"])
        stack = subprocess.check_output(["pulumi", "stack", "output"], text=True)

        if proc.returncode:
            raise RuntimeError(f"`pulumi up` failed for {stack_name}")
        os.environ["PULUMI_STACK_CONTEXT"] = stack_name
        tests_failed_before_module = request.session.testsfailed
        # logger.info(f"Tests failed before: {tests_failed_before_module}")
        yield
        tests_failed_during_module = (
            request.session.testsfailed - tests_failed_before_module
        )
        logger.info(f"New tests failed: {tests_failed_during_module}")

        # if we say "always delete" this will delete
        # if we say "don't always delete, this will only delete if no failures occured"
        if (
            pytestconfig.getoption("always_delete_stack") is True
            or tests_failed_during_module == 0
        ):
            logger.info("Tearing down stack")
            subprocess_runner(["pulumi", "down", "-y", "--non-interactive"])
            subprocess_runner(
                ["pulumi", "stack", "rm", stack_name, "-y", "--non-interactive"]
            )
        else:
            logger.warning(
                f"There were errors. The stack {stack_name} will be preserved. Please check logs."
            )
    else:
        stack = subprocess.check_output(
            ["pulumi", "stack"],
            text=True,
        ).split("\n")[0]
        logger.info(stack)
        yield


@pytest.fixture
def dr_client(session_env_vars):
    return dr.Client()


@pytest.fixture
def llm_deployment_id():
    return LLMDeployment().id


DATA_FILES = {
    "lending_club_profile": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Profile.csv",
    "lending_club_target": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Target.csv",
    "lending_club_transactions": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Transactions.csv",
    "diabetes": "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv",
    "mpg": "https://s3.us-east-1.amazonaws.com/datarobot_public_datasets/auto-mpg.csv",
}


@pytest.fixture(scope="module")
def url_lending_club_profile():
    return DATA_FILES["lending_club_profile"]


@pytest.fixture(scope="module")
def url_lending_club_target():
    return DATA_FILES["lending_club_target"]


@pytest.fixture(scope="module")
def url_lending_club_transactions():
    return DATA_FILES["lending_club_transactions"]


@pytest.fixture(scope="module")
def url_diabetes():
    return DATA_FILES["diabetes"]


@pytest.fixture(scope="module")
def url_mpg():
    return DATA_FILES["mpg"]


@pytest_asyncio.fixture(scope="module")
async def dataset_loaded(url_diabetes: str, analyst_db: AnalystDB) -> AnalystDataset:
    df = pd.read_csv(url_diabetes)
    # Replace non-JSON compliant values
    df = df.replace([float("inf"), -float("inf")], None)  # Replace infinity with None
    df = df.where(pd.notnull(df), None)  # Replace NaN with None

    # Create dataset dictionary
    dataset = AnalystDataset(
        name=os.path.splitext(os.path.basename(url_diabetes))[0],
        data=df,
    )
    await analyst_db.register_dataset(dataset, data_source=DataSourceType.FILE)
    return dataset


@pytest_asyncio.fixture(scope="module")
async def analyst_db() -> AnalystDB:
    analyst_db = await AnalystDB.create(
        user_id="test_user_123",
        db_path=".",
        dataset_db_name="datasets",
        chat_db_name="chats",
    )
    return analyst_db

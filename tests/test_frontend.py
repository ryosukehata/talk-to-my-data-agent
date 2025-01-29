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

import contextlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, cast

import pytest
from streamlit.testing.v1 import AppTest

from utils.schema import DataDictionary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def cd(new_dir: Path) -> Any:
    """Changes the current working directory to the given path and restores the old directory on exit."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


@pytest.fixture
def application(
    pulumi_up: Any,
    subprocess_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> Any:
    stack_name = subprocess.check_output(
        ["pulumi", "stack", "--show-name"],
        text=True,
    ).split("\n")[0]
    with cd(Path("frontend")):
        subprocess_runner(
            ["pulumi", "stack", "select", stack_name, "--non-interactive"]
        )
        # and ensure we can access `frontend` as if we were running from inside
        sys.path.append(".")
        logger.info(subprocess.check_output(["pulumi", "stack", "output"]))

        yield AppTest.from_file("Connect & Explore.py", default_timeout=180)


@pytest.fixture
def app_post_database_load(application: AppTest, pulumi_up: Any) -> AppTest:
    logger.info(os.getcwd())
    at = application.run()
    expander = next(i for i in at.expander if i.label.lower() == "database")
    # Test assumes access to a database with LENDING_CLUB_PROFILE table
    expander.multiselect[0].select("LENDING_CLUB_PROFILE")
    expander.button[0].click().run(timeout=120)
    return at


def test_database_queried(app_post_database_load: AppTest) -> None:
    success_message = "✅ Data processed and dictionaries generated successfully!"
    assert app_post_database_load.success[1].value == success_message


def test_data_dictionary_generated(app_post_database_load: AppTest) -> None:
    dictionaries = cast(
        list[DataDictionary], app_post_database_load.session_state.data_dictionaries
    )
    for dictionary in dictionaries:
        column_decriptions = dictionary.column_descriptions
        logger.info(column_decriptions)
        assert len(column_decriptions)
        assert "No description available" not in [
            c.description for c in column_decriptions
        ]

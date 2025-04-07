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

import json
import sys
from typing import Dict, Generator, MutableMapping
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

sys.path.append("frontend")

from frontend.datarobot_connect import (
    DataRobotCredentials,
    DataRobotTokenManager,
)


@pytest.fixture(autouse=True)
def mock_dr_client() -> Generator[MagicMock, None, None]:
    """Mock DataRobot client for testing."""
    with patch("datarobot.Client") as mock_client:
        client_instance = MagicMock()
        client_instance.token = "test-token"
        client_instance.endpoint = "https://test-endpoint.datarobot.com"
        client_instance.get = MagicMock(
            return_value=MagicMock(
                ok=True,
                json=MagicMock(
                    return_value={"firstName": "Test", "lastName": "User", "uid": "123"}
                ),
            )
        )
        mock_client.return_value = client_instance
        mock_client.__enter__.return_value = client_instance
        mock_client.__exit__.return_value = None

        yield mock_client


@pytest.fixture
def mock_streamlit_session_state() -> Generator[
    MutableMapping[str, object], None, None
]:
    """Mock Streamlit session state for testing."""
    mock_dict: Dict[str, object] = {}
    with patch.object(st, "session_state", mock_dict) as mock_state:
        mock_state_typed: MutableMapping[str, object] = mock_state
        yield mock_state_typed


@pytest.fixture
def mock_st_javascript() -> Generator[MagicMock, None, None]:
    """Mock the Streamlit JavaScript function for testing."""
    with patch("frontend.datarobot_connect.st_javascript") as mock_js:
        mock_js.return_value = json.dumps(
            {
                "data": [
                    {"key": "test-key-1", "expireAt": None},
                    {"key": "test-key-2", "expireAt": "2025-01-01"},
                ]
            }
        )
        yield mock_js


@patch("frontend.datarobot_connect.DataRobotTokenManager._get_current_credentials")
@patch("frontend.datarobot_connect.DataRobotTokenManager._set_user_credentials")
@patch("frontend.datarobot_connect.DataRobotTokenManager._set_user_info")
@pytest.mark.asyncio
async def test_display_info_with_user(
    mock_set_user_info: MagicMock,
    mock_set_user_credentials: MagicMock,
    mock_get_current_credentials: MagicMock,
) -> None:
    """Test the display_info method with a user."""
    token_manager = DataRobotTokenManager.__new__(DataRobotTokenManager)
    token_manager._user_creds = DataRobotCredentials(token="token", endpoint="endpoint")

    mock_stc = MagicMock()

    mock_session_state: Dict[str, str | None] = {}
    mock_session_state["datarobot_firstName"] = "John"
    mock_session_state["datarobot_lastName"] = "Doe"

    with patch("streamlit.session_state", mock_session_state):
        await token_manager.display_info(mock_stc)

        mock_stc.write.assert_called_once_with("Hello John Doe")
        mock_stc.text_input.assert_not_called()


@patch("frontend.datarobot_connect.DataRobotTokenManager._get_current_credentials")
@patch("frontend.datarobot_connect.DataRobotTokenManager._set_user_credentials")
@patch("frontend.datarobot_connect.DataRobotTokenManager._set_user_info")
@pytest.mark.asyncio
async def test_display_info_without_last_name(
    mock_set_user_info: MagicMock,
    mock_set_user_credentials: MagicMock,
    mock_get_current_credentials: MagicMock,
) -> None:
    """Test the display_info method without a last name."""
    token_manager = DataRobotTokenManager.__new__(DataRobotTokenManager)
    token_manager._user_creds = DataRobotCredentials(token=None, endpoint="endpoint")
    token_manager._original_creds = DataRobotCredentials(
        token="test-token", endpoint="test-endpoint"
    )

    mock_stc = MagicMock()

    mock_session_state: Dict[str, str | None] = {}
    mock_session_state["datarobot_firstName"] = "John"
    mock_session_state["datarobot_lastName"] = None

    with patch("streamlit.session_state", mock_session_state):
        await token_manager.display_info(mock_stc)

        mock_stc.write.assert_called_once_with("Hello John")
        mock_stc.text_input.assert_called_once()

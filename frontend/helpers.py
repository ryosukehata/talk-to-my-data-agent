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
import logging
import time
import traceback
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Coroutine, TypeVar

from streamlit.runtime.state import SessionStateProxy
from typing_extensions import ParamSpec


# Configure logging with custom formatter
class CustomJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "json_data"):
            try:
                if (
                    isinstance(record.json_data, dict)
                    and "messages" in record.json_data
                ):
                    formatted_messages = []
                    for msg in record.json_data["messages"]:
                        formatted_msg = {
                            "role": msg["role"],
                            "content": (
                                msg["content"].replace("\n", "\\n")[:100] + "..."
                                if len(msg["content"]) > 100
                                else msg["content"]
                            ),
                        }
                        formatted_messages.append(formatted_msg)

                    clean_payload = {
                        "model": record.json_data.get("model", ""),
                        "messages": formatted_messages,
                        "response_format": record.json_data.get("response_format", {}),
                        "stream": record.json_data.get("stream", False),
                    }
                    record.msg = f"\nOpenAI Request Payload:\n{json.dumps(clean_payload, indent=2)}"
                else:
                    record.msg = f"\n{json.dumps(record.json_data, indent=2)}"
            except (TypeError, ValueError) as e:
                record.msg = f"Error formatting JSON: {str(e)}\nOriginal data: {record.json_data}"
        return super().format(record)


# Configure logging
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomJsonFormatter())
root_logger = logging.getLogger()
root_logger.handlers = [console_handler]
logger = logging.getLogger("DataAnalystApp")


# Helper functions
def format_json(obj: Any) -> str:
    try:
        if hasattr(obj, "dict"):
            obj = obj.dict()
        if isinstance(obj, dict) and "messages" in obj:
            formatted_obj = obj.copy()
            for msg in formatted_obj["messages"]:
                if len(msg.get("content", "")) > 100:
                    msg["content"] = msg["content"][:100] + "..."
            return json.dumps(
                formatted_obj, indent=2, sort_keys=True, default=str, ensure_ascii=False
            )
        return json.dumps(
            obj, indent=2, sort_keys=True, default=str, ensure_ascii=False
        )
    except Exception as e:
        return f"Error formatting JSON: {str(e)}\nOriginal object: {str(obj)}"


T = TypeVar("T")
P = ParamSpec("P")


def log_api_call(
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        separator = f"\n{'=' * 80}\n"

        logger.info(
            f"{separator}API CALL START: {func.__name__} [{request_id}]{separator}"
        )

        try:
            formatted_args = [
                arg.dict() if hasattr(arg, "dict") else arg for arg in args
            ]
            formatted_kwargs = {
                k: v.dict() if hasattr(v, "dict") else v for k, v in kwargs.items()
            }

            input_log = (
                f"INPUT PARAMETERS [{request_id}]\n"
                "------------------------\n"
                f"Function: {func.__name__}\n"
                f"Timestamp: {datetime.now().isoformat()}\n\n"
                "Arguments:\n"
                f"{format_json(formatted_args)}\n\n"
                "Keyword Arguments:\n"
                f"{format_json(formatted_kwargs)}\n"
            )
            logger.debug(input_log)

            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            if hasattr(result, "request_options"):
                request_options = result.request_options
                formatted_options = {
                    "method": request_options.get("method"),
                    "url": request_options.get("url"),
                    "files": request_options.get("files"),
                    "json_data": request_options.get("json_data", {}),
                }
                logger.debug(
                    f"Request options:\n{json.dumps(formatted_options, indent=2, ensure_ascii=False)}\n"
                )

            output_log = (
                f"OUTPUT RESULTS [{request_id}]\n"
                "------------------------\n"
                f"Function: {func.__name__}\n"
                f"Execution Time: {execution_time:.2f} seconds\n\n"
                "Response:\n"
                f"{format_json(result)}\n"
            )
            logger.debug(output_log)

            logger.info(
                f"{separator}API CALL COMPLETE: {func.__name__} [{request_id}]{separator}"
            )
            return result

        except Exception as e:
            error_log = (
                f"ERROR IN API CALL [{request_id}]\n"
                "------------------------\n"
                f"Function: {func.__name__}\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {str(e)}\n\n"
                "Stack Trace:\n"
            )
            logger.error(error_log, exc_info=True)
            raise

    return wrapper


# Add enhanced error logging function
def log_error_details(error: BaseException, context: dict[str, Any]) -> None:
    """Log detailed error information with context

    Args:
        error: The exception that occurred
        context: Dictionary containing error context
    """
    error_details = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        **context,
    }

    logger.error(
        f"\nERROR DETAILS\n=============\n{json.dumps(error_details, indent=2, default=str)}"
    )


empty_session_state = {
    "initialized": True,
    "datasets": [],
    "cleansed_data": [],
    "data_dictionaries": [],
    "selected_catalog_datasets": [],
    "data_source": None,
    "file_uploader_key": 0,
    "processed_file_ids": [],
    "chat_messages": [],
    "chat_input_key": 0,
    "debug_mode": True,
}


def state_empty(session_state: SessionStateProxy) -> None:
    session_state.update(deepcopy(empty_session_state))  # type: ignore


def state_init(session_state: SessionStateProxy) -> None:
    if "initialized" not in session_state:
        state_empty(session_state)

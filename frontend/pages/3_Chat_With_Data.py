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

import asyncio
import json
import logging
import sys
import time
import traceback
import warnings
from datetime import datetime
from typing import Any, Callable, Coroutine, TypeVar, cast

import pandas as pd
import streamlit as st
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from typing_extensions import ParamSpec

sys.path.append("..")


# Import FastAPI functions directly
from app_settings import PAGE_ICON, DataSource, apply_custom_css, get_page_logo

from utils.api import (
    get_business_analysis,
    rephrase_message,
    run_analysis,
    run_charts,
    run_database_analysis,
)
from utils.schema import (
    AnalystChatMessage,
    AnalystDataset,
    ChatRequest,
    CleansedDataset,
    DataDictionary,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunBusinessAnalysisRequest,
    RunBusinessAnalysisResult,
    RunChartsRequest,
    RunChartsResult,
    RunDatabaseAnalysisRequest,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Add after imports, before session state initialization
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize session state variables at the very beginning of the file
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.datasets = []
    st.session_state.cleansed_data = []
    st.session_state.data_dictionaries = []
    st.session_state.chat_messages = []
    st.session_state.chat_input_key = 0
    st.session_state.debug_mode = True
    st.session_state.data_source = None
elif "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
elif "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0


# Page config
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
apply_custom_css()


def clear_chat() -> None:
    st.session_state.chat_messages = []
    st.session_state.chat_input_key += 1


# Sidebar with New Chat button only
with st.sidebar:
    st.title("Chat Controls")

    # Add New Chat button with callback
    st.button("New Chat", on_click=clear_chat, use_container_width=True)


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

            input_log = f"""
INPUT PARAMETERS [{request_id}]
------------------------
Function: {func.__name__}
Timestamp: {datetime.now().isoformat()}

Arguments:
{format_json(formatted_args)}

Keyword Arguments:
{format_json(formatted_kwargs)}
"""
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
                    f"""
Request options:
{json.dumps(formatted_options, indent=2, ensure_ascii=False)}
"""
                )

            output_log = f"""
OUTPUT RESULTS [{request_id}]
------------------------
Function: {func.__name__}
Execution Time: {execution_time:.2f} seconds

Response:
{format_json(result)}
"""
            logger.debug(output_log)

            logger.info(
                f"{separator}API CALL COMPLETE: {func.__name__} [{request_id}]{separator}"
            )
            return result

        except Exception as e:
            error_log = f"""
ERROR IN API CALL [{request_id}]
------------------------
Function: {func.__name__}
Error Type: {type(e).__name__}
Error Message: {str(e)}

Stack Trace:
"""
            logger.error(error_log, exc_info=True)
            raise

    return wrapper


# Wrap API functions with logging

rephrase_message = log_api_call(rephrase_message)
run_analysis = log_api_call(run_analysis)
run_charts = log_api_call(run_charts)
get_business_analysis = log_api_call(get_business_analysis)
run_database_analysis = log_api_call(run_database_analysis)


# Add enhanced error logging function
def log_error_details(error: Exception, context: dict[str, Any]) -> None:
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


# Update rephrase_message_and_analysis with enhanced error handling
async def rephrase_message_and_analysis(
    question: str, chat_messages: list[ChatCompletionMessageParam]
) -> None:
    st.session_state.datasets = cast(list[AnalystDataset], st.session_state.datasets)
    st.session_state.chat_messages = cast(
        list[AnalystChatMessage], st.session_state.chat_messages
    )
    st.session_state.cleansed_data = cast(
        list[CleansedDataset], st.session_state.cleansed_data
    )
    st.session_state.data_dictionaries = cast(
        list[DataDictionary], st.session_state.data_dictionaries
    )
    error_context = {"question": question, "chat_history_length": len(chat_messages)}

    try:
        # Create placeholder containers in desired order
        with st.chat_message("assistant", avatar="bot.jpg"):
            # Create containers for each section
            bottom_line_container = st.container()
            analysis_container = st.container()
            charts_container = st.container()
            insights_container = st.container()
            followup_container = st.container()

            # Initialize the assistant message structure
            assistant_message = AnalystChatMessage(
                role="assistant", content="", components=[]
            )

            # Run analysis with enhanced error capture
            analysis_result: RunAnalysisResult
            with st.spinner("Running analysis..."):
                try:
                    if st.session_state.data_source == DataSource.DATABASE:
                        # Convert DataFrames to dictionary format
                        sf_analysis_request = RunDatabaseAnalysisRequest(
                            data=st.session_state.cleansed_data,
                            dictionary=st.session_state.data_dictionaries,
                            question=enhanced_question,
                        )
                        analysis_result = await run_database_analysis(
                            sf_analysis_request
                        )
                    else:
                        analysis_request = RunAnalysisRequest(
                            data=st.session_state.cleansed_data,
                            dictionary=st.session_state.data_dictionaries,
                            question=enhanced_question,
                        )
                        analysis_result = await run_analysis(analysis_request)

                    # Store analysis results in components
                    assistant_message.components.append(analysis_result)

                    # Display analysis results
                    with analysis_container:
                        if analysis_result.code:
                            with st.expander("Analysis Code", expanded=False):
                                # Use SQL language highlighting for database mode
                                language = (
                                    "sql"
                                    if st.session_state.get("data_source")
                                    == DataSource.DATABASE
                                    else "python"
                                )
                                st.code(analysis_result.code, language=language)
                        if analysis_result.data:
                            with st.expander("Analysis Results", expanded=True):
                                st.dataframe(
                                    analysis_result.data.to_df(),
                                    use_container_width=True,
                                )

                except Exception as e:
                    st.error(
                        "Error running initial analysis. "
                        "Try rephrasing the question and "
                        f"running again: {str(e)}"
                    )
                    error_context.update({"component": "analysis"})
                    log_error_details(e, error_context)
                    st.stop()

            # Process charts and business analysis concurrently
            if analysis_result and analysis_result.data:
                try:
                    # Prepare requests
                    chart_request = RunChartsRequest(
                        data=analysis_result.data,
                        question=enhanced_question,
                    )

                    business_request = RunBusinessAnalysisRequest(
                        data=analysis_result.data,
                        dictionary=DataDictionary.from_df(analysis_result.data.to_df()),
                        question=enhanced_question,
                    )

                    # Create and start tasks immediately
                    charts_task = asyncio.create_task(run_charts(chart_request))
                    business_task = asyncio.create_task(
                        get_business_analysis(business_request)
                    )

                    # Process both tasks as they complete
                    with st.spinner("Generating analysis..."):
                        # Create tasks list
                        tasks = [charts_task, business_task]

                        # Wait for each task to complete
                        for coro in asyncio.as_completed(tasks):
                            try:
                                result = await coro

                                # Determine which task completed by checking the result structure
                                if isinstance(result, RunChartsResult) and (
                                    result.fig1 or result.fig2
                                ):
                                    # Charts task completed
                                    assistant_message.components.append(result)
                                    with charts_container:
                                        if result.fig1:
                                            st.plotly_chart(
                                                result.fig1, use_container_width=True
                                            )
                                        if result.fig2:
                                            st.plotly_chart(
                                                result.fig2, use_container_width=True
                                            )

                                elif isinstance(result, RunBusinessAnalysisResult):
                                    # Business analysis task completed
                                    assistant_message.components.append(result)

                                    with bottom_line_container:
                                        with st.expander("Bottom Line", expanded=True):
                                            st.markdown(
                                                (result.bottom_line or "").replace(
                                                    "$", "\\$"
                                                )
                                            )

                                    with insights_container:
                                        if result.additional_insights:
                                            with st.expander(
                                                "Additional Insights", expanded=True
                                            ):
                                                st.markdown(
                                                    result.additional_insights.replace(
                                                        "$", "\\$"
                                                    )
                                                )

                                    with followup_container:
                                        if result.follow_up_questions:
                                            with st.expander(
                                                "Follow-up Questions", expanded=True
                                            ):
                                                for q in result.follow_up_questions:
                                                    st.markdown(
                                                        f"- {q}".replace("$", r"\$")
                                                    )

                            except Exception as e:
                                # Determine which task failed by checking remaining tasks
                                task_type = (
                                    "charts" if charts_task in tasks else "business"
                                )
                                error_context.update(
                                    {
                                        "component": f"concurrent_processing_{task_type}",
                                        "task_type": task_type,
                                    }
                                )
                                log_error_details(e, error_context)

                                # Display error for the specific component
                                if task_type == "charts":
                                    with charts_container:
                                        st.error(f"Error generating charts: {str(e)}")
                                else:
                                    with bottom_line_container:
                                        st.error(
                                            f"Error generating business analysis: {str(e)}"
                                        )

                except Exception as e:
                    error_context.update({"component": "concurrent_processing_setup"})
                    log_error_details(e, error_context)
                    st.error(f"Error setting up analysis: {str(e)}")

            # Store the complete message in session state
            st.session_state.chat_messages.append(assistant_message)

    except Exception as e:
        error_context["component"] = "main_process"
        log_error_details(e, error_context)
        st.error(f"Error processing chat and analysis: {str(e)}")


def render_conversation_history(chat_messages: list[AnalystChatMessage]) -> None:
    def render_bottom_line(
        business_analysis_component: RunBusinessAnalysisResult | None,
    ) -> None:
        with st.expander("Bottom Line", expanded=True):
            if business_analysis_component:
                bottom_line = business_analysis_component.bottom_line
                st.markdown(
                    bottom_line.replace("$", "\\$")
                    if bottom_line != ""
                    else "No bottom line available"
                )
            else:
                st.markdown("No bottom line available")

    def render_analysis(
        analysis_component: RunAnalysisResult | None,
    ) -> None:
        language = (
            "sql"
            if st.session_state.get("data_source") == DataSource.DATABASE
            else "python"
        )
        analysis_container = st.container()
        with analysis_container:
            with st.expander("Analysis Results", expanded=True):
                if analysis_component:
                    if analysis_component.data:
                        analysis_data: AnalystDataset = analysis_component.data
                        df = pd.DataFrame(analysis_data.data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown("Analysis results not available")
                else:
                    st.markdown("Analysis results not available")

            with st.expander("Analysis Code", expanded=False):
                if analysis_component:
                    if analysis_component.code:
                        st.code(analysis_component.code, language=language)
                    else:
                        st.markdown("Analysis code not available")
                else:
                    st.markdown("Analysis code not available")

    def render_charts(charts_component: RunChartsResult | None) -> None:
        with st.expander("Charts", expanded=True):
            if charts_component:
                st.plotly_chart(
                    charts_component.fig1,
                    use_container_width=True,
                    key=f"fig1_{i}",
                )
                st.plotly_chart(
                    charts_component.fig2,
                    use_container_width=True,
                    key=f"fig2_{i}",
                )
            else:
                st.markdown("No charts available")

    def render_additional_insights(
        business_analysis_component: RunBusinessAnalysisResult | None,
    ) -> None:
        with st.expander("Additional Insights", expanded=True):
            if business_analysis_component:
                st.markdown(
                    business_analysis_component.additional_insights.replace("$", "\\$")
                )
            else:
                st.markdown("No additional insights available")

    def render_follow_up_questions(
        business_analysis_component: RunBusinessAnalysisResult | None,
    ) -> None:
        with st.expander("Follow-up Questions", expanded=True):
            if business_analysis_component:
                questions = business_analysis_component.follow_up_questions
                for q in questions:
                    st.markdown(f"- {q}".replace("$", r"\$"))
                if not len(questions):
                    st.markdown("No follow-up questions available")
            else:
                st.markdown("No follow-up questions available")

    for i, message in enumerate(chat_messages):
        # Set avatar based on role
        avatar = "bot.jpg" if message.role == "assistant" else "you.jpg"
        analysis_component, business_analysis_component, charts_component = (
            None,
            None,
            None,
        )
        for component in message.components:
            if isinstance(component, (RunAnalysisResult)):
                analysis_component = component
            elif isinstance(component, RunBusinessAnalysisResult):
                business_analysis_component = component
            elif isinstance(component, RunChartsResult):
                charts_component = component

        with st.chat_message(message.role, avatar=avatar):
            st.markdown(message.content)
            if message.role == "assistant":
                render_bottom_line(business_analysis_component)
                render_analysis(analysis_component)
                render_charts(charts_component)
                render_additional_insights(business_analysis_component)
                render_follow_up_questions(business_analysis_component)


# Main page content (Chat Interface)
st.image(get_page_logo(), width=200)
st.session_state.chat_messages = cast(
    list[AnalystChatMessage], st.session_state.chat_messages
)
if not st.session_state.cleansed_data:
    st.info("Please upload and process data using the sidebar before starting the chat")
else:
    render_conversation_history(st.session_state.chat_messages)
    # Chat input
    if question := st.chat_input("Ask a question about your data"):
        valid_messages: list[ChatCompletionMessageParam] = [
            msg.to_openai_message_param()
            for msg in st.session_state.chat_messages
            if msg.role in ["user", "assistant", "system"] and msg.content.strip()
        ]

        valid_messages.append(
            ChatCompletionUserMessageParam(role="user", content=question)
        )
        chat_request = ChatRequest(messages=valid_messages)
        chat_response = asyncio.run(rephrase_message(chat_request))

        enhanced_question = chat_response if chat_response else question
        user_message = AnalystChatMessage(
            role="user", content=enhanced_question, components=[]
        )
        st.session_state.chat_messages.append(user_message)

        # Display user message with custom avatar
        with st.chat_message("user", avatar="you.jpg"):
            st.markdown(enhanced_question)
        # TODO: finalise the ChatMessage typing
        # Process chat and display assistant response
        asyncio.run(rephrase_message_and_analysis(enhanced_question, valid_messages))

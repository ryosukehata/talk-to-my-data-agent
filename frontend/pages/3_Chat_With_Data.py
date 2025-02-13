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
import logging
import sys
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, cast

import streamlit as st
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import ValidationError
from streamlit.delta_generator import DeltaGenerator

sys.path.append("..")
# Import FastAPI functions directly
from app_settings import PAGE_ICON, DataSource, apply_custom_css, display_page_logo
from helpers import log_api_call, log_error_details, state_init

from utils.api import (
    get_business_analysis,
    rephrase_message,
    run_analysis,
    run_charts,
    run_database_analysis,
)
from utils.schema import (
    AnalysisError,
    AnalystChatMessage,
    AnalystDataset,
    ChatRequest,
    DataDictionary,
    EnhancedQuestionGeneration,
    GetBusinessAnalysisRequest,
    GetBusinessAnalysisResult,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunChartsRequest,
    RunChartsResult,
    RunDatabaseAnalysisRequest,
    RunDatabaseAnalysisResult,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Add after imports, before session state initialization
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize session state variables
state_init(st.session_state)

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

    enable_chart_generation = st.checkbox(
        "Generate charts in conversation", value=True, key="enable_chart_generation"
    )
    enable_business_insights = st.checkbox(
        "Enable business insights and follow up questions in conversation",
        value=True,
        key="enable_business_insights",
    )

    # Add New Chat button with callback
    st.button("New Chat", on_click=clear_chat, use_container_width=True)


# Wrap API functions with logging

rephrase_message = log_api_call(rephrase_message)
run_analysis = log_api_call(run_analysis)
run_charts = log_api_call(run_charts)
get_business_analysis = log_api_call(get_business_analysis)
run_database_analysis = log_api_call(run_database_analysis)


@dataclass
class RenderContainers:
    """Containers for UI elements"""

    rephrase: DeltaGenerator
    bottom_line: DeltaGenerator
    analysis: DeltaGenerator
    charts: DeltaGenerator
    insights: DeltaGenerator
    followup: DeltaGenerator


class UnifiedRenderer:
    """Handles rendering for both historical and live messages"""

    def __init__(self, is_live: bool = False):
        self.is_live = is_live
        self._containers: RenderContainers | None = None

    def set_containers(self, containers: RenderContainers) -> None:
        """Set containers for live rendering"""
        self._containers = containers

    @property
    def containers(self) -> RenderContainers:
        if self._containers is None:
            raise ValueError("Containers not initialized")
        return self._containers

    def render_message(
        self,
        message: AnalystChatMessage,
        within_chat_context: bool = False,
    ) -> None:
        """
        Render a single message with all its components
        within_chat_context: If True, assumes we're already inside a chat_message context
        """
        if not within_chat_context:
            with st.chat_message(
                message.role,
                avatar="bot.jpg" if message.role == "assistant" else "you.jpg",
            ):
                self._render_message_content(message)
        else:
            self._render_message_content(message)

    def _render_message_content(self, message: AnalystChatMessage) -> None:
        """Internal method to render message content and components"""
        # Render main content
        if message.role == "user":
            # For user messages, just render the content
            st.markdown(message.content)
        else:
            # For assistant messages, only render the main content if there's no EnhancedQuestionGeneration
            has_enhanced = any(
                isinstance(comp, EnhancedQuestionGeneration)
                for comp in message.components
            )
            if not has_enhanced:
                st.markdown(message.content)

        if message.role == "assistant":
            # Sort components by type for consistent rendering order
            enhanced_q = None
            analysis_result = None
            charts_result = None
            business_result = None
            exception = None

            for component in message.components:
                if isinstance(component, EnhancedQuestionGeneration):
                    enhanced_q = component
                elif isinstance(
                    component, (RunAnalysisResult, RunDatabaseAnalysisResult)
                ):
                    analysis_result = component
                elif isinstance(component, RunChartsResult):
                    charts_result = component
                elif isinstance(component, GetBusinessAnalysisResult):
                    business_result = component
                elif isinstance(component, AnalysisError):
                    exception = component

            # Render components in order
            if enhanced_q:
                with self.containers.rephrase:
                    st.markdown(enhanced_q.enhanced_user_message)

            if analysis_result:
                self.render_analysis_results(
                    analysis_result,
                    isinstance(analysis_result, RunDatabaseAnalysisResult),
                )

            if charts_result:
                self.render_charts(charts_result)

            if business_result:
                self.render_business_results(business_result)
            if exception:
                self.render_exception(exception)

    def render_analysis_results(
        self, result: RunAnalysisResult | RunDatabaseAnalysisResult, is_database: bool
    ) -> None:
        """Render analysis results and code"""

        with self.containers.analysis:
            if result.status == "error":
                self.render_exception(result.metadata.exception)
                return
            if result.code:
                with st.expander("Analysis Code", expanded=False):
                    language = "sql" if is_database else "python"
                    st.code(result.code, language=language)
            if result.dataset:
                with st.expander("Analysis Results", expanded=True):
                    st.dataframe(result.dataset.to_df(), use_container_width=True)

    def render_charts(self, result: RunChartsResult) -> None:
        """Render charts"""
        with self.containers.charts:
            if result.status == "error":
                self.render_exception(result.metadata.exception)

            index = uuid.uuid4()
            if result.fig1:
                st.plotly_chart(
                    result.fig1,
                    use_container_width=True,
                    key=f"message_{index}_fig1",
                )
            if result.fig2:
                st.plotly_chart(
                    result.fig2,
                    use_container_width=True,
                    key=f"message_{index}_fig2",
                )

    def render_business_results(self, result: GetBusinessAnalysisResult) -> None:
        """Render business analysis results"""
        if result.status == "error":
            with self.containers.bottom_line:
                if result.metadata is not None and result.metadata.exception_str:
                    st.error(
                        f"Error running business analysis\n{result.metadata.exception_str}"
                    )
                else:
                    st.error("Error running business analysis")
        with self.containers.bottom_line:
            with st.expander("Bottom Line", expanded=True):
                st.markdown((result.bottom_line or "").replace("$", r"\$"))

        with self.containers.insights:
            if result.additional_insights:
                with st.expander("Additional Insights", expanded=True):
                    st.markdown(result.additional_insights.replace("$", r"\$"))

        with self.containers.followup:
            if result.follow_up_questions:
                with st.expander("Follow-up Questions", expanded=True):
                    for q in result.follow_up_questions:
                        st.markdown(f"- {q}".replace("$", r"\$"))

    def render_exception(self, exception: AnalysisError | None) -> None:
        if (
            exception is None
            or exception.exception_history is None
            or len(exception.exception_history) == 0
        ):
            st.error("An error occurred during analysis. Please retry")
            return
        last_exception = exception.exception_history[-1]
        st.error(f"Error: {last_exception.exception_str}")
        if last_exception.code is not None:
            with st.expander("Last Executed Code"):
                st.code(last_exception.code)


async def execute_business_analysis_and_charts(
    analysis_result: RunAnalysisResult | RunDatabaseAnalysisResult,
    enhanced_message: str,
) -> tuple[
    RunChartsResult | BaseException | None,
    GetBusinessAnalysisResult | BaseException | None,
]:
    analysis_result.dataset = cast(AnalystDataset, analysis_result.dataset)
    # Prepare both requests
    chart_request = RunChartsRequest(
        dataset=analysis_result.dataset,
        question=enhanced_message,
    )

    business_request = GetBusinessAnalysisRequest(
        dataset=analysis_result.dataset,
        dictionary=DataDictionary.from_analyst_df(analysis_result.dataset.to_df()),
        question=enhanced_message,
    )

    if (
        st.session_state.enable_chart_generation
        and st.session_state.enable_business_insights
    ):
        # Run both analyses concurrently
        return await asyncio.gather(
            run_charts(chart_request),
            get_business_analysis(business_request),
            return_exceptions=True,
        )
    elif st.session_state.enable_chart_generation:
        charts_result = await run_charts(chart_request)
        return charts_result, None
    else:
        business_result = await get_business_analysis(business_request)
        return None, business_result


# Usage for historical messages
def render_conversation_history(messages: list[AnalystChatMessage]) -> None:
    renderer = UnifiedRenderer(is_live=False)
    for message in messages:
        renderer.render_message(message)


# Usage for live message
async def run_complete_analysis(
    chat_request: ChatRequest, session_state: Any, error_context: dict[str, Any]
) -> None:
    """Run the complete analysis pipeline"""
    renderer = UnifiedRenderer(is_live=True)

    with st.chat_message("assistant", avatar="bot.jpg"):
        containers = RenderContainers(
            rephrase=st.container(),
            bottom_line=st.container(),
            analysis=st.container(),
            charts=st.container(),
            insights=st.container(),
            followup=st.container(),
        )
        renderer.set_containers(containers)
        try:
            # Initial message processing
            try:
                enhanced_message = await rephrase_message(chat_request)
            except ValidationError:
                st.error("LLM Error, please retry")
                return

            interpretation_message = AnalystChatMessage(
                role="assistant",
                content=enhanced_message,
                components=[
                    EnhancedQuestionGeneration(enhanced_user_message=enhanced_message)
                ],
            )

            # Render initial message
            renderer.render_message(interpretation_message, within_chat_context=True)

            # Create working message copy
            assistant_message = interpretation_message.model_copy()

            # Run main analysis
            with st.spinner("Thinking..."):
                with st.spinner("Generating Code..."):
                    try:
                        is_database = session_state.data_source == DataSource.DATABASE
                        analysis_result = await (
                            run_database_analysis(
                                RunDatabaseAnalysisRequest(
                                    datasets=session_state.datasets,
                                    dictionaries=session_state.data_dictionaries,
                                    question=enhanced_message,
                                ),
                            )
                            if is_database
                            else run_analysis(
                                RunAnalysisRequest(
                                    datasets=session_state.datasets,
                                    dictionaries=session_state.data_dictionaries,
                                    question=enhanced_message,
                                ),
                            )
                        )

                        assistant_message.components.append(analysis_result)
                        renderer.render_analysis_results(analysis_result, is_database)

                    except Exception as e:
                        error_context.update({"component": "analysis"})
                        log_error_details(e, error_context)
                        st.error(
                            f"Error running initial analysis. Try rephrasing: {str(e)}"
                        )
                        return
                # Run concurrent analyses if we have initial results
                if (
                    analysis_result
                    and analysis_result.dataset
                    and (
                        st.session_state.enable_chart_generation
                        or st.session_state.enable_business_insights
                    )
                ):
                    with st.spinner("Generating Insights..."):
                        try:
                            (
                                charts_result,
                                business_result,
                            ) = await execute_business_analysis_and_charts(
                                analysis_result, enhanced_message
                            )

                            # Handle concurrent results
                            if isinstance(charts_result, BaseException):
                                error_context.update(
                                    {"component": "concurrent_processing_charts"}
                                )
                                log_error_details(charts_result, error_context)
                                with containers.charts:
                                    st.error(
                                        f"Error generating charts: {str(charts_result)}"
                                    )
                            elif charts_result is None:
                                pass
                            else:
                                assistant_message.components.append(charts_result)
                                renderer.render_charts(charts_result)

                            if isinstance(business_result, BaseException):
                                error_context.update(
                                    {"component": "concurrent_processing_business"}
                                )
                                log_error_details(business_result, error_context)
                                with containers.bottom_line:
                                    st.error(
                                        f"Error generating business analysis: {str(business_result)}"
                                    )
                            elif business_result is None:
                                pass
                            else:
                                assistant_message.components.append(business_result)
                                renderer.render_business_results(business_result)

                        except Exception as e:
                            error_context.update(
                                {"component": "concurrent_processing_setup"}
                            )
                            log_error_details(e, error_context)
                            st.error(f"Error setting up analysis: {str(e)}")

                # Store the complete message
                session_state.chat_messages.append(assistant_message)

        except Exception as e:
            error_context["component"] = "main_process"
            log_error_details(e, error_context)
            st.error(f"Error processing chat and analysis: {str(e)}")


async def main() -> None:
    # Main page content (Chat Interface)
    display_page_logo()
    st.session_state.chat_messages = cast(
        list[AnalystChatMessage], st.session_state.chat_messages
    )

    if not st.session_state.datasets:
        st.info(
            "Please upload and process data using the sidebar before starting the chat"
        )
    else:
        # Render existing chat history
        renderer = UnifiedRenderer(is_live=False)
        for message in st.session_state.chat_messages:
            with st.chat_message(
                message.role,
                avatar="bot.jpg" if message.role == "assistant" else "you.jpg",
            ):
                containers = RenderContainers(
                    rephrase=st.container(),
                    bottom_line=st.container(),
                    analysis=st.container(),
                    charts=st.container(),
                    insights=st.container(),
                    followup=st.container(),
                )
                renderer.set_containers(containers)
                renderer.render_message(message, within_chat_context=True)
        # Handle new chat input
        if question := st.chat_input("Ask a question about your data"):
            # Create and add user message
            user_message = AnalystChatMessage(
                role="user", content=question, components=[]
            )
            st.session_state.chat_messages.append(user_message)

            # Display user's original message
            with st.chat_message("user", avatar="you.jpg"):
                st.markdown(question)

            # Prepare chat messages
            valid_messages: list[ChatCompletionMessageParam] = [
                msg.to_openai_message_param()
                for msg in st.session_state.chat_messages
                if msg.content.strip()
            ]
            valid_messages.append(
                ChatCompletionUserMessageParam(role="user", content=question)
            )
            # Create chat request and run analysis
            chat_request = ChatRequest(messages=valid_messages)
            error_context = {
                "question": question,
                "chat_history_length": len(valid_messages),
            }
            await run_complete_analysis(chat_request, st.session_state, error_context)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())

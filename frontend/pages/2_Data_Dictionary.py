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
from typing import cast

import streamlit as st
from pandas import DataFrame

sys.path.append("..")
from app_settings import PAGE_ICON, apply_custom_css, display_page_logo
from helpers import state_init

from utils.schema import DataDictionary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables
state_init(st.session_state)

# Page config
st.set_page_config(page_title="Data Dictionary", page_icon=PAGE_ICON, layout="wide")

# Custom CSS
apply_custom_css()

display_page_logo()
st.title("Dictionary")


def save(dictionary: DataDictionary, edited_df: DataFrame) -> None:
    edited = DataDictionary.from_application_df(edited_df, dictionary.name)
    st.session_state.data_dictionaries = [
        edited if d.name == edited.name else d
        for d in st.session_state.data_dictionaries
    ]


async def main() -> None:
    if (
        "data_dictionaries" not in st.session_state
        or len(st.session_state.data_dictionaries) == 0
    ):
        logger.warning("No data dictionaries found in session state")
        st.info(
            "Please upload and process data from the main page to view the data dictionary"
        )
    else:
        # Add debug logging
        logger.info("Data Dictionary page loaded")
        logger.info(f"Session state keys: {st.session_state.keys()}")

        logger.info(f"Found {len(st.session_state.data_dictionaries)} dictionaries")

        st.session_state.data_dictionaries = cast(
            list[DataDictionary], st.session_state.data_dictionaries
        )
        for dictionary in st.session_state.data_dictionaries:
            st.subheader(dictionary.name)
            logger.info(f"Processing dictionary for {dictionary.name}")

            try:
                # Convert dictionary to DataFrame
                dict_df = dictionary.to_application_df()
                logger.info(
                    f"Created DataFrame for {dictionary.name} with shape {dict_df.shape}"
                )

                # Make dictionary editable
                edited_df = st.data_editor(
                    dict_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"dict_editor_{dictionary.name}",
                )

                col1, col2, col3 = st.columns([2, 3, 1])

                with col3:
                    st.button(
                        label="Save changes",
                        on_click=save,
                        args=(dictionary, edited_df),
                        key=f"dict_save_{dictionary.name}",
                        use_container_width=True,
                    )

                with col1:
                    # Download button for dictionary
                    csv = edited_df.to_csv(index=False)
                    st.download_button(
                        label="Download Data Dictionary",
                        data=csv,
                        file_name=f"{dictionary.name}_dictionary.csv",
                        mime="text/csv",
                        key=f"download_dict_{dictionary.name}",
                    )

            except Exception as e:
                logger.error(
                    f"Error processing dictionary for {dictionary.name}: {str(e)}",
                    exc_info=True,
                )
                st.error(f"Error displaying dictionary for {dictionary.name}: {str(e)}")

            st.markdown("---")

    # Add helpful tips
    with st.sidebar:
        st.markdown(
            """
        ### Using the Data Dictionary
        
        The data dictionary provides detailed information about each column in your datasets:
        
        - **Column**: The name of the column
        - **Data Type**: The type of data in the column
        - **Description**: A description of what the data represents
        
        You can:
        - Edit descriptions directly in the table
        - Download the dictionary as CSV
        - Use the information to better understand your data
        """
        )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())

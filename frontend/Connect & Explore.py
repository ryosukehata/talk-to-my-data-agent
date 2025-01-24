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
import os
import sys
import warnings
from typing import Any, cast

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

sys.path.append("..")

from app_settings import (
    PAGE_ICON,
    DataSource,
    apply_custom_css,
    get_database_loader_message,
    get_database_logo,
    get_page_logo,
)

from utils.api import (
    cleanse_dataframes,
    download_catalog_datasets,
    get_dictionary,
    list_catalog_datasets,
)
from utils.database_helpers import Database, app_infra
from utils.schema import (
    AnalystDataset,
    CleansedDataset,
    CleansingReport,
    DataDictionary,
)

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize session state variables
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.datasets = []
    st.session_state.cleansed_data = []
    st.session_state.data_dictionaries = []
    st.session_state.data_source = None
    st.session_state.file_uploader_key = 0
    st.session_state.processed_file_ids = []


# Modify process_data to handle coroutine reuse
def process_data_cached(_datasets: list[AnalystDataset]) -> list[CleansedDataset]:
    """
    Wrapper function to handle async processing with caching
    """
    return asyncio.run(cleanse_dataframes(_datasets))


def generate_dictionaries(
    _cleansed_data: list[CleansedDataset],
) -> list[DataDictionary]:
    """
    Wrapper function to handle async dictionary generation
    """
    return asyncio.run(get_dictionary(_cleansed_data))


def process_uploaded_file(file: UploadedFile) -> list[AnalystDataset]:
    """Process a single uploaded file and return a list of (dataset_name, dataframe) tuples

    Args:
        file: The uploaded file object
    Returns:
        list: List of (dataset_name, dataframe) tuples, or empty list if error
    """
    try:
        logger.info(f"Processing uploaded file: {file.name}")
        file_extension = os.path.splitext(file.name)[1].lower()
        results = []

        if file_extension == ".csv":
            df = pd.read_csv(file)
            dataset_name = os.path.splitext(file.name)[0]
            data = cast(list[dict[str, Any]], df.to_dict(orient="records"))
            results.append(AnalystDataset(name=dataset_name, data=data))
            logger.info(
                f"Loaded CSV {dataset_name}: {len(df)} rows, {len(df.columns)} columns"
            )

        elif file_extension in [".xlsx", ".xls"]:
            # Read all sheets
            excel_file = pd.ExcelFile(file)
            base_name = os.path.splitext(file.name)[0]

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                # Use sheet name as dataset name if multiple sheets, otherwise use file name
                dataset_name = (
                    f"{base_name}_{sheet_name}"
                    if len(excel_file.sheet_names) > 1
                    else base_name
                )
                data = cast(list[dict[str, Any]], df.to_dict(orient="records"))
                results.append(AnalystDataset(name=dataset_name, data=data))
                logger.info(
                    f"Loaded Excel sheet {dataset_name}: {len(df)} rows, {len(df.columns)} columns"
                )
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return results

    except Exception as e:
        logger.error(f"Error loading {file.name}: {str(e)}", exc_info=True)
        return []


def clear_data_callback() -> None:
    """Callback function to clear all data from session state and cache"""
    # Clear session state
    st.session_state.datasets = []
    st.session_state.cleansed_data = []
    st.session_state.data_dictionaries = []
    st.session_state.selected_catalog_datasets = []  # Also clear catalog selection
    st.session_state.data_source = None  # Reset data source flag
    st.session_state.processed_file_ids = []
    st.session_state.file_uploader_key += 1  # Used to clear file_uploader on st.rerun()
    st.rerun()


def process_data_and_update_state(datasets: list[AnalystDataset]) -> None:
    new_dataset_names = [ds.name for ds in datasets]

    # Remove existing datasets with the same name
    st.session_state.datasets = [
        ds for ds in st.session_state.datasets if ds.name not in new_dataset_names
    ]
    st.session_state.cleansed_data = [
        ds for ds in st.session_state.cleansed_data if ds.name not in new_dataset_names
    ]

    # Add the new (or updated) datasets to the session state
    st.session_state.datasets.extend(datasets)

    for ds in datasets:
        st.success(f"✓ {ds.name}: {len(ds.to_df())} rows, {len(ds.columns)} columns")

    # Process the new data
    logger.info("Starting data processing")
    if st.session_state.data_source != "database":
        try:
            cleansed_datasets = process_data_cached(datasets)
        except Exception as e:
            logger.error("Data processing failed")
            st.error(f"❌ Error processing data: {str(e)}")
    else:
        cleansed_datasets = [
            CleansedDataset(
                name=ds.name,
                data=ds.data,
                cleaning_report=CleansingReport(
                    columns_cleaned=[], errors=[], warnings=[]
                ),
            )
            for ds in datasets
        ]

    st.session_state.cleansed_data += cleansed_datasets
    logger.info("Data processing successful, generating dictionaries")

    new_dictionaries = []
    # Generate data dictionaries
    try:
        new_dictionaries = generate_dictionaries(cleansed_datasets)
        st.session_state.data_dictionaries += [
            d
            for d in new_dictionaries
            if d.name not in [d.name for d in st.session_state.data_dictionaries]
        ]

    except Exception:
        st.warning(
            "⚠️ Data processed but there were issues generating some dictionaries"
        )
    if len(new_dictionaries) > 0:
        st.success("✅ Data processed and dictionaries generated successfully!")
        st.info(
            "View the generated data dictionaries in the [Data Dictionary](/Data_Dictionary) page"
        )


# Add callback for AI Catalog dataset selection
def catalog_download_callback() -> None:
    """Callback function for AI Catalog dataset download"""
    if (
        "selected_catalog_datasets" in st.session_state
        and st.session_state.selected_catalog_datasets
    ):
        st.session_state.data_source = DataSource.CATALOG
        with st.sidebar:  # Use sidebar context
            with st.spinner("Loading selected datasets..."):
                selected_ids = [
                    ds["id"] for ds in st.session_state.selected_catalog_datasets
                ]
                dataframes = download_catalog_datasets(*selected_ids)

                process_data_and_update_state(dataframes)


def load_from_database_callback() -> None:
    """Callback function for Database table download"""
    # Set flag to indicate data source is a database
    st.session_state.data_source = DataSource.DATABASE
    if (
        "selected_schema_tables" in st.session_state
        and st.session_state.selected_schema_tables
    ):
        with st.sidebar:
            with st.spinner("Loading selected tables..."):
                dataframes = Database.get_data(*st.session_state.selected_schema_tables)

                if not dataframes:
                    st.error(f"Failed to load data from {app_infra.database}")
                    return

                process_data_and_update_state(dataframes)


def uploaded_file_callback(uploaded_files: list[UploadedFile]) -> None:
    """Callback function for file uploads"""
    # Set flag to indicate data source is a file
    st.session_state.data_source = DataSource.FILE

    with st.spinner("Loading and processing files..."):
        # Process uploaded files
        for file in uploaded_files:
            if file.file_id not in st.session_state.processed_file_ids:
                dataset_results = process_uploaded_file(file)
                process_data_and_update_state(dataset_results)
                st.session_state.processed_file_ids.append(file.file_id)


# Page config
st.set_page_config(page_title="Connect Data", page_icon=PAGE_ICON, layout="wide")


# Custom CSS
apply_custom_css()

# Sidebar for data upload and processing
with st.sidebar:
    st.title("Connect")

    # Load Files expander containing file upload and AI Catalog
    with st.expander("Load Files", expanded=True):
        # File upload section
        col1, col2, col3 = st.columns([1, 4, 2])
        with col1:
            st.image("csv_File_Logo.svg", width=25)
        with col2:
            st.write("**Load Data Files**")
        uploaded_files = st.file_uploader(
            "Select 1 or multiple files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            disabled=st.session_state.data_source == DataSource.DATABASE,
            key=st.session_state.file_uploader_key,
        )
        if uploaded_files:
            uploaded_file_callback(uploaded_files)

        # AI Catalog section
        st.subheader("☁️   DataRobot AI Catalog")

        # Get datasets from catalog
        with st.spinner("Loading datasets from AI Catalog..."):
            datasets = [i.model_dump() for i in list_catalog_datasets()]

        # Create form for dataset selection
        with st.form("catalog_selection_form", border=False):
            selected_catalog_datasets = st.multiselect(
                "Select datasets from AI Catalog",
                options=datasets,
                format_func=lambda x: f"{x['name']} ({x['size']})",
                help="You can select multiple datasets",
                key="selected_catalog_datasets",
                disabled=st.session_state.data_source == DataSource.DATABASE,
            )

            # Form submit button
            submit_button = st.form_submit_button(
                "Load Datasets",
                on_click=catalog_download_callback,
                disabled=st.session_state.data_source == DataSource.DATABASE,
            )

            # Process form submission
            if submit_button and len(selected_catalog_datasets) > 0:
                # The callback will handle the download and processing
                pass
            elif submit_button:
                st.warning("Please select at least one dataset")

    # Database expander
    with st.expander("Database", expanded=False):
        get_database_logo(app_infra)

        schema_tables = Database.get_tables()

        # Create form for Database table selection
        with st.form("table_selection_form", border=False):
            selected_schema_tables = st.multiselect(
                label=get_database_loader_message(app_infra),
                options=schema_tables,
                help="You can select multiple tables",
                key="selected_schema_tables",
                disabled=st.session_state.data_source is not None
                and st.session_state.data_source != DataSource.DATABASE,
            )

            # Form submit button
            submit_button = st.form_submit_button(
                "Load Selected Tables",
                use_container_width=False,
                on_click=load_from_database_callback,
                disabled=st.session_state.data_source is not None
                and st.session_state.data_source != DataSource.DATABASE,
            )

            if submit_button and not selected_schema_tables:
                st.warning("Please select at least one table")

    # Add Clear Data button after the Database expander
    st.sidebar.button(
        "Clear Data",
        on_click=clear_data_callback,
        type="secondary",
        use_container_width=False,
    )

# Main content area
st.image(get_page_logo(), width=200)
st.title("Explore")

# Main content area - conditional rendering based on cleansed data
if not st.session_state.cleansed_data:
    st.info("Upload and process your data using the sidebar to get started")
else:
    for ds_clean in st.session_state.cleansed_data:
        st.subheader(f"{ds_clean.name}")

        # Display cleaning report in expander
        with st.expander("View Cleaning Report"):
            report = ds_clean.cleaning_report
            if report.columns_cleaned:
                st.write("**Columns Cleaned:**")
                st.write(", ".join(report.columns_cleaned))

                if report.warnings:
                    st.write("**Warnings:**")
                    for warning in report.warnings:
                        st.write(f"- {warning}")

                if report.errors:
                    st.error("**Errors:**")
                    for error in report.errors:
                        st.write(f"- {error}")

        # Display dataframe with column filters
        df_display: pd.DataFrame = pd.DataFrame.from_records(ds_clean.data)

        # Create column filters
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input(
                "Search columns",
                key=f"search_{ds_clean.name}",
                help="Filter columns by name",
            )
        with col2:
            n_rows = int(
                st.number_input(
                    "Rows to display",
                    min_value=1,
                    max_value=len(df_display),
                    value=min(10, len(df_display)),
                    step=1,
                    key=f"n_rows_{ds_clean.name}",
                )
            )

        # Filter columns based on search
        if search:
            cols = [col for col in df_display.columns if search.lower() in col.lower()]
        else:
            cols = df_display.columns.tolist()

        # Display filtered dataframe
        st.dataframe(df_display[cols].head(n_rows), use_container_width=True)

        # Download button
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="Download Cleansed Data",
                data=csv,
                file_name=f"{ds_clean.name}_cleansed.csv",
                mime="text/csv",
                key=f"download_{ds_clean.name}",
            )

        st.markdown("---")

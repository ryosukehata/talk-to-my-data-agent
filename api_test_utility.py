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

# type: ignore
import json
import os
import time
import warnings
from datetime import datetime
from typing import Any

import inquirer
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from inquirer import List as InquirerList
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import FastAPI functions directly
from utils.rest_api import (  # type: ignore[attr-defined]
    cleanse_dataframes,
    get_business_analysis,
    get_dictionaries,
    rephrase_message,
    run_analysis,
    run_charts,
    suggest_questions,
)
from utils.schema import (
    AnalystDataset,
    ChatRequest,
    RunAnalysisRequest,
    RunBusinessAnalysisRequest,
    RunChartsRequest,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize rich console for better output formatting
console = Console()

# Available data files
DATA_FILES = {
    "lending_club_profile": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Profile.csv",
    "lending_club_target": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Target.csv",
    "lending_club_transactions": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Transactions.csv",
    "diabetes": "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv",
    "mpg": "https://s3.us-east-1.amazonaws.com/datarobot_public_datasets/auto-mpg.csv",
}


def select_files() -> list[str]:
    """Allow user to select multiple files from the available options"""
    questions = [
        inquirer.Checkbox(
            "files",
            message="Select the files you want to process (use spacebar to select)",
            choices=list(DATA_FILES.keys()),
            # Set default selections for lending club files
            default=["mpg"],
        )
    ]

    answers = inquirer.prompt(questions)
    return [DATA_FILES[file] for file in answers["files"]]


def load_dataframes(files: list[str]) -> list[dict[str, Any]]:
    """Load selected files into dataframes and prepare them for API"""
    datasets = []
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        for file_path in files:
            task = progress.add_task(
                f"Loading {os.path.basename(file_path)}...", total=None
            )

            try:
                # First try reading with default encoding (UTF-8)
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                try:
                    # If UTF-8 fails, try latin-1 encoding
                    df = pd.read_csv(file_path, encoding="latin-1")
                    console.print(
                        f"[yellow]Note: {os.path.basename(file_path)} loaded using latin-1 encoding[/yellow]"
                    )
                except Exception as e:
                    console.print(f"[red]Error loading {file_path}: {str(e)}[/red]")
                    progress.update(task, completed=True)
                    continue
            except Exception as e:
                console.print(f"[red]Error loading {file_path}: {str(e)}[/red]")
                progress.update(task, completed=True)
                continue

            # Replace non-JSON compliant values
            df = df.replace(
                [float("inf"), -float("inf")], None
            )  # Replace infinity with None
            df = df.where(pd.notnull(df), None)  # Replace NaN with None

            # Create dataset dictionary
            dataset = {
                "name": os.path.splitext(os.path.basename(file_path))[0],
                "data": df.to_dict("records"),
            }
            datasets.append(dataset)

            console.print(
                f"✓ Successfully loaded {dataset['name']} with {len(df)} rows and {len(df.columns)} columns"
            )
            progress.update(task, completed=True)

    elapsed_time = time.time() - start_time
    console.print(f"\n[cyan]Loading time: {elapsed_time:.2f} seconds[/cyan]")
    return datasets


async def cleanse_datasets(datasets: list[AnalystDataset]) -> dict[str, Any] | None:
    """Use API function directly to cleanse datasets"""
    try:
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Cleansing datasets...", total=None)

            # Call the API function directly
            result = await cleanse_dataframes(datasets)

            progress.update(task, completed=True)

        elapsed_time = time.time() - start_time
        console.print(f"\n[cyan]Cleansing time: {elapsed_time:.2f} seconds[/cyan]")
        return result.model_dump()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return None


async def main() -> None:
    start_time = time.time()
    console.print("[bold blue]Data Analyst API Test Utility[/bold blue]")
    console.print("Select files to process and test with the API\n")

    # File selection
    selected_files = select_files()
    if not selected_files:
        console.print("[yellow]No files selected. Exiting...[/yellow]")
        return

    # Load dataframes
    console.print("\n[bold]Loading selected files...[/bold]")
    datasets = [AnalystDataset(**df) for df in load_dataframes(selected_files)]

    if not datasets:
        console.print("[red]No datasets were successfully loaded. Exiting...[/red]")
        return

    # Cleanse datasets
    console.print("\n[bold]Cleansing datasets...[/bold]")
    result = await cleanse_datasets(datasets)

    if result:
        # Display results
        console.print("\n[bold green]Cleansing Results:[/bold green]")
        for dataset in result["datasets"]:
            console.print(f"\n[bold]{dataset['name']}[/bold]")
            console.print(
                f"Columns cleaned: {len(dataset['cleaning_report']['columns_cleaned'])}"
            )

            # Create and populate Rich table for the dataset
            df = pd.DataFrame(dataset["data"])
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=True
            )

            # Add columns to the table
            for column in df.columns:
                # Determine column style based on dtype
                style = "cyan"  # default style
                if pd.api.types.is_numeric_dtype(df[column]):
                    style = "yellow"
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    style = "green"
                table.add_column(str(column), style=style)

            # Add rows to the table (first 10 rows)
            for _, row in df.head(10).iterrows():
                # Format numeric values to 2 decimal places and handle None/NaN
                formatted_row = []
                for value in row:
                    if pd.isna(value) or value is None:
                        formatted_row.append("")
                    elif isinstance(value, (float, np.float64)):
                        formatted_row.append(f"{value:.2f}")
                    else:
                        formatted_row.append(str(value))
                table.add_row(*formatted_row)

            # Display the table
            console.print(
                f"\n[bold cyan]First 10 rows of cleaned {dataset['name']}:[/bold cyan]"
            )
            console.print(table)
            console.print("\n" + "-" * 80 + "\n")  # Separator between datasets

            # Display warnings and errors
            if dataset["cleaning_report"]["warnings"]:
                console.print("\nWarnings:")
                for warning in dataset["cleaning_report"]["warnings"]:
                    console.print(f"  • {warning}")

            if dataset["cleaning_report"]["errors"]:
                console.print("\n[red]Errors:[/red]")
                for error in dataset["cleaning_report"]["errors"]:
                    console.print(f"  • {error}")

        # Get and display data dictionary
        console.print("\n[bold]Generating Data Dictionary...[/bold]")
        dict_start_time = time.time()

        # Add debug logging
        console.print(
            f"\n[cyan]Debug: Number of datasets in result: {len(result['datasets'])}[/cyan]"
        )
        for dataset in result["datasets"]:
            console.print(
                f"[cyan]Debug: Dataset '{dataset['name']}' has {len(dataset['data'])} records[/cyan]"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Creating data dictionary...", total=None)
            datasets = [AnalystDataset(**dataset) for dataset in result["datasets"]]

            # Add more debug logging
            console.print(
                f"\n[cyan]Debug: Dictionary request contains {len(datasets)} datasets[/cyan]"
            )
            for dataset in datasets:
                console.print(
                    f"[cyan]Debug: Request dataset '{dataset.name}' has {len(dataset.data)} records[/cyan]"
                )

            dictionary_result = await get_dictionaries(datasets)
            progress.update(task, completed=True)

        dict_elapsed_time = time.time() - dict_start_time

        # After getting dictionary_result
        console.print(
            f"\n[cyan]Debug: Dictionary result type: {type(dictionary_result)}[/cyan]"
        )
        console.print(
            f"[cyan]Debug: Dictionary result keys: {dictionary_result.keys() if isinstance(dictionary_result, dict) else 'Not a dict'}[/cyan]"
        )

        # Convert dictionary_result to dict if it's a Pydantic model
        if hasattr(dictionary_result, "dict"):
            dictionary_result = dictionary_result.model_dump()
            console.print(
                f"[cyan]Debug: Converted dictionary result keys: {dictionary_result.keys()}[/cyan]"
            )

        # Add debug logging for dictionaries content
        console.print(
            f"\n[cyan]Debug: Dictionaries content: {dictionary_result.get('dictionaries', [])}[/cyan]"
        )

        # Group entries by dataset - MODIFY THIS SECTION
        dataset_groups = {}
        # The data is in the 'dictionaries' key, not 'data_dictionary'
        for result_dict in dictionary_result.get("dictionaries", []):
            dataset_name = result_dict["name"]
            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            # The column definitions are in the 'dictionary' key of each result
            dataset_groups[dataset_name].extend(result_dict["dictionary"])

        # Display organized dictionary
        console.print("\n[bold green]Data Dictionary:[/bold green]")

        # Display metadata summary
        total_columns = sum(len(entries) for entries in dataset_groups.values())
        console.print("\n[bold cyan]Data Dictionary Summary:[/bold cyan]")
        summary_table = Table(
            show_header=True, header_style="bold magenta", show_lines=True
        )
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")

        summary_table.add_row("Total Datasets", str(len(dataset_groups)))
        summary_table.add_row("Total Columns Defined", str(total_columns))

        # Add dataset-specific counts
        for dataset_name, entries in dataset_groups.items():
            numeric_cols = sum(
                1
                for e in entries
                if e["data_type"].lower() in ["int64", "float64", "numeric"]
            )
            categorical_cols = sum(
                1
                for e in entries
                if e["data_type"].lower() in ["object", "category", "string"]
            )
            summary_table.add_row(
                f"Columns in {dataset_name}",
                f"Total: {len(entries)} (Numeric: {numeric_cols}, Categorical: {categorical_cols})",
            )

        console.print(summary_table)
        console.print("\n[bold cyan]Detailed Column Definitions:[/bold cyan]")

        # Create Rich table for each dataset's dictionary
        for dataset_name, entries in dataset_groups.items():
            console.print(f"\n[bold cyan]Dataset: {dataset_name}[/bold cyan]")

            # Create Rich table
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=True
            )
            table.add_column("Column", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Description", style="green")

            # Update this section to match the new dictionary structure
            for entry in entries:
                # Add row to table using the correct keys
                table.add_row(
                    entry["column"],  # Changed from column_name
                    entry["data_type"],
                    entry["description"],
                )

            # Display the table
            console.print(table)
            console.print("\n")  # Add spacing between datasets

        console.print(
            f"\n[cyan]Data dictionary generation time: {dict_elapsed_time:.2f} seconds[/cyan]"
        )

        # Add question suggestion functionality
        console.print("\n[bold]Suggesting Analysis Questions...[/bold]")
        question_start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Generating questions...", total=None)
            questions_result = await suggest_questions(datasets)
            progress.update(task, completed=True)

        question_elapsed_time = time.time() - question_start_time

        # Convert questions_result to dict if it's a Pydantic model
        if hasattr(questions_result, "dict"):
            questions_result = questions_result.dict()

        # Display suggested questions
        console.print("\n[bold green]Suggested Analysis Questions:[/bold green]")
        suggested_questions = [
            q["question"] for q in questions_result.get("questions", [])
        ][:3]  # Get just the question text from first 3 questions
        for i, question in enumerate(suggested_questions, 1):
            console.print(f"{i}. {question}")

        # Create question selection prompt
        questions = [
            InquirerList(
                "selected_question",
                message="Select a question to analyze",
                choices=[
                    *[f"{i + 1}. {q}" for i, q in enumerate(suggested_questions)],
                    "4. Enter my own question",
                ],
            )
        ]

        answer = inquirer.prompt(questions)

        if answer["selected_question"].startswith("4."):
            custom_question = [
                inquirer.Text("custom", message="Enter your analysis question")
            ]
            custom_answer = inquirer.prompt(custom_question)
            selected_question = custom_answer["custom"]
        else:
            # Extract the question text from the selected option
            selected_question = answer["selected_question"].split(". ", 1)[1]

        console.print(
            f"\n[bold cyan]Selected Question:[/bold cyan] {selected_question}"
        )
        console.print(
            f"\n[cyan]Question suggestion time: {question_elapsed_time:.2f} seconds[/cyan]"
        )

        # Add chat enhancement step
        console.print("\n[bold]Enhancing question with chat analysis...[/bold]")

        # Create chat history table
        history_table = Table(
            show_header=True, header_style="bold magenta", show_lines=True
        )
        history_table.add_column("Original Question", style="cyan")
        history_table.add_column("Enhanced Question", style="yellow")

        # Add previous questions to history if they exist
        if "question_history" not in result:
            result["question_history"] = []

        # Display question history
        if result["question_history"]:
            console.print("\n[bold cyan]Question History:[/bold cyan]")
            for hist in result["question_history"]:
                history_table.add_row(hist["original"], hist["enhanced"])
            console.print(history_table)

        # Create chat request with history context
        chat_messages = []
        # Add history context if available
        for hist in result["question_history"][
            -2:
        ]:  # Include last 2 questions for context
            chat_messages.extend(
                [
                    {"role": "user", "content": hist["original"]},
                    {"role": "assistant", "content": hist["enhanced"]},
                ]
            )
        # Add current question
        chat_messages.append({"role": "user", "content": selected_question})

        chat_request = ChatRequest(messages=chat_messages)
        chat_result = await rephrase_message(chat_request)
        # Add new question to history
        result["question_history"].append(
            {
                "original": selected_question,
                "enhanced": chat_result or selected_question,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Display current question enhancement
        console.print("\n[bold cyan]Current Question Enhancement:[/bold cyan]")
        current_table = Table(
            show_header=True, header_style="bold magenta", show_lines=True
        )
        current_table.add_column("Original Question", style="cyan")
        current_table.add_column("Enhanced Question", style="yellow")
        current_table.add_row(
            selected_question,
            chat_result or selected_question,
        )
        console.print(current_table)

        # Start analysis loop
        analysis_success = False
        while not analysis_success:
            try:
                # Run analysis
                console.print("\n[bold]Running Analysis...[/bold]")
                analysis_start_time = time.time()

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    task = progress.add_task("Analyzing data...", total=None)

                    # Create analysis request with cleaned data
                    analysis_request = RunAnalysisRequest(
                        datasets={
                            dataset["name"]: dataset["data"]
                            for dataset in result["datasets"]
                        },
                        dictionaries={
                            dataset["name"]: [
                                {
                                    "column": col["column"],
                                    "description": col["description"],
                                    "data_type": col["data_type"],
                                }
                                for col in dataset.get("dictionary", [])
                            ]
                            for dataset in dictionary_result.get("dictionaries", [])
                        },
                        question=chat_result or selected_question,
                    )

                    try:
                        analysis_result = await run_analysis(analysis_request)
                        analysis_result = analysis_result.model_dump()
                        progress.update(task, completed=True)

                        # Display analysis execution details
                        console.print(
                            "\n[bold cyan]Analysis Execution Details:[/bold cyan]"
                        )
                        if "metadata" in analysis_result:
                            metadata = analysis_result["metadata"]
                            if "code_generation" in metadata:
                                console.print(
                                    f"Code Generation Attempts: {metadata['code_generation']['attempts']}"
                                )
                                if metadata["code_generation"]["validation_history"]:
                                    console.print("Validation History:")
                                    for idx, error in enumerate(
                                        metadata["code_generation"][
                                            "validation_history"
                                        ],
                                        1,
                                    ):
                                        console.print(f"  {idx}. {error}")

                            console.print(
                                f"Datasets Analyzed: {metadata.get('datasets_analyzed', 'N/A')}"
                            )
                            console.print(
                                f"Total Rows Analyzed: {metadata.get('total_rows_analyzed', 'N/A')}"
                            )
                            console.print(
                                f"Total Columns Analyzed: {metadata.get('total_columns_analyzed', 'N/A')}"
                            )

                            if "stdout" in metadata:
                                console.print("\n[bold]Analysis Output:[/bold]")
                                console.print(metadata["stdout"])
                            if "stderr" in metadata:
                                console.print("\n[bold red]Analysis Errors:[/bold red]")
                                console.print(metadata["stderr"])

                        analysis_success = True  # Mark as successful

                        # Display analysis results
                        console.print("\n[bold green]Analysis Results:[/bold green]")

                        # Display generated code
                        if analysis_result.get("code"):
                            console.print(
                                "\n[bold cyan]Generated Python Code:[/bold cyan]"
                            )
                            console.print("```python")
                            console.print(analysis_result["code"])
                            console.print("```")

                        # Display analysis summary
                        if analysis_result.get("summary"):
                            console.print("\n[bold cyan]Analysis Summary:[/bold cyan]")
                            console.print(analysis_result["summary"])

                        # Display analysis data results
                        if "data" in analysis_result and analysis_result["data"]:
                            console.print(
                                "\n[bold cyan]Analysis Results Data:[/bold cyan]"
                            )
                            result_df = pd.DataFrame(analysis_result["data"])

                            # Format numeric columns
                            numeric_cols = result_df.select_dtypes(
                                include=["float64", "float32"]
                            ).columns
                            for col in numeric_cols:
                                result_df[col] = result_df[col].round(2)

                            console.print(result_df.to_string(index=False))

                        # Display visualizations
                        if analysis_result.get("visualizations"):
                            console.print("\n[bold cyan]Visualizations:[/bold cyan]")
                            for viz in analysis_result["visualizations"]:
                                console.print(f"\n{viz['title']}")
                                console.print(viz["data"])

                        # Display insights
                        if analysis_result.get("insights"):
                            console.print("\n[bold cyan]Key Insights:[/bold cyan]")
                            for idx, insight in enumerate(
                                analysis_result["insights"], 1
                            ):
                                console.print(f"{idx}. {insight}")

                    except Exception as e:
                        progress.update(task, completed=True)
                        console.print("\n[bold red]Analysis Failed[/bold red]")

                        # Create error details table
                        error_table = Table(
                            show_header=True, header_style="bold red", show_lines=True
                        )
                        error_table.add_column("Error Details", style="red")
                        error_table.add_column("Information", style="yellow")

                        # Extract error details if available
                        if hasattr(e, "detail") and isinstance(e.detail, dict):
                            error_context = e.detail.get("context", {})

                            # Add basic error info
                            error_table.add_row(
                                "Error Type", error_context.get("error_type", "Unknown")
                            )
                            error_table.add_row(
                                "Error Message",
                                error_context.get("error_message", str(e)),
                            )

                            # Add attempt information
                            if "attempts" in error_context:
                                error_table.add_row(
                                    "Total Attempts", str(error_context["attempts"])
                                )

                            # Display the table
                            console.print("\n[bold red]Error Details:[/bold red]")
                            console.print(error_table)

                            # Display code generation history if available
                            if error_context.get("code_history"):
                                console.print(
                                    "\n[bold cyan]Code Generation History:[/bold cyan]"
                                )
                                for attempt in error_context["code_history"]:
                                    console.print(
                                        f"\n[bold]Attempt {attempt['attempt']} at {attempt['timestamp']}:[/bold]"
                                    )
                                    if "error" in attempt:
                                        console.print(
                                            f"[red]Error: {attempt['error']}[/red]"
                                        )
                                    console.print("\nGenerated Code:")
                                    console.print("```python")
                                    console.print(attempt["code"])
                                    console.print("```")
                                    if "stdout" in attempt and attempt["stdout"]:
                                        console.print("\nOutput:")
                                        console.print(attempt["stdout"])
                                    if "stderr" in attempt and attempt["stderr"]:
                                        console.print("\nErrors:")
                                        console.print(f"[red]{attempt['stderr']}[/red]")
                        else:
                            # Simple error case
                            error_table.add_row("Error Message", str(e))
                            console.print(error_table)

                        # Ask user what they want to do
                        questions = [
                            InquirerList(
                                "error_action",
                                message="How would you like to proceed?",
                                choices=[
                                    "1. Try a different question",
                                    "2. Exit the program",
                                ],
                            )
                        ]

                        error_answer = inquirer.prompt(questions)

                        if error_answer["error_action"].startswith("2"):
                            console.print("\n[yellow]Exiting program...[/yellow]")
                            return
                        else:
                            # Get new question from user
                            custom_question = [
                                inquirer.Text(
                                    "custom", message="Enter your new analysis question"
                                )
                            ]
                            custom_answer = inquirer.prompt(custom_question)
                            selected_question = custom_answer["custom"]

            except Exception as e:
                console.print(f"\n[bold red]Unexpected error: {str(e)}[/bold red]")
                return

        # Only proceed with business analysis and charts if analysis was successful
        if analysis_success:
            analysis_start_time = time.time()
            analysis_elapsed_time = time.time() - analysis_start_time
            console.print(
                f"\n[cyan]Analysis time: {analysis_elapsed_time:.2f} seconds[/cyan]"
            )

            # Run business analysis
            console.print("\n[bold]Running Business Analysis...[/bold]")
            # analysis_start_time = time.time()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "Analyzing business implications...", total=None
                )

                try:
                    # Create business analysis request
                    if "data" in analysis_result and isinstance(
                        analysis_result["data"], list
                    ):
                        business_request = RunBusinessAnalysisRequest(
                            dataset=analysis_result["data"],
                            dictionary=[
                                {
                                    "column": col["column"],
                                    "description": col["description"],
                                    "data_type": col["data_type"],
                                }
                                for dataset in dictionary_result.get("dictionaries", [])
                                for col in dataset.get("dictionary", [])
                            ],
                            question=chat_result or selected_question,
                        )

                        # Get business analysis
                        business_analysis = await get_business_analysis(
                            business_request
                        )
                        business_analysis = business_analysis.model_dump()
                        progress.update(task, completed=True)

                        # Display business analysis results
                        console.print("\n[bold green]Business Analysis:[/bold green]")

                        # Display The Bottom Line
                        console.print("\n[bold cyan]The Bottom Line:[/bold cyan]")
                        console.print(business_analysis["bottom_line"])

                        # Display Additional Insights
                        console.print("\n[bold cyan]Additional Insights:[/bold cyan]")
                        console.print(business_analysis["additional_insights"])

                        # Display Follow-up Questions
                        console.print("\n[bold cyan]Follow-up Questions:[/bold cyan]")
                        for i, question in enumerate(
                            business_analysis["follow_up_questions"], 1
                        ):
                            console.print(f"{i}. {question}")

                        # Add business analysis to results
                        result["business_analysis"] = business_analysis

                except Exception as e:
                    console.print("\n[bold red]Business Analysis Failed:[/bold red]")
                    console.print(f"Error: {str(e)}")

            # Run charts
            console.print("\n[bold]Generating Charts...[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Creating visualizations...", total=None)

                try:
                    # Get the analysis results data
                    analysis_data = analysis_result.get("data", [])

                    # Create charts request
                    charts_request = RunChartsRequest(
                        dataset=analysis_data,
                        question=chat_result or selected_question,
                    )

                    # Run charts
                    charts_result = await run_charts(charts_request)
                    charts_result = charts_result.model_dump()
                    progress.update(task, completed=True)

                    # Open charts in browser if available
                    if charts_result.get("fig1") and charts_result.get("fig2"):
                        console.print(
                            "\n[bold cyan]Opening Charts in Browser...[/bold cyan]"
                        )

                        try:
                            # Create HTML file for charts
                            html_content = f"""
                            <html>
                            <head>
                                <title>Analysis Charts</title>
                                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                                <style>
                                    .chart-container {{ margin: 20px; padding: 20px; }}
                                    body {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                                    h2 {{ text-align: center; color: #444; }}
                                </style>
                            </head>
                            <body>
                                <h2>{chat_result or selected_question}</h2>
                                <div class="chart-container">
                                    <div id="chart1" style="width: 100%; height: 600px;"></div>
                                </div>
                                <div class="chart-container">
                                    <div id="chart2" style="width: 100%; height: 600px;"></div>
                                </div>
                                <script>
                                    var fig1 = {charts_result["fig1"].to_json()};
                                    var fig2 = {charts_result["fig2"].to_json()};
                                    Plotly.newPlot('chart1', fig1.data, fig1.layout);
                                    Plotly.newPlot('chart2', fig2.data, fig2.layout);
                                </script>
                            </body>
                            </html>
                            """

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            html_file = f"analysis_charts_{timestamp}.html"

                            with open(html_file, "w", encoding="utf-8") as f:
                                f.write(html_content)

                            import webbrowser

                            webbrowser.open(f"file://{os.path.abspath(html_file)}")
                            console.print(f"Charts saved to: {html_file}")

                        except Exception as e:
                            console.print(
                                f"\n[red]Error saving/opening charts: {str(e)}[/red]"
                            )
                    else:
                        console.print(
                            "\n[yellow]Warning: One or both charts were not generated successfully[/yellow]"
                        )

                except Exception as e:
                    console.print("\n[bold red]Chart Generation Failed:[/bold red]")
                    console.print(f"Error: {str(e)}")

            # After completing the full analysis cycle, prompt for follow-up
            while True:
                # Get follow-up questions from business analysis if available
                follow_up_questions = []
                if (
                    "business_analysis" in result
                    and "follow_up_questions" in result["business_analysis"]
                ):
                    follow_up_questions = result["business_analysis"][
                        "follow_up_questions"
                    ]

                # Create question selection prompt
                questions = [
                    InquirerList(
                        "selected_question",
                        message="Would you like to analyze another question?",
                        choices=[
                            *[
                                f"{i + 1}. {q}"
                                for i, q in enumerate(follow_up_questions)
                            ],
                            f"{len(follow_up_questions) + 1}. Enter my own question",
                            f"{len(follow_up_questions) + 2}. Exit",
                        ],
                    )
                ]

                answer = inquirer.prompt(questions)

                # Check if user wants to exit
                if answer["selected_question"].endswith("Exit"):
                    break

                # Get the new question
                if answer["selected_question"].startswith(
                    f"{len(follow_up_questions) + 1}."
                ):
                    custom_question = [
                        inquirer.Text("custom", message="Enter your analysis question")
                    ]
                    custom_answer = inquirer.prompt(custom_question)
                    selected_question = custom_answer["custom"]
                else:
                    # Extract the question text from the selected option
                    selected_question = answer["selected_question"].split(". ", 1)[1]

                console.print(
                    f"\n[bold cyan]Selected Question:[/bold cyan] {selected_question}"
                )

                # Run the analysis cycle again with the new question
                # Chat enhancement
                console.print("\n[bold]Enhancing question with chat analysis...[/bold]")

                # Create chat request with history context
                chat_messages = []
                # Add history context if available
                for hist in result["question_history"][
                    -2:
                ]:  # Include last 2 questions for context
                    chat_messages.extend(
                        [
                            {"role": "user", "content": hist["original"]},
                            {"role": "assistant", "content": hist["enhanced"]},
                        ]
                    )
                # Add current question
                chat_messages.append({"role": "user", "content": selected_question})

                chat_request = ChatRequest(messages=chat_messages)
                chat_result = await rephrase_message(chat_request)
                # Add new question to history
                result["question_history"].append(
                    {
                        "original": selected_question,
                        "enhanced": chat_result or selected_question,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Display current question enhancement
                console.print("\n[bold cyan]Current Question Enhancement:[/bold cyan]")
                current_table = Table(
                    show_header=True, header_style="bold magenta", show_lines=True
                )
                current_table.add_column("Original Question", style="cyan")
                current_table.add_column("Enhanced Question", style="yellow")
                current_table.add_row(
                    selected_question,
                    chat_result or selected_question,
                )
                console.print(current_table)

                # Reset analysis_success flag for the new question
                analysis_success = False
                # Continue with the main analysis loop
                break

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"cleansing_results_{timestamp}.json"

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=serialize_result)
        console.print(f"\nDetailed results saved to: {result_file}")

    total_elapsed_time = time.time() - start_time
    console.print(
        f"\n[cyan]Total execution time: {total_elapsed_time:.2f} seconds[/cyan]"
    )


def serialize_result(obj):
    """Convert non-serializable objects to serializable format"""
    if isinstance(obj, go.Figure):
        return {"type": "plotly_figure", "data": obj.to_dict()}
    elif isinstance(obj, pd.DataFrame):
        return {"type": "dataframe", "data": obj.to_dict("records")}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

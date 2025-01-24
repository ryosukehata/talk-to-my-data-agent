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

import decimal
import json
import os
import time
import warnings
from datetime import date, datetime
from json import JSONEncoder
from typing import Any, Dict, List

import inquirer
import numpy as np
import pandas as pd
import snowflake.connector
from inquirer import List as InquirerList
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from utils.rest_api import (  # type: ignore[attr-defined]
    get_dictionary,
    rephrase_message,
    run_database_analysis,
    suggest_questions,
)
from utils.schema import (
    AnalystDataset,
    ChatRequest,
    RunDatabaseAnalysisRequest,
)

console = Console()

warnings.filterwarnings("ignore")


# Add custom JSON encoder
class DecimalEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, decimal.Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


def load_snowflake_credentials() -> Dict[str, str]:
    """Load Snowflake credentials from .env file"""

    return {
        "user": os.getenv("SNOWFLAKE_USER"),  # type: ignore[dict-item]
        "password": os.getenv("SNOWFLAKE_PASSWORD"),  # type: ignore[dict-item]
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),  # type: ignore[dict-item]
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),  # type: ignore[dict-item]
        "database": os.getenv("SNOWFLAKE_DATABASE"),  # type: ignore[dict-item]
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),  # type: ignore[dict-item]
    }


def connect_to_snowflake(
    credentials: Dict[str, str],
) -> snowflake.connector.SnowflakeConnection:
    """Create connection to Snowflake"""
    try:
        conn = snowflake.connector.connect(
            user=credentials["user"],
            password=credentials["password"],
            account=credentials["account"],
            warehouse=credentials["warehouse"],
            database=credentials["database"],
            schema=credentials["schema"],
        )
        console.print("[green]Successfully connected to Snowflake![/green]")
        return conn
    except Exception as e:
        console.print(f"[red]Error connecting to Snowflake: {str(e)}[/red]")
        raise


def serialize_snowflake_data(data: Any) -> Any:
    """Convert Snowflake data types to JSON serializable formats

    Args:
        data: Data to serialize

    Returns:
        JSON serializable data
    """
    if isinstance(data, (datetime, pd.Timestamp)):
        return data.isoformat()
    elif isinstance(data, (date, pd.Period)):
        # TODO: does this make sense?
        return data.isoformat()  # type: ignore
    elif isinstance(data, decimal.Decimal):
        return float(data)
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif isinstance(data, dict):
        return {k: serialize_snowflake_data(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [serialize_snowflake_data(x) for x in data]
    elif pd.isna(data):
        return None
    return data


def get_table_sample(
    conn: snowflake.connector.SnowflakeConnection, table: str, sample_size: int = 10
) -> List[Dict[str, Any]]:
    """Get a random sample of rows from a Snowflake table

    Args:
        conn: Snowflake connection
        table: Table name
        sample_size: Number of rows to sample

    Returns:
        List of dictionaries containing the sampled rows
    """
    try:
        # Use TABLESAMPLE to get random rows efficiently
        query = f"""
        SELECT *
        FROM {table}
        TABLESAMPLE BERNOULLI ({sample_size} ROWS)
        """

        # Execute query and fetch results
        cursor = conn.cursor(snowflake.connector.DictCursor)
        cursor.execute(query)
        raw_data: list[dict[str, Any]] = cursor.fetchall()  # type: ignore[assignment]

        # Serialize the data to ensure JSON compatibility
        sample_data = [
            {k: serialize_snowflake_data(v) for k, v in row.items()} for row in raw_data
        ]

        return sample_data

    except Exception as e:
        console.print(f"[red]Error getting sample from {table}: {str(e)}[/red]")
        raise
    finally:
        if cursor:
            cursor.close()


def get_table_metadata(
    conn: snowflake.connector.SnowflakeConnection, table: str
) -> Dict[str, Any]:
    """Get metadata for a table including column info and sample data

    Args:
        conn: Snowflake connection
        table: Table name

    Returns:
        Dictionary containing table metadata and sample data
    """
    try:
        # Get column information using DESCRIBE TABLE
        cursor = conn.cursor()
        cursor.execute(f"DESCRIBE TABLE {table}")
        columns = [
            {"name": row[0], "type": row[1], "nullable": row[3] == "Y"}
            for row in cursor.fetchall()
        ]

        # Get sample data
        sample_data = get_table_sample(conn, table)

        return {"table_name": table, "columns": columns, "sample_data": sample_data}
    except Exception as e:
        console.print(f"[red]Error getting metadata for {table}: {str(e)}[/red]")
        raise
    finally:
        if cursor:
            cursor.close()


def display_query_results(results: List[Dict[str, Any]], max_rows: int = 20) -> None:
    """Display query results in a formatted table"""
    if not results:
        console.print("[yellow]No results returned[/yellow]")
        return

    # Create results table
    results_table = Table(
        show_header=True, header_style="bold magenta", show_lines=True
    )

    # Add columns
    for col in results[0].keys():
        results_table.add_column(str(col))

    # Add rows (limit to max_rows)
    for row in results[:max_rows]:
        # Format values
        formatted_row = []
        for val in row.values():
            if isinstance(val, (float, decimal.Decimal)):
                formatted_row.append(f"{float(val):.2f}")
            elif val is None:
                formatted_row.append("")
            else:
                formatted_row.append(str(val))
        results_table.add_row(*formatted_row)

    console.print(results_table)

    if len(results) > max_rows:
        console.print(
            f"\n[yellow]Showing first {max_rows} of {len(results)} rows[/yellow]"
        )


def get_available_tables(conn: snowflake.connector.SnowflakeConnection) -> List[str]:
    """Get list of available tables in the schema"""
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[1] for row in cursor.fetchall()]
        cursor.close()
        return tables
    except Exception as e:
        console.print(f"[red]Error getting tables: {str(e)}[/red]")
        raise


def select_tables(tables: List[str]) -> list[str]:
    """Allow user to select tables from the available options"""
    questions = [
        inquirer.Checkbox(
            "tables",
            message="Select the tables you want to analyze (use spacebar to select)",
            choices=tables,
        )
    ]

    answers = inquirer.prompt(questions)
    return answers["tables"]  # type: ignore[no-any-return]


async def main() -> None:
    start_time = time.time()
    console.print("[bold blue]Snowflake Analysis Test Utility[/bold blue]")

    try:
        # Load credentials and connect to Snowflake
        credentials = load_snowflake_credentials()

        conn = connect_to_snowflake(credentials)

        # Get and select tables
        available_tables = get_available_tables(conn)
        if not available_tables:
            console.print("[yellow]No tables found in schema. Exiting...[/yellow]")
            return

        console.print("\n[bold]Available Tables:[/bold]")
        for table in available_tables:
            console.print(f"  • {table}")

        selected_tables = select_tables(available_tables)
        if not selected_tables:
            console.print("[yellow]No tables selected. Exiting...[/yellow]")
            return

        # Get metadata for selected tables
        console.print("\n[bold]Getting metadata from selected tables...[/bold]")
        tables_metadata = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            for table in selected_tables:
                task = progress.add_task(f"Getting metadata for {table}...", total=None)
                tables_metadata[table] = get_table_metadata(conn, table)
                progress.update(task, completed=True)
                console.print(f"✓ Got metadata for {table}")

        # Generate data dictionary
        console.print("\n[bold]Generating Data Dictionary...[/bold]")
        dict_start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Creating data dictionary...", total=None)

            try:
                # Prepare request for get_dictionary
                datasets = [
                    AnalystDataset(
                        name=table, data=tables_metadata[table]["sample_data"]
                    )
                    for table in selected_tables
                ]

                dictionary_result = await get_dictionary(datasets)
                progress.update(task, completed=True)

                # Display dictionary results
                console.print(
                    "\n[bold green]Data Dictionary Generated Successfully[/bold green]"
                )

                # Create a table to display the dictionary
                dict_table = Table(
                    show_header=True, header_style="bold magenta", show_lines=True
                )
                dict_table.add_column("Table", style="cyan")
                dict_table.add_column("Column", style="yellow")
                dict_table.add_column("Type", style="green")
                dict_table.add_column("Description", style="white")

                # Format dictionary result for display
                formatted_dict: dict[str, Any] = {}
                for dict_entry in dictionary_result:
                    table_name = dict_entry.name
                    for col_info in dict_entry.dictionary:
                        dict_table.add_row(
                            table_name,
                            col_info.column,
                            col_info.data_type,
                            col_info.description,
                        )
                        # Also store in formatted dict for later use
                        if table_name not in formatted_dict:
                            formatted_dict[table_name] = {}
                        formatted_dict[table_name][col_info.column] = col_info

                console.print(dict_table)

                # Store formatted dictionary for later use
                dictionary_result = formatted_dict  # type: ignore

            except Exception as e:
                progress.update(task, completed=True)
                console.print(
                    "\n[bold red]Error Generating Data Dictionary:[/bold red]"
                )
                console.print(f"Error: {str(e)}")
                console.print("\nDebug Information:")
                console.print(f"Number of tables: {len(selected_tables)}")
                for table in selected_tables:
                    sample_count = len(tables_metadata[table]["sample_data"])
                    console.print(f"Table {table}: {sample_count} sample rows")
                return

        dict_elapsed_time = time.time() - dict_start_time
        console.print(
            f"\n[cyan]Dictionary generation time: {dict_elapsed_time:.2f} seconds[/cyan]"
        )

        # Get question suggestions
        console.print("\n[bold]Suggesting Analysis Questions...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Generating questions...", total=None)
            questions_result = await suggest_questions(datasets)
            progress.update(task, completed=True)

        # Display suggested questions
        console.print("\n[bold green]Suggested Analysis Questions:[/bold green]")
        suggested_questions = questions_result[:3]  # Get first 3 questions

        for i, question in enumerate(suggested_questions, 1):
            console.print(f"{i}. {question.question}")

        # Create question selection prompt
        questions = [
            InquirerList(
                "selected_question",
                message="Select a question to analyze",
                choices=[
                    *[
                        f"{i + 1}. {q.question}"
                        for i, q in enumerate(suggested_questions)
                    ],
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
            selected_question = answer["selected_question"].split(". ", 1)[1]

        console.print(
            f"\n[bold cyan]Selected Question:[/bold cyan] {selected_question}"
        )

        # Add chat enhancement step
        console.print("\n[bold]Enhancing question with chat analysis...[/bold]")

        # Create chat request with history context
        chat_messages = [{"role": "user", "content": selected_question}]
        chat_request = ChatRequest(messages=chat_messages)
        chat_result = await rephrase_message(chat_request)

        enhanced_question = chat_result if chat_result else selected_question

        # Display current question enhancement
        console.print("\n[bold cyan]Enhanced Question:[/bold cyan]")
        console.print(enhanced_question)

        # Run Snowflake analysis
        console.print("\n[bold]Running Snowflake Analysis...[/bold]")
        max_attempts = 3

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing data...", total=None)

            # Create analysis request
            analysis_request = RunDatabaseAnalysisRequest(
                data={
                    table: metadata["sample_data"]
                    for table, metadata in tables_metadata.items()
                },
                dictionary=dictionary_result,
                question=enhanced_question,
                warehouse=credentials["warehouse"],
                database=credentials["database"],
                db_schema=credentials["schema"],
            )

            try:
                _analysis_result = await run_database_analysis(analysis_request)
                analysis_result = _analysis_result.model_dump()
                progress.update(task, completed=True)

                # Always display the last generated SQL code
                if analysis_result.get("code"):
                    console.print("\n[bold cyan]Generated SQL Query:[/bold cyan]")
                    console.print("```sql")
                    console.print(analysis_result["code"])
                    console.print("```")
                elif analysis_result.get("last_generated_code"):
                    console.print(
                        "\n[bold red]Last Generated SQL Query (Failed):[/bold red]"
                    )
                    console.print("```sql")
                    console.print(analysis_result["last_generated_code"])
                    console.print("```")

                # Display query description if available
                if analysis_result.get("description"):
                    console.print("\n[bold cyan]Query Description:[/bold cyan]")
                    console.print(analysis_result["description"])

                # Display results based on status
                if analysis_result["status"] == "success":
                    console.print("\n[bold green]Query Results:[/bold green]")
                    display_query_results(analysis_result["data"])

                    # Display execution metadata
                    if "metadata" in analysis_result:
                        console.print("\n[bold cyan]Execution Details:[/bold cyan]")
                        metadata = analysis_result["metadata"]
                        console.print(
                            f"Execution Time: {metadata['execution_time']:.2f} seconds"
                        )
                        console.print(
                            f"Rows Returned: {metadata['query_metadata']['row_count']}"
                        )
                        console.print(
                            f"Attempts: {metadata['attempts']} of {max_attempts}"
                        )

                        # If there were multiple attempts but eventual success, show info
                        if metadata["attempts"] > 1:
                            console.print(
                                "\n[yellow]Note: Query succeeded after retries[/yellow]"
                            )
                            if "error_history" in metadata:
                                console.print("Previous attempts:")
                                for error in metadata["error_history"]:
                                    console.print(
                                        f"  • Attempt {error['attempt']}: {error['error']}"
                                    )
                else:
                    console.print("\n[bold red]Query Failed[/bold red]")
                    console.print(f"Error: {analysis_result['error']}")

                    # Display detailed error history
                    if "error_history" in analysis_result:
                        console.print("\n[bold red]Error History:[/bold red]")
                        for error in analysis_result["error_history"]:
                            console.print(
                                f"\nAttempt {error['attempt']} at {error['timestamp']}:"
                            )
                            console.print(f"Error Type: {error['error_type']}")
                            console.print(f"Error Message: {error['error']}")
                            if error.get("code"):
                                console.print("\nFailed SQL:")
                                console.print("```sql")
                                console.print(error["code"])
                                console.print("```")

            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"\n[bold red]Analysis Failed: {str(e)}[/bold red]")
                if "analysis_result" in locals():
                    console.print("\n[bold red]Last Known State:[/bold red]")
                    console.print(json.dumps(analysis_result, indent=2))

        chat_history = [
            {"role": "user", "content": enhanced_question},
            {"role": "assistant", "content": analysis_result.get("description", "")},
        ]
        while True:
            # Create question selection prompt
            questions = [
                InquirerList(
                    "next_action",
                    message="Would you like to analyze another question?",
                    choices=["1. Ask another question", "2. Exit"],
                )
            ]

            next_action = inquirer.prompt(questions)

            if next_action["next_action"].startswith("2"):
                console.print("\n[yellow]Exiting analysis...[/yellow]")
                break

            # Get new question
            custom_question = [
                inquirer.Text("custom", message="Enter your analysis question")
            ]
            custom_answer = inquirer.prompt(custom_question)
            selected_question = custom_answer["custom"]

            console.print(f"\n[bold cyan]New Question:[/bold cyan] {selected_question}")

            # Enhance question with chat
            console.print("\n[bold]Enhancing question with chat analysis...[/bold]")

            # Create chat request with history context
            chat_context = (
                chat_history + [{"role": "user", "content": selected_question}]
            )[-2:]

            chat_request = ChatRequest(messages=chat_context)
            chat_result = await rephrase_message(chat_request)

            enhanced_question = chat_result if chat_result else selected_question

            console.print("\n[bold cyan]Enhanced Question:[/bold cyan]")
            console.print(enhanced_question)

            # Run new analysis
            console.print("\n[bold]Running Snowflake Analysis...[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Analyzing data...", total=None)

                # Create new analysis request
                analysis_request = RunDatabaseAnalysisRequest(
                    data={
                        table: metadata["sample_data"]
                        for table, metadata in tables_metadata.items()
                    },
                    dictionary=dictionary_result,
                    question=enhanced_question,
                    warehouse=credentials["warehouse"],
                    database=credentials["database"],
                    db_schema=credentials["schema"],
                )

                try:
                    _analysis_result = await run_database_analysis(analysis_request)
                    analysis_result = _analysis_result.model_dump()
                    progress.update(task, completed=True)

                    # Display SQL code
                    if analysis_result.get("code"):
                        console.print("\n[bold cyan]Generated SQL Query:[/bold cyan]")
                        console.print("```sql")
                        console.print(analysis_result["code"])
                        console.print("```")
                    elif analysis_result.get("last_generated_code"):
                        console.print(
                            "\n[bold red]Last Generated SQL Query (Failed):[/bold red]"
                        )
                        console.print("```sql")
                        console.print(analysis_result["last_generated_code"])
                        console.print("```")

                    # Display description
                    if analysis_result.get("description"):
                        console.print("\n[bold cyan]Query Description:[/bold cyan]")
                        console.print(analysis_result["description"])
                        chat_history = (
                            chat_history
                            + [{"role": "user", "content": enhanced_question}]
                            + [
                                {
                                    "role": "assistant",
                                    "content": analysis_result.get("description", ""),
                                }
                            ]
                        )
                    # Display results
                    if analysis_result["status"] == "success":
                        console.print("\n[bold green]Query Results:[/bold green]")
                        display_query_results(analysis_result["data"])

                        # Display execution metadata
                        if "metadata" in analysis_result:
                            console.print("\n[bold cyan]Execution Details:[/bold cyan]")
                            metadata = analysis_result["metadata"]
                            console.print(
                                f"Execution Time: {metadata['execution_time']:.2f} seconds"
                            )
                            console.print(
                                f"Rows Returned: {metadata['query_metadata']['row_count']}"
                            )
                            console.print(
                                f"Attempts: {metadata['attempts']} of {max_attempts}"
                            )

                            if metadata["attempts"] > 1:
                                console.print(
                                    "\n[yellow]Note: Query succeeded after retries[/yellow]"
                                )
                                if "error_history" in metadata:
                                    console.print("Previous attempts:")
                                    for error in metadata["error_history"]:
                                        console.print(
                                            f"  • Attempt {error['attempt']}: {error['error']}"
                                        )
                    else:
                        console.print("\n[bold red]Query Failed[/bold red]")
                        console.print(f"Error: {analysis_result['error']}")

                        if "error_history" in analysis_result:
                            console.print("\n[bold red]Error History:[/bold red]")
                            for error in analysis_result["error_history"]:
                                console.print(
                                    f"\nAttempt {error['attempt']} at {error['timestamp']}:"
                                )
                                console.print(f"Error Type: {error['error_type']}")
                                console.print(f"Error Message: {error['error']}")
                                if error.get("code"):
                                    console.print("\nFailed SQL:")
                                    console.print("```sql")
                                    console.print(error["code"])
                                    console.print("```")

                except Exception as e:
                    progress.update(task, completed=True)
                    console.print(f"\n[bold red]Analysis Failed: {str(e)}[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    finally:
        if "conn" in locals():
            conn.close()

    total_elapsed_time = time.time() - start_time
    console.print(
        f"\n[cyan]Total execution time: {total_elapsed_time:.2f} seconds[/cyan]"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

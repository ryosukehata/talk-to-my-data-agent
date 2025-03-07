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
from abc import ABC
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Optional, cast

import duckdb
import polars as pl

from utils.logging_helper import get_logger
from utils.schema import (
    AnalystChatMessage,
    AnalystDataset,
    ChatJSONEncoder,
    CleansedColumnReport,
    CleansedDataset,
    DataDictionary,
)

logger = get_logger("ApplicationDB")

# increment this number if the database schema has changed to prevent conflicts with existing deployments
# this will force reinitialisation - all tables will be dropped
ANALYST_DATABASE_VERSION = 2


class DatasetType(Enum):
    STANDARD = "standard"
    CLEANSED = "cleansed"
    DICTIONARY = "dictionary"


class DataSourceType(Enum):
    FILE = "file"
    DATABASE = "database"
    CATALOG = "catalog"
    GENERATED = "generated"


@dataclass
class DatasetMetadata:
    name: str
    dataset_type: DatasetType
    original_name: (
        str  # For cleansed/dictionary datasets, links to their original dataset
    )
    created_at: datetime
    columns: list[str]
    row_count: int
    data_source: DataSourceType


class BaseDuckDBHandler(ABC):
    """Abstract base class defining the common async DuckDB interface."""

    def __init__(
        self,
        *,
        user_id: str | None = None,
        db_path: Path | None = None,
        name: str | None = None,
        db_version: int
        | None = 1,  # should be updated after updating db tables structure
    ) -> None:
        """Initialize database path and create tables."""
        self.db_version = db_version
        self.user_id = user_id
        self.db_path = self.get_db_path(user_id=user_id, db_path=db_path, name=name)

    async def _create_db_version_table(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        # create db_versuion table
        await self.execute_query(
            conn,
            """
            CREATE TABLE IF NOT EXISTS db_version (
                version INTEGER PRIMARY KEY
            )
            """,
        )
        # insert new version
        await self.execute_query(
            conn, "INSERT OR IGNORE INTO db_version VALUES (?)", [self.db_version]
        )

    async def _initialize_database(self) -> None:
        """Initialize database tables and extensions."""
        async with self._get_connection() as conn:
            # check if db_version table exist
            db_version_result = await self.execute_query(
                conn,
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = 'db_version');",
            )
            old_db_version_table = await asyncio.get_running_loop().run_in_executor(
                None, db_version_result.fetchone
            )
            if old_db_version_table and old_db_version_table[0]:
                # get db version
                db_version_result = await self.execute_query(
                    conn, "SELECT version FROM db_version"
                )
                db_version_row = await asyncio.get_running_loop().run_in_executor(
                    None, db_version_result.fetchone
                )
                if db_version_row:
                    db_version = db_version_row[0]
                    if db_version != self.db_version:
                        # drop all tables
                        tables_result = await self.execute_query(
                            conn,
                            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';",
                        )
                        table_rows = await asyncio.get_running_loop().run_in_executor(
                            None, tables_result.fetchall
                        )
                        for (table_name,) in table_rows:
                            await self.execute_query(
                                conn, f'DROP TABLE IF EXISTS "{table_name}";'
                            )

                        await self._create_db_version_table(conn)
            else:
                await self._create_db_version_table(conn)

    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[duckdb.DuckDBPyConnection, Any]:
        """Async context manager for database connections."""
        loop = asyncio.get_running_loop()
        conn = await loop.run_in_executor(None, duckdb.connect, self.db_path)
        try:
            yield conn
        finally:
            await loop.run_in_executor(None, conn.close)

    async def execute_query(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        params: list[Any] | None = None,
    ) -> duckdb.DuckDBPyConnection:
        """Execute a query asynchronously."""
        loop = asyncio.get_running_loop()
        if params:
            return await loop.run_in_executor(None, lambda: conn.execute(query, params))
        return await loop.run_in_executor(None, lambda: conn.execute(query))

    @staticmethod
    def get_db_path(
        db_path: Path | None = None, user_id: str | None = None, name: str | None = None
    ) -> Path:
        """Return the database path for a given user."""
        path = Path(db_path or ".")
        name = f"{name or 'app'}_db{'_' + user_id if user_id else ''}.db"
        return path / name


class DatasetHandler(BaseDuckDBHandler):
    async def _initialize_database(self) -> None:
        """Initialize database tables and metadata tracking."""
        await super()._initialize_database()
        async with self._get_connection() as conn:
            # Create metadata table
            await self.execute_query(
                conn,
                """
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    table_name VARCHAR PRIMARY KEY,
                    dataset_type VARCHAR,
                    original_name VARCHAR,
                    created_at TIMESTAMP,
                    columns JSON,
                    row_count INTEGER,
                    data_source VARCHAR,
                )
                """,
            )
            # Create cleansing reports table
            await self.execute_query(
                conn,
                """
                CREATE TABLE IF NOT EXISTS cleansing_reports (
                    dataset_name VARCHAR,
                    report JSON,
                    PRIMARY KEY (dataset_name)
                )
                """,
            )

    async def register_dataframe(
        self,
        df: pl.DataFrame,
        name: str,
        dataset_type: DatasetType,
        data_source: DataSourceType,
        original_name: str | None = None,
    ) -> None:
        """
        Register a Polars DataFrame with explicit dataset type tracking.

        Args:
            df: The dataframe to register
            name: Name for the table
            dataset_type: Type of dataset (STANDARD, CLEANSED, or DICTIONARY)
            original_name: For CLEANSED/DICTIONARY types, the name of the original dataset
            data_source: The source of the data (DataSourceType.FILE, DataSourceType.DATABASE, or DataSourceType.CATALOG)
        """
        logger.info(f"Registering dataframe {name} as {dataset_type.value}")

        if await self.table_exists(name):
            raise ValueError(f"Table '{name}' already exists in the database")

        # For cleansed/dictionary datasets, verify original exists
        if dataset_type in (DatasetType.CLEANSED, DatasetType.DICTIONARY):
            if not original_name:
                raise ValueError(
                    f"original_name required for {dataset_type.value} datasets"
                )
            if not await self.table_exists(original_name):
                raise ValueError(f"Original dataset '{original_name}' not found")

        async with self._get_connection() as conn:
            # Create the table
            arrow_table = df.to_arrow()

            def create_table() -> None:
                conn.register("temp_view", arrow_table)
                conn.execute(f"CREATE TABLE '{name}' AS SELECT * FROM temp_view")
                conn.unregister("temp_view")

            await asyncio.get_running_loop().run_in_executor(None, create_table)

            # Store metadata
            metadata = DatasetMetadata(
                name=name,
                dataset_type=dataset_type,
                original_name=original_name or name,
                created_at=datetime.now(),
                columns=list(df.columns),
                row_count=len(df),
                data_source=data_source,
            )

            await self.execute_query(
                conn,
                """
                INSERT INTO dataset_metadata
                (table_name, dataset_type, original_name, created_at, columns, row_count, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    metadata.name,
                    metadata.dataset_type.value,
                    metadata.original_name,
                    metadata.created_at,
                    json.dumps(metadata.columns),
                    metadata.row_count,
                    metadata.data_source.value,
                ],
            )

    async def list_datasets(
        self,
        dataset_type: DatasetType | None = None,
        data_source: DataSourceType | None = None,
    ) -> list[DatasetMetadata]:
        """
        List all datasets, optionally filtered by dataset type and/or data source.

        Args:
            dataset_type: Optional filter by dataset type (STANDARD, CLEANSED, DICTIONARY)
            data_source: Optional filter by data source (FILE, DATABASE, CATALOG)

        Returns:
            List of DatasetMetadata for matching datasets
        """
        async with self._get_connection() as conn:
            query = """
                SELECT
                    table_name, dataset_type, original_name,
                    created_at, columns, row_count, data_source
                FROM dataset_metadata
            """
            params = []
            where_clauses = []

            if dataset_type:
                where_clauses.append("dataset_type = ?")
                params.append(dataset_type.value)

            if data_source:
                where_clauses.append("data_source = ?")
                params.append(data_source.value)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            result = await self.execute_query(conn, query, params)
            rows = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchall
            )

            return [
                DatasetMetadata(
                    name=row[0],
                    dataset_type=DatasetType(row[1]),
                    original_name=row[2],
                    created_at=row[3],
                    columns=json.loads(row[4]),
                    row_count=row[5],
                    data_source=DataSourceType(row[6]),
                )
                for row in rows
            ]

    async def get_dataset_type(self, name: str) -> DatasetType:
        """Get the type of a dataset."""
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                "SELECT dataset_type FROM dataset_metadata WHERE table_name = ?",
                [name],
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchone
            )
            if not row:
                raise ValueError(f"Dataset '{name}' not found")
            return DatasetType(row[0])

    async def table_exists(self, name: str) -> bool:
        """Check if a table exists in the database."""
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT COUNT(*)
                FROM dataset_metadata
                WHERE table_name = ?
                """,
                [name],
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchone
            )
            return bool(row and row[0])

    async def get_related_datasets(self, name: str) -> dict[str, list[str]]:
        """
        Get all related datasets (cleansed versions and data dictionaries)
        for a given standard dataset.
        """
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT table_name, dataset_type
                FROM dataset_metadata
                WHERE original_name = ?
                """,
                [name],
            )
            rows = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchall
            )

            related: dict[str, list[str]] = {"cleansed": [], "dictionary": []}

            for table_name, dtype in rows:
                if dtype == DatasetType.CLEANSED.value:
                    related["cleansed"].append(table_name)
                elif dtype == DatasetType.DICTIONARY.value:
                    related["dictionary"].append(table_name)

            return related

    async def get_dataframe(
        self,
        name: str,
        expected_type: DatasetType | None = None,
        max_rows: int | None = None,
    ) -> pl.DataFrame:
        """
        Retrieve a registered table as a Polars DataFrame.

        Args:
            name: Name of the dataset to retrieve
            expected_type: Optional type validation - will raise error if dataset is not of expected type

        Returns:
            Polars DataFrame containing the dataset

        Raises:
            ValueError: If dataset doesn't exist or is of wrong type
        """
        logger.info(f"Retrieving dataframe {name}")

        # First verify the dataset exists and check its type
        if not await self.table_exists(name):
            raise ValueError(f"Dataset '{name}' not found")

        if expected_type:
            actual_type = await self.get_dataset_type(name)
            if actual_type != expected_type:
                raise ValueError(
                    f"Dataset '{name}' is of type {actual_type.value}, "
                    f"expected {expected_type.value}"
                )

        # Retrieve the data
        async with self._get_connection() as conn:
            try:
                result = await self.execute_query(
                    conn,
                    f'SELECT * FROM "{name}"'
                    + (f" LIMIT {max_rows}" if max_rows is not None else ""),
                )
                arrow_table = await asyncio.get_running_loop().run_in_executor(
                    None, result.arrow
                )
                return cast(pl.DataFrame, pl.from_arrow(arrow_table))
            except duckdb.CatalogException as e:
                raise ValueError(f"Error retrieving dataset '{name}': {str(e)}") from e

    async def store_cleansing_report(
        self, dataset_name: str, reports: list[CleansedColumnReport]
    ) -> None:
        """Store cleansing reports in the metadata table asynchronously."""
        async with self._get_connection() as conn:
            report_json = json.dumps([report.model_dump() for report in reports])
            await self.execute_query(
                conn,
                """
                INSERT OR REPLACE INTO cleansing_reports (dataset_name, report)
                VALUES (?, ?)
                """,
                [dataset_name, report_json],
            )

    async def get_cleansing_report(
        self, dataset_name: str
    ) -> Optional[list[CleansedColumnReport]]:
        """Retrieve cleansing reports asynchronously."""
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                "SELECT report FROM cleansing_reports WHERE dataset_name = ?",
                [dataset_name],
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchone
            )
            if row:
                reports_data = json.loads(row[0])
                return [CleansedColumnReport(**report) for report in reports_data]
            return []

    async def delete_dataset(self, name: str) -> None:
        """
        Delete a specific dataset and its metadata.
        Will not delete related datasets (cleansed/dictionary versions).

        Args:
            name: Name of the dataset to delete

        Raises:
            ValueError: If dataset doesn't exist
        """
        if not await self.table_exists(name):
            raise ValueError(f"Dataset '{name}' not found")

        async with self._get_connection() as conn:
            # Delete the actual table
            await self.execute_query(conn, f'DROP TABLE IF EXISTS "{name}"')

            # Delete metadata
            await self.execute_query(
                conn, "DELETE FROM dataset_metadata WHERE table_name = ?", [name]
            )

            # Delete any cleansing reports
            await self.execute_query(
                conn, "DELETE FROM cleansing_reports WHERE dataset_name = ?", [name]
            )

        logger.info(f"Deleted dataset {name}")

    async def delete_related_datasets(self, name: str) -> None:
        """
        Delete all related datasets (cleansed and dictionary) for a given standard dataset.
        Does not delete the original dataset itself.
        """
        related = await self.get_related_datasets(name)

        for dataset_list in related.values():
            for dataset_name in dataset_list:
                await self.delete_dataset(dataset_name)

        logger.info(f"Deleted all related datasets for {name}")

    async def delete_dataset_with_related(self, name: str) -> None:
        """
        Delete a dataset and all its related datasets (cleansed and dictionary versions).
        """
        # Delete related datasets first
        await self.delete_related_datasets(name)
        # Then delete the main dataset
        await self.delete_dataset(name)

    async def delete_all_datasets(
        self, dataset_type: DatasetType | None = None
    ) -> None:
        """
        Delete all datasets of a specific type, or all datasets if type is None.
        """
        datasets = await self.list_datasets(dataset_type)

        for dataset in datasets:
            await self.delete_dataset(dataset.name)

        type_str = f" of type {dataset_type.value}" if dataset_type else ""
        logger.info(f"Deleted all datasets{type_str}")

    async def delete_empty_datasets(self) -> None:
        """
        Delete all datasets that have 0 rows.
        """
        datasets = await self.list_datasets()

        for dataset in datasets:
            if dataset.row_count == 0:
                await self.delete_dataset(dataset.name)

        logger.info("Deleted all empty datasets")


class ChatHandler(BaseDuckDBHandler):
    """Async handler for chat-related operations."""

    async def _initialize_database(self) -> None:
        """Initialize chat-related tables."""
        await super()._initialize_database()
        async with self._get_connection() as conn:
            await self.execute_query(
                conn,
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    user_id VARCHAR NOT NULL,
                    chat_name VARCHAR NOT NULL,
                    chat_messages JSON,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (user_id, chat_name)
                )
                """,
            )

    async def save_chat(
        self, chat_name: str, messages: list[AnalystChatMessage]
    ) -> None:
        """Save or update a chat conversation."""
        logger.info(f"Saving chat {chat_name} for user {self.user_id}")

        async with self._get_connection() as conn:
            messages_json = json.dumps(messages, cls=ChatJSONEncoder)
            current_time = datetime.now()

            await self.execute_query(
                conn,
                """
                INSERT INTO chat_history
                    (user_id, chat_name, chat_messages, created_at, updated_at)
                VALUES
                    (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, chat_name) DO UPDATE SET
                    chat_messages = excluded.chat_messages,
                    updated_at = ?
                """,
                [
                    self.user_id,
                    chat_name,
                    messages_json,
                    current_time,  # for created_at in INSERT
                    current_time,  # for updated_at in INSERT
                    current_time,  # for updated_at in UPDATE
                ],
            )

    async def get_chat(self, chat_name: str) -> list[AnalystChatMessage]:
        """Retrieve a specific chat conversation."""
        logger.info(f"Retrieving chat {chat_name} for user {self.user_id}")

        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT chat_messages, created_at, updated_at
                FROM chat_history
                WHERE user_id = ? AND chat_name = ?
                """,
                [self.user_id, chat_name],
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchone()
            )

            if row:
                messages_json = json.loads(row[0])
                # Assuming messages_json is a list of dictionaries
                return [
                    AnalystChatMessage.model_validate(message)  # Don't unpack with **
                    for message in messages_json
                ]
            return []

    async def get_chat_names(self) -> list[str]:
        """Get all chat names for the user."""
        logger.info(f"Retrieving chat names for user {self.user_id}")

        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT chat_name
                FROM chat_history
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                [self.user_id],
            )
            rows = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchall()
            )
            return [row[0] for row in rows]

    async def rename_chat(self, old_name: str, new_name: str) -> None:
        """Rename a chat history entry asynchronously."""
        async with self._get_connection() as conn:
            await self.execute_query(
                conn,
                """
                UPDATE chat_history
                SET chat_name = ?
                WHERE user_id = ? AND chat_name = ?
                """,
                [new_name, self.user_id, old_name],
            )

    async def get_all_users(self) -> list[dict[str, Any]]:
        """Get list of all users and their last activity asynchronously."""
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT user_id, updated_at
                FROM chat_history
                ORDER BY updated_at DESC
                """,
            )
            rows = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchall
            )
            return [{"user_id": row[0], "last_active": row[1]} for row in rows]

    async def delete_chat(self, chat_name: str) -> None:
        """Delete a specific chat conversation."""
        logger.info(f"Deleting chat {chat_name}")

        async with self._get_connection() as conn:
            await self.execute_query(
                conn,
                """
                DELETE FROM chat_history
                WHERE user_id = ? AND chat_name = ?
                """,
                [self.user_id, chat_name],
            )

    async def delete_all_chats(self) -> None:
        """Delete all chats for the user."""
        logger.info(f"Deleting all chats for user {self.user_id}")

        async with self._get_connection() as conn:
            await self.execute_query(
                conn, "DELETE FROM chat_history WHERE user_id = ?", [self.user_id]
            )


class AnalystDB:
    dataset_handler: DatasetHandler
    chat_handler: ChatHandler
    user_id: str
    db_path: Path
    dataset_db_name: str
    chat_db_name: str
    db_version: int

    @classmethod
    async def create(
        cls,
        user_id: str,
        db_path: Path,
        dataset_db_name: str = "dataset",
        chat_db_name: str = "chat",
        db_version: int | None = ANALYST_DATABASE_VERSION,
    ) -> "AnalystDB":
        self = cls.__new__(cls)
        self.dataset_handler = DatasetHandler(
            user_id=user_id,
            db_path=db_path,
            name=dataset_db_name,
            db_version=db_version,
        )
        self.chat_handler = ChatHandler(
            user_id=user_id, db_path=db_path, name=chat_db_name, db_version=db_version
        )
        self.user_id = user_id
        self.db_path = db_path
        self.dataset_db_name = dataset_db_name
        self.chat_db_name = chat_db_name
        await self.initialize()
        return self

    async def initialize(self) -> None:
        """Initialize both database handlers."""
        await self.dataset_handler._initialize_database()
        await self.chat_handler._initialize_database()

    # Dataset operations
    async def register_dataset(
        self, df: AnalystDataset | CleansedDataset, data_source: DataSourceType
    ) -> None:
        if isinstance(df, CleansedDataset):
            is_cleansed = True
            await self.dataset_handler.store_cleansing_report(
                df.name, df.cleaning_report
            )
        else:
            is_cleansed = False
        try:
            await self.dataset_handler.register_dataframe(
                df.to_df(),
                f"{df.name}_cleansed" if is_cleansed else df.name,
                dataset_type=(
                    DatasetType.CLEANSED if is_cleansed else DatasetType.STANDARD
                ),
                data_source=data_source,
                original_name=df.name,
            )
        except Exception as e:
            logger.warning(f"Error registering dataset: {e}")

    async def get_dataset(
        self, name: str, max_rows: int | None = 10000
    ) -> AnalystDataset:
        data = AnalystDataset(
            data=await self.dataset_handler.get_dataframe(
                name, expected_type=DatasetType.STANDARD, max_rows=max_rows
            ),
            name=name,
        )
        return data

    async def get_cleansed_dataset(
        self, name: str, max_rows: int | None = 10000
    ) -> CleansedDataset:
        data = AnalystDataset(
            data=await self.dataset_handler.get_dataframe(
                f"{name}_cleansed",
                expected_type=DatasetType.CLEANSED,
                max_rows=max_rows,
            )
        )
        cleansing_report = await self.dataset_handler.get_cleansing_report(name)
        return CleansedDataset(dataset=data, cleaning_report=cleansing_report)

    async def register_data_dictionary(self, data_dictionary: DataDictionary) -> None:
        try:
            return await self.dataset_handler.register_dataframe(
                data_dictionary.to_application_df(),
                name=f"{data_dictionary.name}_dict",
                dataset_type=DatasetType.DICTIONARY,
                data_source=DataSourceType.GENERATED,
                original_name=data_dictionary.name,
            )
        except Exception as e:
            logger.warning(
                f"Failed to register data dictionary {data_dictionary.name}: {e}"
            )

    async def get_data_dictionary(self, name: str) -> DataDictionary | None:
        try:
            df = await self.dataset_handler.get_dataframe(
                f"{name}_dict", expected_type=DatasetType.DICTIONARY
            )
            return DataDictionary.from_application_df(df, name=name)
        except Exception as e:
            logger.error(f"Failed to get data dictionary {name}: {e}")
            return None

    async def get_cleansing_report(
        self, dataset_name: str
    ) -> list[CleansedColumnReport] | None:
        return await self.dataset_handler.get_cleansing_report(dataset_name)

    async def list_analyst_datasets(
        self, data_source: DataSourceType | None = None
    ) -> list[str]:
        """
        List all standard datasets, optionally filtered by data source.

        Args:
            data_source: Optional filter by data source (FILE, DATABASE, CATALOG)

        Returns:
            List of dataset names
        """
        datasets = await self.dataset_handler.list_datasets(
            dataset_type=DatasetType.STANDARD, data_source=data_source
        )
        return [dataset.name for dataset in datasets]

    async def delete_table(self, table_name: str) -> None:
        logger.info(f"Deleting table: {table_name} and related datasets")
        await self.dataset_handler.delete_dataset_with_related(table_name)

    async def delete_dictionary(self, dataset_name: str) -> None:
        logger.info(f"Deleting dictionary for: {dataset_name}")
        await self.dataset_handler.delete_dataset(f"{dataset_name}_dict")

    async def delete_all_tables(self) -> None:
        await self.dataset_handler.delete_all_datasets()

    # Chat operations
    async def save_chat(
        self, chat_messages: list[AnalystChatMessage], chat_name: str | None = None
    ) -> None:
        now = datetime.now().isoformat()
        chat_name = chat_name if chat_name else f"chat_{now}"
        return await self.chat_handler.save_chat(
            chat_name=chat_name, messages=chat_messages
        )

    async def update_chat(
        self,
        chat_message: AnalystChatMessage,
        chat_name: str | None = None,
        mode: Literal["append", "overwrite"] = "append",
    ) -> None:
        if not chat_name:
            raise ValueError("chat_name must be provided to update a chat")
        current_chat_content = await self.get_chat(name=chat_name) or []

        if mode == "append":
            chat_messages = current_chat_content + [chat_message]
            await self.save_chat(chat_name=chat_name, chat_messages=chat_messages)
            return None  # Return early to avoid calling update_chat again
        elif mode == "overwrite":
            current_chat_content[-1] = chat_message
            await self.save_chat(
                chat_name=chat_name, chat_messages=current_chat_content
            )
            return None  # Return early to avoid calling update_chat again
        else:
            raise ValueError("Invalid mode. Use 'append' or 'overwrite'.")

    async def get_chat_names(
        self,
    ) -> list[str]:
        return await self.chat_handler.get_chat_names()

    async def rename_chat(self, old_name: str, new_name: str) -> None:
        return await self.chat_handler.rename_chat(old_name=old_name, new_name=new_name)

    async def get_chat(self, name: str) -> list[AnalystChatMessage] | None:
        chat_history = await self.chat_handler.get_chat(name)
        return chat_history

    async def get_users(self) -> list[dict[str, Any]]:
        return await self.chat_handler.get_all_users()

    async def delete_all_chats(self) -> None:
        await self.chat_handler.delete_all_chats()

    async def delete_chat(self, name: str) -> None:
        return await self.chat_handler.delete_chat(name)

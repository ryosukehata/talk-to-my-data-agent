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

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Generator, List, Union, cast

import datarobot as dr
import httpx
import pandas as pd
import polars as pl
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from utils.analyst_db import AnalystDB, DatasetMetadata, DataSourceType
from utils.database_helpers import Database

sys.path.append("..")

from utils.api import (
    AnalysisGenerationError,
    download_catalog_datasets,
    list_catalog_datasets,
    process_data_and_update_state,
    run_complete_analysis,
)
from utils.schema import (
    AiCatalogDataset,
    AnalystChatMessage,
    AnalystDataset,
    ChatCreate,
    ChatMessagePayload,
    ChatRequest,
    ChatResponse,
    ChatUpdate,
    CleansedDataset,
    DataDictionary,
    DictionaryCellUpdate,
    FileUploadResponse,
    LoadDatabaseRequest,
)


async def get_database(user_id: str) -> AnalystDB:
    analyst_db = await AnalystDB.create(
        user_id=user_id,
        db_path=Path("/tmp"),
        dataset_db_name="datasets.db",
        chat_db_name="chat.db",
    )
    return analyst_db


# Dependency to provide the initialized database
async def get_initialized_db(request: Request) -> AnalystDB:
    if (
        not hasattr(request.state.session, "analyst_db")
        or request.state.session.analyst_db is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Database not initialized. Please call /initialize first.",
        )

    return cast(AnalystDB, request.state.session.analyst_db)


# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst API",
    description="""
    An intelligent API for data analysis that provides capabilities including:
    - Dataset management (upload CSV/Excel files, connect to databases, access AI catalog)
    - Data cleansing and standardization
    - Data dictionary creation and management
    - Chat-based data analysis conversations
    - Python code generation
    - Chart creation
    - Business insights generation
    
    Available endpoint groups:
    - /api/v1/catalog: Access AI catalog datasets
    - /api/v1/database: Database connection and table management
    - /api/v1/datasets: Upload, retrieve, and manage datasets
    - /api/v1/dictionaries: Manage data dictionaries
    - /api/v1/chats: Create and manage chat conversations for data analysis

    The API uses OpenAI's GPT models for intelligent analysis and response generation.
    """,
    version="1.0.0",
    contact={"name": "API Support", "email": "support@example.com"},
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    debug=True,  # Stack traces will be exposed for 500 responses
)


script_name = os.environ.get("SCRIPT_NAME", "")
router = APIRouter(prefix=f"{script_name}/api/v1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Add custom OpenAPI schema
def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]


class SessionState(object):
    _state: dict[str, Any]

    def __init__(self, state: dict[str, Any] | None = None):
        if state is None:
            state = {}
        super().__setattr__("_state", state)

    def __setattr__(self, key: Any, value: Any) -> None:
        self._state[key] = value

    def __getattr__(self, key: Any) -> Any:
        try:
            return self._state[key]
        except KeyError:
            message = "'{}' object has no attribute '{}'"
            raise AttributeError(message.format(self.__class__.__name__, key))

    def __delattr__(self, key: Any) -> None:
        del self._state[key]

    def update(self, state: dict[str, Any]) -> None:
        self._state.update(state)


session_store: dict[str, SessionState] = {}
session_lock = asyncio.Lock()


@contextmanager
def use_user_token(request: Request) -> Generator[None, None, None]:
    """Context manager to temporarily use the user's DataRobot token."""
    if not request.state.session.datarobot_api_token:
        yield
        return

    with dr.Client(
        token=request.state.session.datarobot_api_token,
        endpoint=request.state.session.datarobot_api_endpoint,
    ):
        yield


@app.middleware("http")
async def add_session_middleware(request: Request, call_next):  # type: ignore
    request_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    if request.method in request_methods:
        # Initialize the session
        session_state, session_id, user_id, new_user_id = await _initialize_session(
            request
        )
        request.state.session = session_state

        # Initialize database in the session
        await _initialize_database(request, user_id, new_user_id)

        # Fetch DataRobot credentials if needed
        await _fetch_datarobot_credentials(request, session_state)

    # Process the request
    response: Response = await call_next(request)

    if request.method in request_methods:
        # Set session cookie if needed
        _set_session_cookie(
            response, user_id, session_id, request.cookies.get("session_fastapi")
        )

    return response


async def _initialize_session(
    request: Request,
) -> tuple[SessionState, str, str | None, str]:
    """Initialize the session state and return the session ID and user ID."""
    # Create a new session state with default values
    session_state = SessionState()
    empty_session_state = {
        "datarobot_api_token": None,
        "datarobot_api_endpoint": None,
        "analyst_db": None,
    }
    session_state.update(deepcopy(empty_session_state))

    # Try to get user ID from cookie
    user_id = None
    session_fastapi_cookie = request.cookies.get("session_fastapi")
    if session_fastapi_cookie:
        try:
            user_id = base64.b64decode(session_fastapi_cookie.encode()).decode()
        except Exception:
            pass  # If decoding fails, continue without user_id

    # Generate a new user ID if needed
    new_user_id = str(uuid.uuid4())[:36]

    # Determine session ID
    if session_fastapi_cookie:
        session_id = session_fastapi_cookie
    elif user_id:
        session_id = base64.b64encode(user_id.encode()).decode()
    else:
        session_id = base64.b64encode(new_user_id.encode()).decode()

    # Get or create session in store
    async with session_lock:
        existing_session = session_store.get(session_id)
        if existing_session:
            return existing_session, session_id, user_id, new_user_id
        else:
            session_store[session_id] = session_state
            return session_state, session_id, user_id, new_user_id


async def _initialize_database(
    request: Request, user_id: str | None, new_user_id: str
) -> None:
    """Initialize the database in the session if not already initialized."""
    if (
        not hasattr(request.state.session, "analyst_db")
        or request.state.session.analyst_db is None
    ):
        async with session_lock:
            if user_id:
                request.state.session.analyst_db = await get_database(user_id[:36])
            else:
                request.state.session.analyst_db = await get_database(new_user_id)


async def _fetch_datarobot_credentials(
    request: Request, session_state: SessionState
) -> None:
    """Fetch DataRobot credentials if no user_id is available."""
    session_cookie = request.cookies.get("session")
    if not session_cookie:
        return  # No session cookie, can't fetch credentials

    try:
        async with httpx.AsyncClient(cookies={"session": session_cookie}) as client:
            # Get account info
            api_response = await client.get("/api/v2/account/info/")
            if api_response.status_code != 200:
                return

            account_info = api_response.json()

            # Store user info in session state
            if "uid" in account_info:
                for k, v in account_info.items():
                    setattr(session_state, f"datarobot_{k}", v)

            # Get API keys
            api_keys_response = await client.get("/api/v2/account/apiKeys/")
            if api_keys_response.status_code != 200:
                return

            api_keys_data = api_keys_response.json()

            # Find first non-expiring key
            if "data" in api_keys_data:
                keys = [
                    key["key"]
                    for key in api_keys_data["data"]
                    if key["expireAt"] is None
                ]
                if keys:
                    # Store the API token in session state
                    datarobot_api_token = keys[0]
                    datarobot_api_endpoint = os.environ.get("DATAROBOT_ENDPOINT")

                    # Initialize session state with DR credentials
                    session_state.datarobot_api_token = datarobot_api_token
                    session_state.datarobot_api_endpoint = datarobot_api_endpoint
    except Exception:
        pass


def _set_session_cookie(
    response: Response,
    user_id: str | None,
    session_id: str,
    session_fastapi_cookie: str | None,
) -> None:
    """Set the session cookie if needed."""
    if user_id and not session_fastapi_cookie:
        encoded_uid = base64.b64encode(user_id.encode()).decode()
        response.set_cookie(key="session_fastapi", value=encoded_uid, httponly=True)
    elif not session_fastapi_cookie and not user_id:
        response.set_cookie(key="session_fastapi", value=session_id, httponly=True)


@router.get("/catalog/datasets")
async def get_catalog_datasets(limit: int = 100) -> list[AiCatalogDataset]:
    return list_catalog_datasets(limit)


@router.get("/database/tables")
async def get_database_tables() -> list[str]:
    return Database.get_tables()


async def process_and_update(
    dataset_names: List[str], analyst_db: AnalystDB, datasource_type: DataSourceType
) -> None:
    async for _ in process_data_and_update_state(
        dataset_names, analyst_db, datasource_type
    ):
        pass


@router.post("/datasets/upload")
async def upload_files(
    request: Request,
    background_tasks: BackgroundTasks,
    analyst_db: AnalystDB = Depends(get_initialized_db),
    files: List[UploadFile] | None = None,
    catalog_ids: str | None = Form(None),
) -> list[FileUploadResponse]:
    dataset_names = []
    response: list[FileUploadResponse] = []
    if files:
        for file in files:
            try:
                file_size = file.size or 0
                contents = await file.read()
                if file.filename is None:
                    continue

                file_extension = os.path.splitext(file.filename)[1].lower()

                if file_extension == ".csv":
                    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
                    dataset_name = os.path.splitext(file.filename)[0]
                    data = cast(list[dict[str, Any]], df.to_dict(orient="records"))
                    dataset = AnalystDataset(name=dataset_name, data=data)

                    # Register dataset with the database
                    await analyst_db.register_dataset(
                        dataset, DataSourceType.FILE, file_size=file_size
                    )

                    # Add to processing queue
                    dataset_names.append(dataset.name)

                    file_response: FileUploadResponse = {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size": len(contents),
                        "dataset_name": dataset_name,
                    }
                    response.append(file_response)

                elif file_extension in [".xlsx", ".xls"]:
                    base_name = os.path.splitext(file.filename)[0]
                    contents = await file.read()
                    excel_file = io.BytesIO(contents)
                    sheet_names = pl.read_excel(
                        excel_file, sheet_id=None
                    )  # Get available sheet names
                    if isinstance(sheet_names, dict):
                        for sheet_name in sheet_names:
                            data = pl.read_excel(excel_file, sheet_name=sheet_name)
                            dataset_name = f"{base_name}_{sheet_name}"
                            dataset = AnalystDataset(name=dataset_name, data=data)
                            await analyst_db.register_dataset(
                                dataset, DataSourceType.FILE, file_size=file_size
                            )
                            # Add to processing queue
                            dataset_names.append(dataset.name)

                            excel_sheet_response: FileUploadResponse = {
                                "filename": file.filename,
                                "content_type": file.content_type,
                                "size": len(contents),
                                "dataset_name": dataset_name,
                            }
                            response.append(excel_sheet_response)
                    else:
                        dataset_name = base_name
                        dataset = AnalystDataset(name=dataset_name, data=data)
                        await analyst_db.register_dataset(
                            dataset, DataSourceType.FILE, file_size=file_size
                        )
                        # Add to processing queue
                        dataset_names.append(dataset.name)

                        excel_file_response: FileUploadResponse = {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size": len(contents),
                            "dataset_name": dataset_name,
                        }
                        response.append(excel_file_response)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

            except Exception as e:
                error_response: FileUploadResponse = {
                    "filename": file.filename or "unknown_file",
                    "error": str(e),
                }
                response.append(error_response)

    # Process the data in the background (cleansing and dictionary generation)
    if dataset_names:
        background_tasks.add_task(
            process_and_update, dataset_names, analyst_db, DataSourceType.FILE
        )

    if catalog_ids:
        id_list: list[str] = json.loads(catalog_ids)
        if id_list:
            with use_user_token(request):
                dataframes = await download_catalog_datasets(id_list, analyst_db)
                background_tasks.add_task(
                    process_and_update, dataframes, analyst_db, DataSourceType.CATALOG
                )

    return response


@router.post("/database/select")
async def load_from_database(
    data: LoadDatabaseRequest,
    background_tasks: BackgroundTasks,
    analyst_db: AnalystDB = Depends(get_initialized_db),
    sample_size: int = 5000,
) -> list[str]:
    dataset_names = []

    # Load the data from the database
    if data.table_names:
        dataframes = await Database.get_data(
            *data.table_names, analyst_db=analyst_db, sample_size=sample_size
        )
        dataset_names.extend(dataframes)

    # Process the data in the background (cleansing and dictionary generation)
    if dataset_names:
        background_tasks.add_task(
            process_and_update, dataset_names, analyst_db, DataSourceType.DATABASE
        )

    return dataset_names


@router.get("/dictionaries")
async def get_dictionaries(
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> list[DataDictionary]:
    # Get datasets from the database
    dataset_names = await analyst_db.list_analyst_datasets()

    # Fetch dictionaries for each dataset
    dictionaries = []
    for name in dataset_names:
        dictionary = await analyst_db.get_data_dictionary(name)
        if dictionary:
            dictionaries.append(dictionary)
        else:
            dictionaries.append(
                {"name": name, "column_descriptions": [], "in_progress": True}  # type: ignore
            )

    return dictionaries if dictionaries else []


@router.get("/datasets/{name}/metadata")
async def get_dataset_metadata(
    name: str, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> DatasetMetadata:
    """
    Get metadata for a dataset by name from the database.

    Args:
        name: The name of the dataset

    Returns:
        A dictionary containing dataset metadata including type, creation date, columns, and row count

    Raises:
        HTTPException: If the dataset doesn't exist or cannot be retrieved
    """
    try:
        return await analyst_db.get_dataset_metadata(name)
    except ValueError as e:
        raise HTTPException(
            status_code=404, detail=f"Dataset metadata not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving dataset metadata: {str(e)}"
        )


@router.get("/datasets/{name}/cleansed")
async def get_cleansed_dataset(
    name: str,
    skip: int = 0,
    limit: int = 10000,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> CleansedDataset:
    """
    Get a cleansed dataset by name from the database with pagination support.

    Args:
        name: The name of the dataset
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return (for pagination)

    Returns:
        The cleansed dataset with cleaning report, containing a subset of records based on skip and limit

    Raises:
        HTTPException: If the dataset doesn't exist or cannot be retrieved
    """
    try:
        # Calculate max_rows based on skip + limit
        max_rows = skip + limit if skip + limit > 0 else None

        # Retrieve the dataset with the calculated max_rows
        cleansed_dataset = await analyst_db.get_cleansed_dataset(
            name, max_rows=max_rows
        )

        # Apply skip if needed (max_rows in get_cleansed_dataset only handles the limit)
        if skip > 0 and cleansed_dataset.dataset.to_df().shape[0] > skip:
            # Create a new dataset with skipped rows
            skipped_df = cleansed_dataset.dataset.to_df().slice(skip, limit)
            cleansed_dataset.dataset = AnalystDataset(
                name=cleansed_dataset.name, data=skipped_df
            )
        elif skip > 0:
            # If skip is greater than the number of rows, return an empty dataset
            cleansed_dataset.dataset = AnalystDataset(
                name=cleansed_dataset.name,
                data=cleansed_dataset.dataset.to_df().slice(0, 0),
            )

        return cleansed_dataset
    except ValueError as e:
        raise HTTPException(
            status_code=404, detail=f"Cleansed dataset not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving cleansed dataset: {str(e)}"
        )


@router.delete("/datasets", status_code=200)
async def delete_datasets(
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> None:
    await analyst_db.delete_all_tables()


@router.delete("/dictionaries/{name}", status_code=200)
async def delete_dictionary(
    name: str, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> None:
    await analyst_db.delete_table(name)


@router.get("/dictionaries/{name}/download")
async def download_dictionary(
    name: str, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> Response:
    """
    Download a dictionary as a CSV file.

    Args:
        name: Name of the dataset whose dictionary to download

    Returns:
        CSV file attachment
    """
    dictionary = await analyst_db.get_data_dictionary(name)

    if not dictionary:
        raise HTTPException(status_code=404, detail=f"Dictionary '{name}' not found")

    # Convert the dictionary to a DataFrame
    df = dictionary.to_application_df()

    # Convert to CSV
    csv_content = io.StringIO()
    df.write_csv(csv_content)

    # Create response with CSV attachment
    response = Response(content=csv_content.getvalue())
    response.headers["Content-Disposition"] = (
        f"attachment; filename={name}_dictionary.csv"
    )
    response.headers["Content-Type"] = "text/csv"

    return response


@router.patch("/dictionaries/{name}/cells")
async def update_dictionary_cell(
    name: str,
    update: DictionaryCellUpdate,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> DataDictionary:
    dictionary = await analyst_db.get_data_dictionary(name)

    if not dictionary:
        raise HTTPException(status_code=404, detail=f"Dictionary '{name}' not found")

    if update.rowIndex < 0 or update.rowIndex >= len(dictionary.column_descriptions):
        raise HTTPException(
            status_code=400, detail=f"Row index {update.rowIndex} is out of range"
        )

    column_description = dictionary.column_descriptions[update.rowIndex]
    if hasattr(column_description, update.field):
        setattr(column_description, update.field, update.value)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Field '{update.field}' not found in dictionary row",
        )

    await analyst_db.delete_dictionary(dictionary.name)
    await analyst_db.register_data_dictionary(dictionary)

    return dictionary


@router.post("/chats")
async def create_chat(
    chat: ChatCreate,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> dict[str, str]:
    """Create a new chat with optional data source"""

    chat_id = await analyst_db.create_chat(
        chat_name=chat.name,
        data_source=chat.data_source,
    )

    return {"id": chat_id}


@router.get("/chats")
async def get_chats(
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> list[dict[str, Any]]:
    """Get all chats"""
    chat_list = await analyst_db.get_chat_list()

    return [
        {
            "id": chat["id"],
            "name": chat["name"],
            "data_source": chat.get("data_source", "catalog"),
            "created_at": chat["created_at"],
        }
        for chat in chat_list
    ]


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> ChatResponse:
    """Get a specific chat by ID"""
    chat = await analyst_db.get_chat_messages(chat_id=chat_id)

    return {
        "id": chat_id,
        "messages": chat,
    }


@router.put("/chats/{chat_id}")
async def update_chat(
    chat_id: str, chat: ChatUpdate, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> dict[str, str]:
    """Update a chat's name and/or data source"""
    response_messages = []

    # Update chat name if provided
    if chat.name:
        await analyst_db.rename_chat(chat_id, chat.name)
        response_messages.append("renamed")

    # Update data source if provided
    if chat.data_source:
        await analyst_db.update_chat_data_source(chat_id, chat.data_source)
        response_messages.append("updated data source")

    if not response_messages:
        return {"message": f"No changes made to chat with ID {chat_id}"}

    return {
        "message": f"Chat with ID {chat_id} was {' and '.join(response_messages)} successfully"
    }


@router.delete("/chats/{chat_id}", status_code=200)
async def delete_chat(
    chat_id: str, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> dict[str, str]:
    """Delete a chat"""
    # Delete the chat
    await analyst_db.delete_chat(chat_id=chat_id)

    return {"message": f"Chat with ID {chat_id} deleted successfully"}


@router.get("/chats/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> list[AnalystChatMessage]:
    """Get messages for a specific chat"""

    chat = await analyst_db.get_chat_messages(chat_id=chat_id)

    return chat


@router.delete("/chats/{chat_id}/messages")
async def delete_chat_messages(
    chat_id: str,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> list[AnalystChatMessage]:
    """Delete all messages for a specific chat"""

    await analyst_db.chat_handler.update_chat(messages=[], chat_id=chat_id)

    return cast(list[AnalystChatMessage], [])


@router.delete("/chats/{chat_id}/messages/{index}")
async def delete_chat_message(
    request: Request,
    chat_id: str,
    index: int,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> list[AnalystChatMessage]:
    """Delete a specific message and its preceding/following pair (user/assistant) from a chat"""
    if not hasattr(request.state.session, "chats"):
        request.state.session.chats = {}

    messages = (await analyst_db.get_chat_messages(chat_id=chat_id)) or []

    # Make sure the index is valid
    if index < 0 or index >= len(messages):
        raise HTTPException(status_code=400, detail=f"Invalid message index: {index}")

    # Determine which messages to delete
    target_message = messages[index]

    # For assistant messages, also delete the preceding user message
    if target_message.role == "assistant" and index > 0:
        # Find the preceding user message
        preceding_index = index - 1
        while preceding_index >= 0:
            if messages[preceding_index].role == "user":
                # Delete both messages
                del messages[max(index, preceding_index)]  # Delete higher index first
                del messages[min(index, preceding_index)]
                break
            preceding_index -= 1
    # For user messages, also delete the following assistant message
    elif target_message.role == "user":
        # Find the following assistant message
        following_index = index + 1
        while following_index < len(messages):
            if messages[following_index].role == "assistant":
                # Delete both messages
                del messages[max(index, following_index)]  # Delete higher index first
                del messages[min(index, following_index)]
                break
            following_index += 1
        # If no following assistant message was found, just delete the user message
        if following_index >= len(messages):
            del messages[index]

    await analyst_db.chat_handler.update_chat(
        messages=messages,
        chat_id=chat_id,
    )

    return cast(list[AnalystChatMessage], list(messages))


@router.post("/chats/messages")
async def create_new_chat_message(
    payload: ChatMessagePayload,
    background_tasks: BackgroundTasks,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> dict[str, Union[str, list[AnalystChatMessage], None]]:
    """Create a new chat and post a message to it"""

    # Create a new chat
    chat_id = await analyst_db.create_chat(
        chat_name="New Chat",
    )

    # Create the user message
    user_message = AnalystChatMessage(
        role="user", content=payload.message, components=[]
    )

    await analyst_db.update_chat(
        chat_id=chat_id,
        chat_message=user_message,
        mode="append",
    )

    # Create valid messages for the chat request
    valid_messages: list[ChatCompletionMessageParam] = [
        user_message.to_openai_message_param()
    ]

    # Add the current message
    valid_messages.append(
        ChatCompletionUserMessageParam(role="user", content=payload.message)
    )

    # Create the chat request
    chat_request = ChatRequest(messages=valid_messages)

    # Run the analysis in the background
    background_tasks.add_task(
        run_complete_analysis_task,
        chat_request,
        payload.data_source,
        analyst_db,
        chat_id,
        payload.enable_chart_generation,
        payload.enable_business_insights,
    )

    chat_list = await analyst_db.get_chat_list()
    chat_name = next((n["name"] for n in chat_list if n["id"] == chat_id), None)
    messages = await analyst_db.get_chat_messages(chat_id=chat_id)

    chat = {
        "id": chat_id,
        "name": chat_name,
        "messages": messages,
    }

    return chat


@router.post("/chats/{chat_id}/messages")
async def create_chat_message(
    chat_id: str,
    payload: ChatMessagePayload,
    background_tasks: BackgroundTasks,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> dict[str, Union[str, list[AnalystChatMessage], None]]:
    """Post a message to a specific chat"""

    # Create the user message
    user_message = AnalystChatMessage(
        role="user", content=payload.message, components=[]
    )

    await analyst_db.update_chat(
        chat_id=chat_id,
        chat_message=user_message,
        mode="append",
    )

    # Create valid messages for the chat request
    valid_messages: list[ChatCompletionMessageParam] = [
        user_message.to_openai_message_param()
    ]

    # Add the current message
    valid_messages.append(
        ChatCompletionUserMessageParam(role="user", content=payload.message)
    )

    # Create the chat request
    chat_request = ChatRequest(messages=valid_messages)

    # Run the analysis in the background
    background_tasks.add_task(
        run_complete_analysis_task,
        chat_request,
        payload.data_source,
        analyst_db,
        chat_id,
        payload.enable_chart_generation,
        payload.enable_business_insights,
    )

    chat_list = await analyst_db.get_chat_list()
    chat_name = next((n["name"] for n in chat_list if n["id"] == chat_id), None)
    messages = await analyst_db.get_chat_messages(chat_id=chat_id)

    chat = {
        "id": chat_id,
        "name": chat_name,
        "messages": messages,
    }

    return chat


async def run_complete_analysis_task(
    chat_request: ChatRequest,
    data_source: str,
    analyst_db: AnalystDB,
    chat_id: str,
    enable_chart_generation: bool,
    enable_business_insights: bool,
) -> None:
    """Run the complete analysis pipeline"""
    source = DataSourceType(data_source)
    datasets_names = []
    if source == DataSourceType.DATABASE:
        datasets_names = await analyst_db.list_analyst_datasets(source)
    else:
        datasets_names = (
            await analyst_db.list_analyst_datasets(DataSourceType.CATALOG)
        ) + (await analyst_db.list_analyst_datasets(DataSourceType.FILE))
    run_analysis_iterator = run_complete_analysis(
        chat_request=chat_request,
        data_source=source,
        datasets_names=datasets_names,
        analyst_db=analyst_db,
        chat_id=chat_id,
        enable_chart_generation=enable_chart_generation,
        enable_business_insights=enable_business_insights,
    )

    async for message in run_analysis_iterator:
        if isinstance(message, AnalysisGenerationError):
            break
        else:
            pass


app.include_router(router)

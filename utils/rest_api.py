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
import sys
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from utils.analyst_db import AnalystDB

sys.path.append("..")

from utils.api import (
    cleanse_dataframe,
    download_catalog_datasets,
    get_business_analysis,
    get_dictionary,
    list_catalog_datasets,
    rephrase_message,
    run_analysis,
    run_charts,
    run_database_analysis,
    suggest_questions,
)
from utils.database_helpers import Database
from utils.schema import (
    AiCatalogDataset,
    AnalystDataset,
    ChatRequest,
    CleansedDataset,
    DataDictionary,
    GetBusinessAnalysisRequest,
    GetBusinessAnalysisResult,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunChartsRequest,
    RunChartsResult,
    RunDatabaseAnalysisRequest,
    RunDatabaseAnalysisResult,
    ValidatedQuestion,
)

analyst_db: AnalystDB | None = None
init_lock = asyncio.Lock()


# Your provided database initialization function
async def get_database(user_id: str) -> AnalystDB:
    analyst_db = await AnalystDB.create(
        user_id=user_id,
        db_path=Path("/tmp"),
        dataset_db_name="datasets.db",
        chat_db_name="chat.db",
    )
    return analyst_db


# Dependency to provide the initialized database
async def get_initialized_db() -> AnalystDB:
    if analyst_db is None:
        raise HTTPException(
            status_code=400,
            detail="Database not initialized. Please call /initialize first.",
        )
    return analyst_db


# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst API",
    description="""
    An intelligent API for data analysis that provides capabilities including:
    - Data cleansing and standardization
    - Data dictionary generation
    - Question suggestions
    - Python code generation
    - Chart creation
    - Business analysis

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Initialization endpoint to set up the database with user_id
@app.post("/initialize")
async def initialize_database(user_id: str) -> dict[str, str]:
    global analyst_db
    async with init_lock:
        if analyst_db is not None:
            return {"message": "Database already initialized"}
        analyst_db = await get_database(user_id)
    return {"message": "Database initialized"}


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


@app.get("/list_catalog_datasets")
async def list_catalog_datasets_endpoint(limit: int = 100) -> list[AiCatalogDataset]:
    return list_catalog_datasets(limit)


@app.post("/download_catalog_datasets")
async def download_catalog_datasets_endpoint(
    dataset_ids: list[str], analyst_db: AnalystDB = Depends(get_initialized_db)
) -> list[str]:
    return await download_catalog_datasets(dataset_ids, analyst_db=analyst_db)


@app.get("/get_database_tables")
async def get_database_tables_endpoint() -> list[str]:
    return Database.get_tables()


@app.get("/get_database_data")
async def get_database_data_endpoint(
    table_names: list[str],
    analyst_db: AnalystDB = Depends(get_initialized_db),
    sample_size: int = 5000,
) -> list[str]:
    return await Database.get_data(
        *table_names, analyst_db=analyst_db, sample_size=sample_size
    )


@app.post("/cleanse_dataframe")
async def cleanse_dataframes_endpoint(
    dataset: AnalystDataset,
) -> CleansedDataset:
    return await cleanse_dataframe(dataset)


@app.post("/get_dictionary")
async def get_dictionaries_endpoint(
    dataset: AnalystDataset,
) -> DataDictionary:
    return await get_dictionary(dataset)


@app.post("/suggest_questions")
async def suggest_questions_endpoint(
    datasets: list[AnalystDataset],
) -> list[ValidatedQuestion]:
    return await suggest_questions(datasets)


@app.post("/run_charts")
async def run_charts_endpoint(request: RunChartsRequest) -> RunChartsResult:
    return await run_charts(request)


@app.post("/get_business_analysis")
async def get_business_analysis_endpoint(
    request: GetBusinessAnalysisRequest,
) -> GetBusinessAnalysisResult:
    return await get_business_analysis(request)


@app.post("/rephrase_message")
async def rephrase_message_endpoint(request: ChatRequest) -> str:
    return await rephrase_message(request)


@app.post("/run_analysis")
async def run_analysis_endpoint(
    request: RunAnalysisRequest, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> RunAnalysisResult:
    return await run_analysis(request=request, analyst_db=analyst_db)


@app.post("/run_database_analysis")
async def run_database_analysis_endpoint(
    request: RunDatabaseAnalysisRequest,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> RunDatabaseAnalysisResult:
    return await run_database_analysis(request=request, analyst_db=analyst_db)

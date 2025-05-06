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

import logging
import os
import sys

import uvicorn
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.append(".")

from utils.rest_api import app

# Configure logging to filter out the health check logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out "GET /" health check logs
        return "GET / HTTP/1.1" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

script_name = os.environ.get("SCRIPT_NAME", "")
react_build_dir = "dist"

# Mount static files directory
app.mount(
    f"{script_name}/assets",
    StaticFiles(directory=os.path.join(react_build_dir, "assets")),
    name="assets",
)


# Serve favicon
@app.get("/datarobot_favicon.png")
async def serve_favicon() -> FileResponse:
    return FileResponse(os.path.join(react_build_dir, "datarobot_favicon.png"))


# client side routes
@app.get(f"{script_name}/data")
@app.get(f"{script_name}/chats")
@app.get(f"{script_name}/chats/{{chat_id}}")
@app.get(f"{script_name}/")
async def serve_root() -> FileResponse:
    """Serve the React index.html for the root route."""
    return FileResponse(os.path.join(react_build_dir, "index.html"))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        access_log=True,
        log_level="warning",
    )

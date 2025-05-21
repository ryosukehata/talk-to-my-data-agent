import json
from datetime import datetime
from typing import Any

import datarobot as dr
import httpx
from datarobot.client import RESTClientObject
from pydantic import ValidationError

from utils.logging_helper import get_logger
from utils.resources import LLMDeployment

logger = get_logger()


def initialize_deployment() -> tuple[RESTClientObject, str]:
    try:
        dr_client = dr.Client()
        chat_agent_deployment_id = LLMDeployment().id
        deployment_chat_base_url = (
            dr_client.endpoint + f"/deployments/{chat_agent_deployment_id}/"
        )
        return dr_client, deployment_chat_base_url
    except ValidationError as e:
        raise ValueError(
            "Unable to load Deployment ID."
            "If running locally, verify you have selected the correct "
            "stack and that it is active using `pulumi stack output`. "
            "If running in DataRobot, verify your runtime parameters have been set correctly."
        ) from e


async def async_submit_actuals_to_datarobot(
    association_id: str, telemetry_json: dict[str, Any] | None = None
) -> None:
    dr_client, deployment_chat_base_url = initialize_deployment()
    deployment_chat_actuals_url = deployment_chat_base_url + "actuals/fromJSON/"
    telemetry_json["endTimestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "data": [
            {
                "associationId": association_id,
                "actualValue": json.dumps(telemetry_json, ensure_ascii=False),
            }
        ]
    }
    headers = dr_client.headers
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                deployment_chat_actuals_url, json=payload, headers=headers, timeout=5
            )
            logger.info("Actuals posted (async).")
        except Exception as e:
            logger.error(f"Failed posting actuals: {e}")

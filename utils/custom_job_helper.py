import datarobot as dr
import httpx

from utils.logging_helper import get_logger

logger = get_logger()


def _log_httpx_error(context: str, exc: Exception):
    message = None
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            message = exc.response.json().get("message")
        except Exception:
            pass
        logger.error(f"{context}: {exc} | message: {message}")
    else:
        logger.error(f"{context}: {exc}")


def get_custom_job(custom_job_id: str) -> dict | None:
    """
    Get a DataRobot custom job by ID. Returns job dict if exists, else None.
    Args:
        custom_job_id (str): The ID of the custom job.
    Returns:
        dict | None: The job dict if found, else None.
    """
    import datarobot as dr
    import httpx

    dr_client = dr.Client()
    url = f"{dr_client.endpoint}/customJobs/{custom_job_id}/"
    headers = dr_client.headers
    try:
        response = httpx.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_custom_job_by_name(job_name: str) -> dict | None:
    """
    Get a DataRobot custom job by name. Returns job dict if exists, else None.
    Args:
        job_name (str): The name of the custom job.
    Returns:
        dict | None: The job dict if found, else None.
    """
    import datarobot as dr
    import httpx

    dr_client = dr.Client()
    url = f"{dr_client.endpoint}/customJobs/"
    headers = dr_client.headers
    try:
        response = httpx.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("data", [])
        for job in jobs:
            if job.get("name") == job_name:
                return job
        return None
    except Exception:
        return None


def create_custom_job_schedule(
    custom_job_id: str,
    minute: list[int],
    hour: list[int],
    day_of_month: list[str],
    month: list[str],
    day_of_week: list[str],
) -> str:
    """
    Create a schedule for a DataRobot custom job.

    Args:
        custom_job_id (str): The ID of the custom job.
        minute (list[int]): List of minutes (e.g., [0]).
        hour (list[int]): List of hours (e.g., [20]).
        day_of_month (list[str]): List of day(s) of month (e.g., ["*"]).
        month (list[str]): List of month(s) (e.g., ["*"]).
        day_of_week (list[str]): List of day(s) of week (e.g., ["*"]).

    Returns:
        str: The ID of the created schedule.
    """
    dr_client = dr.Client()
    headers = dr_client.headers
    url = f"{dr_client.endpoint}/customJobs/{custom_job_id}/schedules/"
    schedule = {
        "minute": minute,
        "hour": hour,
        "dayOfMonth": day_of_month,
        "month": month,
        "dayOfWeek": day_of_week,
    }
    payload = {"schedule": schedule}
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"Created schedule {response.json()['id']}")
        logger.info(f"Schedule updated at {response.json()['updatedAt']}")
        return response.json()["id"]
    except Exception as e:
        _log_httpx_error("Failed to create custom job schedule", e)
        raise


def run_custom_job(
    custom_job_id: str,
    runtime_parameters: list[dict] | None = None,
    description: str = "",
) -> str:
    """
    Trigger a run for a DataRobot custom job and return the jobStatusId.

    Args:
        custom_job_id (str): The ID of the custom job.
        runtime_parameters (list[dict], optional): List of runtime parameter dicts. Each dict should have keys 'fieldName', 'value', and 'type'.
        description (str, optional): Description for the job run.

    Returns:
        str: The jobStatusId from the response.
    """
    dr_client = dr.Client()
    headers = dr_client.headers
    url = f"{dr_client.endpoint}/customJobs/{custom_job_id}/runs/"
    payload = {"description": description}
    if runtime_parameters is not None:
        payload["runtimeParameterValues"] = runtime_parameters
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()["id"]
    except Exception as e:
        _log_httpx_error("Failed to run custom job", e)
        raise


def poll_custom_job_run_status(
    custom_job_id: str,
    custom_run_id: str,
    poll_interval: float = 5.0,
    timeout: float = 1800.0,
) -> None:
    """
    Poll the running status of a DataRobot custom job. Raises exception if not succeeded.

    Args:
        custom_job_id (str): The ID of the custom job.
        custom_run_id (str): The ID of the custom job run.
        poll_interval (float): Time in seconds between polling attempts.
        timeout (float): Maximum time in seconds to wait for job completion.

    Raises:
        RuntimeError: If the job does not finish with 'succeeded' status or times out.
    """
    import time

    dr_client = dr.Client()
    headers = dr_client.headers
    url = f"{dr_client.endpoint}/customJobs/{custom_job_id}/runs/{custom_run_id}/"
    start_time = time.time()
    while True:
        try:
            response = httpx.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            status = response.json().get("status")
            logger.info(f"Custom job {custom_job_id} status: {status}")
            if status == "succeeded":
                return
            elif status in ["failed", "interrupted", "canceling", "canceled"]:
                raise RuntimeError(
                    f"Custom job {custom_job_id} finished with status: {status}"
                )
            elif time.time() - start_time > timeout:
                raise TimeoutError(f"Polling timed out for custom job {custom_job_id}")
            time.sleep(poll_interval)
        except Exception as e:
            _log_httpx_error("Error polling custom job status", e)
            raise


def list_custom_job_schedules(custom_job_id: str) -> str | None:
    """
    List all schedules for a DataRobot custom job and return the first schedule's ID.

    Args:
        custom_job_id (str): The ID of the custom job.

    Returns:
        str | None: The ID of the first schedule (schedule_id) or None if no schedules are found.
    """
    dr_client = dr.Client()
    headers = dr_client.headers
    url = f"{dr_client.endpoint}/customJobs/{custom_job_id}/schedules/"
    try:
        response = httpx.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            logger.warning(f"No schedules found for custom job {custom_job_id}")
            return None
        return data[0]["id"]
    except Exception as e:
        _log_httpx_error("Failed to list custom job schedules", e)
        raise


def update_custom_job_schedule(
    custom_job_id: str,
    schedule_id: str,
    minute: list[int],
    hour: list[int],
    day_of_month: list[str],
    month: list[str],
    day_of_week: list[str],
) -> str:
    """
    Update an existing schedule for a DataRobot custom job and return the updatedAt timestamp.

    Args:
        custom_job_id (str): The ID of the custom job.
        schedule_id (str): The ID of the schedule to update.
        minute (list[int]): List of minutes (e.g., [0]).
        hour (list[int]): List of hours (e.g., [20, 11]).
        day_of_month (list[str]): List of day(s) of month (e.g., ["*"]).
        month (list[str]): List of month(s) (e.g., ["*"]).
        day_of_week (list[str]): List of day(s) of week (e.g., [1, 2, 3]).

    Returns:
        str: The updatedAt timestamp from the response.
    """
    dr_client = dr.Client()
    headers = dr_client.headers
    url = f"{dr_client.endpoint}/customJobs/{custom_job_id}/schedules/{schedule_id}/"
    schedule = {
        "minute": minute,
        "hour": hour,
        "dayOfMonth": day_of_month,
        "month": month,
        "dayOfWeek": day_of_week,
    }
    payload = {"schedule": schedule}
    try:
        response = httpx.patch(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()["updatedAt"]
    except Exception as e:
        _log_httpx_error("Failed to update custom job schedule", e)
        raise


def create_or_update_custom_job_schedule(
    custom_job_id: str,
    minute: list[int],
    hour: list[int],
    day_of_month: list[str],
    month: list[str],
    day_of_week: list[str],
) -> str:
    """
    Create or update a schedule for a DataRobot custom job.
    If a schedule exists, update it; otherwise, create a new one.

    Args:
        custom_job_id (str): The ID of the custom job.
        minute (list[int]): List of minutes (e.g., [0]).
        hour (list[int]): List of hours (e.g., [20]).
        day_of_month (list[str]): List of day(s) of month (e.g., ["*"]).
        month (list[str]): List of month(s) (e.g., ["*"]).
        day_of_week (list[str]): List of day(s) of week (e.g., ["*"]).

    Returns:
        str: The schedule ID.
    """
    schedule_id = list_custom_job_schedules(custom_job_id)
    if schedule_id is None:
        return create_custom_job_schedule(
            custom_job_id, minute, hour, day_of_month, month, day_of_week
        )
    else:
        updated_at = update_custom_job_schedule(
            custom_job_id, schedule_id, minute, hour, day_of_month, month, day_of_week
        )
        logger.info(f"Updated schedule {schedule_id} at {updated_at}")
        return schedule_id


def delete_custom_job_schedule(custom_job_id: str, schedule_id: str) -> bool:
    """
    Delete a schedule for a DataRobot custom job.

    Args:
        custom_job_id (str): The ID of the custom job.
        schedule_id (str): The ID of the schedule to delete.

    Returns:
        bool: True if deletion was successful (status code 204), False otherwise.
    """
    dr_client = dr.Client()
    headers = dr_client.headers
    url = f"{dr_client.endpoint}/customJobs/{custom_job_id}/schedules/{schedule_id}/"
    try:
        response = httpx.delete(url, headers=headers, timeout=10)
        if response.status_code == 204:
            return True
        else:
            _log_httpx_error(
                "Failed to delete custom job schedule",
                httpx.HTTPStatusError(
                    f"Unexpected status code: {response.status_code}",
                    request=response.request,
                    response=response,
                ),
            )
            return False
    except Exception as e:
        _log_httpx_error("Failed to delete custom job schedule", e)
        raise


def delete_all_custom_job_schedule(custom_job_id: str) -> bool:
    """
    Delete all schedules associated with a DataRobot custom job.

    Args:
        custom_job_id (str): The ID of the custom job.

    Returns:
        bool: True if all schedules were deleted (no schedule remains), False otherwise.
    """
    while True:
        schedule_id = list_custom_job_schedules(custom_job_id)
        if not schedule_id:
            return True
        deleted = delete_custom_job_schedule(custom_job_id, schedule_id)
        if not deleted:
            return False
        # Check again if any schedule remains
        if not list_custom_job_schedules(custom_job_id):
            return True
    return False

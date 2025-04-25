"""
Script to check Pulumi output for a specific job and schedule, and delete the schedule if found.

Usage:
    python pulumi_job_schedule_cleanup.py <PROJECT_NAME>

- PROJECT_NAME: The project name to search for in the Pulumi output.
"""
import sys
import subprocess
import json

from utils.custom_job_helper import delete_custom_job_schedule


def main():
    if len(sys.argv) != 2:
        print("Usage: python pulumi_job_schedule_cleanup.py <PROJECT_NAME>")
        sys.exit(1)
    project_name = sys.argv[1]
    job_key = f"Usage Export Job [{project_name}]"
    schedule_key = "CUSTOM_JOB_SCHEDULE_ID"

    # Fetch Pulumi stack output as JSON
    try:
        result = subprocess.run(
            ["pulumi", "stack", "output", "--json"],
            cwd="../infra",
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = json.loads(result.stdout)
    except Exception as e:
        print(f"Error running Pulumi CLI: {e}")
        sys.exit(1)

    # Extract values
    custom_job_id = outputs.get(job_key)
    schedule_id = outputs.get(schedule_key)

    if custom_job_id and schedule_id:
        print(f"Found both {job_key} and {schedule_key} in Pulumi output. Deleting schedule...")
        success = delete_custom_job_schedule(custom_job_id=custom_job_id, schedule_id=schedule_id)
        if success:
            print("Schedule deleted successfully.")
        else:
            print("Failed to delete schedule.")
    else:
        print(f"Did not find both {job_key} and {schedule_key} in Pulumi output. No action taken.")

if __name__ == "__main__":
    main()

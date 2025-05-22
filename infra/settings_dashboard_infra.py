import hashlib
from pathlib import Path

from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.apps import ApplicationSourceArgs
from datarobot_pulumi_utils.schema.exec_envs import RuntimeEnvironments

from .settings_main import PROJECT_ROOT

dashboard_path = PROJECT_ROOT / "resources" / "app_usage_dashboard"

dashboard_source_args = ApplicationSourceArgs(
    resource_name=f"Data Analyst Dashboard Source [{PROJECT_NAME}]",
    base_environment_id=RuntimeEnvironments.PYTHON_312_APPLICATION_BASE.value.id,
).model_dump(mode="json", exclude_none=True)

dashboard_resource_name: str = f"Data Analyst Dashboard [{PROJECT_NAME}]"


def get_dashboard_files() -> tuple[list[tuple[str, str]], str]:
    """Only include the files that's in the app directory"""
    files_to_include: list[Path] = []
    for f in dashboard_path.glob("**/*"):
        if (
            f.is_file()
            and "__pycache__" not in f.parts
            and not (f.name.endswith(".pyc") or f.name.endswith(".pyo"))
            and f.name != ".DS_Store"
            and f.name != "run_local.sh"
        ):
            files_to_include.append(f)
    hasher = hashlib.sha256()
    files_to_include.sort()
    for file_path in files_to_include:
        try:
            with open(file_path, "rb") as file_content:
                while True:
                    chunk = file_content.read(4096)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except FileNotFoundError:
            pass

    content_hash = hasher.hexdigest()

    # Prepare the list of tuples for Pulumi
    source_files = [
        (str(file_path), file_path.relative_to(dashboard_path).as_posix())
        for file_path in files_to_include
    ]

    return source_files, content_hash

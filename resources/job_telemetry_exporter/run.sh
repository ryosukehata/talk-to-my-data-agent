#!/bin/bash

echo "Job Starting: ($0)"

echo "===== Runtime Parameters ======"
echo "LLM Deployment ID: ${LLM_DEPLOYMENT_ID}"
echo "DATAROBOT_APPLICATION_ID: ${DATAROBOT_APPLICATION_ID}"
echo "DATASET_TRACE_ID: ${DATASET_TRACE_ID}"
echo "DATASET_ACCESS_LOG_ID: ${DATASET_ACCESS_LOG_ID}"
echo "MODE: ${MODE}"
echo "===== Generic Variables ==========================="
echo "CURRENT_CUSTOM_JOB_RUN_ID: $CURRENT_CUSTOM_JOB_RUN_ID"
echo "CURRENT_CUSTOM_JOB_ID:     $CURRENT_CUSTOM_JOB_ID"
echo "DATAROBOT_ENDPOINT:        $DATAROBOT_ENDPOINT"
echo "DATAROBOT_API_TOKEN:       Use the environment variable \$DATAROBOT_API_TOKEN"
echo "==================================================="

# Determine script directory and change into it first
dir_path=$(dirname "$0")
echo "Entrypoint is at $dir_path - cd into it"
cd "$dir_path" || exit 1 # Exit if cd fails

# install the required Python packages
echo "===== Installing Python packages ======"
pip install --no-cache-dir -r requirements.txt
echo "===== Finished installing Python packages ======"

# Verify python3 exists (already in correct directory)
if command -v python3 &>/dev/null; then
    echo "python3 is installed and available."
else
    echo "Error: python3 is not installed or not available."
    exit 1
fi

python_file="job.py"
if [ -f "$python_file" ]; then
    echo "Found $python_file .. running it"
    python3 ./job.py
else
    echo "File $python_file does not exist"
    exit 1
fi

#!/bin/bash

# Get the absolute path to the directory where this script resides
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Construct the absolute path to the 'modules' directory
MODULES_DIR="${SCRIPT_DIR}"

# Prepend the modules directory to PYTHONPATH. Create PYTHONPATH if it doesn't exist.
# This allows Python to find modules like 'config' directly when imported from other files within 'modules'.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "--- Running job.py with PYTHONPATH=${PYTHONPATH} ---"

# Execute the python script using its absolute path
python3 "${SCRIPT_DIR}/job.py"

EXIT_CODE=$?
echo "--- Script finished with exit code ${EXIT_CODE} ---"
exit ${EXIT_CODE}

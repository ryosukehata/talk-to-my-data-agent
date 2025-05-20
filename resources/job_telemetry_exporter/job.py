import logging

from modules import config
from modules.trace_helper import run_trace_update_workflow
from modules.usage_helper import run_usage_update_flow

logger = logging.getLogger(__name__)

LLM_DEPLOYMENT_ID = config.LLM_DEPLOYMENT_ID
DATAROBOT_APPLICATION_ID = config.DATAROBOT_APPLICATION_ID
DATASET_TRACE_ID = config.DATASET_TRACE_ID
DATASET_ACCESS_LOG_ID = config.DATASET_ACCESS_LOG_ID
MODE = config.MODE


def main():
    logger.info("Starting job")
    logger.info(f"LLM Deployment ID: {LLM_DEPLOYMENT_ID}")
    logger.info(f"APP ID: {DATAROBOT_APPLICATION_ID}")
    logger.info(f"TRACE ID: {DATASET_TRACE_ID}")
    logger.info(f"ACCESS LOG ID: {DATASET_ACCESS_LOG_ID}")
    logger.info(f"MODE: {MODE}")  # not used eventually

    logger.info("Starting export trace data")
    path_trace = run_trace_update_workflow()
    logger.info("Updated trace dataset.")
    logger.info("Starting update usage data")
    path_usage = run_usage_update_flow()
    logger.info("Updated usage dataset.")

    logger.info("Finished job")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()

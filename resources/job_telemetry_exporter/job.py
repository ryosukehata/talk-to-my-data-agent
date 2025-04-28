import logging

from modules import config

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
    logger.info(f"MODE: {MODE}")

    logger.info("Finished job")

import logging

from modules import config

logger = logging.getLogger(__name__)

LLM_DEPLOYMENT_ID = config.LLM_DEPLOYMENT_ID
APP_ID = config.APP_ID
TRACE_ID = config.TRACE_ID
ACCESS_LOG_ID = config.ACCESS_LOG_ID
MODE = config.MODE


def main():
    logger.info("Starting job")
    logger.info(f"LLM Deployment ID: {LLM_DEPLOYMENT_ID}")
    logger.info(f"APP ID: {APP_ID}")
    logger.info(f"TRACE ID: {TRACE_ID}")
    logger.info(f"ACCESS LOG ID: {ACCESS_LOG_ID}")
    logger.info(f"MODE: {MODE}")

    logger.info("Finished job")

import logging


def _enable_logging(log_level: int = logging.INFO):
    logging.basicConfig(
        level=log_level,  # Set to DEBUG level to see debug logs
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )




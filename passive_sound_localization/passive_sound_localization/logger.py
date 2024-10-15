from passive_sound_localization.models.configs.logging import LoggingConfig
import logging


def setup_logger(logging_config: LoggingConfig, enable_logging: bool):
    """Sets up the logger based on configuration."""
    logger = logging.getLogger()
    if enable_logging:
        level = getattr(logging, logging_config.level.upper(), logging.INFO)
    else:
        level = logging.CRITICAL  # Suppress logs if logging is disabled

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

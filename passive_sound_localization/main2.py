import os
import hydra
import wave
import logging
import numpy as np

from config.config import Config
from logger import setup_logger

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    # Setup logger
    setup_logger(cfg.logging, cfg.feature_flags.enable_logging)
    logger = logging.getLogger(__name__)
    logger.info("Running main2.py...")

if __name__ == "__main__":
    main()
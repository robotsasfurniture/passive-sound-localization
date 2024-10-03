import hydra
from hydra.core.config_store import ConfigStore
from configs.config import Config
import logging

from logger import setup_logger


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config):
    # Setup logger
    setup_logger(cfg.logging, cfg.feature_flags.enable_logging)
    logger = logging.getLogger(__name__)
    logger.info("Starting main script.")

if __name__ == "__main__":
    main()
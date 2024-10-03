from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from config.feature_flags_config import FeatureFlagsConfig
from config.logging_config import LoggingConfig

@dataclass
class Config:
    feature_flags: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
from dataclasses import dataclass
from configs.feature_flags_config import FeatureFlagsConfig
from configs.logging_config import LoggingConfig

@dataclass
class Config:
    feature_flags: FeatureFlagsConfig = FeatureFlagsConfig()
    logging: LoggingConfig = LoggingConfig()
from dataclasses import dataclass

@dataclass
class FeatureFlagsConfig:
    enable_logging: bool = True
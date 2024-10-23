from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureFlagsConfig:
    enable_logging: bool = True
from dataclasses import dataclass

@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
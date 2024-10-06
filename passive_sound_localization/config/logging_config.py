from dataclasses import dataclass

@dataclass
class LoggingConfig:
    level: str = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
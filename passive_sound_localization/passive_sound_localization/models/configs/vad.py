from dataclasses import dataclass

@dataclass(frozen=True)
class VADConfig:
    enabled: bool = True
    aggressiveness: int = 2  # 0-3, where 3 is the most aggressive
    frame_duration_ms: int = 30  # Frame duration in milliseconds (10, 20, or 30)
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RealtimeAudioStreamerConfig:
    sample_rate: int = 44100
    channels: int = 1
    chunk: int = 1024
    # device_indices: List[int] = field(default_factory=lambda: [2, 3, 4, 5]) # Lab configuration
    device_indices: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4]
    )  # Nico's laptop (configuration with soundcard)
    device_indices: List[int] = field(
        default_factory=lambda: [2, 4, 6, 8]
    )  # Nico's laptop (configuration with soundcard)
    # device_indices: List[int] = field(default_factory=lambda: [4, 6, 8, 10]) # Nico's laptop (configuration 1)
    # device_indices: List[int] = field(default_factory=lambda: [2, 4, 6, 8]) # Nico's laptop (configuration 2)

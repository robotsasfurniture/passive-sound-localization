from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class LocalizationConfig:
    speed_of_sound: float = 343.0  # Speed of sound in m/s
    mic_distance: float = 0.1  # Distance between microphones in meters
    sample_rate: int = 16000  # Sample rate of the audio in Hz
    fft_size: int = 1024  # Size of FFT to use

    mic_positions: List[List[float]] = field(
        default_factory=lambda: [
            [0.0000, 0.4500],
            [0.4500, 0.0000],
            [0.0000, -0.4500],
            [-0.4500, 0.0000]
        ]
    ) # Mic position coordinates in meters

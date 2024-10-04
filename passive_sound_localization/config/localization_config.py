import numpy as np
from dataclasses import dataclass

@dataclass
class LocalizationConfig:
    speed_of_sound: float = 343.0  # Speed of sound in m/s
    mic_distance: float = 0.1      # Distance between microphones in meters
    sample_rate: int = 16000       # Sample rate of the audio in Hz
    fft_size: int = 1024           # Size of FFT to use
    angle_resolution: int = 1      # Angle resolution in degrees
    mic_positions: np.ndarray = np.array([[0, 0, 0], [0.1, 0, 0]])  # Default 2-mic linear array
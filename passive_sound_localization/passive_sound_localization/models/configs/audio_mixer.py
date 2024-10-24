from dataclasses import dataclass

@dataclass(frozen=True)
class AudioMixerConfig:
    mic_count: int = 2  # Default to 2 microphones
    sample_rate: int = 16000  # Sample rate in Hz
    chunk_size: int = 1024  # Number of frames per buffer
    record_seconds: int = 5  # Duration of recording in seconds
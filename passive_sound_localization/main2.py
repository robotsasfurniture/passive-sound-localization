import os
import hydra
import wave
import logging
import numpy as np

from config.config import Config
from logger import setup_logger

from audio_mixer import AudioMixer

def load_audio_data(file_path: str, sample_rate: int) -> np.ndarray:
    """Load audio data from a WAV file."""
    with wave.open(file_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
    return audio_data

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    # Setup logger
    setup_logger(cfg.logging, cfg.feature_flags.enable_logging)
    logger = logging.getLogger(__name__)
    logger.info("Running main2.py...")

    # Initialize components with configurations
    audio_mixer = AudioMixer(cfg.audio_mixer)

    try:
        while True:
            # Record and save audio
            audio_mixer.record_audio()
            audio_mixer.save_audio_files()

            # Load multi-channel audio data for localization
            multi_channel_paths = [
                os.path.join("audio_files", "multi_channel", f"output_channel_{i+1}.wav")
                for i in range(cfg.audio_mixer.mic_count)
            ]
            multi_channel_data = [
                load_audio_data(path, cfg.audio_mixer.sample_rate)
                for path in multi_channel_paths
            ]

    except KeyboardInterrupt:
        logger.info("Shutting down.")

if __name__ == "__main__":
    main()
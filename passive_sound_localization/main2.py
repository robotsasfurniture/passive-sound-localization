import os
import hydra
import wave
import logging
import numpy as np

from passive_sound_localization.config.config import Config
from passive_sound_localization.logger import setup_logger

from passive_sound_localization.audio_mixer import AudioMixer
from passive_sound_localization.vad import VoiceActivityDetector
from passive_sound_localization.transcriber import Transcriber
from passive_sound_localization.localization import SoundLocalizer
from passive_sound_localization.visualizer import Visualizer


def load_audio_data(file_path: str, sample_rate: int) -> np.ndarray:
    """Load audio data from a WAV file."""
    with wave.open(file_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
    return audio_data


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    # Setup logger
    setup_logger(cfg.logging, cfg.feature_flags.enable_logging)
    logger = logging.getLogger(__name__)
    logger.info("Running main.py")

    # Initialize components with configurations
    # audio_mixer = AudioMixer(cfg.audio_mixer)
    vad = VoiceActivityDetector(cfg.vad)
    # transcriber = Transcriber(cfg.transcriber)
    localizer = SoundLocalizer(cfg.localization)
    audio_mixer = AudioMixer(cfg.audio_mixer)
    vad = VoiceActivityDetector(cfg.vad)
    transcriber = Transcriber(cfg.transcriber)
    visualizer = Visualizer(cfg.visualizer)

    try:
        while True:
            # Record and save audio
            # audio_mixer.record_audio()
            # audio_mixer.save_audio_files()

            # Load mixed audio for VAD and transcription
            mixed_audio_path = os.path.join(
                "audio_files", "single_channel", "output.wav"
            )
            mixed_audio_data = load_audio_data(
                mixed_audio_path, cfg.audio_mixer.sample_rate
            )

            # # Load multi-channel audio data for localization
            multi_channel_paths = [
                os.path.join(
                    "audio_files", "multi_channel", f"output_channel_{i+1}.wav"
                )
                for i in range(cfg.audio_mixer.mic_count)
            ]
            multi_channel_data = [
                load_audio_data(path, cfg.audio_mixer.sample_rate)
                for path in multi_channel_paths
            ]

            visualizer.open_loading_screen()

            if vad.is_speaking(mixed_audio_data):
                # Do audio transcription and sound localization
                # transcription_text = transcriber.transcribe(mixed_audio_path)
                localization_results = localizer.localize(
                    multi_channel_data, cfg.audio_mixer.sample_rate
                )

                for result in localization_results:
                    logger.info(
                        f"Estimated source at angle: {result.angle} degrees, distance: {result.distance} meters"
                    )
                    coordinate_repersentation = (
                        localizer.computer_cartesian_coordinates(
                            result.distance, result.angle
                        )
                    )
                    visualizer.plot(coordinate_repersentation)

            # multi_channel_paths = [
            #     os.path.join("audio_files", "multi_channel", f"output_channel_{i+1}.wav")
            #     for i in range(cfg.audio_mixer.mic_count)
            # ]
            # multi_channel_data = [
            #     load_audio_data(path, cfg.audio_mixer.sample_rate)
            #     for path in multi_channel_paths
            # ]

            if vad.is_speaking(mixed_audio_data):
                # Do audio transcription and sound localization
                transcription = transcriber.transcribe(mixed_audio_path)
                return
            else:
                logger.debug("No speech detected.")

    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()

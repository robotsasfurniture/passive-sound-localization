import webrtcvad
from localization import NewSoundLocalizer, SoundLocalizer
from models.configs.realtime_streamer import (
    RealtimeAudioStreamerConfig,
)
from models.configs.logging import LoggingConfig
from visualizer import Visualizer
from models.configs.localization import LocalizationConfig
from realtime_audio_streamer import RealtimeAudioStreamer
import logging
from logger import setup_logger

setup_logger(LoggingConfig(), enable_logging=True)
logger = logging.getLogger(__name__)


localization_config = LocalizationConfig()

streamer_manager = RealtimeAudioStreamer(RealtimeAudioStreamerConfig())
localizer = SoundLocalizer(
    localization_config,
    Visualizer(microphone_positions=localization_config.mic_positions),
)
# localizer = NewSoundLocalizer(
#     localization_config,
#     Visualizer(microphone_positions=localization_config.mic_positions),
# )


def voice_activity_detection(sound_data_chunk, frame_duration_ms=30, sample_rate=16000):
    """
    Detects voice activity in a given chunk of sound data using py-webrtcvad.

    Parameters:
    - sound_data_chunk: A list of audio bytes representing a single chunk from multiple sources.
    - frame_duration_ms: Duration of each frame in milliseconds (10, 20, or 30).
    - sample_rate: Sample rate of the audio in Hz (must be 16,000 Hz for webrtcvad).

    Returns:
    - bool: True if voice activity is detected in any of the sound sources; otherwise, False.
    """
    frame_length = int(
        sample_rate * frame_duration_ms / 1000
    )  # Calculate frame length in bytes
    vad = webrtcvad.Vad(mode=3)  # Aggressiveness level for VAD (0-3)
    frame_size = int(
        sample_rate * frame_duration_ms / 1000 * 2
    )  # Frame size in bytes (2 bytes per sample for 16-bit PCM)

    for sound_bytes in sound_data_chunk:
        # Ensure the sound data is a multiple of frame_size
        if len(sound_bytes) < frame_size:
            logger.warning("Frame size too small, skipping this chunk.")
            continue  # Skip chunks that are too small

        # Split sound_bytes into frames of frame_size
        frames = [
            sound_bytes[i : i + frame_size]
            for i in range(0, len(sound_bytes), frame_size)
            if len(sound_bytes[i : i + frame_size]) == frame_size
        ]

        # Check each frame for speech
        for frame in frames:
            if vad.is_speech(frame, sample_rate):
                logger.info("Voice activity detected.")
                return True

    logger.info("No voice activity detected.")
    return False

while True:
    did_get = False
    with streamer_manager as streamer:
        total_results = []
        for audio_streams in streamer.audio_generator():
            try:
                if audio_streams is None:
                    logger.error("Audio streams are None")
                    continue

                if not voice_activity_detection(audio_streams, sample_rate=16000):
                    continue

                #  Stream audio data and pass it to the localizer
                localization_stream = localizer.localize_stream(
                    audio_streams,
                )

                for localization_results in localization_stream:
                    if len(localization_results) > 0:
                        logger.info(f"Localization results: {localization_results[0]}")
                    

            except Exception as e:
                print(f"Realtime Localization error: {e}")

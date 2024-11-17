import numpy as np
from localization import SoundLocalizer
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

while True:
    did_get = False
    with streamer_manager as streamer:
        total_results = []
        for audio_streams in streamer.audio_generator():
            try:
                if audio_streams is None:
                    logger.error("Audio streams are None")
                    continue

                #  Stream audio data and pass it to the localizer
                localization_stream = localizer.localize_stream(
                    audio_streams,
                )

                for localization_results in localization_stream:
                    logger.info(f"Localization results: {localization_results}")

            except Exception as e:
                print(f"Realtime Localization error: {e}")

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from localization import LocalizationResult, SoundLocalizer
from models.configs.realtime_streamer import (
    RealtimeAudioStreamerConfig,
)
from models.configs.logging import LoggingConfig
from models.configs.openai_websocket import (
    OpenAIWebsocketConfig,
)
from models.configs.localization import LocalizationConfig
from realtime_openai_websocket import OpenAIResponseType, OpenAIWebsocketClient
from realtime_audio_streamer import RealtimeAudioStreamer
import logging
from logger import setup_logger
import sys
import os

setup_logger(LoggingConfig(), enable_logging=True)
logger = logging.getLogger(__name__)

streamer_manager = RealtimeAudioStreamer(
    RealtimeAudioStreamerConfig(
        sample_rate=44100,
        chunk=882,
        # device_indices=[1],
    )
)

openai_ws_client = OpenAIWebsocketClient(
    OpenAIWebsocketConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
)

localizer = SoundLocalizer(LocalizationConfig())


def send_audio_continuously(
    client: OpenAIWebsocketClient,
    streamer: RealtimeAudioStreamer,
    logger: logging.Logger,
):
    logger.info("Sending audio to OpenAI")
    for audio_streams in streamer.audio_generator():
        if audio_streams is None:
            continue

        client.send_audio(streamer.resample_stream(audio_streams[0]))
        client.store_audio(audio_streams)


def receive_text_messages(
    client: OpenAIWebsocketClient,
    localizer: SoundLocalizer,
    publisher: Callable,
    logger: logging.Logger,
):
    logger.info("OpanAI: Listening to audio stream")
    locations = deque(maxlen=10)
    while True:
        try:
            response = client.receive_response()

            match response["type"]:
                case OpenAIResponseType.NONE:
                    continue

                case OpenAIResponseType.COMPLETED:
                    logger.info(f"Received text response: {response['text']}")
                    if response["text"].strip() == "MOVE_TO":
                        logger.info("Received command: MOVE_TO")

                        locations = []
                        for audio_chunk in response["audio_chunks"]:
                            localization_results = localizer.localize(audio_chunk)
                            locations.extend(localization_results)

                        if len(locations) == 0:
                            logger.error("No locations to publish")
                            continue

                        # Take the mean of all locations
                        best_location = LocalizationResult(
                            angle=sum(location.angle for location in locations)
                            / len(locations),
                            distance=sum(location.distance for location in locations)
                            / len(locations),
                        )

                        logger.info(f"Publishing location: {best_location}")
                        publisher(best_location)

                case _:
                    logger.error(f"Received unknown response type: {response['type']}")

            # if response["type"] == OpenAIResponseType.TEXT:
            #     logger.info(f"Received text response: {response['text']}")
            #     if response["text"].strip() == "MOVE_TO":
            #         logger.info("Received command: MOVE_TO")

            #         if len(locations) > 0:
            #             location = locations.pop()
            #             logger.info(f"Publishing location: {location}")
            #             publisher(location)
            #         else:
            #             logger.error("No locations to publish")

            # elif response["type"] == OpenAIResponseType.AUDIO:
            #     logger.info(f"Received audio chunks: {len(response['audio_chunks'])}")

            #     for audio_chunks in response["audio_chunks"]:
            #         localization_results = localizer.localize(audio_chunks)
            #         locations.extend(localization_results)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Error receiving response: {str(e)}")
            logger.error(f"{exc_type}, {fname}, {exc_tb.tb_lineno}")


with streamer_manager as streamer, openai_ws_client as client:
    with ThreadPoolExecutor(max_workers=2) as executor:
        logger.info("Threading log")
        executor.submit(send_audio_continuously, client, streamer, logger)
        executor.submit(
            receive_text_messages,
            client,
            None,
            lambda x: print(x),
            logger,
        )

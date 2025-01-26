from typing import Callable
from passive_sound_localization.logger import setup_logger
from passive_sound_localization.models.configs import Config

from passive_sound_localization.localization import SoundLocalizer
from passive_sound_localization.realtime_audio_streamer import RealtimeAudioStreamer
from passive_sound_localization.realtime_openai_websocket import (
    OpenAIResponseType,
    OpenAIWebsocketClient,
)
from passive_sound_localization.visualizer import Visualizer
from passive_sound_localization_msgs.msg import LocalizationResult

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from rclpy.node import Node
import logging
import rclpy


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

                        # if len(locations) > 0:
                        #     location = locations.pop()
                        #     logger.info(f"Publishing location: {location}")
                        #     publisher(location)
                        # else:
                        #     logger.error("No locations to publish")

                # case OpenAIResponseType.AUDIO:
                #     logger.info(
                #         f"Received audio chunks: {len(response['audio_chunks'])}"
                #     )

                #     for audio_chunks in response["audio_chunks"]:
                #         localization_results = localizer.localize(audio_chunks)
                #         locations.extend(localization_results)

                case _:
                    logger.error(f"Received unknown response type: {response['type']}")

        except Exception as e:
            logger.error(f"Error receiving response: {str(e)}")


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("localization.speed_of_sound", 343.0),
                ("localization.sample_rate", 24000),
                ("localization.fft_size", 1024),
                ("localization.mic_array_x", [0.00]),
                ("localization.mic_array_y", [0.00]),
                ("logging.level", "INFO"),
                ("feature_flags.enable_logging", True),
                ("openai_websocket.api_key", ""),
                (
                    "openai_websocket.websocket_url",
                    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
                ),
                ("realtime_streamer.sample_rate", 24000),
                ("realtime_streamer.channels", 1),
                ("realtime_streamer.chunk", 1024),
                ("realtime_streamer.device_indices", [2, 3, 4, 5]),
            ],
        )

        self.config: Config = Config.build_configs(self)

        # Setup logger
        setup_logger(self.config.logging, self.config.feature_flags.enable_logging)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Running Localization Node")

        # Setup ROS publisher
        self.publisher = self.create_publisher(
            LocalizationResult, "localization_results", 10
        )

        # Initialize components with configurations
        self.localizer = SoundLocalizer(
            model_path="<insert model path>",
            sampling_rate=16000, 
            visualizer=Visualizer(self.config.localization.mic_positions)
        )
        self.streamer = RealtimeAudioStreamer(self.config.realtime_streamer)
        self.openai_ws_client = OpenAIWebsocketClient(self.config.openai_websocket)

        # Start processing audio
        self.process_audio()

    def process_audio(self):
        self.logger.info("Processing audio...")

        with self.streamer as streamer, self.openai_ws_client as client:
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.logger.info("Threading log")
                executor.submit(send_audio_continuously, client, streamer, self.logger)
                executor.submit(
                    receive_text_messages,
                    client,
                    self.localizer,
                    lambda x: self.publish_results(x),
                    self.logger,
                )

    def publish_results(self, localization_results):
        # Publish results to ROS topic
        self.logger.info(f"Publishing results: {localization_results}")
        msg = LocalizationResult()
        msg.angle = float(localization_results.angle)
        msg.distance = float(localization_results.distance)
        self.logger.info(f"Publishing: angle={msg.angle}, distance={msg.distance}")
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    localization_node = LocalizationNode()

    try:
        rclpy.spin(localization_node)
    except KeyboardInterrupt:
        localization_node.logger.info("Shutting down.")
    finally:
        localization_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

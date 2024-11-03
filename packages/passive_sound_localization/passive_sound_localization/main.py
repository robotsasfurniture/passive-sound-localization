from passive_sound_localization.logger import setup_logger
from passive_sound_localization.models.configs import Config

from passive_sound_localization.audio_mixer import AudioMixer
from passive_sound_localization.localization import SoundLocalizer
from passive_sound_localization.realtime_audio_streamer import RealtimeAudioStreamer
from passive_sound_localization.realtime_openai_websocket import OpenAIWebsocketClient
from passive_sound_localization_msgs.msg import LocalizationResult

from concurrent.futures import ThreadPoolExecutor
from rclpy.node import Node
import numpy as np
import logging
import rclpy

commands = []
locations = []
def send_audio_continuously(client, single_channel_generator):
    print("Threading...")
    for single_channel_audio in single_channel_generator:
        client.send_audio(single_channel_audio)


def receive_text_messages(client, logger):
    logger.info("OpanAI: Listening to audio stream")
    while True:
        try:
            command = client.receive_text_response()
            if command:
                logger.info(f"Received command: {command}")
                commands.append(command)
        except Exception as e:
            print(f"Error receiving response: {e}")
            break  # Exit loop if server disconnects

def realtime_localization(multi_channel_stream, localizer, logger):
    logger.info("Localization: Listening to audio stream")
    for audio_data in multi_channel_stream:
        #  Stream audio data and pass it to the localizer
        localization_stream = localizer.localize_stream(
            [audio_data[k] for k in audio_data.keys()]
        )

        for localization_results in localization_stream:
            locations.append(localization_results)

def command_executor(publisher, logger):
    logger.info("Executor: listening for command")
    while True:
        if len(commands) > 0:
            logger.info(f"Got command, locations: {locations}")
            commands.pop()
            if len(locations) > 0:
                publisher(locations.pop())
        

class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("audio_mixer.sample_rate", 16000),
                ("audio_mixer.chunk_size", 1024),
                ("audio_mixer.record_seconds", 5),
                ("audio_mixer.mic_count", 4),
                ("vad.enabled", True),
                ("vad.aggressiveness", 2),
                ("vad.frame_duration_ms", 30),
                ("transcriber.api_key", ""),
                ("transcriber.model_name", "whisper-1"),
                ("transcriber.language", "en"),
                ("localization.speed_of_sound", 343.0),
                ("localization.sample_rate", 16000),
                ("localization.fft_size", 1024),
                ("localization.mic_array_x", [0.00]),
                ("localization.mic_array_y", [0.00]),
                ("localization.mic_distance", 0.05),
                ("localization.angle_resolution", 1),
                ("logging.level", "INFO"),
                ("feature_flags.enable_logging", True),
                ("openai_websocket.api_key", ""),
                (
                    "openai_websocket.websocket_url",
                    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
                ),
                # ("openai_websocket.instructions", ""),
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
        self.localizer = SoundLocalizer(self.config.localization)
        self.audio_mixer = AudioMixer(self.config.audio_mixer)
        self.streamer = RealtimeAudioStreamer(
            sample_rate=self.config.localization.sample_rate,
            channels=1,
            chunk=self.config.audio_mixer.chunk_size,
        )
        self.openai_ws_client = OpenAIWebsocketClient(self.config.openai_websocket)


        self.logger.info("Ending config of localization node")

        # Start processing audio
        self.process_audio()

    def process_audio(self):
        self.logger.info("Processing audio...")

        self.logger.info("About to run streamer...")
        with self.streamer as streamer: 
            with self.openai_ws_client as client:
                multi_channel_stream = streamer.multi_channel_gen()
                single_channel_stream = streamer.single_channel_gen()

                # TODO: Clean up threading so the localization is properly integrated
                with ThreadPoolExecutor(max_workers=4) as executor:
                    self.logger.info("Threading log")
                    executor.submit(send_audio_continuously, client, single_channel_stream)
                    executor.submit(receive_text_messages, client, self.logger)
                    executor.submit(realtime_localization, multi_channel_stream, self.localizer, self.logger)
                    executor.submit(command_executor, lambda x: self.publish_results(x), self.logger)                    


    def publish_results(self, localization_results):
        # Publish results to ROS topic
        self.logger.info(f"Publishing results {localization_results}")
        location_result = localization_results[0]
        try:
            msg = LocalizationResult()
            msg.angle = float(location_result.angle)
            msg.distance = float(location_result.distance)
            self.logger.info(
                f"Publishing: angle={msg.angle}, distance={msg.distance}"
            )
        except e as Exception:
            self.logger.info(f"{str(e)}")
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

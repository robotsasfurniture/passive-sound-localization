from typing import Generator
from passive_sound_localization.logger import setup_logger
from passive_sound_localization.models.configs import Config

from passive_sound_localization.localization import SoundLocalizer
from passive_sound_localization.realtime_audio_streamer import RealtimeAudioStreamer
from passive_sound_localization.realtime_openai_websocket import OpenAIWebsocketClient
from passive_sound_localization_msgs.msg import LocalizationResult

from concurrent.futures import ThreadPoolExecutor
from rclpy.node import Node
import logging
import rclpy

from queue import Queue

commands = Queue(maxsize=10)
locations = Queue(maxsize=10)

def send_audio_continuously(client, single_channel_generator):
    print("Threading...")
    for single_channel_audio in single_channel_generator:
        client.send_audio(single_channel_audio)


def receive_text_messages(client, logger):
    logger.info("OpanAI: Listening to audio stream")
    while True:
        try:
            command = client.receive_text_response()
            if command and command.strip() == "MOVE_TO":
                print(command)
                logger.info(f"Received command: {command}")
                
                if commands.full():
                    commands.get()
                    commands.task_done()
                commands.put(command)
        except Exception as e:
            print(f"Error receiving response: {e}")
            break  # Exit loop if server disconnects

def realtime_localization(multi_channel_stream, localizer, logger):
    logger.info("Localization: Listening to audio stream")
    try:
        did_get = True
        for audio_data in multi_channel_stream:
            #  Stream audio data and pass it to the localizer
            localization_stream = localizer.localize_stream(
                [audio_data[k] for k in audio_data.keys()]
            )

            for localization_results in localization_stream:
                if locations.full():
                    locations.get()
                
                locations.put(localization_results)
            
            if did_get:
                locations.task_done()
            did_get = False
                
    except Exception as e:
        print(f"Realtime Localization error: {e}")

def command_executor(publisher, logger):
    logger.info("Executor: listening for command")
    while True:
        try:
            if commands.qsize() > 0:
                logger.info(f"Got command, locations: {locations}")
                commands.get()
                commands.task_done()
                if locations.qsize() > 0:
                    publisher(locations.get())
            
                locations.task_done()
        except Exception as e:
            print(f"Command executor error: {e}")


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
                ("realtime_streamer.device_indices", [2, 3, 4, 5])
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
        self.streamer = RealtimeAudioStreamer(self.config.realtime_streamer)
        self.openai_ws_client = OpenAIWebsocketClient(self.config.openai_websocket)

        # Start processing audio
        self.process_audio()

    def process_audio(self):
        self.logger.info("Processing audio...")

        with self.streamer as streamer, self.openai_ws_client as client:
            multi_channel_stream = streamer.multi_channel_gen()
            single_channel_stream = streamer.single_channel_gen()

            with ThreadPoolExecutor(max_workers=4) as executor:
                self.logger.info("Threading log")
                executor.submit(send_audio_continuously, client, single_channel_stream)
                executor.submit(receive_text_messages, client, self.logger)
                executor.submit(realtime_localization, multi_channel_stream, self.localizer, self.logger)
                executor.submit(command_executor, lambda x: self.publish_results(x), self.logger)

    def publish_results(self, localization_results):
        # Publish results to ROS topic
        msg = LocalizationResult()
        msg.angle = localization_results.angle
        msg.distance = localization_results.distance
        self.get_logger().info(
            f"Publishing: angle={msg.angle}, distance={msg.distance}"
        )
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

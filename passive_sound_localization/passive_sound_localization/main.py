import numpy as np
from passive_sound_localization.logger import setup_logger
from passive_sound_localization.models.configs import Config

import logging
from passive_sound_localization.audio_mixer import AudioMixer
from passive_sound_localization.localization import SoundLocalizer
from passive_sound_localization.realtime_audio_streamer import RealtimeAudioStreamer

import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Import necessary message types


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
                ("localization.mic_distance", 10),
                ("localization.sample_rate", 16000),
                ("localization.fft_size", 1024),
                ("localization.mic_array_x", [0]),
                ("localization.mic_array_y", [0]),
                ("logging.level", "INFO"),
                ("feature_flags.enable_logging", True),
            ],
        )

        self.config: Config = Config.build_configs(self)

        # Setup logger
        setup_logger(self.config.logging, self.config.feature_flags.enable_logging)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Running Localization Node")

        # Setup ROS publisher
        self.publisher = self.create_publisher(String, "localization_results", 10)

        # Initialize components with configurations
        self.localizer = SoundLocalizer(self.config.localization)
        self.audio_mixer = AudioMixer(self.config.audio_mixer)
        self.streamer = RealtimeAudioStreamer(
            sample_rate=self.config.localization.sample_rate,
            channels=1,
            chunk=self.config.audio_mixer.chunk_size,
        )

        # Start processing audio
        self.process_audio()

    def process_audio(self):
        self.logger.info("Processing audio...")

        with self.streamer as streamer:
            multi_channel_stream = streamer.multi_channel_gen()

            for audio_data in multi_channel_stream:
                #  Stream audio data and pass it to the localizer
                localization_stream = self.localizer.localize_stream(
                    [audio_data[k] for k in audio_data.keys()]
                )

                for localization_results in localization_stream:
                    for result in localization_results:
                        self.logger.info(
                            f"Estimated source at angle: {result.angle} degrees, distance: {result.distance} meters"
                        )
                        coordinate_representation = (
                            self.localizer.compute_cartesian_coordinates(
                                result.distance, result.angle
                            )
                        )
                        self.logger.info(
                            f"Cartesian Coordinates: x={coordinate_representation[0]}, y={coordinate_representation[1]}"
                        )

                    # Publish results to ROS topic
                    self.publish_results(localization_results)

    def publish_results(self, localization_results):
        # Publish results to ROS topic
        msg = String()
        msg.data = str(localization_results)
        self.get_logger().info(f"Publishing: {msg.data}")
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

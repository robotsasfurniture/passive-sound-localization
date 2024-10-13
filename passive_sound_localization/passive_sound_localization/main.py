from passive_sound_localization.logger import setup_logger
from passive_sound_localization.models.configs import Config

import os
import logging
from passive_sound_localization.audio_mixer import AudioMixer
from passive_sound_localization.localization import SoundLocalizer
from passive_sound_localization.transcriber import Transcriber
from passive_sound_localization.vad import VoiceActivityDetector

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
                ("audio_mixer.mic_count", 2),
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
                ("localization.angle_resolution", 1),
                ("localization.mic_array_x", [0]),
                ("localization.mic_array_y", [0]),
                ("localization.mic_array_z", [0]),
                ("logging.level", "INFO"),
                ("feature_flags.enable_logging", True),
            ],
        )

        self.config: Config = Config.build_configs(self)

        # Setup logger
        setup_logger(self.config.logging, self.config.feature_flags.enable_logging)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Running Localization Node")

        # Initialize components with configurations
        self.vad = VoiceActivityDetector(self.config.vad)
        self.localizer = SoundLocalizer(self.config.localization)
        self.audio_mixer = AudioMixer(self.config.audio_mixer)
        self.transcriber = Transcriber(self.config.transcriber)
        # self.visualizer = Visualizer(cfg.visualizer)

        # Timer for periodic checks
        self.timer = self.create_timer(
            1.0, self.process_audio
        )  # Adjust timer duration as needed

    def process_audio(self):
        # Load mixed audio for VAD and transcription
        self.audio_mixer.record_audio()
        multi_channel_data = self.audio_mixer.multi_channel_data()

        # self.visualizer.open_loading_screen()

        if self.vad.is_speaking(multi_channel_data):
            # Perform sound localization
            localization_results = self.localizer.localize(
                multi_channel_data, self.config.audio_mixer.sample_rate
            )

            for result in localization_results:
                self.logger.info(
                    f"Estimated source at angle: {result.angle} degrees, distance: {result.distance} meters"
                )
                coordinate_representation = (
                    self.localizer.computer_cartesian_coordinates(
                        result.distance, result.angle
                    )
                )
                # self.visualizer.plot(coordinate_representation)

        if self.vad.is_speaking(multi_channel_data):
            # Do audio transcription if needed
            transcription = self.transcriber.transcribe(
                self.audio_mixer.mix_audio_channels()
            )
            return
        else:
            self.logger.debug("No speech detected.")


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

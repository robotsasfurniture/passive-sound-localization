from passive_sound_localization.models.configs.audio_mixer import (
    AudioMixerConfig,
)
from passive_sound_localization.models.configs.localization import (
    LocalizationConfig,
)
from passive_sound_localization.models.configs.transcriber import (
    TranscriberConfig,
)
from passive_sound_localization.models.configs.vad import (
    VADConfig,
)
from passive_sound_localization.models.configs.logging import (
    LoggingConfig,
)
from passive_sound_localization.models.configs.feature_flags import (
    FeatureFlagsConfig,
)
from dataclasses import dataclass, field


@dataclass
class Config:
    audio_mixer: AudioMixerConfig = field(default_factory=AudioMixerConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    feature_flags: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)

    def build_configs(self) -> "Config":
        def calculate_mic_positions() -> list[list[float]]:
            mic_array_x = self.get_parameter("localization.mic_array_x").value
            mic_array_y = self.get_parameter("localization.mic_array_y").value
            mic_array_z = self.get_parameter("localization.mic_array_z").value

            if len(mic_array_x) != len(mic_array_y) or len(mic_array_x) != len(
                mic_array_z
            ):
                raise ValueError("Mic array dimensions must match.")

            return list(zip(mic_array_x, mic_array_y, mic_array_z))

        return Config(
            audio_mixer=AudioMixerConfig(
                sample_rate=self.get_parameter("audio_mixer.sample_rate").value,
                chunk_size=self.get_parameter("audio_mixer.chunk_size").value,
                record_seconds=self.get_parameter("audio_mixer.record_seconds").value,
                mic_count=self.get_parameter("audio_mixer.mic_count").value,
            ),
            vad=VADConfig(
                enabled=self.get_parameter("vad.enabled").value,
                aggressiveness=self.get_parameter("vad.aggressiveness").value,
                frame_duration_ms=self.get_parameter("vad.frame_duration_ms").value,
            ),
            transcriber=TranscriberConfig(
                api_key=self.get_parameter("transcriber.api_key").value,
                model_name=self.get_parameter("transcriber.model_name").value,
                language=self.get_parameter("transcriber.language").value,
            ),
            localization=LocalizationConfig(
                speed_of_sound=self.get_parameter("localization.speed_of_sound").value,
                mic_distance=self.get_parameter("localization.mic_distance").value
                / 100,
                sample_rate=self.get_parameter("localization.sample_rate").value,
                fft_size=self.get_parameter("localization.fft_size").value,
                angle_resolution=self.get_parameter(
                    "localization.angle_resolution"
                ).value,
                mic_positions=calculate_mic_positions(),
            ),
            logging=LoggingConfig(
                level=self.get_parameter("logging.level").value,
            ),
            feature_flags=FeatureFlagsConfig(
                enable_logging=self.get_parameter("feature_flags.enable_logging").value
            ),
        )

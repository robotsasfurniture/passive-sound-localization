from passive_sound_localization.models.configs.localization import (
    LocalizationConfig,
)
from passive_sound_localization.models.configs.logging import (
    LoggingConfig,
)
from passive_sound_localization.models.configs.feature_flags import (
    FeatureFlagsConfig,
)

from passive_sound_localization.models.configs.openai_websocket import (
    OpenAIWebsocketConfig,
)

# from models.configs.localization import (
#     LocalizationConfig,
# ) # Need import paths like this to test audio streaming with `realtime_audio.py`
# from models.configs.logging import (
#     LoggingConfig,
# ) # Need import paths like this to test audio streaming with `realtime_audio.py`
# from models.configs.feature_flags import (
#     FeatureFlagsConfig,
# ) # Need import paths like this to test audio streaming with `realtime_audio.py`


from dataclasses import dataclass, field


@dataclass
class Config:
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    feature_flags: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)
    openai_websocket: OpenAIWebsocketConfig = field(
        default_factory=OpenAIWebsocketConfig
    )

    def build_configs(self) -> "Config":
        def calculate_mic_positions() -> list[list[float]]:
            mic_array_x = self.get_parameter("localization.mic_array_x").value
            mic_array_y = self.get_parameter("localization.mic_array_y").value

            if len(mic_array_x) != len(mic_array_y):
                raise ValueError("Mic array dimensions must match.")

            return list(zip(mic_array_x, mic_array_y))

        return Config(
            localization=LocalizationConfig(
                speed_of_sound=self.get_parameter("localization.speed_of_sound").value,
                mic_distance=self.get_parameter("localization.mic_distance").value
                / 100,
                sample_rate=self.get_parameter("localization.sample_rate").value,
                fft_size=self.get_parameter("localization.fft_size").value,
                mic_positions=calculate_mic_positions(),
            ),
            logging=LoggingConfig(
                level=self.get_parameter("logging.level").value,
            ),
            feature_flags=FeatureFlagsConfig(
                enable_logging=self.get_parameter("feature_flags.enable_logging").value
            ),
            openai_websocket=OpenAIWebsocketConfig(
                api_key=self.get_parameter("openai_websocket.api_key").value,
                websocket_url=self.get_parameter(
                    "openai_websocket.websocket_url"
                ).value,
                instructions=self.get_parameter("openai_websocket.instructions").value,
            ),
        )

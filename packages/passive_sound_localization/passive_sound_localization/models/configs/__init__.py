# from passive_sound_localization.models.configs.localization import (
#     LocalizationConfig,
# )
# from passive_sound_localization.models.configs.logging import (
#     LoggingConfig,
# )
# from passive_sound_localization.models.configs.feature_flags import (
#     FeatureFlagsConfig,
# )

# from passive_sound_localization.models.configs.openai_websocket import (
#     OpenAIWebsocketConfig,
# )

# from passive_sound_localization.models.configs.realtime_streamer import RealtimeAudioStreamerConfig

from models.configs.localization import (
    LocalizationConfig,
) # Need import paths like this to test audio streaming with `realtime_audio.py`
from models.configs.logging import (
    LoggingConfig,
) # Need import paths like this to test audio streaming with `realtime_audio.py`
from models.configs.feature_flags import (
    FeatureFlagsConfig,
) # Need import paths like this to test audio streaming with `realtime_audio.py`
from models.configs.realtime_streamer import RealtimeAudioStreamerConfig # Need import paths like this to test audio streaming with `realtime_audio.py`
from models.configs.openai_websocket import OpenAIWebsocketConfig # Need import paths like this to test audio streaming with `realtime_audio.py`


from dataclasses import dataclass, field


@dataclass
class Config:
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    feature_flags: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)
    openai_websocket: OpenAIWebsocketConfig = field(
        default_factory=OpenAIWebsocketConfig
    )
    realtime_streamer: RealtimeAudioStreamerConfig = field(default_factory=RealtimeAudioStreamerConfig)

    def build_configs(self) -> "Config":
        def check_chunk_sizes() -> int:
            localization_fft_size = self.get_parameter("localization.fft_size").value
            realtime_streamer_chunk = self.get_parameter("realtime_streamer.chunk").value

            if localization_fft_size != realtime_streamer_chunk:
                raise ValueError("Localization FFT size and RealtimeAudioStreamer chunk size should match")
            
            return localization_fft_size

        def check_sample_rates() -> int:
            localization_sample_rate = self.get_parameter("localization.sample_rate").value
            realtime_streamer_sample_rate = self.get_parameter("realtime_streamer.sample_rate").value

            if localization_sample_rate != realtime_streamer_sample_rate:
                raise ValueError("Localization and RealtimeAudioStreamer sample rates should match")
            
            return localization_sample_rate
        
        def calculate_mic_positions() -> list[list[float]]:
            mic_array_x = self.get_parameter("localization.mic_array_x").value
            mic_array_y = self.get_parameter("localization.mic_array_y").value

            if len(mic_array_x) != len(mic_array_y):
                raise ValueError("Mic array dimensions must match.")

            return list(zip(mic_array_x, mic_array_y))

        return Config(
            localization=LocalizationConfig(
                speed_of_sound=self.get_parameter("localization.speed_of_sound").value,
                sample_rate=check_sample_rates(),
                fft_size=check_chunk_sizes(),
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
            ),
            realtime_streamer=RealtimeAudioStreamerConfig(
                sample_rate=check_sample_rates(),
                channels=self.get_parameter("realtime_streamer.channels").value,
                chunk=check_chunk_sizes(),
                device_indices=self.get_parameter("realtime_streamer.device_indices").value
            )
        )

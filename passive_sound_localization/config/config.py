from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from passive_sound_localization.config.feature_flags_config import FeatureFlagsConfig
from passive_sound_localization.config.logging_config import LoggingConfig
from passive_sound_localization.config.audio_mixer_config import AudioMixerConfig
from passive_sound_localization.config.vad_config import VADConfig
from passive_sound_localization.config.transcriber_config import TranscriberConfig
from passive_sound_localization.config.localization_config import LocalizationConfig
from passive_sound_localization.config.visualizer_config import VisualizerConfig


@dataclass
class Config:
    feature_flags: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    audio_mixer: AudioMixerConfig = field(default_factory=AudioMixerConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from config.feature_flags_config import FeatureFlagsConfig
from config.logging_config import LoggingConfig
from config.audio_mixer_config import AudioMixerConfig
from config.vad_config import VADConfig
from config.transcriber_config import TranscriberConfig

@dataclass
class Config:
    feature_flags: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    audio_mixer: AudioMixerConfig = field(default_factory=AudioMixerConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
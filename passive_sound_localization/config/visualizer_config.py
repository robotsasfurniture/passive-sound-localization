from dataclasses import dataclass
from passive_sound_localization.config.localization_config import LocalizationConfig
from typing import List, Tuple
from dataclasses import field


@dataclass
class VisualizerConfig:
    continue_execution: bool = False
    microphone_positions: List[List[float]] = field(
        default_factory=lambda: LocalizationConfig().mic_positions
    )

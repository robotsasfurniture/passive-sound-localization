import logging
from typing import Dict, Generator, List, Optional
import soundcard as sc
import numpy as np
import threading

# from passive_sound_localization.models.configs.realtime_streamer import (
#     RealtimeAudioStreamerConfig,
# )

from models.configs.realtime_streamer import (
    RealtimeAudioStreamerConfig,
)  # Only needed to run with `realtime_audio.py`

logger = logging.getLogger(__name__)


class RealtimeAudioStreamer:
    def __init__(self, config: RealtimeAudioStreamerConfig):
        self.sample_rate: int = config.sample_rate
        self.channels: int = config.channels
        self.chunk: int = config.chunk
        self.device_indices = config.device_indices
        self.streams: Dict[int, np.ndarray] = {}
        self.streaming: bool = False

        print(self.device_indices)

    def __enter__(self):
        microphones: List[sc._Microphone] = sc.all_microphones()
        self.streams = {
            device_index: np.zeros((self.chunk, self.channels), dtype=np.float32)
            for device_index in self.device_indices
        }
        self.streaming = True

        # Start a thread to continuously record audio
        self.recording_thread = threading.Thread(
            target=self.record_audio, args=(microphones,)
        )
        self.recording_thread.start()

        return self

    def record_audio(self, microphones: List[sc._Microphone]):
        while self.streaming:
            for device_index in self.device_indices:
                self.streams[device_index] = microphones[device_index].record(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    numframes=self.chunk,
                )

    def __exit__(self, *args):
        self.streaming = False
        self.recording_thread.join()  # Wait for the recording thread to finish

    def get_stream(self, device_index: int) -> Optional[bytes]:
        """Retrieve the audio stream for a specific device index."""
        if device_index in self.device_indices:
            return self.streams[device_index].tobytes()
        else:
            print(f"Device index {device_index} not found.")
            return None

    def multi_channel_gen(self) -> Generator[Dict[int, bytes], None, None]:
        try:
            while self.streaming:
                audio_arrays = []
                for device_index in self.device_indices:
                    audio_arrays.append(self.get_stream(device_index))

                yield {
                    device_index: audio
                    for device_index, audio in zip(self.device_indices, audio_arrays)
                }

        except Exception as e:
            print(f"Error in multi_channel_gen: {e}")

    def merge_streams(self, streams: List[np.ndarray]) -> np.ndarray:
        return np.sum(streams, axis=0) / len(streams)

    def single_channel_gen(self) -> Generator[bytes, None, None]:
        try:
            while self.streaming:
                yield self.merge_streams(list(self.streams.values())).tobytes()
        except Exception as e:
            print(f"Error in single_channel_gen: {e}")

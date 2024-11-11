import logging
from typing import Dict, Generator, List, Optional
import soundcard as sc
import numpy as np
import threading
from pydub import AudioSegment
from io import BytesIO


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
        microphones = sc.all_microphones()
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

    def record_audio(self, microphones):
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
            return np.nan_to_num(self.streams[device_index]).tobytes()
        else:
            print(f"Device index {device_index} not found.")
            return None

    def multi_channel_gen(self) -> Generator[Optional[Dict[int, bytes]], None, None]:
        try:
            while self.streaming:
                audio_arrays = []
                for device_index in self.device_indices:
                    audio_arrays.append(self.get_stream(device_index))

                # Return none if any audio is None or empty bytes
                if any(audio == b"" or audio is None for audio in audio_arrays):
                    yield None

                yield {
                    device_index: audio
                    for device_index, audio in zip(self.device_indices, audio_arrays)
                }

        except Exception as e:
            print(f"Error in multi_channel_gen: {e}")

    def merge_streams(self, streams: List[np.ndarray]) -> np.ndarray:
        return np.sum(streams, axis=0) / len(streams)
    
    def resample_stream(self, stream: bytes, target_sample_rate: int = 24000, sample_width: int=2) -> bytes:
        try:
            audio = AudioSegment.from_file(BytesIO(stream))

            # Resample to 24kHz mono pcm16
            return audio.set_frame_rate(target_sample_rate).set_channels(self.channels).set_sample_width(sample_width).raw_data

        except Exception as e:
            print(f"Error in resample_stream: {e}")
            return b""


    def single_channel_gen(self) -> Generator[Optional[bytes], None, None]:
        try:
            while self.streaming:
                stream = self.get_stream(self.device_indices[0])
                if stream == b"" or stream is None:
                    yield None

                resampled_stream = self.resample_stream(stream)

                if resampled_stream != b"":
                    yield resampled_stream
                else:
                    yield None
        except Exception as e:
            print(f"Error in single_channel_gen: {e}")
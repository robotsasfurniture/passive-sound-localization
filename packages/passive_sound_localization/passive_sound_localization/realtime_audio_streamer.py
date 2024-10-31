import logging
from typing import Dict, Generator, List, Optional
from pyaudio import PyAudio, paInt16, Stream
import numpy as np

from passive_sound_localization.models.configs.realtime_streamer import RealtimeAudioStreamerConfig

logger = logging.getLogger(__name__)

class RealtimeAudioStreamer:
    def __init__(self, config: RealtimeAudioStreamerConfig):
        self.sample_rate: int = config.sample_rate
        self.channels: int = config.channels
        self.chunk: int = config.chunk
        self.device_indices = config.device_indices
        self.format = paInt16
        self.audio: Optional[PyAudio] = None
        self.streams: List[Optional[Stream]] = []
        self.streaming: bool = False

    def __enter__(self):
        self.audio = PyAudio()

        self.streams = [
            self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk,
            )
            for device_index in self.device_indices
        ]
        self.streaming = True
        return self

    def __exit__(self, *args):
        self.streaming = False
        for stream in self.streams:
            if stream:
                stream.stop_stream()
                stream.close()
        self.streams = []

        if self.audio:
            self.audio.terminate()
            self.audio = None

    def _mix_audio_chunks(self, audio_arrays: List[np.ndarray]) -> np.ndarray:
        if not audio_arrays:
            return np.array([], dtype=np.int16)
        mixed_data = np.sum(audio_arrays, axis=0) / len(audio_arrays)
        mixed_data = np.clip(mixed_data, -32768, 32767).astype(np.int16)
        return mixed_data

    def multi_channel_gen(self) -> Generator[Dict[int, bytes], None, None]:
        try:
            while self.streaming:
                audio_data = {}
                for device_index, stream in zip(self.device_indices, self.streams):
                    try:
                        data = stream.read(self.chunk, exception_on_overflow=False)
                        audio_data[device_index] = data
                    except Exception as e:
                        print(f"Error reading from device {device_index}: {e}")
                if audio_data:
                    yield audio_data
        except Exception as e:
            print(f"Error in audio_generator: {e}")

    def single_channel_gen(self) -> Generator[bytes, None, None]:
        try:
            while self.streaming:
                audio_arrays = []
                for device_index, stream in zip(self.device_indices, self.streams):
                    try:
                        data = stream.read(self.chunk, exception_on_overflow=False)
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        audio_arrays.append(audio_array)
                    except Exception as e:
                        print(f"Error reading from device {device_index}: {e}")
                if audio_arrays:
                    mixed_data = self._mix_audio_chunks(audio_arrays)
                    yield mixed_data.tobytes()
        except Exception as e:
            print(f"Error in mixed_audio_generator: {e}")

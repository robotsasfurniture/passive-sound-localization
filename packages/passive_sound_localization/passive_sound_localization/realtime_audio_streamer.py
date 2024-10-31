import logging
from typing import Dict, Generator, List, Optional
from pyaudio import PyAudio, paInt16, Stream
from scipy.signal import resample
import numpy as np

from passive_sound_localization.models.configs.realtime_streamer import RealtimeAudioStreamerConfig

logger = logging.getLogger(__name__)


class InvalidDeviceIndexError(Exception):
    pass


# TODO: Make it take in Hydra config
class RealtimeAudioStreamer:
    def __init__(self, config: RealtimeAudioStreamerConfig):
        self.sample_rate: int = config.sample_rate
        self.channels: int = config.channels
        self.chunk: int = config.chunk
        self.device_indices = config.device_indices
        # self.device_indices: List[int] = [2, 3, 4, 5] # Lab configuration
        # self.device_indices: List[int] = [4, 6, 8, 10] # Nico's laptop (configuration 1)
        # self.device_indices: List[int] = [2, 4, 6, 8] # Nico's laptop (configuration 2)
        self.format = paInt16
        self.audio: Optional[PyAudio] = None
        self.streams: List[Optional[Stream]] = []
        self.streaming: bool = False
        self.original_sample_rates: Dict[int, int] = {2: 48000, 4: 48000, 6: 48000, 8: 48000}

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

    def _resample_audio(
        self, audio_data: bytes, original_sample_rate: int, target_sample_rate: int
    ) -> bytes:
        if original_sample_rate == target_sample_rate:
            return audio_data

        number_of_samples = round(
            len(audio_data) * float(target_sample_rate) / original_sample_rate
        )
        resampled_audio = resample(audio_data, number_of_samples)
        return resampled_audio.astype(np.int16).tobytes()

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
                        # TODO: fix resampling
                        resampled_data = self._resample_audio(
                            data,
                            self.sample_rate,
                            self.sample_rate,
                        )
                        audio_data[device_index] = resampled_data
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
                        resampled_data = self._resample_audio(
                            audio_data=data,
                            original_sample_rate=self.sample_rate,
                            target_sample_rate=self.sample_rate,
                        )
                        audio_array = np.frombuffer(resampled_data, dtype=np.int16)
                        audio_arrays.append(audio_array)
                    except Exception as e:
                        print(f"Error reading from device {device_index}: {e}")
                if audio_arrays:
                    mixed_data = self._mix_audio_chunks(audio_arrays)
                    yield mixed_data.tobytes()
        except Exception as e:
            print(f"Error in mixed_audio_generator: {e}")

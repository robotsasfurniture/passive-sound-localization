import logging
from typing import Generator, List
import numpy as np
import threading
from pydub import AudioSegment
import pyaudio

from models.configs.realtime_streamer import (
    RealtimeAudioStreamerConfig,
)

logger = logging.getLogger(__name__)

import pyaudio
import threading

class RealtimeAudioStreamer:
    def __init__(self, config: RealtimeAudioStreamerConfig):
        """
        Initializes the AudioStreamer class.

        Parameters:
            mic_indices (list): List of microphone indices (int) to stream audio from.
            rate (int): The sample rate in Hz (default is 44100).
            chunk_size (int): The number of frames per buffer (default is 1024).
        """
        self.mic_indices = config.mic_indices
        self.sample_rate = config.sample_rate
        self.chunk_size = config.chunk_size
        self.is_running = False
        self.pyaudio_instance = pyaudio.PyAudio()
        self.streams = []
        for mic_index in self.mic_indices:
            stream = self.pyaudio_instance.open(
                rate=self.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=self.chunk_size
            )
            self.streams.append(stream)
        self.thread_lock = threading.Lock()

    def __enter__(self):
        self.is_running = True
        return self

    def audio_generator(self) -> Generator[List[bytes], None, None]:
        """
        Thread-safe generator that yields a list of audio data chunks from all microphones.

        Yields:
            list: A list where each element is the byte data from a microphone in the same
                  order as `mic_indices`.
        """
        while self.is_running:
            with self.thread_lock:
                yield [stream.read(self.chunk_size) for stream in self.streams]

    def __exit__(self, *args, **kwargs):
        """Stops all audio streams and joins threads."""
        self.is_running = False
        self.pyaudio_instance.terminate()

    def resample_stream(self, stream: bytes, target_sample_rate: int = 24000) -> bytes:
        try:
            np_stream = np.frombuffer(stream, dtype=np.int16).astype(np.int16).tobytes()
            audio_segment = AudioSegment(
                np_stream, frame_rate=self.sample_rate, sample_width=2, channels=1
            )
            audio_segment = audio_segment.set_frame_rate(
                target_sample_rate
            ).set_channels(1)
            return audio_segment.raw_data

        except Exception as e:
            print(f"Error in resample_stream: {e}")
            return b""

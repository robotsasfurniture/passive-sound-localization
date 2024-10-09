from passive_sound_localization.config.audio_mixer_config import AudioMixerConfig
import numpy as np
import logging
import pyaudio
import wave
import os

logger = logging.getLogger(__name__)


class AudioMixer:
    def __init__(self, config: AudioMixerConfig):
        self.config = config
        self.audio_interface = pyaudio.PyAudio()
        self.streams = []
        self.frames_per_channel = [[] for _ in range(self.config.mic_count)]
        self.output_dir_single = os.path.join("audio_files", "single_channel")
        self.output_dir_multi = os.path.join("audio_files", "multi_channel")
        os.makedirs(self.output_dir_single, exist_ok=True)
        os.makedirs(self.output_dir_multi, exist_ok=True)

    def get_device_indices(self):
        """Get the device indices for the available microphones."""
        mic_indices = []
        info = self.audio_interface.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount")
        logger.debug(f"Number of audio devices: {num_devices}")
        for i in range(num_devices):
            device_info = self.audio_interface.get_device_info_by_host_api_device_index(
                0, i
            )
            if device_info.get("maxInputChannels") > 0:
                mic_indices.append(i)
                logger.debug(
                    f"Found microphone: {device_info.get('name')} at index {i}"
                )
                if len(mic_indices) == self.config.mic_count:
                    break
        if len(mic_indices) < self.config.mic_count:
            logger.error(
                f"Only found {len(mic_indices)} microphones. {self.config.mic_count} required."
            )
            raise RuntimeError("Insufficient number of microphones available.")
        logger.debug(f"Microphone device indices: {mic_indices}")
        return mic_indices

    def open_streams(self):
        """Open streams for each microphone."""
        mic_indices = self.get_device_indices()
        for i in range(self.config.mic_count):
            stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=mic_indices[i],
                frames_per_buffer=self.config.chunk_size,
            )
            self.streams.append(stream)
            logger.debug(f"Opened stream for microphone {i+1}")

    def close_streams(self):
        """Close all audio streams."""
        for stream in self.streams:
            stream.stop_stream()
            stream.close()
        self.audio_interface.terminate()
        logger.debug("Closed all audio streams.")

    def record_audio(self):
        """Record audio from all microphones."""
        self.open_streams()
        logger.info("Recording audio...")
        num_chunks = int(
            self.config.sample_rate
            / self.config.chunk_size
            * self.config.record_seconds
        )
        for _ in range(num_chunks):
            for idx, stream in enumerate(self.streams):
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                self.frames_per_channel[idx].append(data)
        logger.info("Finished recording.")
        self.close_streams()

    def save_audio_files(self):
        """Save individual channels and mixed audio as WAV files."""
        # Save individual channels
        for i, frames in enumerate(self.frames_per_channel):
            output_path = os.path.join(
                self.output_dir_multi, f"output_channel_{i+1}.wav"
            )
            wf = wave.open(output_path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(self.audio_interface.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(b"".join(frames))
            wf.close()
            logger.info(f"Saved audio for channel {i+1} to {output_path}")

        # Mix the audio
        logger.info("Mixing audio channels.")
        mixed_frames = self.mix_audio_channels()
        # Save mixed audio
        output_path = os.path.join(self.output_dir_single, "output.wav")
        wf = wave.open(output_path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.audio_interface.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.config.sample_rate)
        wf.writeframes(mixed_frames)
        wf.close()
        logger.info(f"Saved mixed audio to {output_path}")

    def mix_audio_channels(self) -> bytes:
        """Mix audio frames from all channels."""
        # Convert frames to numpy arrays
        channels_data = []
        for frames in self.frames_per_channel:
            audio_data = b"".join(frames)
            samples = np.frombuffer(audio_data, dtype=np.int16)
            channels_data.append(samples)
            logger.debug(f"Channel data length: {len(samples)}")

        # Truncate to the shortest length
        min_length = min(len(data) for data in channels_data)
        channels_data = [data[:min_length] for data in channels_data]

        # Stack channels and compute the mean
        stacked_data = np.vstack(channels_data)
        mixed_data = np.mean(stacked_data, axis=0)
        mixed_data = mixed_data.astype(np.int16)
        logger.debug(f"Mixed audio data length: {len(mixed_data)}")

        return mixed_data.tobytes()

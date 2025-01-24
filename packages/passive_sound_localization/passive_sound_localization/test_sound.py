import wave
from models.configs.logging import LoggingConfig
from logger import setup_logger
import numpy as np
import logging
import glob
from scipy.io.wavfile import read
from scipy.signal import resample
from localization import NewSoundLocalizer, SoundLocalizer
from visualizer import Visualizer
from models.configs.localization import LocalizationConfig
import webrtcvad
import os

setup_logger(LoggingConfig(), enable_logging=True)
logger = logging.getLogger(__name__)

# Folders
folders = [
    "back",
    "front",
    "left",
    "right",
    "front-left",
    "front-right",
    "back-left",
    "back-right",
]

SAMPLE_RATE = 16000


def voice_activity_detection(sound_data_chunk, frame_duration_ms=30, sample_rate=16000):
    """
    Detects voice activity in a given chunk of sound data using py-webrtcvad.

    Parameters:
    - sound_data_chunk: A list of audio bytes representing a single chunk from multiple sources.
    - frame_duration_ms: Duration of each frame in milliseconds (10, 20, or 30).
    - sample_rate: Sample rate of the audio in Hz (must be 16,000 Hz for webrtcvad).

    Returns:
    - bool: True if voice activity is detected in any of the sound sources; otherwise, False.
    """
    frame_length = int(
        sample_rate * frame_duration_ms / 1000
    )  # Calculate frame length in bytes
    vad = webrtcvad.Vad(mode=3)  # Aggressiveness level for VAD (0-3)
    frame_size = int(
        sample_rate * frame_duration_ms / 1000 * 2
    )  # Frame size in bytes (2 bytes per sample for 16-bit PCM)

    for sound_bytes in sound_data_chunk:
        # Ensure the sound data is a multiple of frame_size
        if len(sound_bytes) < frame_size:
            logger.warning("Frame size too small, skipping this chunk.")
            continue  # Skip chunks that are too small

        # Split sound_bytes into frames of frame_size
        frames = [
            sound_bytes[i : i + frame_size]
            for i in range(0, len(sound_bytes), frame_size)
            if len(sound_bytes[i : i + frame_size]) == frame_size
        ]

        # Check each frame for speech
        for frame in frames:
            if vad.is_speech(frame, sample_rate):
                logger.info("Voice activity detected.")
                return True

    logger.info("No voice activity detected.")
    return False


def read_wav_to_bytes(file_path, chunk_size=1024, target_sample_rate=44100):
    # Open the WAV file
    with wave.open(file_path, "rb") as wav_file:
        # Get properties of the WAV file
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()

        logger.info(f"Sample rate: {sample_rate} Hz")
        logger.info(f"Channels: {num_channels}")
        logger.info(f"Sample width: {sample_width} bytes")
        logger.info(f"Number of frames: {num_frames}")

        # Read all audio frames
        raw_audio = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_audio, dtype=np.int16)

        # Resample if the sample rate does not match the target
        if sample_rate != target_sample_rate:
            logger.info(
                f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz..."
            )
            duration = num_frames / sample_rate
            target_num_frames = int(duration * target_sample_rate)
            audio_data = resample(audio_data, target_num_frames).astype(np.int16)

        # Chunk the audio data
        data_bytes = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size].tobytes()
            data_bytes.append(chunk)

    return data_bytes


for folder in folders:
    # Load sound files
    # files_path = f"../../../sounds/{folder}/"
    # files = glob.glob(files_path + "/*.wav")

    files_path = f"/sounds/{folder}/"
    pattern = os.getcwd() + files_path + '*.wav'
    # files = glob.glob(files_path + "/*.wav")
    files = glob.glob(pattern)

    sound_data = []
    for file in files:
        sound_data.append(read_wav_to_bytes(file, target_sample_rate=SAMPLE_RATE))

    localization_config = LocalizationConfig(sample_rate=SAMPLE_RATE)
    localizer = SoundLocalizer(
        localization_config,
        Visualizer(
            microphone_positions=localization_config.mic_positions,
            title=folder,
        ),
    )
    # localizer = NewSoundLocalizer(
    #     localization_config,
    #     Visualizer(
    #         microphone_positions=localization_config.mic_positions,
    #         title=folder,
    #     ),
    # )

    # Now loop through a zip of the sub lists in sound_data
    for sound_data_chunk in zip(*sound_data):

        # Check for voice activity detection
        if not voice_activity_detection(sound_data_chunk, sample_rate=SAMPLE_RATE):
            continue

        for result in localizer.localize_stream(sound_data_chunk):
            logger.info(result)
    
    if folder == "back":
        break

from passive_sound_localization.config.vad_config import VADConfig
import numpy as np
import webrtcvad
import logging

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(self, config: VADConfig):
        self.config = config
        self.vad = webrtcvad.Vad(self.config.aggressiveness)
        self.frame_duration_ms = self.config.frame_duration_ms

    def is_speaking(self, audio_data: np.ndarray, sample_rate=16000) -> bool:
        """
        Determines whether someone is speaking in the provided audio data.

        Parameters:
        - audio_data: The mixed single-channel audio data as a NumPy array of int16 samples.
        - sample_rate: The sample rate of the audio data (default is 16000 Hz).

        Returns:
        - True if speech is detected; False otherwise.
        """
        if not self.config.enabled:
            logger.info("VAD is disabled. Assuming speech is present.")
            return True

        logger.debug("Performing voice activity detection.")

        # Ensure frame duration is valid
        if self.frame_duration_ms not in [10, 20, 30]:
            logger.error("Invalid frame duration. Must be 10, 20, or 30 milliseconds.")
            raise ValueError("Invalid frame duration for VAD.")

        # Calculate frame size in samples
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        if len(audio_data) < frame_size:
            logger.warning("Audio data is shorter than one frame.")
            return False

        # Convert audio data to bytes
        audio_bytes = audio_data.tobytes()

        # Iterate over the audio data in frames
        is_speech_detected = False
        num_frames = len(audio_bytes) // (frame_size * 2)  # 2 bytes per int16 sample
        for i in range(num_frames):
            start = i * frame_size * 2
            end = start + frame_size * 2
            frame = audio_bytes[start:end]
            if len(frame) < frame_size * 2:
                logger.debug("Incomplete frame detected at the end of audio data.")
                break
            is_speech = self.vad.is_speech(frame, sample_rate)
            logger.debug(f"Frame {i+1}/{num_frames}: Speech detected = {is_speech}")
            if is_speech:
                is_speech_detected = True
                logger.info("Speech detected in audio.")
                break  # Early exit if speech is detected

        if not is_speech_detected:
            logger.info("No speech detected in audio.")
        return is_speech_detected

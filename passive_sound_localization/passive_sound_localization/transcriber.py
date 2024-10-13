from passive_sound_localization.models.configs.transcriber import TranscriberConfig
from openai import OpenAI, OpenAIError
import logging
import os

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, config: TranscriberConfig):
        self.config = config
        logger.info(f"API Key: {self.config.api_key}")
        self.openai_client = OpenAI(api_key=self.config.api_key)

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio from bytes using OpenAI Whisper.

        Parameters:
        - audio_bytes: The audio data in bytes to be transcribed.

        Returns:
        - The transcribed text as a string.
        """
        logger.info("Transcribing audio bytes.")

        try:
            response = self.openai_client.audio.transcriptions.create(
                model=self.config.model_name,
                file=audio_bytes,
                language=self.config.language,
            )
            transcription = response["text"]
            logger.info("Transcription successful.")
        except OpenAIError as e:
            logger.error(f"OpenAI API error during transcription: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during transcription: {e}")
            raise e

        logger.debug(f"Transcription result: {transcription}")
        return transcription

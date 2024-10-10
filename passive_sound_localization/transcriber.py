from passive_sound_localization.config.transcriber_config import TranscriberConfig
from openai import OpenAI, OpenAIError
import logging
import os

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, config: TranscriberConfig):
        self.config = config
        logger.info(f"API Key: {self.config.api_key}")
        self.openai_client = OpenAI(api_key=self.config.api_key)

    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribes audio from a file using OpenAI Whisper.

        Parameters:
        - audio_file_path: The file path to the audio file to be transcribed.

        Returns:
        - The transcribed text as a string.
        """
        logger.info(f"Transcribing audio file: {audio_file_path}")

        # Check if the file exists
        if not os.path.isfile(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            with open(audio_file_path, "rb") as audio_file:
                response = self.openai_client.audio.transcriptions.create(
                    model=self.config.model_name,
                    file=audio_file,
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

from typing import List
import numpy as np
from passive_sound_localization.visualizer import Visualizer
import torch
from scipy.signal import spectrogram

class LocalizationResult:
    def __init__(self, angle: float, distance: float):
        self.angle = angle
        self.distance = distance

class SoundLocalizer:
    def __init__(self, model_path: str, sampling_rate: int = 16000, visualizer: Visualizer = None):
        """
        Initializes the sound localizer.

        :param model_path: Path to the pre-trained PyTorch model for localization.
        :param sampling_rate: Sampling rate of the audio streams.
        """
        self.model = torch.load(model_path)
        self.sampling_rate = sampling_rate
        self.visualizer = visualizer
        self.visualizer.open_loading_screen()
        
    def _generate_spectrogram(self, stream: np.ndarray) -> torch.Tensor:
        """
        Generates a spectrogram from an audio stream.

        :param stream: Input audio stream.
        :return: Torch tensor of the spectrogram.
        """
        f, t, Sxx = spectrogram(stream, fs=self.sampling_rate)
        Sxx = np.log1p(Sxx)  # Use log scaling for better representation
        Sxx = np.expand_dims(Sxx, axis=0)  # Add a channel dimension
        return torch.tensor(Sxx, dtype=torch.float32)

    @staticmethod
    def calculate_distance(x: float, y: float) -> float:
        """
        Calculates the distance to the target given x and y coordinates.
        :param x: X coordinate of the target.
        :param y: Y coordinate of the target.
        :return: Distance to the target.
        """
        return np.sqrt(x**2 + y**2)

    @staticmethod
    def calculate_angle(x: float, y: float) -> float:
        """
        Calculates the angle to the target given x and y coordinates.
        :param x: X coordinate of the target.
        :param y: Y coordinate of the target.
        :return: Angle to the target in degrees.
        """
        return np.arctan2(y, x) * (180 / np.pi)

    def localize(self, streams: np.ndarray) -> List[LocalizationResult]:
        """
        Localizes the sound source given audio streams.

        :param streams: Array of audio streams, each corresponding to a microphone.
        :return: LocalizationResult containing the angle and distance.
        """
        if streams.ndim != 2:
            raise ValueError("Input streams must be a 2D array with shape (num_mics, samples).")
        if streams.shape[0] < 2:
            raise ValueError("At least two streams are required for localization.")

        # Process each stream into a spectrogram
        spectrograms = [self._generate_spectrogram(stream) for stream in streams]
        spectrograms = torch.stack(spectrograms, dim=0)  # Shape: (num_mics, channels, freq, time)

        # Perform prediction with the model
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(spectrograms)  # Shape: (1, 2)

        # Extract the single predicted coordinate
        predicted_coordinates = predictions.squeeze(0).numpy()
        x, y = predicted_coordinates

        # Calculate angle and distance
        distance = self.calculate_distance(x, y)
        angle = self.calculate_angle(x, y)

        self.visualizer.plot(angle=angle, distance=distance, selected_grid_point=predicted_coordinates)

        return [LocalizationResult(angle=angle, distance=distance)]

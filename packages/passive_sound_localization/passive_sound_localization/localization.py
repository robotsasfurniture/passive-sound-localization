import numpy as np
import torch
from scipy.signal import spectrogram
import torch.nn as nn
import torch.nn.functional as F


class SoundLocalizationNet(nn.Module):
    def __init__(self):
        super(SoundLocalizationNet, self).__init__()
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 4 channels for 4 microphones
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling layers to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Transformer encoder for self-attention
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=64, dropout=0, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=2
        )

        # Fully connected layers for regression
        self.fc1 = nn.Linear(
            192 * 64, 128
        )  # Adjust input size based on spectrogram dimensions
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 3)  # Output: 3D coordinates (x, y, z)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Reshape for the transformer layer (batch_size, seq_length, feature_dim)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(
            0, 2, 1
        )  # (batch_size, seq_length, feature_dim)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Flatten the output for fully connected layers
        x = x.view(batch_size, -1)

        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class LocalizationResult:
    def __init__(self, angle: float, distance: float):
        self.angle = angle
        self.distance = distance


class SoundLocalizer:
    def __init__(self, model_path: str, sampling_rate: int = 16000):
        """
        Initializes the sound localizer.

        :param model_path: Path to the pre-trained PyTorch model for localization.
        :param sampling_rate: Sampling rate of the audio streams.
        """
        # Create model instance
        self.model = SoundLocalizationNet()

        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to evaluation mode
        self.sampling_rate = sampling_rate

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

    def _generate_spectrogram(self, stream: np.ndarray) -> torch.Tensor:
        """
        Generates a spectrogram from an audio stream.

        :param stream: Input audio stream.
        :return: Torch tensor of the spectrogram.
        """
        _, _, sxx = spectrogram(stream, fs=self.sampling_rate)
        sxx_normalized = (sxx - np.mean(sxx)) / np.std(sxx)
        return torch.from_numpy(sxx_normalized).float()

    def localize(self, streams: np.ndarray) -> LocalizationResult:
        """
        Localizes the sound source given audio streams.

        :param streams: Array of audio streams, each corresponding to a microphone.
        :return: LocalizationResult containing the angle and distance.
        """
        if streams.ndim != 2:
            raise ValueError(
                "Input streams must be a 2D array with shape (num_mics, samples)."
            )
        if streams.shape[0] < 2:
            raise ValueError("At least two streams are required for localization.")

        # Process each stream into a spectrogram
        spectrograms = [self._generate_spectrogram(stream) for stream in streams]
        spectrograms = torch.stack(spectrograms, dim=0)  # Shape: (num_mics, freq, time)

        # Add batch dimension
        spectrograms = spectrograms.unsqueeze(0)  # Shape: (1, num_mics, freq, time)

        # Perform prediction with the model
        with torch.no_grad():
            predictions = self.model(spectrograms)  # Shape: (1, 2)

        # Extract the single predicted coordinate
        predicted_coordinates = predictions.squeeze(0).numpy()
        x, y, _ = predicted_coordinates  # ignore the third dimension

        # Calculate angle and distance
        distance = self.calculate_distance(x, y)
        angle = self.calculate_angle(x, y)

        return LocalizationResult(angle=angle, distance=distance)

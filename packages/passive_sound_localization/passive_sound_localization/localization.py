from typing import List, Iterator, Optional, Tuple

# from passive_sound_localization.models.configs.localization import LocalizationConfig
# from passive_sound_localization.visualizer import Visualizer

from models.configs.localization import (
    LocalizationConfig,
)  # Only needed to run with `realtime_audio.py`
from visualizer import Visualizer
from dataclasses import dataclass
import numpy as np
import logging


logger = logging.getLogger(__name__)


class NoMicrophonePositionsError(Exception):
    """Exception raised when there are no microphone positions"""

    def __init__(self) -> None:
        super().__init__("No microphone positions were configured")


class TooFewMicrophonePositionsError(Exception):
    """Exception raised when there are less than 2 microphone positions"""

    def __init__(self, num_mic_positions: int) -> None:
        super().__init__(
            f"There should be at least 2 microphone positions. Currently only {num_mic_positions} microphone position(s) were configured"
        )


class MicrophonePositionShapeError(Exception):
    """Exception raised when microphone positions don't match (x,y) pairs"""

    def __init__(self, mic_position_shape: int) -> None:
        super().__init__(
            f"The microphone positions should be in (x,y) pairs. Currently the microphone positions come in pairs of shape {mic_position_shape}"
        )


class NoMicrophoneStreamsError(Exception):
    """Exception raised when there are no microphone streams"""

    def __init__(self) -> None:
        super().__init__("No microphone streams were passed for localization")

    pass


class TooFewMicrophoneStreamsError(Exception):
    """Exception raised when there are less than 2 microphone streams"""

    def __init__(self, num_mic_streams: int) -> None:
        super().__init__(
            f"There should be at least 2 microphone streams. Currently there are only {num_mic_streams} microphone streams"
        )


class MicrophoneStreamSizeMismatchError(Exception):
    """Exception raised when the number of microphone streams doesn't match the number of microphone positions"""

    def __init__(self, num_mics: int, num_mic_streams) -> None:
        super().__init__(
            f"The number of microphone streams should match the number of microphone positions. Currently there are {num_mic_streams} microphone streams and {num_mics} microphone positions"
        )


@dataclass(frozen=True)
class LocalizationResult:
    distance: float  # Estimated distance to the sound source in meters
    angle: float  # Estimated angle to the sound source in degrees


class SoundLocalizer:
    def __init__(self, config: LocalizationConfig, visualizer: Visualizer):
        # TODO: Make sure that mic position ordering matches the ordering of the microphone streams/device indices
        self.mic_positions = np.array(
            config.mic_positions, dtype=np.float32
        )  # Get mic positions from config
        if self.mic_positions.shape[0] == 0:
            raise NoMicrophonePositionsError()

        if self.mic_positions.shape[0] < 2:
            raise TooFewMicrophonePositionsError(
                num_mic_positions=self.mic_positions.size
            )

        if self.mic_positions.shape[1] < 2:
            raise MicrophonePositionShapeError(
                mic_position_shape=self.mic_positions.shape[1]
            )

        self.speed_of_sound: float = config.speed_of_sound
        self.sample_rate: int = config.sample_rate
        self.fft_size: int = config.fft_size
        self.num_mics: int = self.mic_positions.shape[
            0
        ]  # To be set when data is received
        self.visualize: bool = config.visualize

        # Generate circular plane of grid points for direction searching
        self.grid_points = self._generate_circular_grid()
        self.visualizer = visualizer

        if self.visualize:
            self.visualizer.set_grid_points(self.grid_points)

        # Precompute delays and phase shifts
        self.distances_to_mics, self.delays = self._compute_all_delays()
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)
        self.phase_shifts = self._compute_all_phase_shifts(self.freqs)

        # Initialize buffer for streaming
        self.buffer: Optional[np.ndarray[np.float32]] = None
        self.total_results = []

    def localize_stream(
        self,
        multi_channel_stream: List[bytes],
        num_sources: int = 1,
    ) -> Iterator[List[LocalizationResult]]:
        """
        Performs real-time sound source localization on streaming multi-channel audio data.

        Parameters:
        - multi_channel_stream: An iterator that yields lists of numpy arrays containing audio data from each microphone.
        - num_sources: The number of sound sources to localize (default is 1 for single-source localization).

        Yields:
        - List of LocalizationResult objects for each processed audio chunk.
        """
        num_mic_streams = len(multi_channel_stream)

        if num_mic_streams == 0:
            raise NoMicrophoneStreamsError()
        if num_mic_streams < 2:
            logger.error("At least two microphones are required for localization.")
            raise TooFewMicrophoneStreamsError()

        if self.num_mics != num_mic_streams:
            raise MicrophoneStreamSizeMismatchError(
                num_mics=self.num_mics, num_mic_streams=num_mic_streams
            )

        # Convert buffers into numpy arrays
        multi_channel_data = [
            np.frombuffer(data, dtype=np.float32) for data in multi_channel_stream
        ]

        # Stack the multi-channel data into a 2D array (num_mics x num_samples) and replace any na values with zeroes
        data = np.nan_to_num(np.vstack(multi_channel_data))

        # Initialize buffer if it's the first chunk
        if self.buffer is None:
            self.buffer = data
        else:
            # Append new data to the buffer
            self.buffer = np.hstack((self.buffer, data))

            # If buffer exceeds fft_size, trim it
            if self.buffer.shape[1] > self.fft_size:
                self.buffer = self.buffer[:, -self.fft_size :]

        # Compute the cross-power spectrum of the buffered signals
        cross_spectrum = self._compute_cross_spectrum(
            self.buffer, fft_size=self.fft_size
        )

        # List to hold localization results for each source
        results = []

        # Iteratively localize each source
        for _ in range(num_sources):
            best_direction, estimated_distance, source_idx = (
                self._search_best_direction(cross_spectrum)
            )
            if best_direction is not None:
                # Convert direction into an angle for the result
                estimated_angle = self._calculate_estimated_angle(best_direction)

                # Append the localization result for the source
                results.append(
                    LocalizationResult(
                        distance=estimated_distance, angle=estimated_angle
                    )
                )

                # Remove the contribution of the localized source to find the next source
                cross_spectrum = self._remove_source_contribution(
                    cross_spectrum, source_idx
                )

        self.total_results.extend(results)

        if len(self.total_results) > 5:
            average_distance = np.mean(
                [result.distance for result in self.total_results]
            )
            average_angle = np.mean([result.angle for result in self.total_results])
            self.total_results.pop(0)

        if self.visualize:
            logger.info(
                f"Visualizing localization results for source {_} with angle {estimated_angle} and distance {estimated_distance}"
            )

            average_point = self.compute_cartesian_coordinates(
                average_distance, average_angle
            )

            self.visualizer.plot(
                angle=estimated_angle,
                distance=estimated_distance,
                selected_grid_point=self.grid_points[source_idx],
                average_point=average_point,
            )

        logger.debug(f"Localization results for current chunk: {results}")
        yield results

    def _compute_cross_spectrum(
        self, mic_signals: np.ndarray[np.float32], fft_size: int = 1024
    ) -> np.ndarray[np.complex128]:
        """Compute the cross-power spectrum between microphone pairs."""
        # Correct shape: (num_mics, num_mics, fft_size // 2 + 1) for the rfft result

        mic_signals = mic_signals.astype(np.float64)

        # Compute the FFT of each microphone signal
        mic_fft = np.fft.rfft(mic_signals, fft_size)

        # Compute the cross-power spectrum for each microphone pair using broadcasting
        cross_spectrum = mic_fft[:, np.newaxis, :] * np.conj(mic_fft[np.newaxis, :, :])

        return cross_spectrum

    def _generate_circular_grid(
        self,
        offset: float = 0.45,
        radius: float = 1.0,
        num_points_radial: int = 50,
        num_points_angular: int = 360,
    ) -> np.ndarray[np.float32]:
        """Generate a grid of points on a circular plane, optimized for speed."""
        # Create radial distances from 0 to the specified radius
        r = np.linspace(offset, radius + offset, num_points_radial, dtype=np.float32)

        # Create angular values from 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, num_points_angular, dtype=np.float32)

        # Compute x and y directly using broadcasting without creating a meshgrid
        r = r[:, np.newaxis]  # Convert r to column vector for broadcasting
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Return the points stacked as (x, y) pairs
        return np.column_stack((x.ravel(), y.ravel()))

    def _calculate_estimated_angle(self, best_direction: Tuple[float, float]) -> float:
        delta_x = best_direction[0]
        delta_y = best_direction[1] - self.mic_positions[0, 1]
        return np.degrees(np.arctan2(delta_y, delta_x))

    def _search_best_direction(
        self, cross_spectrum: np.ndarray[np.complex128]
    ) -> Tuple[np.ndarray, np.float32, int]:
        """Search the circular grid for the direction with maximum beamformer output."""
        energies = self._compute_beamformer_energies(cross_spectrum)
        best_direction_idx = np.argmax(energies)
        best_direction = self.grid_points[best_direction_idx]
        estimated_distance = np.min(self.distances_to_mics[best_direction_idx])
        print(
            f"Position of closest mic: {self.mic_positions[np.argmin(self.distances_to_mics[best_direction_idx])]}"
        )
        return best_direction, estimated_distance, best_direction_idx

    def _compute_all_phase_shifts(
        self, freqs: np.ndarray[np.float32]
    ) -> np.ndarray[np.complex128]:
        """
        Precompute phase shifts for all grid points, microphone pairs, and frequency bins.
        """
        # Compute tau (time delays between microphone pairs for all grid points)
        tau = (
            self.delays[:, :, np.newaxis] - self.delays[:, np.newaxis, :]
        )  # Shape: (num_grid_points, num_mics, num_mics)

        # Compute phase shifts for all frequencies
        phase_shifts = np.exp(
            -1j
            * 2
            * np.pi
            * tau[:, :, :, np.newaxis]
            * freqs[np.newaxis, np.newaxis, np.newaxis, :]
        )  # Shape: (num_grid_points, num_mics, num_mics, num_freq_bins)
        return phase_shifts

    def _compute_all_delays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute delays for all grid points and microphones.
        """
        # Compute distances from each microphone to each grid point
        mic_positions_2d = self.mic_positions[:, :2]  # Shape: (num_mics, 2)

        # Calculate distances between microphones and grid points
        distances_to_mics = np.linalg.norm(
            mic_positions_2d[np.newaxis, :, :] - self.grid_points[:, np.newaxis, :],
            axis=2,
        )  # Shape: (num_grid_points, num_mics)

        # Compute delays: distances divided by speed of sound
        delays = (
            distances_to_mics / self.speed_of_sound
        )  # Shape: (num_grid_points, num_mics)

        return distances_to_mics, delays

    def _compute_beamformer_energies(
        self, cross_spectrum: np.ndarray[np.complex128]
    ) -> np.ndarray:
        """Compute the beamformer energy given the cross-spectrum and delays."""
        cross_spectrum_expanded = cross_spectrum[np.newaxis, :, :, :]
        # Multiply and sum over mics and frequency bins
        product = (
            cross_spectrum_expanded * self.phase_shifts
        )  # Shape: (num_grid_points, num_mics, num_mics, num_freq_bins)
        energies = np.abs(np.sum(product, axis=(1, 2, 3)))  # Shape: (num_grid_points,)
        return energies

    def _remove_source_contribution(
        self, cross_spectrum: np.ndarray[np.complex128], source_idx: int
    ) -> np.ndarray[np.complex128]:
        """
        Remove the contribution of a localized source using vectorized operations.
        """
        # Get the phase shifts for the localized source
        phase_shift = self.phase_shifts[
            source_idx
        ]  # Shape: (num_mics, num_mics, num_freq_bins)

        # Subtract the contribution from the cross-spectrum
        cross_spectrum -= phase_shift
        return cross_spectrum

    def compute_cartesian_coordinates(self, distance, angle):
        """
        Compute the Cartesian coordinates of a sound source given its distance and angle.
        """
        x = distance * np.cos(np.radians(angle))
        y = distance * np.sin(np.radians(angle))
        return x, y

import pytest
from passive_sound_localization.models.configs.localization import LocalizationConfig
from passive_sound_localization.localization import SoundLocalizer
import numpy as np


# Pytest fixture that by default returns SoundLocalizer with an empty LocalizationConfig object
@pytest.fixture
def localizer():
    def _localizer(config=None):
        if config is None:
            config = LocalizationConfig()
        return SoundLocalizer(config)
    return _localizer

# Test that the localizer correctly initializes given a valid configuration
def test_sound_localizer_initialization(localizer):
    mic_positions = [[-0.5, 0], [0.5, 0]]

    config = LocalizationConfig(
        mic_positions=mic_positions,
        speed_of_sound=343.0,
        sample_rate=16000,
        fft_size=1024
    )

    loc = localizer(config)

    assert np.array_equal(loc.mic_positions, np.array(mic_positions, dtype=np.float32))
    assert loc.speed_of_sound == 343.0
    assert loc.sample_rate == 16000
    assert loc.fft_size == 1024


# Test that the localizer return an error when there are less than 2 microphones
def test_sound_localizer_insufficient_microphones(localizer):
    # Only one microphone
    mic_positions = [[0, 0]]
    config = LocalizationConfig(
        mic_positions=mic_positions,
        speed_of_sound=343.0,
        sample_rate=16000,
        fft_size=1024
    )
    loc = localizer(config)
    
    # Mock stream with one microphone
    def multi_channel_stream():
        yield [np.zeros(1024, dtype=np.float32).tobytes()]
    
    with pytest.raises(ValueError):
        for s in multi_channel_stream():
            results = list(loc.localize_stream(s))

# Test that the circular grid generates the correct amount of grid points shaped as (x,y) pairs
def test_generate_circular_grid(localizer):
    loc = localizer()
    circular_grid = loc._generate_circular_grid()

    assert circular_grid.shape == (18000, 2) # 50 radial points * 360 angular points = 18000 points

def test_compute_distances_to_mics(localizer):
    loc = localizer()
    distances_to_mics, _ = loc._compute_all_delays()

    assert distances_to_mics.shape == (len(loc.grid_points), 4)

def test_compute_delays(localizer):
    loc = localizer()
    _, delays = loc._compute_all_delays()

    assert delays.shape == (len(loc.grid_points), 4)

def test_compute_cross_power_spectrum(localizer):
    config = LocalizationConfig(
        mic_positions=np.array([
            [0.0000, 0.4500],
            [0.4500, 0.0000],
            [0.0000, -0.4500],
            [-0.4500, 0.0000]
        ]),
        speed_of_sound=343.0,
        sample_rate=16000,
        fft_size=1024
    )
    loc = localizer(config)
    # Prepare test data
    mic_signals = np.random.rand(loc.num_mics, loc.fft_size).astype(np.float32)

    cross_spectrum = loc._compute_cross_spectrum(mic_signals)

    assert cross_spectrum.shape == (loc.num_mics, loc.num_mics, loc.fft_size // 2 + 1)

def test_calculate_phase_shifts(localizer):
    pass

def test_compute_beamformer_energies(localizer):
    pass


def test_remove_source_contribution(localizer):
    pass

# Test that the compute cartesian coordinates function returns a valid answer
def test_compute_cartesian_coordinates(localizer):
    loc = localizer()
    distance = 5.0
    angle = 45.0
    x, y = loc.compute_cartesian_coordinates(distance, angle)

    assert x == distance * np.cos(np.radians(angle))
    assert y == distance * np.sin(np.radians(angle))
# Successfully initializes and opens an audio stream with default parameters
# import pytest
# from pyaudio import paInt16
# import numpy as np
# from unittest.mock import call, patch, MagicMock
# from passive_sound_localization.realtime_audio_streamer import RealtimeAudioStreamer, InvalidDeviceIndexError

# # Correctly enters async context manager with defaults
# @pytest.mark.asyncio
# async def test_initialize_and_open_audio_stream_with_defaults():
#     with patch('passive_sound_localization.realtime_audio_streamer.PyAudio') as mock_pyaudio:
#         mock_audio = MagicMock()
#         mock_pyaudio.return_value = mock_audio
#         mock_audio.get_device_info_by_index.return_value = {"defaultSampleRate": 24000}
#         mock_audio.open.return_value = MagicMock()

#         async with RealtimeAudioStreamer() as streamer:
#             assert streamer.audio is not None
#             assert streamer.stream is not None
#             assert streamer.original_sample_rate == 24000
#             assert streamer.streaming is True
        
#             # Additional assertions to ensure the stream is opened with correct parameters
#             mock_audio.open.assert_called_once_with(
#                 format=streamer.format,
#                 channels=streamer.channels,
#                 rate=streamer.original_sample_rate,
#                 input=True,
#                 input_device_index=streamer.device_index,
#                 frames_per_buffer=streamer.chunk
#             )

# # Closes and terminates audio stream and PyAudio instance on exit
# @pytest.mark.asyncio
# async def test_close_and_terminate_audio_stream():
#     with patch('passive_sound_localization.realtime_audio_streamer.PyAudio') as mock_pyaudio:
#         mock_audio = MagicMock()
#         mock_stream = MagicMock()
#         mock_pyaudio.return_value = mock_audio
#         mock_audio.get_device_info_by_index.return_value = {"defaultSampleRate": 24000}
#         mock_audio.open.return_value = mock_stream

#         async with RealtimeAudioStreamer() as streamer:
#             await streamer.__aexit__()
#             assert streamer.audio is None
#             assert streamer.stream is None
#             assert streamer.streaming is False
#             mock_stream.stop_stream.assert_called_once()
#             mock_stream.close.assert_called_once()
#             mock_pyaudio.return_value.terminate.assert_called_once()

# # Handles valid device index without raising exceptions
# @pytest.mark.asyncio
# async def test_handles_valid_device_index():
#     with patch('passive_sound_localization.realtime_audio_streamer.PyAudio') as mock_pyaudio:
#         mock_audio = MagicMock()
#         mock_pyaudio.return_value = mock_audio
#         mock_audio.get_device_info_by_index.return_value = {"defaultSampleRate": 44000}
#         mock_audio.open.return_value = MagicMock()

#         async with RealtimeAudioStreamer(device_index=5):
#             mock_audio.open.assert_called_once_with(
#                 format=paInt16,
#                 channels=1,
#                 rate=44000,
#                 input=True,
#                 input_device_index=5,
#                 frames_per_buffer=1024,
#             )

# # Handles an invalid device index by raising an exceptions
# @pytest.mark.asyncio
# async def test_invalid_device_index_raises_exception():
#     with patch('passive_sound_localization.realtime_audio_streamer.PyAudio') as mock_pyaudio:
#         mock_audio = MagicMock()
#         mock_pyaudio.return_value = mock_audio
#         mock_audio.get_device_info_by_index.side_effect = IndexError

#         with pytest.raises(InvalidDeviceIndexError):
#             async with RealtimeAudioStreamer(device_index=999):
#                 pass

# # Properly resamples audio data when original and target sample rates differ
# @pytest.mark.skip(reason="Want to clean up resample function")
# @pytest.mark.wip
# @pytest.mark.asyncio
# async def test_resample_audio_data():
#     audio_data = np.array([1, 2, 3, 4, 5])
#     original_sample_rate = 24000
#     target_sample_rate = 16000

#     with (
#         patch('passive_sound_localization.realtime_audio_streamer.resample') as mock_resample,
#         patch('passive_sound_localization.realtime_audio_streamer.PyAudio') as mock_pyaudio
#     ):
#         mock_audio = MagicMock()
#         mock_pyaudio.return_value = mock_audio
#         mock_audio.get_device_info_by_index.return_value = {"defaultSampleRate": 24000}
#         mock_resample.return_value = np.array([1, 2, 3, 4])

#         async with RealtimeAudioStreamer(sample_rate=target_sample_rate) as streamer:
#             resampled_audio = streamer._resample_audio(audio_data, original_sample_rate, target_sample_rate)

#             mock_resample.assert_called_once_with(audio_data, 3)
#             assert np.array_equal(resampled_audio, np.array([1, 2, 3, 4]))

# # Properly yields audio data in the audio_generator method while streaming
# @pytest.mark.skip(reason="Need to fix audio generator")
# @pytest.mark.wip
# @pytest.mark.asyncio
# async def test_properly_yields_audio_data():
#     with patch('passive_sound_localization.realtime_audio_streamer.PyAudio') as mock_pyaudio:
#         mock_stream = MagicMock()
#         mock_pyaudio_instance = mock_pyaudio.return_value
#         mock_pyaudio_instance.open.return_value = mock_stream
#         mock_stream.read.side_effect = [b'audio_data_1', b'audio_data_2', b'']

#         async with RealtimeAudioStreamer() as streamer:
#             audio_data = [audio async for audio in streamer.multi_channel_gen()]

#         assert audio_data == [b'resampled_audio_1', b'resampled_audio_2']
#         mock_stream.read.assert_has_calls([call(1024, exception_on_overflow=False), call(1024, exception_on_overflow=False)])
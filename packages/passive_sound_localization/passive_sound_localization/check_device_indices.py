import pyaudio

def check_device_indices() -> None:
    """
    Lists and prints information about all audio input/output devices available on the system.

    This function uses the PyAudio library to query and display detailed information about 
    each audio device connected to the system. The information includes device name, index, 
    supported input/output channels, sample rates, and other device-specific details.

    Returns:
        None
    """
    pyaudio_instance = pyaudio.PyAudio()
    for mic_index in range(pyaudio_instance.get_device_count()):
        device_info = pyaudio_instance.get_device_info_by_index(mic_index)
        if device_info['maxInputChannels'] > 0 and "USB" in device_info['name']:
            print(f"Device info: {device_info}")


if __name__ == "__main__":
    check_device_indices()
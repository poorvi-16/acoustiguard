# detection/audio_capture.py
# This file opens your laptop microphone and reads audio continuously.

import sounddevice as sd
import numpy as np
import queue
import threading

SAMPLE_RATE = 16000   # 16,000 samples per second
CHUNK_SIZE  = 16000   # 1 second of audio per chunk

# A queue is like a pipe - audio goes in one end, our AI reads from the other
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """
    This function is called automatically every time
    the microphone captures a new chunk of audio.
    """
    if status:
        print(f"Audio warning: {status}")
    # Put the audio chunk into the queue
    audio_queue.put(indata.copy())


def start_listening():
    """
    Opens the microphone and starts capturing audio.
    Returns the stream object.
    """
    print(f"Opening microphone at {SAMPLE_RATE}Hz...")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,           # mono
        dtype='float32',
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )
    stream.start()
    print("Microphone open. Listening...")
    return stream


def get_audio_chunk():
    """
    Gets the next chunk of audio from the queue.
    Returns a 1D numpy array of audio samples.
    Waits until audio is available.
    """
    chunk = audio_queue.get()
    return chunk.flatten()  # convert from (16000, 1) to (16000,)


def test_microphone():
    """
    Opens mic for 3 seconds and prints the volume level.
    If numbers change when you speak or clap - mic is working.
    """
    print("Testing microphone for 5 seconds...")
    print("Make some noise - speak or clap!")
    print("-" * 40)

    stream = start_listening()

    for i in range(5):
        chunk = get_audio_chunk()
        volume = np.abs(chunk).mean() * 1000  # scale for readability
        bar = "█" * int(volume * 10)
        print(f"Second {i+1}: Volume = {volume:.2f}  {bar}")

    stream.stop()
    stream.close()
    print("-" * 40)
    print("Microphone test complete!")


if __name__ == "__main__":
    test_microphone()

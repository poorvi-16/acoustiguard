# detection/spectrogram.py
# This file converts raw audio into a frequency map (mel-spectrogram)
# Think of it as turning sound into a picture our AI can read

import numpy as np
import librosa

def audio_to_spectrogram(audio_array, sample_rate=16000):
    """
    Takes a 1D array of audio samples.
    Returns a 2D mel-spectrogram as a numpy array.
    """
    # Create the mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio_array.astype(float),
        sr=sample_rate,
        n_mels=128,        # 128 frequency bands
        fmax=8000          # focus on 0-8kHz (drone range)
    )

    # Convert to decibels (log scale) - makes patterns clearer
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def test_spectrogram():
    """
    Quick test - creates a fake audio signal and converts it.
    If this prints a shape, everything is working.
    """
    print("Testing spectrogram converter...")

    # Create 1 second of fake audio (silence with a tone)
    sample_rate = 16000
    duration = 1  # second
    t = np.linspace(0, duration, sample_rate)
    
    # Simulate a drone-like tone at 400Hz
    fake_drone_audio = np.sin(2 * np.pi * 400 * t) * 0.5

    # Convert to spectrogram
    spec = audio_to_spectrogram(fake_drone_audio, sample_rate)

    print(f"Input audio shape:       {fake_drone_audio.shape}")
    print(f"Output spectrogram shape: {spec.shape}")
    print(f"Min value: {spec.min():.1f}, Max value: {spec.max():.1f}")
    print("SUCCESS - spectrogram converter is working!")

if __name__ == "__main__":
    test_spectrogram()

# data/download_data.py
# Generates a varied, realistic training dataset
# Heavy randomisation to prevent overfitting

import os
import numpy as np
import soundfile as sf
import librosa

os.makedirs("data/raw_audio/drone",     exist_ok=True)
os.makedirs("data/raw_audio/not_drone", exist_ok=True)

SAMPLE_RATE = 16000
DURATION    = 2
N_SAMPLES   = SAMPLE_RATE * DURATION
t           = np.linspace(0, DURATION, N_SAMPLES)

def save_wav(array, path, sr=16000):
    array = array / (np.max(np.abs(array)) + 1e-8) * 0.9
    sf.write(path, array.astype(np.float32), sr)

def add_random_noise(signal, min_level=0.05, max_level=0.25):
    noise = np.random.randn(len(signal))
    level = np.random.uniform(min_level, max_level)
    return signal + noise * level

def add_random_fade(signal):
    # Random amplitude envelope
    fade_len = np.random.randint(1000, 5000)
    fade_in  = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len]  *= fade_in
    signal[-fade_len:] *= fade_out
    return signal

def add_random_reverb(signal):
    # Simulate room echo
    delay  = np.random.randint(200, 800)
    decay  = np.random.uniform(0.1, 0.4)
    echo   = np.zeros_like(signal)
    echo[delay:] = signal[:-delay] * decay
    return signal + echo

print("=" * 50)
print("Generating training dataset with heavy variation")
print("=" * 50)

# ── DRONE CLIPS ───────────────────────────────
# Each profile has a base frequency but every
# single clip is different due to randomisation

drone_configs = [
    # name,           base_freq, count
    ("dji_phantom",   400,       150),
    ("dji_mini",      300,       150),
    ("racing_drone",  600,       150),
    ("mavic_pro",     350,       150),
    ("generic_quad",  450,       150),
    ("hexacopter",    250,       150),
]

total_drone = 0

for name, base_freq, count in drone_configs:
    print(f"Generating {count} clips — {name} "
          f"(base freq: {base_freq}Hz)...")

    for i in range(count):

        # Very wide frequency variation per clip
        freq = base_freq + np.random.uniform(-80, 80)

        # Random harmonic strengths
        h1 = np.random.uniform(0.2, 0.7)
        h2 = np.random.uniform(0.05, 0.3)
        h3 = np.random.uniform(0.02, 0.15)
        h4 = np.random.uniform(0.01, 0.08)

        # Build signal from harmonics
        signal  = np.sin(2 * np.pi * freq       * t) * h1
        signal += np.sin(2 * np.pi * freq * 2   * t) * h2
        signal += np.sin(2 * np.pi * freq * 3   * t) * h3
        signal += np.sin(2 * np.pi * freq * 4   * t) * h4

        # Random slight frequency wobble 
        # (motor speed isn't perfectly constant)
        wobble_rate = np.random.uniform(0.5, 3.0)
        wobble_amt  = np.random.uniform(0, 15)
        freq_wobble = np.sin(2 * np.pi * wobble_rate * t) * wobble_amt
        signal += np.sin(2 * np.pi * (freq + freq_wobble) * t) * 0.1

        # Random amplitude (drone at different distances)
        signal *= np.random.uniform(0.2, 1.0)

        # Always add noise — varying levels
        signal = add_random_noise(signal, 0.03, 0.20)

        # Sometimes add background traffic under drone
        if np.random.random() > 0.4:
            traffic_level = np.random.uniform(0.05, 0.20)
            signal += np.random.randn(N_SAMPLES) * traffic_level
            signal += (np.sin(2 * np.pi * 80 * t) *
                      np.random.uniform(0.03, 0.12))

        # Sometimes add wind
        if np.random.random() > 0.5:
            wind = np.random.randn(N_SAMPLES)
            wind = np.convolve(
                wind,
                np.ones(100)/100,
                mode='same'
            )
            signal += wind * np.random.uniform(0.02, 0.10)

        # Random fade
        if np.random.random() > 0.5:
            signal = add_random_fade(signal)

        # Random reverb (indoor/outdoor variation)
        if np.random.random() > 0.6:
            signal = add_random_reverb(signal)

        path = f"data/raw_audio/drone/{name}_{i:04d}.wav"
        save_wav(signal, path)
        total_drone += 1

print(f"Total drone clips: {total_drone}")

# ── NOT-DRONE CLIPS ───────────────────────────

print("\nGenerating non-drone sounds...")

not_drone_count = 0

# Traffic — 200 clips
print("Generating 200 traffic clips...")
for i in range(200):
    # Low rumble dominant
    signal  = np.random.randn(N_SAMPLES) * 0.4
    signal += np.sin(2 * np.pi * np.random.uniform(60,100) * t) * np.random.uniform(0.1, 0.3)
    signal += np.sin(2 * np.pi * np.random.uniform(100,150) * t) * np.random.uniform(0.05, 0.2)
    # Occasional horn
    if np.random.random() > 0.7:
        horn_freq = np.random.uniform(400, 700)
        horn_start = np.random.randint(0, N_SAMPLES - 3000)
        horn_len   = np.random.randint(1000, 3000)
        signal[horn_start:horn_start+horn_len] += (
            np.sin(2 * np.pi * horn_freq *
                   t[:horn_len]) * 0.3
        )
    save_wav(signal, f"data/raw_audio/not_drone/traffic_{i:04d}.wav")
    not_drone_count += 1

# Wind — 200 clips
print("Generating 200 wind clips...")
for i in range(200):
    # Broadband noise with low-freq emphasis
    signal = np.random.randn(N_SAMPLES)
    # Low pass filter simulation
    kernel_size = np.random.randint(50, 300)
    signal = np.convolve(
        signal,
        np.ones(kernel_size)/kernel_size,
        mode='same'
    )
    signal *= np.random.uniform(0.3, 1.0)
    # Add occasional gust
    if np.random.random() > 0.5:
        gust_start = np.random.randint(0, N_SAMPLES//2)
        gust_len   = np.random.randint(2000, 8000)
        gust       = np.random.randn(gust_len) * 0.5
        signal[gust_start:gust_start+gust_len] += gust
    save_wav(signal, f"data/raw_audio/not_drone/wind_{i:04d}.wav")
    not_drone_count += 1

# Silence / ambient — 200 clips
print("Generating 200 silence/ambient clips...")
for i in range(200):
    # Very quiet noise floor only
    level  = np.random.uniform(0.001, 0.03)
    signal = np.random.randn(N_SAMPLES) * level
    save_wav(signal, f"data/raw_audio/not_drone/silence_{i:04d}.wav")
    not_drone_count += 1

# Crowd / human noise — 200 clips
print("Generating 200 crowd clips...")
for i in range(200):
    signal = np.random.randn(N_SAMPLES) * 0.2
    # Multiple speech-like frequencies
    for f in np.random.uniform(100, 400, 8):
        signal += (np.sin(2 * np.pi * f * t) *
                   np.random.uniform(0.01, 0.06))
    save_wav(signal, f"data/raw_audio/not_drone/crowd_{i:04d}.wav")
    not_drone_count += 1

# Construction / machinery — 200 clips
# These are hardest negatives — 
# machinery has harmonics too
print("Generating 200 machinery clips...")
for i in range(200):
    # Random mechanical frequency 
    # but NOT drone range
    mech_freq = np.random.uniform(20, 150)
    signal  = np.sin(2 * np.pi * mech_freq * t) * 0.4
    signal += np.sin(2 * np.pi * mech_freq*2 * t) * 0.2
    signal += np.random.randn(N_SAMPLES) * 0.3
    # Irregular rhythm
    for _ in range(np.random.randint(2, 8)):
        pos = np.random.randint(0, N_SAMPLES-500)
        signal[pos:pos+500] += (np.random.randn(500) * 0.4)
    save_wav(signal, f"data/raw_audio/not_drone/machinery_{i:04d}.wav")
    not_drone_count += 1

print(f"Total non-drone clips: {not_drone_count}")

print("\n" + "=" * 50)
print("Dataset complete!")
print(f"  Drone:     {total_drone}")
print(f"  Not drone: {not_drone_count}")
print(f"  Total:     {total_drone + not_drone_count}")
print("=" * 50)
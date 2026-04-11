# detection/classifier.py
# Loads the trained model and classifies audio in real time

import numpy as np
import librosa
import tensorflow as tf

SAMPLE_RATE = 16000
N_MELS      = 128
IMG_SIZE    = 128
THRESHOLD   = 0.70  # confidence needed to trigger alert

class DroneClassifier:

    def __init__(self, model_path="model/drone_classifier.tflite"):
        print(f"Loading model from {model_path}...")
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path
        )
        self.interpreter.allocate_tensors()
        self.input_details  = \
            self.interpreter.get_input_details()
        self.output_details = \
            self.interpreter.get_output_details()
        print("Model loaded successfully")

    def predict(self, audio_array):
        """
        Takes 1D audio array.
        Returns (is_drone, confidence_score).
        """
        # Convert to spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio_array.astype(float),
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Resize and prepare for model
        mel_resized = tf.image.resize(
            mel_db[:, :, np.newaxis],
            [IMG_SIZE, IMG_SIZE]
        ).numpy()
        mel_3ch = np.repeat(mel_resized, 3, axis=-1)
        mel_norm = (mel_3ch - mel_3ch.min()) / \
                   (mel_3ch.max() - mel_3ch.min() + 1e-8)

        # Run inference
        input_data = mel_norm[np.newaxis, ...].astype(
            np.float32
        )
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data
        )
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        confidence = float(output[0][0])
        is_drone   = confidence >= THRESHOLD

        return is_drone, confidence


def test_classifier():
    """
    Tests classifier on a synthetic drone sound.
    """
    clf = DroneClassifier()
    t   = np.linspace(0, 1, SAMPLE_RATE)

    print("\nTest 1 — Drone-like sound (400Hz harmonics):")
    drone_audio  = np.sin(2 * np.pi * 400 * t) * 0.5
    drone_audio += np.sin(2 * np.pi * 800 * t) * 0.25
    drone_audio += np.sin(2 * np.pi * 1200 * t) * 0.1
    drone_audio += np.random.randn(SAMPLE_RATE) * 0.02
    is_drone, conf = clf.predict(drone_audio)
    print(f"  Confidence: {conf*100:.1f}%")
    print(f"  Result:     {'🚨 DRONE DETECTED' if is_drone else '✅ No drone'}")

    print("\nTest 2 — Silence:")
    silence = np.random.randn(SAMPLE_RATE) * 0.01
    is_drone, conf = clf.predict(silence)
    print(f"  Confidence: {conf*100:.1f}%")
    print(f"  Result:     {'🚨 DRONE DETECTED' if is_drone else '✅ No drone'}")

    print("\nTest 3 — Traffic noise:")
    traffic  = np.random.randn(SAMPLE_RATE) * 0.3
    traffic += np.sin(2 * np.pi * 80 * t) * 0.2
    is_drone, conf = clf.predict(traffic)
    print(f"  Confidence: {conf*100:.1f}%")
    print(f"  Result:     {'🚨 DRONE DETECTED' if is_drone else '✅ No drone'}")


if __name__ == "__main__":
    test_classifier()


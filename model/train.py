import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pickle

SAMPLE_RATE  = 16000
DURATION     = 2       # seconds
N_MELS       = 128
IMG_SIZE     = 128     # resize spectrogram to 128x128

print("=" * 50)
print("AcoustiGuard CNN Training")
print("=" * 50)

# ─────────────────────────────────────────────
# STEP 1: Load and convert all audio to spectrograms
# ─────────────────────────────────────────────

def audio_to_spectrogram(file_path):
    """Load a .wav file and convert to mel-spectrogram."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE,
                                  duration=DURATION)
        # Pad if too short
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio,
                (0, SAMPLE_RATE * DURATION - len(audio)))

        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS, fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Resize to fixed size for MobileNetV2
        mel_resized = tf.image.resize(
            mel_db[:, :, np.newaxis], [IMG_SIZE, IMG_SIZE]
        ).numpy()

        # MobileNetV2 needs 3 channels (RGB) - repeat the channel
        mel_3ch = np.repeat(mel_resized, 3, axis=-1)

        # Normalize to [0, 1]
        mel_norm = (mel_3ch - mel_3ch.min()) / \
                   (mel_3ch.max() - mel_3ch.min() + 1e-8)

        return mel_norm

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


print("\nLoading dataset...")
X, y = [], []

# Load drone clips (label = 1)
drone_folder = "data/raw_audio/drone"
files = os.listdir(drone_folder)
print(f"Loading {len(files)} drone clips...")

for i, fname in enumerate(files):
    if fname.endswith(".wav"):
        spec = audio_to_spectrogram(
            os.path.join(drone_folder, fname)
        )
        if spec is not None:
            X.append(spec)
            y.append(1)  # 1 = drone
    if i % 50 == 0:
        print(f"  {i}/{len(files)} loaded...")

# Load not-drone clips (label = 0)
not_drone_folder = "data/raw_audio/not_drone"
files = os.listdir(not_drone_folder)
print(f"Loading {len(files)} non-drone clips...")

for i, fname in enumerate(files):
    if fname.endswith(".wav"):
        spec = audio_to_spectrogram(
            os.path.join(not_drone_folder, fname)
        )
        if spec is not None:
            X.append(spec)
            y.append(0)  # 0 = not drone
    if i % 50 == 0:
        print(f"  {i}/{len(files)} loaded...")

X = np.array(X)
y = np.array(y)

print(f"\nDataset loaded:")
print(f"  Total samples: {len(X)}")
print(f"  Drone:         {sum(y == 1)}")
print(f"  Not drone:     {sum(y == 0)}")
print(f"  Input shape:   {X.shape}")

# ─────────────────────────────────────────────
# STEP 2: Split into train and test sets
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples:  {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 3: Build the model
# ─────────────────────────────────────────────

print("\nBuilding MobileNetV2 model...")

# Load MobileNetV2 pretrained on ImageNet
# include_top=False means we add our own final layer
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model weights - 
# we only train the top layers
base_model.trainable = False

# Unfreeze last 20 layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model built successfully")
print(f"Total parameters: "
      f"{model.count_params():,}")

# ─────────────────────────────────────────────
# STEP 4: Train
# ─────────────────────────────────────────────

print("\nTraining... (this takes 5-10 minutes)")
print("Watch the accuracy go up each epoch")
print("-" * 50)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=4,
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ],
    verbose=1
)

# ─────────────────────────────────────────────
# STEP 5: Evaluate
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("Evaluating on test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.1f}%")
print(f"Test Loss:     {loss:.4f}")

# ─────────────────────────────────────────────
# STEP 6: Save the model
# ─────────────────────────────────────────────

os.makedirs("model", exist_ok=True)

# Save full Keras model
model.save("model/drone_classifier.keras")
print("\nModel saved to model/drone_classifier.keras")

# Convert to TFLite (small version for edge deployment)
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model/drone_classifier.tflite", "wb") as f:
    f.write(tflite_model)

tflite_size = os.path.getsize(
    "model/drone_classifier.tflite"
) / 1024 / 1024

print(f"TFLite model saved: {tflite_size:.1f} MB")
print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print(f"Final accuracy: {accuracy*100:.1f}%")
print("=" * 50)
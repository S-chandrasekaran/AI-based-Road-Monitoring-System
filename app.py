import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

# -----------------------------
# MODEL BUILD (SAFE FUNCTIONAL MODEL)
# -----------------------------
def build_model():
    inputs = keras.Input(shape=(128, 128, 3), name="input_layer")

    # Encoder
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    skip1 = x
    x = layers.MaxPooling2D()(x)

    # Middle
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Decoder
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, skip1])

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Conv2D(4, 1, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="safe_unet_model")
    return model


model = build_model()
model.summary()


# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# SAVE MODEL (IMPORTANT: .keras ONLY)
# -----------------------------
print("\nSaving model...")
model.save("model.keras")
print("Model saved successfully")


# -----------------------------
# LOAD MODEL (SAFE + NO HANG)
# -----------------------------
print("\nLoading model...")

start = time.time()

model = keras.models.load_model(
    "model.keras",
    compile=False
)

print("Model loaded in:", round(time.time() - start, 2), "seconds")


# -----------------------------
# TEST PREDICTION (SAFE CHECK)
# -----------------------------
print("\nRunning test prediction...")

dummy_input = np.random.rand(1, 128, 128, 3).astype("float32")

output = model.predict(dummy_input, verbose=0)

print("Output shape:", output.shape)
print("Done successfully ✅")

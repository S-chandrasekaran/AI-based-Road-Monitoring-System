import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# =========================
# CONFIG
# =========================
MODEL_PATH = "model.h5"   # change to your model file
IMG_SIZE = (128, 128)

# =========================
# SAFE MODEL LOADING (FIX)
# =========================
try:
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False  # IMPORTANT: avoids Keras version issues
    )
    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)
    exit()


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# =========================
# PREDICT MASK
# =========================
def predict_mask(image):
    pred = model.predict(image, verbose=0)

    # For segmentation (softmax output)
    mask = np.argmax(pred[0], axis=-1)
    return mask


# =========================
# COLORIZE MASK
# =========================
def colorize_mask(mask):
    colors = [
        [0, 0, 0],        # class 0
        [255, 0, 0],      # class 1
        [0, 255, 0],      # class 2
        [0, 0, 255],      # class 3
    ]

    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        output[mask == i] = color

    return output


# =========================
# OVERLAY FUNCTION
# =========================
def overlay_image(original, mask_color, alpha=0.5):
    return cv2.addWeighted(original, 1 - alpha, mask_color, alpha, 0)


# =========================
# RUN PIPELINE
# =========================
def run(image_path, alpha=0.5):
    image = preprocess_image(image_path)

    original = cv2.imread(image_path)
    original = cv2.resize(original, IMG_SIZE)

    mask = predict_mask(image)
    mask_color = colorize_mask(mask)

    result = overlay_image(original, mask_color, alpha)

    cv2.imshow("Original", original)
    cv2.imshow("Mask", mask_color)
    cv2.imshow("Overlay", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# TEST RUN
# =========================
if __name__ == "__main__":
    image_path = "test.jpg"

    # 🔥 Overlay transparency control (0.10 → 1.00)
    overlay_alpha = 0.4

    run(image_path, alpha=overlay_alpha)

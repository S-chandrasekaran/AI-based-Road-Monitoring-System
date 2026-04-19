import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Road Defect Detection",
    page_icon="🛣️",
    layout="wide"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "road_defect_unet_multiclass.keras")
IMG_SIZE = 128

CLASS_NAMES = {
    1: "Pothole",
    2: "Crack",
    3: "Manhole"
}

CLASS_COLORS = {
    1: [255, 0, 0],
    2: [0, 0, 255],
    3: [0, 255, 0]
}

NUM_CLASSES = 4

# =========================
# METRICS (for model loading)
# =========================
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    return tf.reduce_mean((2. * intersection) /
                          (tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) + 1e-6))

def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true + y_pred, axis=0) - intersection

    return tf.reduce_mean(intersection / (union + 1e-6))

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "dice_coef": dice_coef,
                "iou_metric": iou_metric
            },
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# =========================
# UI
# =========================
st.title("🛣️ Road Defect Detection using U-Net")

if model is None:
    st.error("❌ Model not found or failed to load. Check repo file path.")
    st.stop()

uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])

# =========================
# FUNCTIONS
# =========================
def preprocess(img):
    img = np.array(img.convert("RGB"))
    h, w = img.shape[:2]

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    norm = resized / 255.0
    return img, np.expand_dims(norm.astype(np.float32), axis=0), (h, w)

def predict(img):
    original, input_img, (h, w) = preprocess(img)

    pred = model.predict(input_img, verbose=0)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return original, mask

def colorize(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, v in CLASS_COLORS.items():
        color[mask == k] = v
    return color

# =========================
# MAIN
# =========================
if uploaded_file:
    image = Image.open(uploaded_file)

    with st.spinner("Processing..."):
        original, mask = predict(image)
        color_mask = colorize(mask)

        overlay = cv2.addWeighted(original, 0.6, color_mask, 0.4, 0)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original")

    with col2:
        st.image(mask * 80, caption="Mask")

    with col3:
        st.image(overlay, caption="Overlay")

else:
    st.info("Upload an image to start detection")

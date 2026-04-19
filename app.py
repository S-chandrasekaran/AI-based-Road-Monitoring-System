import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Road Defect Detection",
    page_icon="🛣️",
    layout="wide"
)

IMG_SIZE = 128

CLASS_NAMES = {
    1: "Pothole",
    2: "Crack",
    3: "Manhole"
}

CLASS_COLORS = {
    1: [255, 0, 0],   # Red
    2: [0, 0, 255],   # Blue
    3: [0, 255, 0]    # Green
}

NUM_CLASSES = 4

# =========================================================
# MODEL PATH (CLOUD SAFE)
# =========================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "road_defect_unet_multiclass.keras")


# =========================================================
# CUSTOM LOSS / METRICS (needed for loading model)
# =========================================================
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)

    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return tf.reduce_mean(dice)


def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true + y_pred, axis=0) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return tf.reduce_mean(iou)


# =========================================================
# MODEL LOADING (SAFE + CACHED)
# =========================================================
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
            }
        )
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


model = load_model()


# =========================================================
# PREDICTION
# =========================================================
def predict_mask(image):
    original = np.array(image.convert("RGB"))
    h, w = original.shape[:2]

    resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    input_img = resized.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    pred = model.predict(input_img, verbose=0)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return original, mask


# =========================================================
# COLOR MASK
# =========================================================
def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color

    return color_mask


# =========================================================
# DETECTION SUMMARY
# =========================================================
def get_summary(mask, threshold=100):
    result = {}
    for cls, name in CLASS_NAMES.items():
        count = int(np.sum(mask == cls))
        if count >= threshold:
            result[name] = count
    return result


# =========================================================
# UI
# =========================================================
st.title("🛣️ Road Defect Detection (U-Net)")

if model is None:
    st.error("❌ Model not found or failed to load. Check repo file path.")
    st.stop()

uploaded_file = st.file_uploader("Upload road image", type=["jpg", "png", "jpeg"])

overlay_alpha = st.sidebar.slider("Overlay opacity", 0.1, 1.0, 0.4)

# =========================================================
# MAIN LOGIC
# =========================================================
if uploaded_file:

    image = Image.open(uploaded_file)

    with st.spinner("Running inference..."):
        original, mask = predict_mask(image)
        color_mask = colorize_mask(mask)

        overlay = cv2.addWeighted(original, 1 - overlay_alpha, color_mask, overlay_alpha, 0)
        detections = get_summary(mask)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original", use_container_width=True)

    with col2:
        st.image(mask * 60, caption="Mask", use_container_width=True)

    with col3:
        st.image(overlay, caption="Overlay", use_container_width=True)

    st.subheader("Detection Summary")

    if detections:
        for k, v in detections.items():
            st.warning(f"{k}: {v} pixels")
    else:
        st.success("No major defects detected")

else:
    st.info("Upload an image to start detection")

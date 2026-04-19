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

IMG_SIZE = 128

MODEL_PATH = os.path.join(os.path.dirname(__file__), "road_defect_unet_multiclass.keras")

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

# =========================
# SAFE MODEL LOADING
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found in repo.")
        return None

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,   # IMPORTANT FIX
        )
        return model

    except Exception as e:
        st.error("❌ Model loading failed (Keras version mismatch).")
        st.exception(e)
        return None


model = load_model()

# =========================
# UI
# =========================
st.title("🛣️ Road Defect Detection using U-Net")

st.markdown("""
Upload a road image to detect:
- 🔴 Pothole  
- 🔵 Crack  
- 🟢 Manhole  
""")

st.sidebar.header("⚙️ Settings")
overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.1, 1.0, 0.4)

# =========================
# IMAGE PROCESSING
# =========================
def predict(image):
    original = np.array(image.convert("RGB"))
    h, w = original.shape[:2]

    resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    input_img = resized / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    pred = model.predict(input_img, verbose=0)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return original, mask


def colorize(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, v in CLASS_COLORS.items():
        color[mask == k] = v
    return color


def summary(mask):
    out = {}
    for k, name in CLASS_NAMES.items():
        count = np.sum(mask == k)
        if count > 100:
            out[name] = int(count)
    return out

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("📤 Upload Road Image", type=["jpg", "png", "jpeg"])

if uploaded:

    if model is None:
        st.stop()

    image = Image.open(uploaded)

    with st.spinner("Detecting defects..."):
        original, mask = predict(image)
        color_mask = colorize(mask)

        overlay = cv2.addWeighted(original, 1 - overlay_alpha, color_mask, overlay_alpha, 0)
        detections = summary(mask)

    # =========================
    # DISPLAY
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original", use_container_width=True)

    with col2:
        st.image(mask * 80, caption="Mask", use_container_width=True)

    with col3:
        st.image(overlay, caption="Overlay", use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Detection Summary")

    if not detections:
        st.success("No major defects detected.")
    else:
        for k, v in detections.items():
            st.warning(f"{k}: {v} pixels")

else:
    st.info("Upload an image to start detection.")

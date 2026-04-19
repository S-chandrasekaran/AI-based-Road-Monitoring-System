import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="AI Road Monitoring", layout="centered")

st.title("🚧 AI-Based Road Monitoring System")

# -----------------------------
# SAFE VERSION INFO (IMPORTANT FIX)
# -----------------------------
st.subheader("System Info")

st.write("TensorFlow:", tf.__version__)
st.write("Keras (tf.keras):", tf.keras.__version__)


# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    st.write("Loading model... please wait ⏳")

    start = time.time()

    model = tf.keras.models.load_model(
        "model.keras",
        compile=False
    )

    st.write("Model loaded in:", round(time.time() - start, 2), "seconds")
    return model


model = load_model()


# -----------------------------
# IMAGE UPLOAD
# -----------------------------
st.subheader("Upload Road Image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])


# -----------------------------
# OVERLAY TRANSPARENCY SLIDER
# -----------------------------
st.subheader("Overlay Transparency")
alpha = st.slider("Adjust Transparency", 0.1, 1.0, 0.5)


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    st.write("Processing...")

    input_img = preprocess_image(image)

    pred = model.predict(input_img, verbose=0)

    # fake segmentation visualization (for demo-safe UI)
    mask = np.argmax(pred[0], axis=-1)

    mask_img = Image.fromarray((mask * 60).astype(np.uint8))

    st.subheader("Prediction Output")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_column_width=True)

    with col2:
        st.image(mask_img, caption="Predicted Mask", use_column_width=True)


    # -----------------------------
    # OVERLAY VISUALIZATION
    # -----------------------------
    st.subheader("Overlay Result")

    image_resized = image.resize((128, 128))
    image_np = np.array(image_resized)

    mask_rgb = np.stack([mask * 80]*3, axis=-1)

    overlay = (alpha * image_np + (1 - alpha) * mask_rgb).astype(np.uint8)

    st.image(overlay, caption=f"Overlay (alpha={alpha})", use_column_width=True)


else:
    st.info("Upload an image to start prediction.")

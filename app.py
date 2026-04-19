import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Road Defect Detection (U-Net)", layout="centered")

st.title("🚧 Road Defect Detection System (U-Net Multiclass)")

# =========================
# SYSTEM INFO
# =========================
st.write("TensorFlow Version:", tf.__version__)

# =========================
# LOAD MODEL (FIXED FOR .keras)
# =========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "road_defect_unet_multiclass.keras",
            compile=False
        )
        return model
    except Exception as e:
        st.error("❌ Model loading failed")
        st.exception(e)
        return None

st.write("Loading model... ⏳")
model = load_model()

# =========================
# CLASS LABELS (EDIT IF NEEDED)
# =========================
CLASS_NAMES = {
    0: "Background",
    1: "Crack",
    2: "Pothole",
    3: "Road Damage"
}

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # =========================
    # PREPROCESS
    # =========================
    img = image.resize((128, 128))
    img_array = np.array(img).astype(np.float32)

    # fix grayscale or RGBA
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDICTION
    # =========================
    if model:
        st.write("Running prediction... 🔍")

        pred = model.predict(img_array)

        # multiclass segmentation output
        mask = np.argmax(pred[0], axis=-1)

        st.success("Prediction completed!")

        st.image(mask * 60, caption="Predicted Defect Mask (Raw)")

        # =========================
        # CLASS STATS
        # =========================
        st.subheader("Detected Defects")

        unique, counts = np.unique(mask, return_counts=True)

        for u, c in zip(unique, counts):
            st.write(f"**{CLASS_NAMES.get(u, 'Unknown')}** : {c} pixels")

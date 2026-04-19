import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Road Defect Detection", layout="centered")

# -----------------------------
# SYSTEM INFO
# -----------------------------
st.title("🚧 AI Road Defect Detection System")

st.write("TensorFlow Version:", tf.__version__)
st.write("Keras (tf.keras is used inside TensorFlow)")

MODEL_PATH = "road_defect_unet_multiclass.keras"


# -----------------------------
# LOAD MODEL (SAFE + FIXED)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            return None

        # IMPORTANT FIX:
        # compile=False avoids many Keras deserialization issues
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )

        st.success("✅ Model loaded successfully!")
        return model

    except Exception as e:
        st.error("❌ Model loading failed")
        st.exception(e)
        return None


model = load_model()


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------
# PREDICTION
# -----------------------------
def predict(image):
    if model is None:
        return None

    img = preprocess_image(image)
    prediction = model.predict(img)
    mask = np.argmax(prediction[0], axis=-1)
    return mask


# -----------------------------
# UI
# -----------------------------
st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("🔍 Detect Defects"):

        if model is None:
            st.error("Model not loaded. Fix environment first.")
        else:
            with st.spinner("Processing..."):
                result = predict(image)

            st.success("Prediction completed!")

            st.image(
                result,
                caption="Detected Road Defects (Mask)",
                use_container_width=True
            )

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="AI Road Monitoring System", layout="centered")

# =========================
# SYSTEM INFO (FIXED)
# =========================
st.title("🚧 AI-Based Road Monitoring System")

st.write("TensorFlow Version:", tf.__version__)
st.write("Keras is included inside TensorFlow → use tf.keras only")

# =========================
# LOAD MODEL (SAFE + FIXED)
# =========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "model.h5",        # change if your file name is different
            compile=False      # IMPORTANT FIX for most errors
        )
        return model
    except Exception as e:
        st.error("❌ Model loading failed")
        st.exception(e)
        return None

st.write("Loading model... ⏳ Please wait")
model = load_model()

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # =========================
    # PREPROCESS
    # =========================
    img = image.resize((128, 128))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDICTION
    # =========================
    if model is not None:
        st.write("Running prediction... 🔍")

        prediction = model.predict(img_array)
        mask = np.argmax(prediction, axis=-1)[0]

        st.success("Prediction completed!")

        st.write("Output Mask Shape:", mask.shape)
        st.image(mask * 80, caption="Predicted Road Mask")

    else:
        st.warning("Model not loaded. Check model file path.")

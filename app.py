import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import cv2

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
    1: [255, 0, 0],
    2: [0, 0, 255],
    3: [0, 255, 0]
}

# =========================================================
# MODEL PATH (supports both formats)
# =========================================================
MODEL_PATH_KERAS = "road_defect_unet_multiclass.keras"
MODEL_PATH_H5 = "road_defect_unet.h5"


# =========================================================
# METRICS (only needed if model uses them)
# =========================================================
NUM_CLASSES = 4

def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true + y_pred, axis=0)

    return tf.reduce_mean((2. * intersection) / (union + 1e-6))


def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true + y_pred, axis=0) - intersection

    return tf.reduce_mean((intersection + 1e-6) / (union + 1e-6))


# =========================================================
# MODEL LOADING (SAFE)
# =========================================================
@st.cache_resource
def load_model():
    try:
        path = None

        if os.path.exists(MODEL_PATH_H5):
            path = MODEL_PATH_H5
        elif os.path.exists(MODEL_PATH_KERAS):
            path = MODEL_PATH_KERAS
        else:
            return None, "Model file not found"

        model = tf.keras.models.load_model(
            path,
            custom_objects={
                "dice_coef": dice_coef,
                "iou_metric": iou_metric
            },
            compile=False
        )

        return model, "loaded"

    except Exception as e:
        return None, str(e)


model, model_status = load_model()

# =========================================================
# UI
# =========================================================
st.title("🛣️ Road Defect Detection (U-Net)")

if model is None:
    st.error("❌ Model failed to load")
    st.code(model_status)
    st.stop()

st.success("✅ Model loaded successfully")

uploaded_file = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])


# =========================================================
# FUNCTIONS
# =========================================================
def preprocess(img):
    img = np.array(img.convert("RGB"))
    h, w = img.shape[:2]

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    resized = resized / 255.0
    resized = np.expand_dims(resized, axis=0)

    return img, resized, h, w


def predict(img):
    original, inp, h, w = preprocess(img)

    pred = model.predict(inp, verbose=0)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return original, mask


def color_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for k, v in CLASS_COLORS.items():
        color[mask == k] = v

    return color


# =========================================================
# MAIN
# =========================================================
if uploaded_file:

    image = Image.open(uploaded_file)

    with st.spinner("Processing..."):
        original, mask = predict(image)
        overlay = color_mask(mask)

        blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original", use_container_width=True)

    with col2:
        st.image(mask * 60, caption="Mask", use_container_width=True)

    with col3:
        st.image(blended, caption="Overlay", use_container_width=True)

    st.subheader("Detected Classes")

    unique = np.unique(mask)
    detected = [CLASS_NAMES[i] for i in unique if i in CLASS_NAMES]

    if detected:
        st.write(", ".join(detected))
    else:
        st.write("No defects detected")

else:
    st.info("Upload an image to start detection")

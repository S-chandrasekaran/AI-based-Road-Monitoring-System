import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# =========================================================
# STREAMLIT APP - ROAD DEFECT DETECTION USING U-NET
# Classes:
#   0 = Background
#   1 = Pothole
#   2 = Crack
#   3 = Manhole
# =========================================================

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Road Defect Detection",
    page_icon="🛣️",
    layout="wide"
)
import os

model_path = os.path.join(os.path.dirname(__file__), "road_defect_unet_multiclass.keras")
IMG_SIZE = 128   # MUST match training size

CLASS_NAMES = {
    1: "Pothole",
    2: "Crack",
    3: "Manhole"
}

# Colors in RGB
CLASS_COLORS = {
    1: [255, 0, 0],    # Red = Pothole
    2: [0, 0, 255],    # Blue = Crack
    3: [0, 255, 0]     # Green = Manhole
}

# -------------------------------
# CUSTOM METRICS (needed to load model)
# -------------------------------
NUM_CLASSES = 4

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)

    dice = (2. * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

CLASS_WEIGHTS = tf.constant([0.2, 1.3, 1.0, 1.2], dtype=tf.float32)

def weighted_categorical_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred) * CLASS_WEIGHTS, axis=-1)
    return tf.reduce_mean(loss)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at:\n{MODEL_PATH}")
        return None

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "dice_coef": dice_coef,
            "iou_metric": iou_metric,
            "weighted_categorical_crossentropy": weighted_categorical_crossentropy
        }
    )
    return model

model = load_model()

# -------------------------------
# UI HEADER
# -------------------------------
st.title("🛣️ Road Defect Detection using U-Net")
st.markdown("""
Upload a road image to detect:

- 🔴 **Pothole**
- 🔵 **Crack**
- 🟢 **Manhole**
""")

st.info("This app uses a trained multi-class U-Net segmentation model.")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Settings")
overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.1, 1.0, 0.4, 0.1)
min_pixels = st.sidebar.slider("Minimum pixels to count as detection", 10, 5000, 100, 10)

st.sidebar.markdown("### 🎨 Class Colors")
st.sidebar.markdown("- 🔴 **Pothole**")
st.sidebar.markdown("- 🔵 **Crack**")
st.sidebar.markdown("- 🟢 **Manhole**")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a road image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_mask(input_image_pil):
    original = np.array(input_image_pil.convert("RGB"))
    original_h, original_w = original.shape[:2]

    resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    input_img = resized.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    pred = model.predict(input_img, verbose=0)[0]
    pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)

    # Resize mask back to original image size
    pred_mask_resized = cv2.resize(
        pred_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    return original, pred_mask_resized

# -------------------------------
# COLOR MASK FUNCTION
# -------------------------------
def create_color_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask

# -------------------------------
# DETECTION SUMMARY
# -------------------------------
def get_detection_summary(mask, min_pixels=100):
    detected = {}
    for class_id, class_name in CLASS_NAMES.items():
        pixel_count = int(np.sum(mask == class_id))
        if pixel_count >= min_pixels:
            detected[class_name] = pixel_count
    return detected

# -------------------------------
# MAIN APP
# -------------------------------
if uploaded_file is not None:
    if model is None:
        st.stop()

    image = Image.open(uploaded_file)

    with st.spinner("🔍 Detecting road defects..."):
        original, pred_mask = predict_mask(image)
        color_mask = create_color_mask(pred_mask)
        overlay = cv2.addWeighted(original, 1 - overlay_alpha, color_mask, overlay_alpha, 0)
        detections = get_detection_summary(pred_mask, min_pixels=min_pixels)

    # Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original Image")
        st.image(original, use_container_width=True)

    with col2:
        st.subheader("🎭 Predicted Mask")
        st.image(pred_mask * 60, clamp=True, use_container_width=True)

    with col3:
        st.subheader("🧠 Overlay Detection")
        st.image(overlay, use_container_width=True)

    st.markdown("---")

    # Detection results
    st.subheader("📋 Detection Summary")

    if len(detections) == 0:
        st.success("✅ No significant road defects detected based on threshold.")
    else:
        for defect, pixels in detections.items():
            st.warning(f"**{defect}** detected — Pixel Area: `{pixels}`")

    # Unique classes
    unique_classes = np.unique(pred_mask)
    detected_class_names = [CLASS_NAMES[c] for c in unique_classes if c in CLASS_NAMES]

    st.markdown("### 🧾 Classes Present in Prediction")
    if detected_class_names:
        st.write(", ".join(detected_class_names))
    else:
        st.write("Only background detected.")

    # Optional: raw pixel counts
    with st.expander("📊 Show detailed class pixel counts"):
        for class_id, class_name in CLASS_NAMES.items():
            pixel_count = int(np.sum(pred_mask == class_id))
            st.write(f"**{class_name}**: {pixel_count} pixels")

    # Download overlay
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    success, encoded_img = cv2.imencode(".png", overlay_bgr)
    if success:
        st.download_button(
            label="💾 Download Overlay Result",
            data=encoded_img.tobytes(),
            file_name="road_defect_detection_result.png",
            mime="image/png"
        )

else:
    st.markdown("## 👈 Upload a road image to begin detection")

    st.markdown("""
    ### 📌 How it works
    1. Upload a road image
    2. The U-Net model segments road defects
    3. Results are displayed as:
       - Original Image
       - Predicted Mask
       - Overlay Output
    """)

    st.markdown("""
    ### 🎯 Supported Classes
    - **Pothole**
    - **Crack**
    - **Manhole**
    """)

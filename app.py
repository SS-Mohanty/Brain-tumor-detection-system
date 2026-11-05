import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import urllib.request

# 1. Page Configuration
st.set_page_config(
    page_title="Brain Tumor Classification & XAI",
    page_icon="🧠",
    layout="centered"
)

# 2. Automatically download model from GitHub Release if not present locally
MODEL_PATH = "best_brain_tumor_model_v3.keras"
MODEL_URL = "https://github.com/SS-Mohanty/Brain-tumor-detection-system/releases/download/v1.0/best_brain_tumor_model_v3.keras"  

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model weights (first-time setup)... Please wait."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return load_model(MODEL_PATH)

with st.spinner("Loading AI Model... Please wait."):
    model = get_model()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 3. Web App UI Layout
st.title("🧠 Automated Brain Tumor Classification")
st.write("Upload an MRI scan to perform real-time classification and visualize model attention via Grad-CAM.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a brain MRI scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and process image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Preprocess image to match training dimensions (150x150, normalized to [0,1])
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Run Prediction
    with st.spinner("Analyzing scan through CNN architecture..."):
        preds = model.predict(img_batch)
        pred_idx = np.argmax(preds[0])
        pred_label = class_names[pred_idx]
        confidence = float(preds[0][pred_idx]) * 100
        
    # Display Results First
    st.markdown("---")
    st.subheader("Diagnostic Prediction Results")
    if pred_label == "notumor":
        st.success(f"**Prediction:** {pred_label.upper()} (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"**Prediction:** {pred_label.upper()} Tumor Detected (Confidence: {confidence:.2f}%)")
        
    # Show probability breakdown for all classes
    with st.expander("📊 View Detailed Class Probability Distribution"):
        for idx, name in enumerate(class_names):
            st.write(f"**{name.capitalize()}**: {preds[0][idx]*100:.2f}%")
            
    # Grad-CAM Heatmap Generation with Radial Center Mask
    def generate_gradcam_overlay(img_tensor, img_pil, model, pred_idx, last_conv_layer_name="conv2d_5"):
        import scipy.ndimage as ndimage
        
        img_input = tf.keras.Input(shape=(150, 150, 3))
        x = img_input
        conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
        grad_model = tf.keras.models.Model(inputs=img_input, outputs=[conv_output, x])

        with tf.GradientTape() as tape:
            conv_out, p = grad_model(img_tensor)
            class_chan = p[:, pred_idx]

        grads = tape.gradient(class_chan, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = tf.maximum(heatmap, 0) / max_val
        else:
            heatmap = tf.zeros_like(heatmap)
            
        heatmap = heatmap.numpy()
        heatmap = ndimage.gaussian_filter(heatmap, sigma=1.2)

        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_img = heatmap_img.resize(img_pil.size, resample=Image.Resampling.BICUBIC)
        heatmap_resized = np.array(heatmap_img) / 255.0

        h, w = heatmap_resized.shape
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        radius = min(center_x, center_y) * 0.80
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        spatial_mask = np.clip(1.0 - (dist_from_center / radius), 0, 1)
        spatial_mask[dist_from_center > radius] = 0

        orig_arr = np.array(img_pil) / 255.0
        gray_orig = np.mean(orig_arr, axis=2)
        tissue_mask = gray_orig > 0.08
        
        heatmap_resized = heatmap_resized * tissue_mask * spatial_mask

        if np.max(heatmap_resized) > 0:
            heatmap_resized = heatmap_resized / np.max(heatmap_resized)

        jet = plt.colormaps.get_cmap("jet")
        jet_colors = jet(heatmap_resized)[:, :, :3]
        
        alpha = np.expand_dims(heatmap_resized * 0.65, axis=-1)
        superimposed = (jet_colors * alpha) + (orig_arr * (1 - alpha))
        return np.clip(superimposed, 0, 1)

    # Side-by-Side Comparison Layout
    st.markdown("---")
    st.subheader("🔍 Explainable AI (Side-by-Side Comparison)")
    st.write("Comparing the original input scan against the Grad-CAM attention heatmap overlay.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original MRI Scan**")
        st.image(image, width="stretch")

    with col2:
        st.markdown("**Grad-CAM Attention Map**")
        try:
            overlay_img = generate_gradcam_overlay(img_batch, image, model, pred_idx, "conv2d_5")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(overlay_img)
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error rendering Grad-CAM: {e}")

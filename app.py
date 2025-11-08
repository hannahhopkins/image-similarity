import os
import tempfile
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
import cv2
from colorthief import ColorThief
import matplotlib.pyplot as plt
import io

# -----------------------------
# Helper: Extract ZIP safely
# -----------------------------
def extract_zip_to_temp(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    image_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    image_files = [f for f in image_files if not os.path.basename(f).startswith('.')]
    if not image_files:
        st.warning("⚠️ No valid JPG or PNG images found in the ZIP.")
    else:
        st.success(f"✅ Found {len(image_files)} valid images in extracted ZIP.")
    return image_files

# -----------------------------
# Helper: Extract dominant colors
# -----------------------------
def extract_palette(image, n_colors=10):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        ct = ColorThief(tmp.name)
        palette = ct.get_palette(color_count=n_colors)
    os.remove(tmp.name)
    return palette

# -----------------------------
# Helper: Display color palette
# -----------------------------
def plot_palette(palette, labels=True):
    fig, ax = plt.subplots(figsize=(3, 0.3))
    for i, color in enumerate(palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255))
        if labels:
            hex_val = '#%02x%02x%02x' % color
            ax.text(i + 0.5, 0.5, hex_val, ha='center', va='center', fontsize=5, color='white')
    ax.set_xlim(0, len(palette))
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

# -----------------------------
# Helper: Calculate similarity metrics
# -----------------------------
def calculate_metrics(query_img, ref_img):
    # Resize to match query
    ref_img = ref_img.resize(query_img.size)
    query_gray = np.array(query_img.convert("L"))
    ref_gray = np.array(ref_img.convert("L"))

    # Structural Similarity (SSIM)
    ssim_score = ssim(
        query_gray,
        ref_gray,
        data_range=ref_gray.max() - ref_gray.min()
    )

    # Texture (GLCM contrast)
    glcm1 = graycomatrix((query_gray / 16).astype('uint8'), [1], [0], 256, symmetric=True, normed=True)
    glcm2 = graycomatrix((ref_gray / 16).astype('uint8'), [1], [0], 256, symmetric=True, normed=True)
    tex_sim = 1 - abs(graycoprops(glcm1, 'contrast')[0, 0] - graycoprops(glcm2, 'contrast')[0, 0])

    # Edge similarity
    edges1 = cv2.Canny(np.array(query_gray), 100, 200)
    edges2 = cv2.Canny(np.array(ref_gray), 100, 200)
    edge_overlap = np.sum(edges1 & edges2) / (np.sum(edges1 | edges2) + 1e-6)

    # Brightness difference
    bright_sim = 1 - abs(np.mean(query_gray) - np.mean(ref_gray)) / 255

    return {
        "SSIM": float(ssim_score),
        "Texture": float(tex_sim),
        "Edges": float(edge_overlap),
        "Brightness": float(bright_sim)
    }

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Image Similarity Analyzer")
st.write("Upload a ZIP of reference images and a single query image to compare visual similarity.")

uploaded_zip = st.file_uploader("Upload ZIP of reference images", type=["zip"])
uploaded_query = st.file_uploader("Upload a single query image", type=["jpg", "jpeg", "png"])

if uploaded_zip and uploaded_query:
    with st.spinner("Processing images..."):
        st.write("✅ ZIP uploaded:", uploaded_zip.name)
        st.write("✅ Query image uploaded:", uploaded_query.name)

        ref_images = extract_zip_to_temp(uploaded_zip)
        st.write(f"📸 Found {len(ref_images)} reference images.")

        query_img = Image.open(uploaded_query).convert("RGB")
        results = []

        for ref_path in ref_images:
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                metrics = calculate_metrics(query_img, ref_img)
                avg_score = np.mean(list(metrics.values()))
                results.append((ref_path, avg_score, metrics))
            except Exception as e:
                st.write(f"⚠️ Skipping {ref_path}: {e}")

        st.write(f"✅ Processed {len(results)} comparisons.")

    if not results:
        st.error("No valid image comparisons found. Make sure your ZIP contains only JPG or PNG images.")
    else:
        st.subheader("Top 5 Similar Images")
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:5]

        for ref_path, avg_score, metrics in top_results:
            ref_img = Image.open(ref_path).convert("RGB").resize(query_img.size)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(query_img, caption="Query Image", use_container_width=True)
            with col2:
                st.image(ref_img, caption=f"Match: {os.path.basename(ref_path)}", use_container_width=True)

            # Palette comparison
            q_palette = extract_palette(query_img)
            r_palette = extract_palette(ref_img)
            q_buf = plot_palette(q_palette)
            r_buf = plot_palette(r_palette)
            st.image([q_buf, r_buf], caption=["Query Palette", "Reference Palette"], width=250)

            # Metrics chart (streamlined)
            fig, ax = plt.subplots(figsize=(3, 1.2))
            ax.barh(list(metrics.keys()), list(metrics.values()))
            ax.set_xlim(0, 1)
            ax.set_xlabel("Similarity Score", fontsize=8)
            ax.set_title("Metric Breakdown", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig, use_container_width=False)

            # Text interpretation
            st.markdown(
                f"**Technical Summary:** This match shows {metrics['SSIM']:.2f} structural alignment, "
                f"{metrics['Texture']:.2f} textural coherence, {metrics['Edges']:.2f} edge overlap, "
                f"and {metrics['Brightness']:.2f} brightness balance. Overall score: {avg_score:.2f}."
            )
            st.markdown("---")



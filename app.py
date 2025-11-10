import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import zipfile
import os
import tempfile
from io import BytesIO

# -------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Image Similarity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Image Similarity Analyzer")
st.write("""
Upload a ZIP folder of reference images and a query image.
This app analyzes structural, color, and entropy-based similarity and visualizes color intersections.
""")

# -------------------------------------------------
# Color Palette Extraction
# -------------------------------------------------
def extract_palette(img, n_colors=5):
    img_np = np.array(img)
    img_np = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(img_np)
    colors = np.clip(kmeans.cluster_centers_.astype(int), 0, 255)
    return colors

def create_palette_image(colors, square_size=40):
    cols = len(colors)
    palette = np.zeros((square_size, square_size * cols, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        palette[:, i * square_size:(i + 1) * square_size] = color
    return Image.fromarray(palette)

# -------------------------------------------------
# Image Similarity Metrics
# -------------------------------------------------
def compute_metrics(img1, img2):
    img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    # Structural Similarity
    ssim_score = ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())

    # Color Histogram Similarity (normalized)
    hist1 = cv2.calcHist([np.array(img1)], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([np.array(img2)], [0], None, [256], [0, 256])
    hist_score_raw = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    hist_score = (hist_score_raw + 1.0) / 2.0  # remap from [-1, 1] → [0, 1]

    # Entropy Similarity
    hist1_prob = hist1 / np.sum(hist1)
    hist2_prob = hist2 / np.sum(hist2)
    entropy1 = -np.sum(hist1_prob * np.log2(hist1_prob + 1e-10))
    entropy2 = -np.sum(hist2_prob * np.log2(hist2_prob + 1e-10))
    entropy_sim = 1 - abs(entropy1 - entropy2) / max(entropy1, entropy2)

    return {
        "Structural Alignment": float(ssim_score),
        "Color Histogram": float(hist_score),
        "Entropy Similarity": float(entropy_sim)
    }

# -------------------------------------------------
# Plotly Chart for Metrics
# -------------------------------------------------
def make_metric_chart(metrics):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(metrics.values()),
        y=list(metrics.keys()),
        orientation='h',
        marker_color=['#2ca02c', '#1f77b4', '#ff7f0e']
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        xaxis=dict(range=[0, 1], title="Similarity Score"),
        yaxis=dict(title=""),
        template="simple_white",
        showlegend=False
    )
    return fig

# -------------------------------------------------
# File Upload Interface
# -------------------------------------------------
uploaded_zip = st.file_uploader("Upload a ZIP of Reference Images", type=["zip"])
query_image = st.file_uploader("Upload a Query Image", type=["jpg", "jpeg", "png"])

if uploaded_zip and query_image:
    # Create a temporary directory for the uploaded ZIP
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract ZIP contents
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Recursively search for all valid image files
        ref_paths = []
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    ref_paths.append(os.path.join(root, f))

        if len(ref_paths) == 0:
            st.error("No valid reference images found. Please ensure your ZIP contains JPG or PNG files.")
            st.stop()

        query_img = Image.open(query_image).convert("RGB")
        results = []

        for ref_path in ref_paths:
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                metrics = compute_metrics(query_img, ref_img)
                results.append((ref_path, metrics))
            except Exception as e:
                st.warning(f"Skipped {ref_path}: {e}")

        # Sort results by average similarity
        results.sort(key=lambda x: np.mean(list(x[1].values())), reverse=True)
        top_results = results[:5]

        st.subheader("Top 5 Most Similar Images")

        for i, (ref_path, metrics) in enumerate(top_results):
            ref_img = Image.open(ref_path).convert("RGB")
            col1, col2 = st.columns([2.5, 1], gap="large")

            with col1:
                st.markdown(f"### Match {i + 1}")
                st.image(
                    [query_img, ref_img],
                    caption=["Query Image", f"Reference {i + 1}"],
                    use_container_width=True
                )
                st.plotly_chart(make_metric_chart(metrics), use_container_width=True)

                for m, score in metrics.items():
                    if m == "Structural Alignment":
                        desc = (
                            f"**{m}** — Measures geometric and spatial consistency between the two images. "
                            f"Result: {score:.2f}, indicating "
                            f"{'strong' if score>0.75 else 'moderate' if score>0.5 else 'weak'} structural correspondence."
                        )
                    elif m == "Color Histogram":
                        desc = (
                            f"**{m}** — Compares distribution of colors across the two images. "
                            f"Result: {score:.2f}, showing "
                            f"{'high' if score>0.75 else 'partial' if score>0.5 else 'limited'} chromatic overlap."
                        )
                    elif m == "Entropy Similarity":
                        desc = (
                            f"**{m}** — Evaluates tonal variation and textural complexity. "
                            f"Result: {score:.2f}, suggesting "
                            f"{'similar' if score>0.75 else 'slightly varied' if score>0.5 else 'contrasting'} texture patterns."
                        )
                    st.markdown(desc)

            with col2:
                st.markdown("#### Intersection Palettes")

                q_colors = extract_palette(query_img)
                r_colors = extract_palette(ref_img)

                # Generate intersection palettes
                blended = np.mean([q_colors, r_colors], axis=0)
                shared = np.array([(q_colors[i] + r_colors[i]) / 2 for i in range(min(len(q_colors), len(r_colors)))])
                weighted = (0.6 * q_colors + 0.4 * r_colors)

                # Display compact square palettes
                st.image(create_palette_image(blended), caption="Blended Midpoint")
                st.image(create_palette_image(shared), caption="Shared Hue Range")
                st.image(create_palette_image(weighted), caption="Weighted Hybrid")

                st.caption("These palettes represent intersections between the dominant color clusters of the two images.")

else:
    st.info("Upload your ZIP folder of reference images and a query image to begin analysis.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown("Built with Streamlit, OpenCV, scikit-image, and Plotly.")

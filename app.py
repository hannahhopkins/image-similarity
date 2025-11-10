import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
import plotly.graph_objects as go
import zipfile
import os
import tempfile

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
This tool compares images across multiple visual metrics — structure, color, texture, edge density, hue, and entropy — 
and visualizes the intersections between their color palettes.
""")

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("User Controls")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 10, 5)
resize_refs = st.sidebar.checkbox("Resize reference images to match query", value=True)

# -------------------------------------------------
# Helper Functions
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

def normalize_metric(value):
    return float(np.clip(value, 0, 1))

# -------------------------------------------------
# Image Similarity Metrics
# -------------------------------------------------
def compute_metrics(img1, img2, resize=True):
    if resize:
        img2 = img2.resize(img1.size)

    img1_np = np.array(img1)
    img2_np = np.array(img2)
    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

    # 1. Structural Similarity
    ssim_score = ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())

    # 2. Color Histogram Similarity
    hist1 = cv2.calcHist([img1_np], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    hist2 = cv2.calcHist([img2_np], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    raw_hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    hist_score = (raw_hist_score + 1) / 2  # Normalize from [-1,1] → [0,1]


    # 3. Entropy Similarity
    h1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
    p1 = h1 / np.sum(h1)
    p2 = h2 / np.sum(h2)
    e1 = -np.sum(p1 * np.log2(p1 + 1e-10))
    e2 = -np.sum(p2 * np.log2(p2 + 1e-10))
    entropy_sim = 1 - abs(e1 - e2) / max(e1, e2)

    # 4. Edge/Shape Density Similarity
    edges1 = cv2.Canny(img1_gray, 100, 200)
    edges2 = cv2.Canny(img2_gray, 100, 200)
    edge_score = 1 - np.abs(np.mean(edges1) - np.mean(edges2)) / 255

    # 5. Texture Correlation (GLCM)
    glcm1 = graycomatrix(img1_gray, distances=[5], angles=[0], symmetric=True, normed=True)
    glcm2 = graycomatrix(img2_gray, distances=[5], angles=[0], symmetric=True, normed=True)
    tex1 = graycoprops(glcm1, 'contrast')[0, 0]
    tex2 = graycoprops(glcm2, 'contrast')[0, 0]
    texture_sim = 1 - abs(tex1 - tex2) / max(tex1, tex2)

    # 6. Hue Distribution Similarity
    hsv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2HSV)
    hue_hist1 = cv2.calcHist([hsv1], [0], None, [180], [0, 180])
    hue_hist2 = cv2.calcHist([hsv2], [0], None, [180], [0, 180])
    try:
        raw_hue_score = cv2.EMD(hist_h1.astype(np.float32), hist_h2.astype(np.float32), cv2.DIST_L2)[0]
        hue_score = max(0, min(1, 1 - raw_hue_score))
    except:
        raw_hue_score = cv2.compareHist(hist_h1, hist_h2, cv2.HISTCMP_CORREL)
        hue_score = (raw_hue_score + 1) / 2


    return {
        "Structural Alignment": ssim_score,
        "Color Histogram": hist_score,
        "Entropy Similarity": entropy_sim,
        "Edge Complexity": edge_score,
        "Texture Correlation": texture_sim,
        "Hue Distribution": hue_score
    }

# -------------------------------------------------
# Plotly Chart
# -------------------------------------------------
def make_metric_chart(metrics):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(metrics.values()),
        y=list(metrics.keys()),
        orientation='h',
        marker_color=['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#17becf']
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=40, r=20, t=30, b=10),
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        ref_paths = []
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.startswith("._") or "__MACOSX" in root:
                    continue
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
                metrics = compute_metrics(query_img, ref_img, resize_refs)
                results.append((ref_path, metrics))
            except Exception as e:
                st.warning(f"Skipped {ref_path}: {e}")

        if len(results) == 0:
            st.error("No valid comparisons could be made.")
            st.stop()

        # Sort results by average similarity
        results.sort(key=lambda x: np.mean(list(x[1].values())), reverse=True)
        top_results = results[:top_k]

        st.subheader(f"Top {top_k} Most Similar Images")

        for i, (ref_path, metrics) in enumerate(top_results):
            ref_img = Image.open(ref_path).convert("RGB")
            if resize_refs:
                ref_img = ref_img.resize(query_img.size)

            col1, col2 = st.columns([2.5, 1.2], gap="large")

            with col1:
                st.markdown(f"### Match {i + 1}")
                st.image(
                    [query_img, ref_img],
                    caption=["Query Image", f"Reference {i + 1}"],
                    use_container_width=True
                )
                st.plotly_chart(make_metric_chart(metrics), use_container_width=True)

                for m, score in metrics.items():
                    desc = {
                        "Structural Alignment": "Measures geometric correspondence between images — higher values mean stronger spatial consistency.",
                        "Color Histogram": "Compares color distribution patterns — higher scores reflect closer overall color balance.",
                        "Entropy Similarity": "Evaluates tonal variation and complexity — similar entropy means similar texture richness.",
                        "Edge Complexity": "Assesses visual structure density — how comparable the amount and distribution of edges are.",
                        "Texture Correlation": "Examines micro-patterns and surface qualities — higher means closer textural identity.",
                        "Hue Distribution": "Analyzes dominant hue proportions — closer hue alignment indicates shared color harmonics."
                    }[m]
                    st.markdown(f"**{m} ({score:.2f})** — {desc}")

            with col2:
                st.markdown("#### Intersection Palettes")

                q_colors = extract_palette(query_img, num_colors)
                r_colors = extract_palette(ref_img, num_colors)

                blended = np.mean([q_colors, r_colors], axis=0)
                shared = np.array([(q_colors[i] + r_colors[i]) / 2 for i in range(min(len(q_colors), len(r_colors)))])
                weighted = (0.6 * q_colors + 0.4 * r_colors)

                st.image(create_palette_image(blended), caption="Blended Midpoint")
                st.caption("Represents the average chromatic midpoint between the dominant colors of both images, highlighting shared perceptual tones.")

                st.image(create_palette_image(shared), caption="Shared Hue Range")
                st.caption("Shows hues most common to both images, emphasizing overlapping color families and harmony regions.")

                st.image(create_palette_image(weighted), caption="Weighted Hybrid")
                st.caption("Blends color dominance by weighting the query image slightly more, illustrating how the reference adapts within its chromatic context.")

else:
    st.info("Upload your ZIP folder of reference images and a query image to begin analysis.")

st.markdown("---")
st.markdown("Built with Streamlit, OpenCV, scikit-image, and Plotly.")

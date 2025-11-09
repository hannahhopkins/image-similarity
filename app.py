import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import zipfile, os, tempfile
from io import BytesIO

st.set_page_config(page_title="Image Similarity Analyzer", layout="wide")

# --- Utility: Extract colors using KMeans ---
def extract_palette(img, n_colors=5):
    img_np = np.array(img)
    img_np = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(img_np)
    colors = np.clip(kmeans.cluster_centers_.astype(int), 0, 255)
    return colors

# --- Utility: Compute similarity metrics ---
def compute_metrics(img1, img2):
    img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    
    ssim_score = ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())
    hist_score = cv2.compareHist(
        cv2.calcHist([np.array(img1)], [0], None, [256], [0,256]),
        cv2.calcHist([np.array(img2)], [0], None, [256], [0,256]),
        cv2.HISTCMP_CORREL
    )
    entropy1 = -np.sum(cv2.calcHist([img1_gray], [0], None, [256], [0,256]) / img1_gray.size * 
                       np.log2(cv2.calcHist([img1_gray], [0], None, [256], [0,256]) / img1_gray.size + 1e-10))
    entropy2 = -np.sum(cv2.calcHist([img2_gray], [0], None, [256], [0,256]) / img2_gray.size * 
                       np.log2(cv2.calcHist([img2_gray], [0], None, [256], [0,256]) / img2_gray.size + 1e-10))
    entropy_sim = 1 - abs(entropy1 - entropy2) / max(entropy1, entropy2)

    return {
        "Structural Alignment": float(ssim_score),
        "Color Histogram": float(hist_score),
        "Entropy Similarity": float(entropy_sim)
    }

# --- Utility: Create square palettes ---
def create_palette_image(colors, square_size=50):
    cols = len(colors)
    palette = np.zeros((square_size, square_size * cols, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        palette[:, i * square_size:(i + 1) * square_size] = color
    return Image.fromarray(palette)

# --- Plotly similarity bar chart ---
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
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(range=[0, 1], title="Similarity Score"),
        yaxis=dict(title=""),
        showlegend=False,
        template="simple_white"
    )
    return fig

# --- Streamlit App ---
st.title("🧠 Image Similarity Analyzer with Color Palettes")

uploaded_zip = st.file_uploader("Upload a ZIP folder of reference images (JPG/PNG)", type=["zip"])
query_image = st.file_uploader("Upload your query image", type=["jpg", "jpeg", "png"])

if uploaded_zip and query_image:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        ref_paths = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.lower().endswith((".jpg", ".png"))]
        query_img = Image.open(query_image).convert("RGB")

        results = []
        for ref_path in ref_paths:
            ref_img = Image.open(ref_path).convert("RGB")
            metrics = compute_metrics(query_img, ref_img)
            results.append((ref_path, metrics))

        # Sort by average metric score
        results.sort(key=lambda x: np.mean(list(x[1].values())), reverse=True)
        top_results = results[:5]

        st.subheader("🔝 Top 5 Most Similar Images")

        for i, (ref_path, metrics) in enumerate(top_results):
            ref_img = Image.open(ref_path).convert("RGB")
            col1, col2 = st.columns([2, 1], gap="large")

            with col1:
                st.markdown(f"### Match {i + 1}")
                st.image([query_img, ref_img], caption=["Query Image", f"Reference {i+1}"], use_container_width=True)
                st.plotly_chart(make_metric_chart(metrics), use_container_width=True)

                for m, score in metrics.items():
                    if m == "Structural Alignment":
                        st.markdown(f"**{m}** — Measures shape and spatial consistency between images. "
                                    f"Result: {score:.2f} → indicates {('strong' if score>0.75 else 'moderate' if score>0.5 else 'weak')} correspondence.")
                    elif m == "Color Histogram":
                        st.markdown(f"**{m}** — Evaluates overlap in color distribution patterns. "
                                    f"Result: {score:.2f} → suggests {('high' if score>0.75 else 'partial' if score>0.5 else 'limited')} color similarity.")
                    elif m == "Entropy Similarity":
                        st.markdown(f"**{m}** — Compares image complexity and tonal variation. "
                                    f"Result: {score:.2f} → implies {('comparable' if score>0.75 else 'slightly varied' if score>0.5 else 'contrasting')} visual texture.")

            with col2:
                st.markdown("#### 🎨 Intersection Palettes")
                q_colors = extract_palette(query_img)
                r_colors = extract_palette(ref_img)

                # Palettes
                blended = np.mean([q_colors, r_colors], axis=0)
                shared = np.array([(q_colors[i] + r_colors[i]) / 2 for i in range(min(len(q_colors), len(r_colors)))])
                weighted = (0.6 * q_colors + 0.4 * r_colors)

                # Display as compact square grids
                st.image(create_palette_image(blended), caption="Blended Midpoint")
                st.image(create_palette_image(shared), caption="Shared Hue Range")
                st.image(create_palette_image(weighted), caption="Weighted Hybrid")

                st.markdown("_These palettes represent the dominant color intersections between the query and reference image._")

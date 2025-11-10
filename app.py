import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
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
and visualizes intersections between their color palettes with three distinct methods.
""")

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("User Controls")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 8)
resize_refs = st.sidebar.checkbox("Resize reference images to match query", value=True)
bars_height = st.sidebar.slider("Metric bar chart height (px)", 160, 400, 240, 10)

# -------------------------------------------------
# Helpers: palettes
# -------------------------------------------------
def extract_palette(img: Image.Image, n_colors: int = 8):
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3).astype(np.float32)
    k = max(1, min(n_colors, flat.shape[0] // 50))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(flat)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    counts = counts[order]
    # pad if needed
    while centers.shape[0] < n_colors:
        centers = np.vstack([centers, np.mean(centers, axis=0, keepdims=True)])
        counts = np.append(counts, 0)
    return centers[:n_colors].astype(int), counts[:n_colors].astype(int)

def create_palette_image(colors, square_size=36):
    colors = np.asarray(colors, dtype=np.uint8)
    cols = len(colors)
    palette = np.zeros((square_size, square_size * cols, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        palette[:, i * square_size:(i + 1) * square_size] = c
    return Image.fromarray(palette)

# Lab helpers
def rgb_list_to_lab(rgb_list):
    arr = np.array(rgb_list, dtype=float).reshape(-1, 1, 3) / 255.0
    return rgb2lab(arr).reshape(-1, 3)

def lab_to_rgb_tuple(lab_vec):
    rgb = lab2rgb(lab_vec.reshape(1, 1, 3)).reshape(3,) * 255.0
    return tuple(int(np.clip(x, 0, 255)) for x in rgb)

# Distinct intersection palettes
def blended_midpoint_palette(q_colors, r_colors, n_out=5):
    # For each query color (in Lab), find nearest ref color and take Lab midpoint
    q_lab = rgb_list_to_lab(q_colors)
    r_lab = rgb_list_to_lab(r_colors)
    out = []
    for i in range(min(n_out, len(q_lab))):
        d = np.linalg.norm(r_lab - q_lab[i], axis=1)
        j = int(np.argmin(d)) if len(d) else 0
        mid = (q_lab[i] + r_lab[j]) / 2.0
        out.append(lab_to_rgb_tuple(mid))
    while len(out) < n_out:
        out.append(tuple(map(int, q_colors[0])))
    return out

def shared_hue_palette(q_colors, r_colors, n_out=5, bins=36, min_presence=0.02):
    # Build hue histograms; choose bins where both have presence; then sample colors near those hues
    def hues_of(colors):
        hsv = cv2.cvtColor(np.array(colors, dtype=np.uint8).reshape(-1,1,3), cv2.COLOR_RGB2HSV)
        return hsv.reshape(-1,3)[:,0]  # 0..179

    q_h = hues_of(q_colors)
    r_h = hues_of(r_colors)

    # histograms (0..180)
    q_hist = cv2.calcHist([q_h.astype(np.uint8)], [0], None, [bins], [0, 180]).astype(np.float32)
    r_hist = cv2.calcHist([r_h.astype(np.uint8)], [0], None, [bins], [0, 180]).astype(np.float32)
    if q_hist.sum() > 0: q_hist /= q_hist.sum()
    if r_hist.sum() > 0: r_hist /= r_hist.sum()

    # intersection mask: bins where both have enough presence
    inter = (q_hist.flatten() > min_presence) & (r_hist.flatten() > min_presence)

    # If none, fall back to strongest shared bins by product
    if not np.any(inter):
        scores = (q_hist.flatten() * r_hist.flatten())
        inter_idx = np.argsort(-scores)[:n_out]
    else:
        inter_idx = np.where(inter)[0]
        # pick top bins by min(q,r) so it emphasizes shared presence
        scores = np.minimum(q_hist.flatten()[inter_idx], r_hist.flatten()[inter_idx])
        order = np.argsort(-scores)
        inter_idx = inter_idx[order][:n_out]

    # For each selected bin, average closest colors from q and r
    out = []
    for b in inter_idx:
        target_h = (b + 0.5) * (180 / bins)
        def closest_by_h(colors):
            hsv = cv2.cvtColor(np.array(colors, dtype=np.uint8).reshape(-1,1,3), cv2.COLOR_RGB2HSV).reshape(-1,3)
            h = hsv[:,0].astype(float)
            diff = np.minimum(np.abs(h - target_h), 180 - np.abs(h - target_h))
            k = int(np.argmin(diff))
            return colors[k]
        cq = closest_by_h(q_colors)
        cr = closest_by_h(r_colors)
        avg = ((np.array(cq, dtype=float) + np.array(cr, dtype=float)) / 2.0).astype(int)
        out.append(tuple(np.clip(avg, 0, 255)))
    while len(out) < n_out:
        out.append(tuple(map(int, q_colors[0])))
    return out

def weighted_hybrid_palette(q_colors, q_counts, r_colors, r_counts, n_out=5):
    # Weight by cluster prominence and saturation. Lab blend with weights.
    q_lab = rgb_list_to_lab(q_colors)
    r_lab = rgb_list_to_lab(r_colors)

    # compute saturation weights
    def sat_weights(colors):
        hsv = cv2.cvtColor(np.array(colors, dtype=np.uint8).reshape(-1,1,3), cv2.COLOR_RGB2HSV).reshape(-1,3)
        s = hsv[:,1].astype(float) / 255.0
        return s

    q_sat = sat_weights(q_colors)
    r_sat = sat_weights(r_colors)

    out = []
    for i in range(min(n_out, len(q_lab))):
        qw = (q_counts[i] if i < len(q_counts) else 1) * (0.5 + 0.5*q_sat[i])
        # nearest in r
        d = np.linalg.norm(r_lab - q_lab[i], axis=1)
        j = int(np.argmin(d)) if len(d) else 0
        rw = (r_counts[j] if j < len(r_counts) else 1) * (0.5 + 0.5*r_sat[j])
        wsum = max(qw + rw, 1e-6)
        lab_mix = (q_lab[i]*qw + r_lab[j]*rw) / wsum
        out.append(lab_to_rgb_tuple(lab_mix))
    while len(out) < n_out:
        out.append(tuple(map(int, q_colors[0])))
    return out

# -------------------------------------------------
# Metrics
# -------------------------------------------------
def normalize_01(x):
    return float(np.clip(x, 0.0, 1.0))

def compute_metrics(img1: Image.Image, img2: Image.Image, resize=True):
    if resize:
        img2 = img2.resize(img1.size)

    a = np.array(img1.convert("RGB"))
    b = np.array(img2.convert("RGB"))
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

    # 1) Structural Similarity (needs data_range)
    dr = max(1.0, float(b_gray.max() - b_gray.min()))
    ssim_score = normalize_01(ssim(a_gray.astype(float), b_gray.astype(float), data_range=dr))

    # 2) Color Histogram similarity (3D RGB hist, correlation mapped [-1,1]→[0,1])
    h1 = cv2.calcHist([a], [0,1,2], None, [8,8,8], [0,256]*3)
    h2 = cv2.calcHist([b], [0,1,2], None, [8,8,8], [0,256]*3)
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    raw_hist_corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    color_hist_sim = normalize_01((raw_hist_corr + 1.0) / 2.0)

    # 3) Entropy similarity (Shannon)
    gh1 = cv2.calcHist([a_gray], [0], None, [256], [0,256]).astype(np.float64)
    gh2 = cv2.calcHist([b_gray], [0], None, [256], [0,256]).astype(np.float64)
    gh1 /= (gh1.sum() + 1e-12)
    gh2 /= (gh2.sum() + 1e-12)
    e1 = -(gh1 * np.log2(gh1 + 1e-12)).sum()
    e2 = -(gh2 * np.log2(gh2 + 1e-12)).sum()
    entropy_sim = normalize_01(1.0 - abs(e1 - e2) / max(e1, e2, 1e-6))

    # 4) Edge complexity similarity (Canny edge density)
    e1_map = cv2.Canny(a_gray, 100, 200)
    e2_map = cv2.Canny(b_gray, 100, 200)
    edge_sim = normalize_01(1.0 - abs(e1_map.mean() - e2_map.mean()) / 255.0)

    # 5) Texture similarity (GLCM contrast distance mapped to similarity)
    try:
        glcm1 = graycomatrix(a_gray, distances=[5], angles=[0], symmetric=True, normed=True)
        glcm2 = graycomatrix(b_gray, distances=[5], angles=[0], symmetric=True, normed=True)
        t1 = graycoprops(glcm1, 'contrast')[0,0]
        t2 = graycoprops(glcm2, 'contrast')[0,0]
        texture_sim = normalize_01(1.0 - abs(t1 - t2) / max(t1, t2, 1e-6))
    except Exception:
        texture_sim = 0.0

    # 6) Hue Distribution similarity (EMD first, correlation fallback; map to [0,1])
    hsv1 = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
    hue1 = hsv1[:,:,0].astype(np.uint8)
    hue2 = hsv2[:,:,0].astype(np.uint8)
    hue_hist1 = cv2.calcHist([hue1], [0], None, [50], [0,180]).astype(np.float32)
    hue_hist2 = cv2.calcHist([hue2], [0], None, [50], [0,180]).astype(np.float32)
    if hue_hist1.sum() > 0: hue_hist1 /= hue_hist1.sum()
    if hue_hist2.sum() > 0: hue_hist2 /= hue_hist2.sum()

    # EMD requires signature format: (weight, bin_index)
    try:
        # Build signatures: Nx2 (weight, position)
        sig1 = np.hstack([hue_hist1, np.arange(hue_hist1.shape[0], dtype=np.float32).reshape(-1,1)]).astype(np.float32)
        sig2 = np.hstack([hue_hist2, np.arange(hue_hist2.shape[0], dtype=np.float32).reshape(-1,1)]).astype(np.float32)
        emd_val = cv2.EMD(sig1, sig2, cv2.DIST_L2)[0]  # smaller is better
        # map EMD to similarity; rough scale by max expected ~1.0
        hue_sim = normalize_01(1.0 - float(emd_val))
    except Exception:
        raw_hue_corr = cv2.compareHist(hue_hist1, hue_hist2, cv2.HISTCMP_CORREL)
        hue_sim = normalize_01((raw_hue_corr + 1.0) / 2.0)

    metrics = {
        "Structural Alignment": ssim_score,
        "Color Histogram": color_hist_sim,
        "Entropy Similarity": entropy_sim,
        "Edge Complexity": edge_sim,
        "Texture Correlation": texture_sim,
        "Hue Distribution": hue_sim
    }
    return metrics

# -------------------------------------------------
# Plotly Bar
# -------------------------------------------------
def make_metric_chart(metrics: dict, height: int):
    names = list(metrics.keys())
    vals  = [float(metrics[k]) for k in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker_color=['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#17becf'],
        text=[f"{v:.2f}" for v in vals], textposition='outside'
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=20, b=10),
        xaxis=dict(range=[0,1], title="Similarity (0–1)"),
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

        # recursive gather with macOS artifact filtering
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

        if not results:
            st.error("No valid comparisons could be made.")
            st.stop()

        # Sort by average similarity
        results.sort(key=lambda x: np.mean(list(x[1].values())), reverse=True)
        top_results = results[:top_k]

        st.subheader(f"Top {top_k} Most Similar Images")

        for i, (ref_path, metrics) in enumerate(top_results):
            ref_img = Image.open(ref_path).convert("RGB")
            if resize_refs:
                ref_img = ref_img.resize(query_img.size)

            col1, col2 = st.columns([2.6, 1.4], gap="large")

            with col1:
                st.markdown(f"### Match {i+1}: {os.path.basename(ref_path)}")
                st.image([query_img, ref_img], caption=["Query Image", f"Reference {i+1}"], use_container_width=True)
                st.plotly_chart(make_metric_chart(metrics, bars_height), use_container_width=True)

                # concise technical blurbs
                blurbs = {
                    "Structural Alignment": "Compares luminance/contrast/structure (SSIM). Higher means layouts and form relationships are closely aligned.",
                    "Color Histogram": "Compares the overall RGB distribution (3D histogram). Higher means global color balance is similar.",
                    "Entropy Similarity": "Compares information density/tonal variability. Higher means comparable visual complexity.",
                    "Edge Complexity": "Compares edge density via Canny. Higher means a similar amount and spread of contours.",
                    "Texture Correlation": "Compares GLCM contrast (micro-patterns). Higher means similar surface rhythm.",
                    "Hue Distribution": "Compares hue emphasis across the spectrum (EMD fallback to correlation). Higher means dominant hue families align."
                }
                for k in metrics:
                    st.markdown(f"**{k} ({metrics[k]:.2f})** — {blurbs[k]}")

            with col2:
                st.markdown("#### Intersection Palettes")

                q_cols, q_counts = extract_palette(query_img, num_colors)
                r_cols, r_counts = extract_palette(ref_img, num_colors)

                blended = blended_midpoint_palette(q_cols, r_cols, n_out=min(6, num_colors))
                shared  = shared_hue_palette(q_cols, r_cols, n_out=min(6, num_colors))
                hybrid  = weighted_hybrid_palette(q_cols, q_counts, r_cols, r_counts, n_out=min(6, num_colors))

                st.image(create_palette_image(blended),  caption="Blended Midpoint")
                st.caption("Lab midpoint of nearest color pairs. This emphasizes a balanced chromatic bridge between both palettes.")

                st.image(create_palette_image(shared),   caption="Shared Hue")
                st.caption("Only hues both images emphasize are shown, chosen from intersection bins of the hue histograms. This reveals true overlap in hue families.")

                st.image(create_palette_image(hybrid),   caption="Weighted Hybrid")
                st.caption("Lab blend weighted by cluster prominence and saturation. This favors influential colors and preserves chroma energy differences.")

else:
    st.info("Upload your ZIP folder of reference images and a query image to begin analysis.")

st.markdown("---")
st.markdown("Built with Streamlit, OpenCV, scikit-image, and Plotly.")

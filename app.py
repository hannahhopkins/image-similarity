import os
import zipfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("Image Similarity Analyzer")
st.write(
    "Upload a ZIP of reference images and a query image. "
    "The app compares them across multiple visual metrics and shows three different intersection palettes."
)

# -----------------------------
# Sidebar controls (with per-setting descriptions)
# -----------------------------
st.sidebar.header("User Controls")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 6)
resize_refs = st.sidebar.checkbox("Resize reference images to match query", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Hue Similarity Settings")

hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, step=6)
st.sidebar.caption(
    "Hue bins split the color wheel into equal slices (e.g., 36 bins ≈ 10° per slice). "
    "More bins capture subtler hue differences; fewer bins smooth over small variations."
)

sat_thresh = st.sidebar.slider("Saturation mask threshold", 0.0, 1.0, 0.15, 0.01)
st.sidebar.caption(
    "Only pixels above this saturation are used for hue comparison. "
    "Raising this reduces the influence of gray/washed-out areas."
)

val_thresh = st.sidebar.slider("Value mask threshold", 0.0, 1.0, 0.15, 0.01)
st.sidebar.caption(
    "Only pixels above this brightness are used for hue comparison. "
    "Raising this reduces the influence of very dark pixels."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Palette")
hybrid_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.6, 0.05)
st.sidebar.caption(
    "Controls how much the query image influences the hybrid palette. "
    "A higher value keeps the hybrid closer to the query’s colors; a lower value leans toward the reference."
)

# -----------------------------
# Helpers: I/O and filtering
# -----------------------------
def load_image(fp_or_file):
    """Open image safely as RGB."""
    try:
        img = Image.open(fp_or_file).convert("RGB")
        return img
    except Exception as e:
        return None

def collect_images_from_zip(uploaded_zip) -> List[str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmp_dir)
        refs = []
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                # skip macOS resource forks and metadata folders
                if f.startswith("._") or "__MACOSX" in root:
                    continue
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    refs.append(os.path.join(root, f))
        # copy refs out of temp (Streamlit reruns would close temp otherwise)
        stable_files = []
        for p in refs:
            with open(p, "rb") as fh:
                data = fh.read()
            fd, newp = tempfile.mkstemp(suffix=Path(p).suffix)
            os.close(fd)
            with open(newp, "wb") as out:
                out.write(data)
            stable_files.append(newp)
        return stable_files

# -----------------------------
# Palettes (k-means) and drawing
# -----------------------------
def extract_palette_kmeans(img: Image.Image, n_colors: int) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.dstack([arr, arr, arr]) if arr.ndim == 2 else np.zeros((64, 64, 3), dtype=np.uint8)
    small = cv2.resize(arr, (200, 200), interpolation=cv2.INTER_AREA)
    pts = small.reshape(-1, 3).astype(np.float32)
    k = min(n_colors, max(1, pts.shape[0] // 50))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(pts)
    centers = np.clip(km.cluster_centers_, 0, 255).astype(np.uint8)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    # pad to n_colors by repeating mean if needed
    while centers.shape[0] < n_colors:
        centers = np.vstack([centers, np.mean(centers, axis=0, keepdims=True)])
    return centers[:n_colors]

def palette_strip(colors: np.ndarray, square=40) -> Image.Image:
    """Return a horizontal strip of square swatches."""
    n = colors.shape[0]
    out = np.zeros((square, square * n, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        out[:, i * square:(i + 1) * square, :] = c
    return Image.fromarray(out)

# -----------------------------
# Intersection palettes (distinct methods)
# -----------------------------
def blended_midpoint_palette(q_colors: np.ndarray, r_colors: np.ndarray) -> np.ndarray:
    """RGB midpoint per paired cluster (by nearest in Lab)."""
    if len(q_colors) == 0 or len(r_colors) == 0:
        return q_colors
    # convert to Lab
    q_lab = cv2.cvtColor(q_colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    r_lab = cv2.cvtColor(r_colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    out_lab = []
    for i in range(len(q_lab)):
        d = np.linalg.norm(r_lab - q_lab[i], axis=1)
        j = int(np.argmin(d))
        mid = (q_lab[i] + r_lab[j]) / 2.0
        out_lab.append(mid)
    out_lab = np.vstack(out_lab).reshape(1, -1, 3).astype(np.float32)
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB).reshape(-1, 3)
    return np.clip(out_rgb, 0, 255).astype(np.uint8)

def shared_hue_palette(q_colors: np.ndarray, r_colors: np.ndarray, bins: int) -> np.ndarray:
    """Pick colors nearest to the strongest overlapping hue bins."""
    if len(q_colors) == 0 or len(r_colors) == 0:
        return q_colors
    # Build hue histograms for the palettes themselves
    def hues_from_rgb(colors):
        hsv = cv2.cvtColor(colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        return hsv[:, 0]  # 0..179
    q_h = hues_from_rgb(q_colors)
    r_h = hues_from_rgb(r_colors)
    hist_q, _ = np.histogram(q_h, bins=bins, range=(0, 180))
    hist_r, _ = np.histogram(r_h, bins=bins, range=(0, 180))
    overlap = hist_q.astype(float) * hist_r.astype(float)  # emphasize bins strong in both
    if overlap.sum() == 0:
        # fallback: just average palettes
        return ((q_colors.astype(np.float32) + r_colors.astype(np.float32)) / 2).astype(np.uint8)
    # pick indices of q_colors whose hues fall into top overlap bins
    top_bins = np.argsort(-overlap)
    chosen = []
    for b in top_bins:
        # select one q color whose hue falls into this bin (closest)
        bin_start = (180 / bins) * b
        bin_end = bin_start + (180 / bins)
        # choose q color closest to bin center
        center = (bin_start + bin_end) / 2.0
        diffs = np.minimum(np.abs(q_h - center), 180 - np.abs(q_h - center))
        idx = int(np.argmin(diffs))
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= len(q_colors):
            break
    return q_colors[chosen[: len(q_colors)]]

def weighted_hybrid_palette(q_colors: np.ndarray, r_colors: np.ndarray, weight: float) -> np.ndarray:
    """Weighted blend in Lab with query weight in [0,1]."""
    w = float(np.clip(weight, 0.0, 1.0))
    if len(q_colors) == 0:
        return r_colors
    if len(r_colors) == 0:
        return q_colors
    q_lab = cv2.cvtColor(q_colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    r_lab = cv2.cvtColor(r_colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    # pair each q with nearest r in Lab, then weighted average
    out_lab = []
    for i in range(len(q_lab)):
        d = np.linalg.norm(r_lab - q_lab[i], axis=1)
        j = int(np.argmin(d))
        blend = q_lab[i] * w + r_lab[j] * (1.0 - w)
        out_lab.append(blend)
    out_lab = np.vstack(out_lab).reshape(1, -1, 3)
    out_rgb = cv2.cvtColor(out_lab.astype(np.float32), cv2.COLOR_Lab2RGB).reshape(-1, 3)
    return np.clip(out_rgb, 0, 255).astype(np.uint8)

# -----------------------------
# Metrics
# -----------------------------
def structural_alignment(img1_gray, img2_gray) -> float:
    dr = float(max(1, img2_gray.max() - img2_gray.min()))
    return float(np.clip(ssim(img1_gray, img2_gray, data_range=dr), 0.0, 1.0))

def color_histogram_similarity(img1_rgb, img2_rgb, bins=8) -> float:
    h1 = cv2.calcHist([img1_rgb], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    h2 = cv2.calcHist([img2_rgb], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    raw = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)  # [-1,1]
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

def entropy_similarity(img1_gray, img2_gray) -> float:
    h1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
    p1 = h1 / max(1, np.sum(h1))
    p2 = h2 / max(1, np.sum(h2))
    e1 = float(-np.sum(p1 * np.log2(p1 + 1e-12)))
    e2 = float(-np.sum(p2 * np.log2(p2 + 1e-12)))
    if max(e1, e2) <= 1e-6:
        return 1.0
    return float(np.clip(1.0 - abs(e1 - e2) / max(e1, e2), 0.0, 1.0))

def edge_complexity(img1_gray, img2_gray) -> float:
    e1 = cv2.Canny(img1_gray, 100, 200)
    e2 = cv2.Canny(img2_gray, 100, 200)
    return float(np.clip(1.0 - abs(float(e1.mean()) - float(e2.mean())) / 255.0, 0.0, 1.0))

def texture_correlation(img1_gray, img2_gray) -> float:
    try:
        g1 = graycomatrix(img1_gray, [5], [0], symmetric=True, normed=True)
        g2 = graycomatrix(img2_gray, [5], [0], symmetric=True, normed=True)
        c1 = float(graycoprops(g1, "contrast")[0, 0])
        c2 = float(graycoprops(g2, "contrast")[0, 0])
        if max(c1, c2) <= 1e-9:
            return 1.0
        return float(np.clip(1.0 - abs(c1 - c2) / max(c1, c2), 0.0, 1.0))
    except Exception:
        return 0.5

def hue_distribution_similarity(img1_rgb, img2_rgb, bins: int, s_thresh: float, v_thresh: float) -> float:
    """
    Compare hue histograms with S/V masking.
    Returns similarity in [0,1]. Robust normalization and fallback.
    """
    # to HSV
    hsv1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2HSV)
    S1 = hsv1[:, :, 1] / 255.0
    V1 = hsv1[:, :, 2] / 255.0
    S2 = hsv2[:, :, 1] / 255.0
    V2 = hsv2[:, :, 2] / 255.0

    mask1 = (S1 >= s_thresh) & (V1 >= v_thresh)
    mask2 = (S2 >= s_thresh) & (V2 >= v_thresh)

    h1 = hsv1[:, :, 0][mask1] if np.any(mask1) else hsv1[:, :, 0].flatten()
    h2 = hsv2[:, :, 0][mask2] if np.any(mask2) else hsv2[:, :, 0].flatten()

    # build histograms
    hist_h1, _ = np.histogram(h1, bins=bins, range=(0, 180))
    hist_h2, _ = np.histogram(h2, bins=bins, range=(0, 180))

    if hist_h1.sum() == 0 or hist_h2.sum() == 0:
        return 0.0

    # normalize
    h1n = hist_h1.astype(np.float32) / float(hist_h1.sum())
    h2n = hist_h2.astype(np.float32) / float(hist_h2.sum())

    # cosine similarity (robust, non-negative)
    denom = (np.linalg.norm(h1n) * np.linalg.norm(h2n) + 1e-12)
    sim = float(np.dot(h1n, h2n) / denom)
    # safety clamp
    return float(np.clip(sim, 0.0, 1.0))

# -----------------------------
# Plotly bar chart
# -----------------------------
def plot_metric_bars(metrics: dict):
    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        text=[f"{v:.2f}" for v in vals],
        textposition="outside",
        marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=1)
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Similarity"),
        yaxis=dict(autorange="reversed"),
        height=max(220, 28 * len(names) + 60),
        margin=dict(l=80, r=30, t=10, b=10),
        template="simple_white",
        showlegend=False
    )
    return fig

# -----------------------------
# Metric explainer text (expanded for clarity)
# -----------------------------
METRIC_EXPLAIN = {
    "Structural Alignment":
        "Compares overall layout of light and dark regions (via SSIM). High means similar composition and spatial structure.",
    "Color Histogram":
        "Compares the distribution of colors across the full image (3D RGB histogram). High means overall color balance is similar.",
    "Entropy Similarity":
        "Measures image complexity (variation in tones). High means both images have a similar level of detail and texture richness.",
    "Edge Complexity":
        "Compares how much edge structure (outlines/contours) exists. High means both images have a similar amount of sharp boundaries.",
    "Texture Correlation":
        "Examines micro-patterns using GLCM contrast. High means surface qualities (fine grain vs. smooth) are alike.",
    "Hue Distribution":
        "Looks only at the hue channel (after masking low-saturation and very dark pixels). High means dominant hue families are similar."
}

# -----------------------------
# File upload
# -----------------------------
ref_zip = st.file_uploader("Upload a ZIP of Reference Images", type=["zip"])
query_file = st.file_uploader("Upload a Query Image", type=["jpg", "jpeg", "png"])

if ref_zip and query_file:
    ref_paths = collect_images_from_zip(ref_zip)
    if len(ref_paths) == 0:
        st.error("No valid reference images found. Make sure the ZIP contains JPG or PNG files (not only folders).")
        st.stop()

    query_img = load_image(query_file)
    if query_img is None:
        st.error("Could not open the query image.")
        st.stop()

    # Precompute query arrays
    query_rgb = np.array(query_img)
    query_gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)

    # Evaluate references
    results = []
    for rp in ref_paths:
        ref_img = load_image(rp)
        if ref_img is None:
            continue
        ref_img_proc = ref_img.resize(query_img.size) if resize_refs else ref_img

        ref_rgb = np.array(ref_img_proc)
        ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)

        metrics = {
            "Structural Alignment": structural_alignment(query_gray, ref_gray),
            "Color Histogram": color_histogram_similarity(query_rgb, ref_rgb, bins=8),
            "Entropy Similarity": entropy_similarity(query_gray, ref_gray),
            "Edge Complexity": edge_complexity(query_gray, ref_gray),
            "Texture Correlation": texture_correlation(query_gray, ref_gray),
            "Hue Distribution": hue_distribution_similarity(query_rgb, ref_rgb, hue_bins, sat_thresh, val_thresh)
        }
        overall = float(np.mean(list(metrics.values())))
        results.append((rp, ref_img_proc, metrics, overall))

    if not results:
        st.error("No valid comparisons could be made.")
        st.stop()

    # Sort and display
    results.sort(key=lambda t: t[3], reverse=True)
    top = results[:top_k]

    st.subheader(f"Top {len(top)} Matches")
    for rank, (path, ref_img_proc, metrics, overall) in enumerate(top, start=1):
        col1, col2 = st.columns([2.5, 1.3], gap="large")

        with col1:
            st.markdown(f"### Match {rank}: {Path(path).name} — Overall {overall:.2f}")
            st.image([query_img, ref_img_proc], caption=["Query", "Reference"], use_container_width=True)
            st.plotly_chart(plot_metric_bars(metrics), use_container_width=True)

            with st.expander("Metric Explanations", expanded=False):
                for k in ["Structural Alignment", "Color Histogram", "Entropy Similarity",
                          "Edge Complexity", "Texture Correlation", "Hue Distribution"]:
                    st.markdown(f"**{k} ({metrics[k]:.2f})** — {METRIC_EXPLAIN[k]}")

        with col2:
            st.markdown("Intersection Palettes")

            # Extract palettes
            q_cols = extract_palette_kmeans(query_img, num_colors)
            r_cols = extract_palette_kmeans(ref_img_proc, num_colors)

            # Three distinct intersection palettes
            pal_blend = blended_midpoint_palette(q_cols, r_cols)
            pal_shared = shared_hue_palette(q_cols, r_cols, bins=hue_bins)
            pal_hybrid = weighted_hybrid_palette(q_cols, r_cols, hybrid_weight)

            st.image(palette_strip(pal_blend), caption="Blended Midpoint")
            st.caption(
                "A midpoint blend in a perceptual color space. This shows the average tone where each query color meets its closest reference color."
            )

            st.image(palette_strip(pal_shared), caption="Shared Hue")
            st.caption(
                "Colors drawn from the most overlapping regions of the hue wheel (after masking low-saturation/dark pixels). "
                "This highlights color families emphasized by both images."
            )

            st.image(palette_strip(pal_hybrid), caption="Weighted Hybrid")
            st.caption(
                "A matched, weighted merge in a perceptual space. The query’s influence is set by the sidebar weight; "
                "higher values keep the hybrid close to the query while still reflecting the nearest colors in the reference."
            )

else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

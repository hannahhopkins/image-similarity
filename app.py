import os
import zipfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import cv2
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, lab2rgb
import plotly.graph_objects as go

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Image Similarity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Image Similarity Analyzer")
st.write(
    "Upload a ZIP of reference images and one query image. "
    "This app computes multiple similarity metrics (structure, color, texture, edges, entropy, hue), "
    "and builds three differently-computed intersection palettes."
)

# ---------------------------
# Sidebar Controls + Descriptions
# ---------------------------
st.sidebar.header("Settings")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
st.sidebar.caption("How many of the closest matches to show from your reference set.")

num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 7)
st.sidebar.caption("How many dominant colors to extract from each image using k-means clustering.")

resize_refs = st.sidebar.checkbox("Resize reference images to match query", value=True)
st.sidebar.caption("If enabled, reference images are resized to the query image size for reliable metric comparisons (e.g., SSIM).")

st.sidebar.markdown("---")
st.sidebar.subheader("Hue Similarity Settings")

hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, step=6)
st.sidebar.caption(
    "Divides the hue circle into this many segments when comparing hue distributions. "
    "More bins capture finer hue differences; fewer bins are smoother."
)

sat_thresh = st.sidebar.slider("Saturation mask threshold", 0, 255, 30, step=5)
st.sidebar.caption(
    "Pixels with saturation below this value are ignored for hue-based metrics to reduce grayscale/low-color noise."
)

val_thresh = st.sidebar.slider("Value (brightness) mask threshold", 0, 255, 30, step=5)
st.sidebar.caption(
    "Pixels with value (brightness) below this are ignored in hue metrics to avoid near-black regions dominating."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Palette Setting")

q_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.6, 0.05)
st.sidebar.caption(
    "Controls how strongly the query palette influences the Weighted Hybrid palette. "
    "1.0 uses only the query colors; 0.0 uses only the reference colors; values in between blend them."
)

# ---------------------------
# Utility: Safe image open
# ---------------------------
def safe_open_image(p) -> Image.Image | None:
    try:
        img = Image.open(p).convert("RGB")
        return img
    except UnidentifiedImageError:
        return None
    except Exception:
        return None

# ---------------------------
# Palette helpers
# ---------------------------
def kmeans_palette(pil_img: Image.Image, k: int) -> np.ndarray:
    """Return (k,3) uint8 palette centers."""
    arr = np.array(pil_img.convert("RGB"))
    if arr.size == 0:
        # fallback neutral
        return np.tile(np.array([[127, 127, 127]], dtype=np.uint8), (k, 1))
    flat = arr.reshape(-1, 3).astype(np.float32)
    # guard tiny images
    k_eff = min(k, max(1, flat.shape[0] // 50))
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=0)
    labels = km.fit_predict(flat)
    centers = km.cluster_centers_.astype(np.float32)
    # sort by cluster size desc
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    # pad if needed
    if centers.shape[0] < k:
        pad = np.tile(np.mean(flat, axis=0, keepdims=True), (k - centers.shape[0], 1))
        centers = np.vstack([centers, pad])
    centers = np.clip(centers[:k], 0, 255).astype(np.uint8)
    return centers

def hex_from_rgb(rgb: np.ndarray) -> str:
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def palette_image_hex(colors: np.ndarray, square_size: int = 40, show_hex: bool = True) -> Image.Image:
    """Render a horizontal strip of color squares with optional hex labels underneath."""
    k = colors.shape[0]
    # render only color blocks; hex labels will be shown as caption text list
    strip = np.zeros((square_size, square_size * k, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        strip[:, i * square_size:(i + 1) * square_size, :] = c
    return Image.fromarray(strip)

# ---------------------------
# Metrics
# ---------------------------
def structural_similarity(pil_a: Image.Image, pil_b: Image.Image) -> float:
    a = np.array(pil_a.convert("L")).astype(np.float32)
    b = np.array(pil_b.convert("L")).astype(np.float32)
    dr = b.max() - b.min()
    if dr <= 0:
        dr = 1.0
    return float(np.clip(ssim(a, b, data_range=dr), 0, 1))

def histogram_rgb_correlation(pil_a: Image.Image, pil_b: Image.Image, bins: int = 8) -> float:
    a = np.array(pil_a)
    b = np.array(pil_b)
    h1 = cv2.calcHist([a], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    h2 = cv2.calcHist([b], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    raw = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)  # [-1,1]
    return float(np.clip((raw + 1.0) / 2.0, 0, 1))

def entropy_similarity(pil_a: Image.Image, pil_b: Image.Image) -> float:
    ga = np.array(pil_a.convert("L"))
    gb = np.array(pil_b.convert("L"))
    ha = cv2.calcHist([ga], [0], None, [256], [0, 256])
    hb = cv2.calcHist([gb], [0], None, [256], [0, 256])
    pa = ha / (np.sum(ha) + 1e-8)
    pb = hb / (np.sum(hb) + 1e-8)
    ea = -np.sum(pa * np.log2(pa + 1e-12))
    eb = -np.sum(pb * np.log2(pb + 1e-12))
    if max(ea, eb) <= 0:
        return 0.5
    return float(np.clip(1.0 - abs(ea - eb) / max(ea, eb), 0, 1))

def edge_density_similarity(pil_a: Image.Image, pil_b: Image.Image) -> float:
    ga = np.array(pil_a.convert("L"))
    gb = np.array(pil_b.convert("L"))
    ea = cv2.Canny(ga, 100, 200)
    eb = cv2.Canny(gb, 100, 200)
    da = float(np.mean(ea))
    db = float(np.mean(eb))
    return float(np.clip(1.0 - abs(da - db) / 255.0, 0, 1))

def texture_glcm_similarity(pil_a: Image.Image, pil_b: Image.Image) -> float:
    ga = np.array(pil_a.convert("L"))
    gb = np.array(pil_b.convert("L"))
    try:
        g1 = graycomatrix(ga, [5], [0], symmetric=True, normed=True)
        g2 = graycomatrix(gb, [5], [0], symmetric=True, normed=True)
        c1 = graycoprops(g1, 'contrast')[0, 0]
        c2 = graycoprops(g2, 'contrast')[0, 0]
        denom = max(c1, c2, 1e-6)
        return float(np.clip(1.0 - abs(c1 - c2) / denom, 0, 1))
    except Exception:
        return 0.5

def hue_distribution_similarity(pil_a: Image.Image, pil_b: Image.Image,
                                bins: int, s_thr: int, v_thr: int) -> float:
    """
    Compare hue distributions with masking for low-sat/low-val pixels.
    Returns similarity in [0,1]. Handles empty masks gracefully.
    """
    a = np.array(pil_a)
    b = np.array(pil_b)
    hsv_a = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)

    mask_a = (hsv_a[..., 1] >= s_thr) & (hsv_a[..., 2] >= v_thr)
    mask_b = (hsv_b[..., 1] >= s_thr) & (hsv_b[..., 2] >= v_thr)

    ha = hsv_a[..., 0][mask_a]
    hb = hsv_b[..., 0][mask_b]

    # if either is empty, return neutral 0.5 (unknown)
    if ha.size == 0 or hb.size == 0:
        return 0.5

    # Histogram on 0..180 (OpenCV hue range)
    hist_a, _ = np.histogram(ha, bins=bins, range=(0, 180), density=False)
    hist_b, _ = np.histogram(hb, bins=bins, range=(0, 180), density=False)

    # normalize to probabilities
    ha_sum = hist_a.sum()
    hb_sum = hist_b.sum()
    if ha_sum == 0 or hb_sum == 0:
        return 0.5

    hist_a = hist_a.astype(np.float32) / ha_sum
    hist_b = hist_b.astype(np.float32) / hb_sum

    # correlation in [-1,1] → map to [0,1]
    # use numpy dot with L2 normalization (cosine of distributions) as robust similarity
    denom = (np.linalg.norm(hist_a) * np.linalg.norm(hist_b)) + 1e-12
    sim = float(np.dot(hist_a, hist_b) / denom)  # [0,1]
    return float(np.clip(sim, 0, 1))

# ---------------------------
# Intersection Palettes
# ---------------------------
def nearest_lab_index(lab_array: np.ndarray, lab_color: np.ndarray) -> int:
    d = np.linalg.norm(lab_array - lab_color, axis=1)
    return int(np.argmin(d))

def blended_midpoint_palette(q_cols: np.ndarray, r_cols: np.ndarray, out_k: int) -> np.ndarray:
    """Lab-midpoint of nearest colors (query to reference)."""
    if q_cols.size == 0 or r_cols.size == 0:
        return np.tile(np.array([[127, 127, 127]], dtype=np.uint8), (out_k, 1))
    q_lab = rgb2lab(q_cols.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    r_lab = rgb2lab(r_cols.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    out = []
    for i in range(min(out_k, q_lab.shape[0])):
        j = nearest_lab_index(r_lab, q_lab[i])
        mid = (q_lab[i] + r_lab[j]) / 2.0
        rgb = (lab2rgb(mid.reshape(1, 1, 3)).reshape(3,) * 255.0)
        out.append(np.clip(rgb, 0, 255))
    while len(out) < out_k:
        out.append(q_cols[0].astype(np.float32))
    return np.array(out, dtype=np.uint8)

def shared_hue_intersection_palette(q_cols: np.ndarray, r_cols: np.ndarray,
                                    out_k: int, bins: int, s_thr: int, v_thr: int) -> np.ndarray:
    """
    Build an intersection palette by finding overlapping hue bins between the two palettes.
    We take each palette color, convert to HSV, keep if V>=v_thr and S>=s_thr, bin by hue, find shared bins,
    and for each shared bin, average the closest query/ref color in that bin (in RGB).
    """
    def hue_bin(rgb):
        hsv = cv2.cvtColor(rgb.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if S < s_thr or V < v_thr:
            return None
        bin_idx = min(bins - 1, int(H / (180 / bins)))
        return bin_idx

    q_bins = {}
    for c in q_cols:
        bidx = hue_bin(c)
        if bidx is not None:
            q_bins.setdefault(bidx, []).append(c.astype(np.float32))

    r_bins = {}
    for c in r_cols:
        bidx = hue_bin(c)
        if bidx is not None:
            r_bins.setdefault(bidx, []).append(c.astype(np.float32))

    shared_bins = sorted(set(q_bins.keys()).intersection(set(r_bins.keys())))
    out = []
    for bidx in shared_bins:
        q_list = q_bins[bidx]
        r_list = r_bins[bidx]
        q_mean = np.mean(q_list, axis=0)
        r_mean = np.mean(r_list, axis=0)
        avg = (q_mean + r_mean) / 2.0
        out.append(np.clip(avg, 0, 255))

    # if not enough shared bins, fill with nearest overall averages to ensure visual difference
    if len(out) < out_k:
        # fall back to cross-pairs of closest hues between all colors
        q_lab = rgb2lab(q_cols.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
        r_lab = rgb2lab(r_cols.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
        for i in range(min(out_k - len(out), q_lab.shape[0])):
            j = nearest_lab_index(r_lab, q_lab[i])
            rgb = (lab2rgb(((q_lab[i] + r_lab[j]) / 2.0).reshape(1, 1, 3)).reshape(3,) * 255.0)
            out.append(np.clip(rgb, 0, 255))

    out = np.array(out, dtype=np.float32)
    # take only out_k
    if out.shape[0] > out_k:
        out = out[:out_k]
    # pad
    while out.shape[0] < out_k:
        out = np.vstack([out, q_cols[0].astype(np.float32)])
    return out.astype(np.uint8)

def weighted_hybrid_palette(q_cols: np.ndarray, r_cols: np.ndarray,
                            out_k: int, q_w: float) -> np.ndarray:
    """
    For each query color, find nearest reference color in Lab and blend by q_w (query weight).
    q_w=1 keeps query; q_w=0 keeps reference.
    """
    if q_cols.size == 0 and r_cols.size == 0:
        return np.tile(np.array([[127, 127, 127]], dtype=np.uint8), (out_k, 1))
    if q_cols.size == 0:
        return np.tile(r_cols[0:1], (out_k, 1))
    if r_cols.size == 0:
        return np.tile(q_cols[0:1], (out_k, 1))

    q_lab = rgb2lab(q_cols.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    r_lab = rgb2lab(r_cols.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

    out = []
    for i in range(min(out_k, q_lab.shape[0])):
        j = nearest_lab_index(r_lab, q_lab[i])
        blend = q_w * q_lab[i] + (1.0 - q_w) * r_lab[j]
        rgb = (lab2rgb(blend.reshape(1, 1, 3)).reshape(3,) * 255.0)
        out.append(np.clip(rgb, 0, 255))

    while len(out) < out_k:
        out.append(q_cols[0].astype(np.float32))
    return np.array(out, dtype=np.uint8)

# ---------------------------
# Plotly bar (compact)
# ---------------------------
def plot_metrics_bar(metrics: dict) -> go.Figure:
    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]
    hover = []
    explain = {
        "Structural Alignment": "Structural Similarity (SSIM) across luminance/contrast/structure; measures compositional alignment.",
        "Color Histogram": "RGB distribution correlation; measures overall palette balance.",
        "Entropy Similarity": "Shannon entropy of luminance histograms; measures information density/complexity.",
        "Edge Complexity": "Canny edge density difference; measures outline/contour activity similarity.",
        "Texture Correlation": "GLCM contrast similarity; measures micro-pattern and surface rhythm alignment.",
        "Hue Distribution": "Masked hue histogram similarity with circular handling; measures dominant hue balance."
    }
    for n, v in zip(names, vals):
        hover.append(f"<b>{n}</b><br>{explain.get(n,'')}<br>Score: {v:.2f}")

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=1),
        hovertemplate=hover
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Similarity (0–1)"),
        yaxis=dict(title=""),
        margin=dict(l=80, r=20, t=10, b=10),
        height= 40 * len(names) + 60,
        template="simple_white",
        showlegend=False
    )
    fig.update_yaxes(autorange="reversed")
    return fig

# ---------------------------
# File Upload UI
# ---------------------------
uploaded_zip = st.file_uploader("Upload a ZIP of Reference Images", type=["zip"])
query_file = st.file_uploader("Upload a Query Image", type=["jpg", "jpeg", "png"])

if uploaded_zip and query_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmp_dir)

        ref_paths = []
        for root, _, files in os.walk(tmp_dir):
            if "__MACOSX" in root:
                continue
            for f in files:
                if f.startswith("._"):
                    continue
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    ref_paths.append(os.path.join(root, f))

        if not ref_paths:
            st.error("No valid reference images found. Make sure your ZIP contains JPG or PNG images.")
            st.stop()

        query_img = safe_open_image(query_file)
        if query_img is None:
            st.error("Could not open the query image.")
            st.stop()

        # Compute metrics for all refs
        results = []
        for p in ref_paths:
            ref = safe_open_image(p)
            if ref is None:
                continue
            if resize_refs:
                ref = ref.resize(query_img.size)
            try:
                metrics = {
                    "Structural Alignment": structural_similarity(query_img, ref),
                    "Color Histogram": histogram_rgb_correlation(query_img, ref, bins=8),
                    "Entropy Similarity": entropy_similarity(query_img, ref),
                    "Edge Complexity": edge_density_similarity(query_img, ref),
                    "Texture Correlation": texture_glcm_similarity(query_img, ref),
                    "Hue Distribution": hue_distribution_similarity(query_img, ref, hue_bins, sat_thresh, val_thresh),
                }
                score = float(np.mean(list(metrics.values())))
                results.append((p, ref, metrics, score))
            except Exception as e:
                st.warning(f"Skipped {os.path.basename(p)}: {e}")

        if not results:
            st.error("No valid comparisons could be made.")
            st.stop()

        # Sort by average similarity
        results.sort(key=lambda t: t[3], reverse=True)
        top = results[:top_k]

        st.subheader(f"Top {len(top)} Matches")
        for rank, (path, ref_img, metrics, score) in enumerate(top, start=1):
            st.markdown(f"### Match {rank}: {os.path.basename(path)} — Overall {score:.2f}")
            col1, col2 = st.columns([2.5, 1.2], gap="large")

            with col1:
                st.image([query_img, ref_img], caption=["Query", f"Reference {rank}"], use_container_width=True)
                st.plotly_chart(plot_metrics_bar(metrics), use_container_width=True)
                with st.expander("Metric Explanations."):
                    st.markdown(
                        "**Structural Alignment** — Measures overlap in luminance, contrast, and local structure (SSIM). "
                        "High scores indicate similar composition/arrangement of forms."
                    )
                    st.markdown(
                        "**Color Histogram** — Compares RGB distributions across coarse bins. "
                        "High scores indicate similar overall color balance and channel proportions."
                    )
                    st.markdown(
                        "**Entropy Similarity** — Compares information density (tonal variation). "
                        "High scores indicate comparable texture richness/complexity."
                    )
                    st.markdown(
                        "**Edge Complexity** — Compares the density of detected edges (contours/outlines). "
                        "High scores indicate similar contour activity and structural detail."
                    )
                    st.markdown(
                        "**Texture Correlation** — Uses GLCM contrast to compare micro-pattern rhythm and surface texture. "
                        "High scores indicate similar fine-grained texture behavior."
                    )
                    st.markdown(
                        "**Hue Distribution** — Compares masked hue histograms (low-saturation and low-brightness pixels are ignored). "
                        "High scores indicate similar dominant hue families."
                    )

            with col2:
                st.markdown("Intersection Palettes")

                # Extract per-image palettes (k-means)
                q_cols = kmeans_palette(query_img, num_colors)
                r_cols = kmeans_palette(ref_img, num_colors)

                # Compute three distinct palettes
                blended = blended_midpoint_palette(q_cols, r_cols, out_k=min(5, num_colors))
                shared = shared_hue_intersection_palette(q_cols, r_cols, out_k=min(5, num_colors),
                                                         bins=hue_bins, s_thr=sat_thresh, v_thr=val_thresh)
                hybrid = weighted_hybrid_palette(q_cols, r_cols, out_k=min(5, num_colors), q_w=float(q_weight))

                # Render + captions (hex only)
                st.image(palette_image_hex(blended), caption="Blended Midpoint")
                st.caption(
                    "Lab midpoint between nearest query and reference colors. "
                    "Represents a balanced chromatic blend centered where both images agree."
                )
                st.write(" • ".join([hex_from_rgb(c) for c in blended]))

                st.image(palette_image_hex(shared), caption="Shared Hue Intersection")
                st.caption(
                    "Colors drawn from hue bins present in both palettes (after saturation/value masking). "
                    "Highlights overlapping color families that both images emphasize."
                )
                st.write(" • ".join([hex_from_rgb(c) for c in shared]))

                st.image(palette_image_hex(hybrid), caption="Weighted Hybrid")
                st.caption(
                    f"Nearest-color Lab blend using query weight = {q_weight:.2f}. "
                    "Shows how the reference palette shifts when influenced by the query’s chromatic emphasis."
                )
                st.write(" • ".join([hex_from_rgb(c) for c in hybrid]))

else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

st.markdown("---")
st.markdown("Built with Streamlit, OpenCV, scikit-image, scikit-learn, and Plotly.")

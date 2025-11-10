import os
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide")
st.title("Image Similarity Analyzer")
st.write(
    "Upload a ZIP of reference images and a query image. "
    "The app compares them across structure, color, texture, edges, entropy, and hue, "
    "and shows three different intersection palettes."
)

# -----------------------------------------------------------------------------
# Sidebar controls with short explanations under each setting
# -----------------------------------------------------------------------------
st.sidebar.header("Settings")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
st.sidebar.caption("How many of the most similar reference images to show.")

num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 6)
st.sidebar.caption("How many dominant colors to extract for each image via k-means clustering.")

resize_refs = st.sidebar.checkbox("Resize reference images to match query", value=True)
st.sidebar.caption("When enabled, reference images are resized to the query size for fair, pixel-wise metrics (e.g., SSIM).")

st.sidebar.markdown("---")
st.sidebar.subheader("Hue Similarity Settings")

hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, step=6)
st.sidebar.caption(
    "How finely the hue circle is divided. More bins distinguish smaller hue differences; fewer bins group hues broadly."
)

sat_thresh = st.sidebar.slider("Saturation mask threshold", 0.0, 1.0, 0.15, step=0.01)
st.sidebar.caption(
    "Pixels with saturation below this threshold are treated as gray/neutral and ignored in hue calculations."
)

val_thresh = st.sidebar.slider("Value mask threshold", 0.0, 1.0, 0.15, step=0.01)
st.sidebar.caption(
    "Pixels with value (brightness) below this threshold are ignored in hue calculations to reduce noise from very dark areas."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Palette")
hybrid_query_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.6, step=0.05)
st.sidebar.caption(
    "Controls how much the query palette dominates the Hybrid palette. "
    "At 1.0 the Hybrid follows the query entirely; at 0.0 it follows the reference."
)

# -----------------------------------------------------------------------------
# Helpers: safe image opening and ZIP scanning
# -----------------------------------------------------------------------------
VALID_EXT = {".jpg", ".jpeg", ".png"}

def safe_open_image(path: str):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except UnidentifiedImageError:
        return None

def iter_zip_images(extract_dir: str):
    """Yield valid image paths from a recursively extracted directory, skipping macOS junk."""
    for root, _, files in os.walk(extract_dir):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            if Path(f).suffix.lower() in VALID_EXT:
                yield os.path.join(root, f)

# -----------------------------------------------------------------------------
# Color utilities
# -----------------------------------------------------------------------------
def kmeans_palette(pil_img: Image.Image, k: int):
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    if h * w == 0:
        return []
    X = arr.reshape(-1, 3).astype(np.float32)
    k = min(k, max(1, X.shape[0] // 200))  # guard for very small images
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    centers = np.clip(km.cluster_centers_, 0, 255).astype(np.uint8)
    # order by cluster size
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    colors = centers[order]
    return [tuple(map(int, c)) for c in colors]

def rgb_to_hsv01(rgb_arr_uint8):
    """Convert uint8 RGB array to HSV in [0,1]."""
    hsv = cv2.cvtColor(rgb_arr_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] /= 179.0  # OpenCV hue is [0,179]
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    return hsv

def hue_histogram(pil_img: Image.Image, bins: int, s_thr: float, v_thr: float):
    """Masked hue histogram in [0,1] with circular bins; returns normalized histogram."""
    arr = np.array(pil_img).astype(np.uint8)
    hsv = rgb_to_hsv01(arr)
    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]
    mask = (S >= s_thr) & (V >= v_thr)
    if not np.any(mask):
        # If no valid hue pixels, return uniform mid histogram to avoid constant 0.0
        return np.ones(bins, dtype=np.float32) / bins
    h = H[mask].ravel()
    hist, _ = np.histogram(h, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    else:
        hist[:] = 1.0 / bins
    return hist

def circ_hue_mean(h1, h2):
    """Circular mean of two hue values in [0,1]."""
    a = np.exp(1j * 2 * np.pi * h1)
    b = np.exp(1j * 2 * np.pi * h2)
    m = np.angle((a + b) / 2.0) / (2 * np.pi)
    if m < 0:
        m += 1.0
    return m

# -----------------------------------------------------------------------------
# Metrics (all normalized to 0..1)
# -----------------------------------------------------------------------------
def structural_similarity(pil_a, pil_b):
    a = cv2.cvtColor(np.array(pil_a), cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(np.array(pil_b), cv2.COLOR_RGB2GRAY).astype(np.float32)
    dr = max(1e-6, b.max() - b.min())
    return float(np.clip(ssim(a, b, data_range=dr), 0, 1))

def color_hist_similarity(pil_a, pil_b):
    a = np.array(pil_a)
    b = np.array(pil_b)
    h1 = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    h2 = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    raw = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
    return float(np.clip((raw + 1.0) / 2.0, 0, 1))  # map [-1,1] -> [0,1]

def entropy_similarity(pil_a, pil_b):
    def entropy_gray(img):
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([g], [0], None, [256], [0, 256]).astype(np.float32)
        p = hist / (hist.sum() + 1e-8)
        return float(-np.sum(p * np.log2(p + 1e-12)))
    e1 = entropy_gray(pil_a)
    e2 = entropy_gray(pil_b)
    return float(np.clip(1.0 - abs(e1 - e2) / max(1e-6, max(e1, e2)), 0, 1))

def edge_complexity_similarity(pil_a, pil_b):
    def edge_density(img):
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g, 100, 200)
        return float(e.mean() / 255.0)
    d1 = edge_density(pil_a)
    d2 = edge_density(pil_b)
    return float(np.clip(1.0 - abs(d1 - d2), 0, 1))

def texture_correlation_similarity(pil_a, pil_b):
    def glcm_energy(gray):
        g = gray.astype(np.uint8)
        gl = graycomatrix(g, [2], [0], symmetric=True, normed=True)
        return float(graycoprops(gl, "energy")[0, 0])
    a = cv2.cvtColor(np.array(pil_a), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.array(pil_b), cv2.COLOR_RGB2GRAY)
    e1 = glcm_energy(a)
    e2 = glcm_energy(b)
    return float(np.clip(1.0 - abs(e1 - e2) / max(1e-6, max(e1, e2)), 0, 1))

def hue_distribution_similarity(pil_a, pil_b, bins, s_thr, v_thr):
    h1 = hue_histogram(pil_a, bins, s_thr, v_thr)
    h2 = hue_histogram(pil_b, bins, s_thr, v_thr)
    # cosine similarity (stable, [0..1])
    denom = (np.linalg.norm(h1) * np.linalg.norm(h2)) + 1e-10
    cs = float(np.dot(h1, h2) / denom)
    return float(np.clip(cs, 0, 1))

# -----------------------------------------------------------------------------
# Intersection palettes (distinct methods)
# -----------------------------------------------------------------------------
def plotly_palette(colors, key, title=None):
    """Display a palette as adjacent rectangles with hex on hover (no labels under)."""
    n = len(colors)
    # Build shapes (rectangles) and invisible scatter for hover
    fig = go.Figure()
    shapes = []
    xs = []
    ys = []
    hover = []
    for i, c in enumerate(colors):
        hexv = "#{:02x}{:02x}{:02x}".format(*c)
        x0, x1 = i, i + 1
        shapes.append(dict(type="rect", x0=x0, y0=0, x1=x1, y1=1,
                           line=dict(width=0),
                           fillcolor=hexv))
        xs.append(i + 0.5)
        ys.append(0.5)
        hover.append(hexv)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", marker=dict(opacity=0),
        hovertext=hover, hoverinfo="text", showlegend=False
    ))
    fig.update_layout(
        shapes=shapes, xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=20 if title else 4, b=4), height=60, title=title
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def lab_midpoint_palette(query_colors, ref_colors, out_n):
    """Nearest-hue pairing then midpoint in Lab to emphasize perceptual mid-tones."""
    if not query_colors or not ref_colors:
        return []
    # Convert palettes to HSV for nearest-hue pairing
    q = np.array(query_colors, dtype=np.uint8).reshape(-1, 1, 3)
    r = np.array(ref_colors, dtype=np.uint8).reshape(-1, 1, 3)
    q_hsv = cv2.cvtColor(q, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    r_hsv = cv2.cvtColor(r, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # Convert to Lab for blending
    q_lab = cv2.cvtColor(q, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    r_lab = cv2.cvtColor(r, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    out = []
    for i in range(min(out_n, len(q_lab))):
        # nearest in circular hue
        qh = q_hsv[i, 0] / 179.0
        diffs = []
        for j in range(len(r_hsv)):
            rh = r_hsv[j, 0] / 179.0
            d = min(abs(qh - rh), 1 - abs(qh - rh))
            diffs.append((d, j))
        j = min(diffs, key=lambda t: t[0])[1]
        mid = (q_lab[i] + r_lab[j]) / 2.0
        rgb = cv2.cvtColor(mid.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(3,)
        out.append(tuple(int(x) for x in rgb))
    # pad if needed
    while len(out) < out_n:
        out.append(tuple(query_colors[0]))
    return out

def shared_hue_palette(query_img, ref_img, out_n, s_thr, v_thr, bins):
    """Build palette by finding bins with shared hue energy and sampling circular midpoints."""
    # Compute histograms
    arr_q = np.array(query_img).astype(np.uint8)
    arr_r = np.array(ref_img).astype(np.uint8)
    hsv_q = rgb_to_hsv01(arr_q); hsv_r = rgb_to_hsv01(arr_r)
    Hq, Sq, Vq = hsv_q[...,0], hsv_q[...,1], hsv_q[...,2]
    Hr, Sr, Vr = hsv_r[...,0], hsv_r[...,1], hsv_r[...,2]
    mask_q = (Sq >= s_thr) & (Vq >= v_thr)
    mask_r = (Sr >= s_thr) & (Vr >= v_thr)

    if not np.any(mask_q) or not np.any(mask_r):
        # fallback to kmeans on the images directly if one is too neutral
        return kmeans_palette(query_img, out_n)

    # Hist bins and centers
    hq = Hq[mask_q].ravel(); hr = Hr[mask_r].ravel()
    hist_q, edges = np.histogram(hq, bins=bins, range=(0,1))
    hist_r, _     = np.histogram(hr, bins=bins, range=(0,1))
    centers = (edges[:-1] + edges[1:]) / 2.0
    shared_idx = np.argsort(-(hist_q * hist_r + 1e-9))[:out_n]  # pick bins with strongest product
    out = []
    for ci in shared_idx:
        # center hue & representative S,V for each image
        h_center = centers[ci]
        # sample SV medians from masked pixels nearest to this bin
        def sample_sv(H, S, V, mask):
            if not np.any(mask): return 0.5, 0.5
            Hm = H[mask]
            idx = np.argmin(np.minimum(np.abs(Hm - h_center), 1 - np.abs(Hm - h_center)))
            Sm = S[mask][idx]; Vm = V[mask][idx]
            return float(Sm), float(Vm)
        s_q, v_q = sample_sv(Hq, Sq, Vq, mask_q)
        s_r, v_r = sample_sv(Hr, Sr, Vr, mask_r)
        s_mid = (s_q + s_r) / 2.0; v_mid = (v_q + v_r) / 2.0
        # convert HSV back to RGB
        hsv = np.array([[[h_center*179.0, s_mid*255.0, v_mid*255.0]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(3,)
        out.append(tuple(int(x) for x in rgb))
    # pad if needed
    while len(out) < out_n:
        out.append(tuple(out[-1] if out else (128,128,128)))
    return out

def weighted_hybrid_palette(query_colors, ref_colors, out_n, q_weight):
    """Lab blend with adjustable query weight, per nearest-hue pairing."""
    if not query_colors:
        return []
    if not ref_colors:
        return list(query_colors)[:out_n]
    q = np.array(query_colors, dtype=np.uint8).reshape(-1, 1, 3)
    r = np.array(ref_colors, dtype=np.uint8).reshape(-1, 1, 3)
    q_hsv = cv2.cvtColor(q, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    r_hsv = cv2.cvtColor(r, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    q_lab = cv2.cvtColor(q, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    r_lab = cv2.cvtColor(r, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    out = []
    for i in range(min(out_n, len(q_lab))):
        qh = q_hsv[i, 0] / 179.0
        diffs = []
        for j in range(len(r_hsv)):
            rh = r_hsv[j, 0] / 179.0
            d = min(abs(qh - rh), 1 - abs(qh - rh))
            diffs.append((d, j))
        j = min(diffs, key=lambda t: t[0])[1]
        blend = q_lab[i] * q_weight + r_lab[j] * (1.0 - q_weight)
        rgb = cv2.cvtColor(blend.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(3,)
        out.append(tuple(int(x) for x in rgb))
    while len(out) < out_n:
        out.append(tuple(query_colors[0]))
    return out

# -----------------------------------------------------------------------------
# Plot: metric bars (with unique keys)
# -----------------------------------------------------------------------------
def metric_bar_chart(metrics: dict, key: str):
    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=1),
        text=[f"{v:.2f}" for v in vals], textposition="outside", hoverinfo="skip"
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1], title="Similarity (0–1)"),
        yaxis=dict(title=""),
        height=max(220, 28*len(names)+60),
        margin=dict(l=80, r=20, t=10, b=10),
        template="simple_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# -----------------------------------------------------------------------------
# File upload
# -----------------------------------------------------------------------------
ref_zip = st.file_uploader("Upload a ZIP of reference images", type=["zip"])
query_file = st.file_uploader("Upload a query image", type=["jpg","jpeg","png"])

# -----------------------------------------------------------------------------
# Process
# -----------------------------------------------------------------------------
if ref_zip and query_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(ref_zip, "r") as z:
            z.extractall(tmp_dir)

        ref_paths = [p for p in iter_zip_images(tmp_dir)]
        if not ref_paths:
            st.error("No valid reference images found. Ensure your ZIP has JPG or PNG files.")
            st.stop()

        try:
            query_img = Image.open(query_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not open query image: {e}")
            st.stop()

        results = []
        for p in ref_paths:
            img = safe_open_image(p)
            if img is None:
                continue
            img_use = img.resize(query_img.size) if resize_refs else img

            metrics = {
                "Structural Alignment": structural_similarity(query_img, img_use),
                "Color Histogram":     color_hist_similarity(query_img, img_use),
                "Entropy Similarity":  entropy_similarity(query_img, img_use),
                "Edge Complexity":     edge_complexity_similarity(query_img, img_use),
                "Texture Correlation": texture_correlation_similarity(query_img, img_use),
                "Hue Distribution":    hue_distribution_similarity(query_img, img_use, hue_bins, sat_thresh, val_thresh),
            }
            results.append((p, metrics))

        if not results:
            st.error("No valid comparisons could be made.")
            st.stop()

        # Sort by mean similarity
        results.sort(key=lambda t: float(np.mean(list(t[1].values()))), reverse=True)
        top_results = results[:top_k]

        st.subheader(f"Top {len(top_results)} Most Similar Images")
        for rank, (ref_path, metrics) in enumerate(top_results, start=1):
            ref_img = safe_open_image(ref_path)
            if ref_img is None:
                continue
            if resize_refs:
                ref_img_disp = ref_img.resize(query_img.size)
            else:
                ref_img_disp = ref_img

            c1, c2 = st.columns([2.6, 1.4])
            with c1:
                st.markdown(f"### Match {rank}: {os.path.basename(ref_path)}")
                st.image([query_img, ref_img_disp],
                         caption=["Query", f"Reference {rank}"],
                         use_container_width=True)
                metric_bar_chart(metrics, key=f"bar_{rank}")

                st.markdown("**Metric Explanations**")
                with st.expander("Metric Explanations", expanded=False):
                    st.markdown(
                        "- **Structural Alignment**: Measures overlap in luminance/contrast structure (SSIM). "
                        "High = similar composition and spatial form."
                    )
                    st.markdown(
                        "- **Color Histogram**: 3D RGB histogram correlation (normalized). "
                        "High = similar overall color balance and distribution."
                    )
                    st.markdown(
                        "- **Entropy Similarity**: Compares information density (tonal variation). "
                        "High = similar detail/texture richness."
                    )
                    st.markdown(
                        "- **Edge Complexity**: Compares density of edges/contours. "
                        "High = similar amount and distribution of edges."
                    )
                    st.markdown(
                        "- **Texture Correlation**: GLCM energy proximity (local micro-patterns). "
                        "High = similar surface rhythm/regularity."
                    )
                    st.markdown(
                        "- **Hue Distribution**: Masked hue histogram cosine similarity. "
                        "High = shared dominant hue families; low = distinct chromatic families."
                    )

            with c2:
                st.markdown("Intersection Palettes")

                # Base palettes
                q_palette = kmeans_palette(query_img, num_colors)
                r_palette = kmeans_palette(ref_img_disp, num_colors)

                # 1) Blended Midpoint (Lab)
                mid_palette = lab_midpoint_palette(q_palette, r_palette, out_n=min(num_colors, 6))
                plotly_palette(mid_palette, key=f"mid_{rank}", title="Blended Midpoint")
                st.caption(
                    "Perceptual midpoints in Lab between nearest hue pairs. "
                    "Shows balanced chromatic overlap emphasizing shared tones."
                )

                # 2) Shared Hue (circular mid of shared bins)
                shared_palette = shared_hue_palette(query_img, ref_img_disp,
                                                    out_n=min(num_colors, 6),
                                                    s_thr=sat_thresh, v_thr=val_thresh,
                                                    bins=hue_bins)
                plotly_palette(shared_palette, key=f"shared_{rank}", title="Shared Hue")
                st.caption(
                    "Derived from hue bins that both images emphasize (after saturation/value masking). "
                    "Indicates common hue families and harmony regions."
                )

                # 3) Weighted Hybrid (Lab, adjustable weight)
                hybrid_palette = weighted_hybrid_palette(q_palette, r_palette,
                                                         out_n=min(num_colors, 6),
                                                         q_weight=hybrid_query_weight)
                plotly_palette(hybrid_palette, key=f"hyb_{rank}", title="Weighted Hybrid")
                st.caption(
                    f"Lab blend with query weight = {hybrid_query_weight:.2f}. "
                    "Demonstrates how the reference adapts toward the query’s palette or vice versa."
                )
else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

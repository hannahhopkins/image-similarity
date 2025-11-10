import os
import zipfile
import tempfile
from pathlib import Path
from io import BytesIO
from typing import List, Tuple

import streamlit as st
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("Image Similarity Analyzer")
st.write(
    "Upload a ZIP of reference images and a query image. "
    "The app compares them across multiple metrics (structure, color, texture, edges, entropy, hue) "
    "and shows three distinct intersection palettes."
)

# ---------------------------
# Sidebar controls & help text
# ---------------------------
st.sidebar.header("Options")

top_k = st.sidebar.slider("Top matches to display", 1, 10, 5)
num_colors = st.sidebar.slider("Palette size (k-means colors per image)", 3, 12, 6)
resize_refs = st.sidebar.checkbox("Resize reference images to match query", value=True)

st.sidebar.markdown("### Hue / Palette settings")
hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, help="How finely the hue circle is divided. More bins = finer hue resolution (slower).")
sat_thresh = st.sidebar.slider("Saturation mask threshold", 0.0, 1.0, 0.20, 0.01,
                               help="Pixels with saturation below this are treated as neutral and masked out of hue histograms.")
val_thresh = st.sidebar.slider("Value mask threshold", 0.0, 1.0, 0.20, 0.01,
                               help="Pixels with value (brightness) below this are treated as too dark and masked out of hue histograms.")
hybrid_query_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.60, 0.05,
                                        help="How much the query palette dominates the Lab blend. 1.0 = query only; 0.0 = reference only.")

st.sidebar.markdown("### What these mean")
st.sidebar.caption(
    "**Hue bins**: number of slices around the color wheel used to compare hue distributions.\n\n"
    "**Saturation mask threshold**: filters out low-saturation (nearly gray) pixels before hue comparison.\n\n"
    "**Value mask threshold**: filters out very dark pixels before hue comparison.\n\n"
    "**Hybrid palette: query weight**: how heavily the query colors influence the weighted Lab blend."
)

# ---------------------------
# Utility functions
# ---------------------------
def safe_open(path_or_file) -> Image.Image | None:
    try:
        img = Image.open(path_or_file).convert("RGB")
        return img
    except UnidentifiedImageError:
        return None
    except Exception:
        return None

def extract_zip_paths(uploaded_zip) -> List[str]:
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(tmp_dir)
    image_paths = []
    for root, _, files in os.walk(tmp_dir):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                image_paths.append(os.path.join(root, f))
    return image_paths

# ---------------------------
# Palette extraction (k-means)
# ---------------------------
def extract_palette_kmeans(img: Image.Image, n_colors: int) -> np.ndarray:
    arr = np.array(img.resize((200, 200))).reshape(-1, 3).astype(np.float32)
    if arr.size == 0:
        return np.zeros((n_colors, 3), dtype=np.uint8)
    k = min(n_colors, max(1, arr.shape[0] // 50))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_.astype(np.float32)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    if centers.shape[0] < n_colors:
        pad = np.tile(centers.mean(axis=0, keepdims=True), (n_colors - centers.shape[0], 1))
        centers = np.vstack([centers, pad])
    return np.clip(centers[:n_colors], 0, 255).astype(np.uint8)

def palette_grid(colors: np.ndarray, squares: int | None = None) -> BytesIO:
    """Render a compact square palette image."""
    if squares is None:
        squares = colors.shape[0]
    cols = int(np.ceil(np.sqrt(squares)))
    rows = int(np.ceil(squares / cols))
    sw = 36
    canvas = np.zeros((rows * sw, cols * sw, 3), dtype=np.uint8)
    for i in range(squares):
        r = i // cols
        c = i % cols
        canvas[r*sw:(r+1)*sw, c*sw:(c+1)*sw] = colors[i]
    buf = BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    buf.seek(0)
    return buf

# ---------------------------
# Color space helpers (Lab / HSV)
# ---------------------------
def rgb_to_lab_arr(rgb_arr_uint8: np.ndarray) -> np.ndarray:
    """rgb_arr_uint8: (N,3) uint8 -> Lab (N,3) float"""
    import skimage.color as skc
    rgb01 = (rgb_arr_uint8.astype(np.float32) / 255.0).reshape(-1, 1, 3)
    lab = skc.rgb2lab(rgb01).reshape(-1, 3)
    return lab

def lab_to_rgb_arr(lab_arr: np.ndarray) -> np.ndarray:
    import skimage.color as skc
    lab = lab_arr.reshape(-1, 1, 3)
    rgb01 = skc.lab2rgb(lab).reshape(-1, 3)
    return np.clip((rgb01 * 255).round(), 0, 255).astype(np.uint8)

# ---------------------------
# Intersection palettes
# ---------------------------
def blended_midpoint_palette(query_colors: np.ndarray, ref_colors: np.ndarray, n_out: int) -> np.ndarray:
    # nearest neighbors in Lab, then midpoint in Lab
    q_lab = rgb_to_lab_arr(query_colors)
    r_lab = rgb_to_lab_arr(ref_colors)
    out = []
    for i in range(min(n_out, len(q_lab))):
        d = np.linalg.norm(r_lab - q_lab[i], axis=1)
        j = int(np.argmin(d)) if len(d) else 0
        mid = (q_lab[i] + r_lab[j]) / 2.0
        out.append(mid)
    if not out:
        return query_colors[:n_out]
    out_lab = np.vstack(out)
    return lab_to_rgb_arr(out_lab)[:n_out]

def shared_hue_palette(query_img: Image.Image, ref_img: Image.Image, n_out: int,
                       bins: int, s_thr: float, v_thr: float) -> np.ndarray:
    # Build overlapping hue bins and use HSV medians of overlapping bins as swatches
    def hue_hist_and_bins(img: Image.Image):
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[:, :, 0]            # 0..180
        s = hsv[:, :, 1] / 255.0
        v = hsv[:, :, 2] / 255.0
        mask = (s >= s_thr) & (v >= v_thr)
        hh = h[mask].flatten()
        if hh.size == 0:
            hh = h.flatten()
        hist, edges = np.histogram(hh, bins=bins, range=(0, 180), density=False)
        return hist.astype(np.float32), edges, hsv, mask
    h1, e1, hsv1, m1 = hue_hist_and_bins(query_img)
    h2, e2, hsv2, m2 = hue_hist_and_bins(ref_img)
    # find overlapping bins (both nonzero)
    overlap_idx = np.where((h1 > 0) & (h2 > 0))[0]
    if overlap_idx.size == 0:
        # fallback: take top bins of query
        top_idx = np.argsort(-h1)[:n_out]
        hues = (e1[top_idx] + e1[top_idx+1]) / 2.0
        sv = np.array([0.7, 0.8])  # default S,V
        cols = []
        for hu in hues[:n_out]:
            cols.append(cv2.cvtColor(np.uint8([[[hu, int(sv[0]*255), int(sv[1]*255)]]]), cv2.COLOR_HSV2RGB)[0,0])
        return np.array(cols, dtype=np.uint8)

    # take up to n_out overlapping bins with largest combined counts
    combined = h1 + h2
    chosen = overlap_idx[np.argsort(-combined[overlap_idx])[:n_out]]
    cols = []
    for idx in chosen:
        lo, hi = e1[idx], e1[idx+1]
        # collect pixels in this bin for each image
        mask1 = (hsv1[:, :, 0] >= lo) & (hsv1[:, :, 0] < hi)
        mask2 = (hsv2[:, :, 0] >= lo) & (hsv2[:, :, 0] < hi)
        # combine HSV pixels (masked by S/V thresholds already in hist calc)
        hhsv = []
        if mask1.any():
            hhsv.append(hsv1[mask1])
        if mask2.any():
            hhsv.append(hsv2[mask2])
        if len(hhsv) == 0:
            # fallback median
            hu = (lo + hi) / 2.0
            cols.append(cv2.cvtColor(np.uint8([[[hu, 180, 200]]]), cv2.COLOR_HSV2RGB)[0,0])
        else:
            hhsv = np.vstack(hhsv)
            # median HSV within the overlapping bin
            med = np.median(hhsv, axis=0)
            hu, sa, va = float(med[0]), float(med[1]), float(med[2])
            rgb = cv2.cvtColor(np.uint8([[[hu, sa, va]]]), cv2.COLOR_HSV2RGB)[0,0]
            cols.append(rgb)
    return np.array(cols, dtype=np.uint8)[:n_out]

def weighted_hybrid_palette(query_colors: np.ndarray, ref_colors: np.ndarray, n_out: int, wq: float) -> np.ndarray:
    # convex blend in Lab with user weight
    q_lab = rgb_to_lab_arr(query_colors[:n_out])
    r_lab = rgb_to_lab_arr(ref_colors[:n_out])
    # pad if ref has fewer
    if r_lab.shape[0] < q_lab.shape[0]:
        r_lab = np.vstack([r_lab, np.tile(r_lab.mean(axis=0), (q_lab.shape[0] - r_lab.shape[0], 1))])
    blended = wq * q_lab + (1.0 - wq) * r_lab
    return lab_to_rgb_arr(blended)[:n_out]

# ---------------------------
# Metrics
# ---------------------------
def structural_similarity(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dr = max(1.0, float(g2.max() - g2.min()))
    return float(np.clip(ssim(g1, g2, data_range=dr), 0.0, 1.0))

def color_hist_similarity(img1_rgb: np.ndarray, img2_rgb: np.ndarray, bins=8) -> float:
    h1 = cv2.calcHist([img1_rgb], [0,1,2], None, [bins,bins,bins], [0,256,0,256,0,256])
    h2 = cv2.calcHist([img2_rgb], [0,1,2], None, [bins,bins,bins], [0,256,0,256,0,256])
    cv2.normalize(h1, h1); cv2.normalize(h2, h2)
    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)  # [-1,1]
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

def entropy_similarity(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [256], [0,256]); h1 = h1 / max(1, h1.sum())
    h2 = cv2.calcHist([g2], [0], None, [256], [0,256]); h2 = h2 / max(1, h2.sum())
    e1 = float(-(h1*(np.log2(h1+1e-12))).sum()); e2 = float(-(h2*(np.log2(h2+1e-12))).sum())
    if max(e1, e2) <= 1e-9:
        return 0.0
    return float(np.clip(1.0 - abs(e1-e2)/max(e1,e2), 0.0, 1.0))

def edge_complexity_similarity(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    e1 = cv2.Canny(g1, 100, 200); e2 = cv2.Canny(g2, 100, 200)
    return float(np.clip(1.0 - abs(e1.mean() - e2.mean())/255.0, 0.0, 1.0))

def texture_glcm_similarity(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    g1u = g1.astype(np.uint8); g2u = g2.astype(np.uint8)
    try:
        gl1 = graycomatrix(g1u, [2], [0], symmetric=True, normed=True)
        gl2 = graycomatrix(g2u, [2], [0], symmetric=True, normed=True)
        c1 = float(graycoprops(gl1, "contrast")[0,0])
        c2 = float(graycoprops(gl2, "contrast")[0,0])
        denom = max(c1 + c2, 1e-9)
        sim = 1.0 - abs(c1 - c2) / denom
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 0.0

def hue_distribution_similarity(img1: Image.Image, img2: Image.Image,
                                bins: int, s_thr: float, v_thr: float) -> float:
    # Build masked hue histograms and compute maximum cosine similarity under circular shift
    def masked_hues(img):
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[:,:,0]    # [0,180)
        s = hsv[:,:,1]/255.0
        v = hsv[:,:,2]/255.0
        mask = (s >= s_thr) & (v >= v_thr)
        hh = h[mask]
        if hh.size == 0:
            hh = h.flatten()
        hist, _ = np.histogram(hh, bins=bins, range=(0,180), density=False)
        hist = hist.astype(np.float32)
        if hist.sum() == 0:
            hist[:] = 1.0
        hist /= hist.sum()
        return hist
    h1 = masked_hues(img1)
    h2 = masked_hues(img2)

    def cosine(a,b):
        denom = (np.linalg.norm(a)*np.linalg.norm(b)) + 1e-12
        return float(np.dot(a,b)/denom)

    best = -1.0
    for shift in range(bins):
        sim = cosine(h1, np.roll(h2, shift))
        if sim > best:
            best = sim
    return float(np.clip(best, 0.0, 1.0))

def compute_all_metrics(query_img: Image.Image, ref_img: Image.Image, resize_to_query: bool) -> dict:
    rimg = ref_img.resize(query_img.size) if resize_to_query else ref_img
    q = np.array(query_img); r = np.array(rimg)
    metrics = {
        "Structural Alignment": structural_similarity(q, r),
        "Color Histogram": color_hist_similarity(q, r, bins=8),
        "Entropy Similarity": entropy_similarity(q, r),
        "Edge Complexity": edge_complexity_similarity(q, r),
        "Texture Correlation": texture_glcm_similarity(q, r),
        "Hue Distribution": hue_distribution_similarity(query_img, rimg, hue_bins, sat_thresh, val_thresh),
    }
    return metrics

METRIC_DEFS = {
    "Structural Alignment": "SSIM-based similarity of spatial structure (luminance, contrast, local form).",
    "Color Histogram": "Correlation of 3D RGB histograms (overall color balance).",
    "Entropy Similarity": "Similarity of information density / tonal complexity.",
    "Edge Complexity": "Similarity of edge density and distribution (Canny).",
    "Texture Correlation": "Similarity of micro-patterns via GLCM contrast.",
    "Hue Distribution": "Similarity of dominant hue proportions around the color wheel (circular).",
}

# ---------------------------
# Plotly metric bar
# ---------------------------
def plotly_metric_bar(metrics: dict):
    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]
    tooltips = [f"<b>{n}</b><br>{METRIC_DEFS.get(n,'')}" for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=1),
        hovertemplate=tooltips
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1], title="Similarity"),
        margin=dict(l=80, r=20, t=20, b=20),
        height= 30*len(names) + 60,
        yaxis=dict(autorange="reversed"),
        showlegend=False, template="simple_white"
    )
    return fig

# ---------------------------
# Upload UI
# ---------------------------
uploaded_zip = st.file_uploader("Upload a ZIP of Reference Images", type=["zip"])
query_image = st.file_uploader("Upload a Query Image", type=["jpg","jpeg","png","bmp","tiff","webp"])

if uploaded_zip and query_image:
    ref_paths = extract_zip_paths(uploaded_zip)
    if not ref_paths:
        st.error("No valid reference images found. Make sure your ZIP contains JPG/PNG (not just folders).")
        st.stop()

    qimg = safe_open(query_image)
    if qimg is None:
        st.error("Could not open query image.")
        st.stop()

    # Compute metrics for all refs
    results = []
    for p in ref_paths:
        rimg = safe_open(p)
        if rimg is None:
            continue
        try:
            metrics = compute_all_metrics(qimg, rimg, resize_refs)
            score = float(np.mean(list(metrics.values())))
            results.append((p, rimg, metrics, score))
        except Exception as e:
            st.warning(f"Skipped {p}: {e}")

    if not results:
        st.error("No valid comparisons could be made.")
        st.stop()

    results.sort(key=lambda t: t[3], reverse=True)
    top = results[:top_k]

    st.subheader(f"Top {len(top)} Matches")
    for idx, (path, rimg, metrics, score) in enumerate(top, start=1):
        if resize_refs:
            rimg = rimg.resize(qimg.size)

        col1, col2 = st.columns([2.5, 1.2], gap="large")

        with col1:
            st.markdown(f"### Match {idx}: {os.path.basename(path)} — Overall {score*100:.1f}%")
            st.image([qimg, rimg], caption=["Query", "Reference"], use_container_width=True)
            st.plotly_chart(plotly_metric_bar(metrics), use_container_width=True)

            # metric text
            for name, val in metrics.items():
                st.markdown(f"**{name} ({val:.2f})** — {METRIC_DEFS[name]}")

        with col2:
            st.markdown("#### Intersection Palettes")

            q_cols = extract_palette_kmeans(qimg, num_colors)
            r_cols = extract_palette_kmeans(rimg, num_colors)

            # 1) Blended Midpoint (Lab)
            blended = blended_midpoint_palette(q_cols, r_cols, n_out=min(num_colors, 6))
            st.image(palette_grid(blended), caption="Blended Midpoint")
            st.caption(
                "Nearest colors from each palette are paired in Lab space and averaged. "
                "This emphasizes balanced overlaps in perceptual lightness and chroma."
            )

            # 2) Shared Hue (overlapping bins)
            shared = shared_hue_palette(qimg, rimg, n_out=min(num_colors, 6),
                                        bins=hue_bins, s_thr=sat_thresh, v_thr=val_thresh)
            st.image(palette_grid(shared), caption="Shared Hue")
            st.caption(
                "Built from hue bins that both images share (after masking low-saturation/value pixels). "
                "Each swatch represents a median HSV from an overlapping hue region."
            )

            # 3) Weighted Hybrid (Lab convex blend)
            hybrid = weighted_hybrid_palette(q_cols, r_cols, n_out=min(num_colors, 6), wq=hybrid_query_weight)
            st.image(palette_grid(hybrid), caption="Weighted Hybrid")
            st.caption(
                "Convex blend in Lab with the selected query weight. "
                "Higher query weight retains query character; lower weight brings the reference forward."
            )

else:
    st.info("Upload your ZIP of reference images and a query image to begin.")

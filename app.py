import os
import zipfile
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2lab, lab2rgb

import plotly.graph_objects as go


# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("Image Similarity Analyzer")
st.write(
    "Upload a ZIP of reference images and a query image. The app computes similarity metrics and shows three "
    "distinct intersection palettes with explanations."
)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Options")

top_k = st.sidebar.slider("Matches to display", 1, 12, 5, 1)

num_colors = st.sidebar.slider("Palette size per image", 3, 12, 6, 1)
st.sidebar.caption("Number of dominant colors extracted from each image for palette comparison.")

resize_refs = st.sidebar.checkbox("Resize reference images to match query", value=True)
st.sidebar.caption("Ensures structural comparisons are made at matching resolutions.")

st.sidebar.markdown("---")

hue_bins = st.sidebar.slider("Hue bins (distribution metric)", 12, 72, 36, 6)
st.sidebar.caption("""
Determines how finely the hue wheel is divided.  
Higher values = more sensitive to subtle hue differences.  
Lower values = broader grouping of color families.
""")

sat_threshold = st.sidebar.slider("Saturation mask threshold", 0.0, 1.0, 0.15, 0.05)
st.sidebar.caption("""
Pixels with saturation below this value are excluded from hue analysis.  
Use to avoid noise from grayscale or muted regions.
""")

val_threshold = st.sidebar.slider("Value mask threshold", 0.0, 1.0, 0.15, 0.05)
st.sidebar.caption("""
Pixels below this brightness level are excluded from hue analysis.  
Helps prevent shadows/dark regions from distorting hue measurements.
""")

st.sidebar.markdown("---")

hybrid_query_weight = st.sidebar.slider("Hybrid palette: query weight", 0.5, 0.9, 0.7, 0.05)
st.sidebar.caption("""
Controls how strongly the query image influences the Weighted Hybrid palette.  
Higher values = palette shifts toward the query's dominant colors.  
Lower values = more balanced blend between query and reference palettes.
""")


# ---------------------------
# Helpers
# ---------------------------
def safe_open_image(path_or_file):
    try:
        img = Image.open(path_or_file).convert("RGB")
        return img
    except Exception:
        return None


def list_images_from_zip(uploaded_zip) -> list[str]:
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(tmp_dir)
    paths = []
    for root, _, files in os.walk(tmp_dir):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, f))
    return paths


def kmeans_palette(pil_img: Image.Image, k: int) -> tuple[list[tuple[int,int,int]], list[int]]:
    # downsample for speed/robustness
    small = pil_img.resize((220, 220))
    arr = np.array(small).reshape(-1, 3).astype(np.float32)
    k = min(k, max(2, arr.shape[0] // 200))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = [tuple(map(int, centers[i])) for i in order]
    counts = [int(counts[i]) for i in order]
    return centers, counts


def palette_image(colors: list[tuple[int,int,int]], square=38) -> Image.Image:
    if len(colors) == 0:
        colors = [(128,128,128)]
    w = square * len(colors)
    h = square
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        canvas[:, i*square:(i+1)*square] = np.array(c, dtype=np.uint8)
    return Image.fromarray(canvas)


# ---------------------------
# Metrics (all return 0..1)
# ---------------------------
def structural_alignment(img1: Image.Image, img2: Image.Image) -> float:
    a = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    dr = max(1, b.max() - b.min())
    return float(np.clip(ssim(a, b, data_range=dr), 0, 1))


def color_histogram_similarity(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(img1)
    b = np.array(img2)
    h1 = cv2.calcHist([a], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    h2 = cv2.calcHist([b], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    raw = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)  # [-1,1]
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))


def _hue_histogram(pil_img: Image.Image, bins: int, s_thr: float, v_thr: float) -> np.ndarray:
    hsv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[..., 0]  # [0,180)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    mask = (s >= s_thr) & (v >= v_thr)
    if np.count_nonzero(mask) == 0:
        hist = np.zeros(bins, dtype=np.float32)
        hist[0] = 1.0  # neutral spike to avoid empties
        return hist
    h_sel = h[mask]  # 0..180
    # map to bin indices on circle [0,180)
    bin_idx = np.floor(h_sel / (180.0 / bins)).astype(int)
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    hist = np.bincount(bin_idx, minlength=bins).astype(np.float32)
    hist /= max(1.0, hist.sum())
    return hist


def hue_distribution_similarity(img1: Image.Image, img2: Image.Image, bins: int, s_thr: float, v_thr: float) -> float:
    h1 = _hue_histogram(img1, bins, s_thr, v_thr)
    h2 = _hue_histogram(img2, bins, s_thr, v_thr)
    # circular cross-correlation max over all rotations
    best = 0.0
    for k in range(bins):
        sim = float(np.dot(h1, np.roll(h2, k)))
        if sim > best:
            best = sim
    # best already in [0,1] if both are normalized
    return float(np.clip(best, 0.0, 1.0))


def entropy_similarity(img1: Image.Image, img2: Image.Image) -> float:
    g1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [256], [0,256]).astype(np.float64)
    h2 = cv2.calcHist([g2], [0], None, [256], [0,256]).astype(np.float64)
    p1 = h1 / max(1.0, h1.sum())
    p2 = h2 / max(1.0, h2.sum())
    e1 = float(-np.sum(p1 * np.log2(p1 + 1e-12)))
    e2 = float(-np.sum(p2 * np.log2(p2 + 1e-12)))
    if max(e1, e2) <= 1e-9:
        return 1.0
    return float(np.clip(1.0 - abs(e1 - e2) / max(e1, e2), 0.0, 1.0))


def edge_complexity(img1: Image.Image, img2: Image.Image) -> float:
    g1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    e1 = cv2.Canny(g1, 100, 200)
    e2 = cv2.Canny(g2, 100, 200)
    d = abs(e1.mean() - e2.mean()) / 255.0
    return float(np.clip(1.0 - d, 0.0, 1.0))


def texture_correlation(img1: Image.Image, img2: Image.Image) -> float:
    g1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    gcm1 = graycomatrix(g1, [5], [0], symmetric=True, normed=True)
    gcm2 = graycomatrix(g2, [5], [0], symmetric=True, normed=True)
    t1 = graycoprops(gcm1, 'contrast')[0, 0]
    t2 = graycoprops(gcm2, 'contrast')[0, 0]
    if max(t1, t2) <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - abs(t1 - t2) / max(t1, t2), 0.0, 1.0))


def brightness_similarity(img1: Image.Image, img2: Image.Image) -> float:
    g1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.clip(1.0 - abs(g1.mean() - g2.mean()) / 255.0, 0.0, 1.0))


def compute_metrics(img_query: Image.Image, img_ref: Image.Image) -> dict[str, float]:
    return {
        "Structural Alignment": structural_alignment(img_query, img_ref),
        "Color Histogram": color_histogram_similarity(img_query, img_ref),
        "Hue Distribution": hue_distribution_similarity(img_query, img_ref, bins=hue_bins,
                                                       s_thr=sat_threshold, v_thr=val_threshold),
        "Entropy Similarity": entropy_similarity(img_query, img_ref),
        "Edge Complexity": edge_complexity(img_query, img_ref),
        "Texture Correlation": texture_correlation(img_query, img_ref),
        "Brightness Similarity": brightness_similarity(img_query, img_ref),
    }


# ---------------------------
# Intersection palettes (3 distinct methods)
# ---------------------------
def lab_midpoint_pairs(q_colors, r_colors, out_n=5):
    # Pair each top query color to nearest ref color in Lab, take Lab midpoint, convert back.
    q_lab = rgb2lab(np.array(q_colors, dtype=np.float32).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    r_lab = rgb2lab(np.array(r_colors, dtype=np.float32).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    out = []
    for i in range(min(out_n, len(q_lab))):
        d = np.linalg.norm(r_lab - q_lab[i], axis=1)
        j = int(np.argmin(d))
        mid = (q_lab[i] + r_lab[j]) / 2.0
        rgb = (lab2rgb(mid.reshape(1,1,3)).reshape(3,) * 255.0).clip(0,255).astype(np.uint8)
        out.append(tuple(int(x) for x in rgb))
    while len(out) < out_n:
        out.append(tuple(q_colors[0]))
    return out


def shared_hue_regions(q_colors, r_colors, out_n=5, hue_tol_deg=18.0):
    # Keep only colors whose hues are within a tolerance across palettes; average in HSV space.
    def rgb_to_hsv_tuple(c):
        hsv = cv2.cvtColor(np.array([[c]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0,0,:]
        return hsv.astype(np.float32)

    qh = [rgb_to_hsv_tuple(c) for c in q_colors]
    rh = [rgb_to_hsv_tuple(c) for c in r_colors]
    out = []
    tol = hue_tol_deg / 180.0 * 180.0  # OpenCV hue is 0..180
    for i, qc in enumerate(qh[:max(1, out_n*2)]):  # scan more to find overlaps
        diffs = [min(abs(qc[0]-rc[0]), 180.0-abs(qc[0]-rc[0])) for rc in rh]
        if len(diffs) == 0:
            continue
        j = int(np.argmin(diffs))
        if diffs[j] <= tol:
            # average hue circularly, and average S/V
            h_candidates = np.array([qc[0], rh[j][0]], dtype=np.float32)
            # circular mean on 0..180: convert to unit circle
            ang = h_candidates / 180.0 * 2*np.pi
            ch = np.array([np.cos(ang), np.sin(ang)]).mean(axis=1)
            h_mean = (np.arctan2(ch[1], ch[0]) % (2*np.pi)) / (2*np.pi) * 180.0
            s_mean = float((qc[1] + rh[j][1]) / 2.0)
            v_mean = float((qc[2] + rh[j][2]) / 2.0)
            hsv = np.array([[[h_mean, s_mean, v_mean]]], dtype=np.float32)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)[0,0,:]
            out.append(tuple(int(x) for x in rgb))
            if len(out) >= out_n:
                break
    while len(out) < out_n:
        out.append(tuple(q_colors[min(len(q_colors)-1, len(out))]))
    return out


def weighted_hybrid(q_colors, q_counts, r_colors, r_counts, out_n=5, q_weight=0.7):
    # Weight by cluster prominence and bias toward query colors by q_weight.
    q_lab = rgb2lab(np.array(q_colors, dtype=np.float32).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    r_lab = rgb2lab(np.array(r_colors, dtype=np.float32).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    out = []
    q_counts = np.array(q_counts, dtype=np.float32) + 1e-6
    r_counts = np.array(r_counts, dtype=np.float32) + 1e-6
    for i in range(min(out_n, len(q_lab))):
        d = np.linalg.norm(r_lab - q_lab[i], axis=1)
        if len(d) == 0:
            rgb = (lab2rgb(q_lab[i].reshape(1,1,3)).reshape(3,) * 255.0).clip(0,255).astype(np.uint8)
            out.append(tuple(int(x) for x in rgb))
            continue
        j = int(np.argmin(d))
        wq = q_weight * q_counts[i]
        wr = (1.0 - q_weight) * r_counts[j]
        lab_mix = (wq * q_lab[i] + wr * r_lab[j]) / (wq + wr)
        rgb = (lab2rgb(lab_mix.reshape(1,1,3)).reshape(3,) * 255.0).clip(0,255).astype(np.uint8)
        out.append(tuple(int(x) for x in rgb))
    while len(out) < out_n:
        out.append(tuple(q_colors[0]))
    return out


# ---------------------------
# Plotly metric bar chart
# ---------------------------
def plot_metrics_bar(metrics: dict) -> go.Figure:
    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]
    tooltips = {
        "Structural Alignment": "SSIM-based overlap of luminance, contrast, and structure.",
        "Color Histogram": "Correlation of 3D RGB histograms; maps [-1,1] to [0,1].",
        "Hue Distribution": "Circular similarity of hue histograms with saturation/value masking.",
        "Entropy Similarity": "Closeness of information density (Shannon entropy).",
        "Edge Complexity": "Similarity of overall edge density from Canny edges.",
        "Texture Correlation": "GLCM contrast proximity; micro-pattern resemblance.",
        "Brightness Similarity": "Closeness of mean luminance."
    }
    hover = [f"<b>{n}</b><br>{tooltips.get(n,'')}" for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=1),
        hovertemplate=hover
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1], title="Similarity"),
        yaxis=dict(autorange="reversed"),
        height= max(220, 28*len(names)),
        margin=dict(l=80, r=30, t=20, b=20),
        showlegend=False,
        template="simple_white"
    )
    return fig


# ---------------------------
# Uploads
# ---------------------------
ref_zip = st.file_uploader("Reference images (ZIP)", type=["zip"])
query_file = st.file_uploader("Query image", type=["jpg","jpeg","png"])

if ref_zip and query_file:
    ref_paths = list_images_from_zip(ref_zip)
    if len(ref_paths) == 0:
        st.error("No valid reference images found in the ZIP.")
        st.stop()

    query_img = safe_open_image(query_file)
    if query_img is None:
        st.error("Could not open the query image.")
        st.stop()

    st.subheader("Query")
    st.image(query_img, use_container_width=True)

    # Compute metrics for all references
    results = []
    for p in ref_paths:
        ref_img = safe_open_image(p)
        if ref_img is None:
            continue
        if resize_refs:
            ref_img = ref_img.resize(query_img.size)
        try:
            m = compute_metrics(query_img, ref_img)
        except Exception as e:
            st.warning(f"Skipped {p}: {e}")
            continue
        score = float(np.mean(list(m.values())))
        results.append((p, ref_img, m, score))

    if not results:
        st.error("No valid comparisons could be computed.")
        st.stop()

    results.sort(key=lambda t: t[3], reverse=True)
    top = results[:top_k]

    st.subheader(f"Top {len(top)} Matches")

    for rank, (path, ref_img, metrics, score) in enumerate(top, start=1):
        st.markdown(f"### Match {rank}: {os.path.basename(path)} — Overall {score*100:.1f}%")
        col_l, col_r = st.columns([2.5, 1.2], gap="large")

        with col_l:
            st.image([query_img, ref_img], caption=["Query", "Reference"], use_container_width=True)
            bar = plot_metrics_bar(metrics)
            st.plotly_chart(bar, use_container_width=True)

            # Per-metric concise descriptions
            explanations = {
                "Structural Alignment": "Measures overlap in composition and local structure via SSIM.",
                "Color Histogram": "Measures similarity in overall RGB color distribution.",
                "Hue Distribution": "Measures alignment of dominant hue families after masking low-saturation/dim pixels.",
                "Entropy Similarity": "Measures similarity in information density / complexity.",
                "Edge Complexity": "Compares overall density of detected edges.",
                "Texture Correlation": "Compares micro-pattern contrast using GLCM.",
                "Brightness Similarity": "Compares overall mean brightness."
            }
            for k in metrics:
                st.markdown(f"**{k} ({metrics[k]:.2f})** — {explanations.get(k,'')}")

        with col_r:
            st.markdown("Intersection Palettes")

            # Build palettes for both images
            q_cols, q_cnts = kmeans_palette(query_img, num_colors)
            r_cols, r_cnts = kmeans_palette(ref_img, num_colors)

            # 1) Lab Midpoint Pairs
            pal_mid = lab_midpoint_pairs(q_cols, r_cols, out_n=min(6, num_colors))
            st.image(palette_image(pal_mid), caption="Lab Midpoint Pairs")
            st.caption(
                "For each prominent query color, finds its nearest reference color in Lab space and shows the midpoint. "
                "This highlights shared perceptual tones between both palettes."
            )

            # 2) Shared Hue Regions
            pal_hue = shared_hue_regions(q_cols, r_cols, out_n=min(6, num_colors), hue_tol_deg=360.0/hue_bins)
            st.image(palette_image(pal_hue), caption="Shared Hue Regions")
            st.caption(
                "Selects only hues that occur in both palettes within a small angular tolerance on the hue circle, "
                "then averages them in HSV. This emphasizes truly overlapping hue families."
            )

            # 3) Weighted Hybrid (query-biased)
            pal_hyb = weighted_hybrid(q_cols, q_cnts, r_cols, r_cnts, out_n=min(6, num_colors), q_weight=hybrid_query_weight)
            st.image(palette_image(pal_hyb), caption="Weighted Hybrid")
            st.caption(
                "Blends nearest Lab neighbors using cluster prominence and a query-biased weight. "
                "This shows how the reference colors are pulled toward the query’s dominant palette."
            )

        st.markdown("---")

else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

# app.py
# Image Similarity Analyzer — full Streamlit app
# - Vertical layout, thin separators between matches
# - Sidebar controls + human-friendly explanations
# - Robust ZIP handling (skips __MACOSX and ._ files)
# - Resizes reference images safely (fixes "same dimensions" error)
# - 6 metrics with dropdown "Metric Explanations"
# - Plotly bar chart with unique keys per match
# - Intersection palettes (Blended Midpoint, Shared Hue, Weighted Hybrid) with distinct math
# - Palettes shown as Plotly “square” strips with hover HEX

import os, zipfile, tempfile
from pathlib import Path
from io import BytesIO
from typing import List, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import cv2
from PIL import Image, UnidentifiedImageError

from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2lab, lab2rgb

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("Image Similarity Analyzer")
st.write("Upload a ZIP of reference images and a single query image. The app compares them across multiple visual metrics and visualizes three intersection color palettes.")

# -----------------------------
# Sidebar controls + explanations
# -----------------------------
st.sidebar.header("Settings")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 6)
resize_refs = st.sidebar.checkbox("Resize reference images to match query for metrics", value=True)

st.sidebar.subheader("Hue Similarity Settings")
hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, help="How finely the hue circle is divided when comparing hue distributions.")
st.sidebar.caption("Hue bins: The hue spectrum is circular (red wraps around). More bins = finer color discrimination; fewer bins = broader grouping.")

sat_thresh = st.sidebar.slider("Saturation mask threshold", 0.0, 1.0, 0.1, 0.05, help="Ignore pixels with saturation below this level when computing hue similarity.")
st.sidebar.caption("Saturation mask threshold: Pixels with very low saturation are nearly gray; excluding them keeps the hue comparison meaningful.")

val_thresh = st.sidebar.slider("Value mask threshold", 0.0, 1.0, 0.08, 0.02, help="Ignore pixels darker than this level when computing hue similarity.")
st.sidebar.caption("Value mask threshold: Extremely dark pixels carry little hue information; masking them stabilizes the hue metric.")

st.sidebar.subheader("Hybrid Palette")
hybrid_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.6, 0.05, help="How much the query palette influences the hybrid palette vs. the reference.")
st.sidebar.caption("Hybrid palette: A weighted blend in perceptual Lab space. A higher weight leans the hybrid toward the query’s colors, a lower weight toward the reference.")

# -----------------------------
# Utility functions
# -----------------------------
def read_image_safe(fp) -> Image.Image | None:
    try:
        if isinstance(fp, (str, Path)):
            img = Image.open(fp)
        else:
            img = Image.open(fp)
        return img.convert("RGB")
    except UnidentifiedImageError:
        return None
    except Exception:
        return None

def extract_zip_images(uploaded_zip) -> List[str]:
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(tmp)
    valid = []
    for root, _, files in os.walk(tmp):
        if "__MACOSX" in root:
            continue
        for f in files:
            sf = str(f)
            if sf.startswith("._"):
                continue
            if str(sf).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
                valid.append(os.path.join(root, sf))
    return valid

def ensure_same_size(query_img: Image.Image, ref_img: Image.Image, do_resize: bool) -> Image.Image:
    return ref_img.resize(query_img.size, Image.BILINEAR) if do_resize else ref_img

# -----------------------------
# Metrics (normalized to 0..1)
# -----------------------------
def metric_ssim(q: Image.Image, r: Image.Image) -> float:
    qg = cv2.cvtColor(np.array(q), cv2.COLOR_RGB2GRAY)
    rg = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2GRAY)
    dr = float(rg.max() - rg.min())
    if dr <= 0: dr = 1.0
    return float(np.clip(ssim(qg, rg, data_range=dr), 0.0, 1.0))

def metric_hist_rgb(q: Image.Image, r: Image.Image, bins=8) -> float:
    qa = np.array(q); ra = np.array(r)
    h1 = cv2.calcHist([qa], [0,1,2], None, [bins,bins,bins], [0,256,0,256,0,256])
    h2 = cv2.calcHist([ra], [0,1,2], None, [bins,bins,bins], [0,256,0,256,0,256])
    cv2.normalize(h1, h1); cv2.normalize(h2, h2)
    raw = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)  # [-1,1]
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

def metric_entropy(q: Image.Image, r: Image.Image) -> float:
    qg = cv2.cvtColor(np.array(q), cv2.COLOR_RGB2GRAY)
    rg = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2GRAY)
    def ent(gray):
        h = cv2.calcHist([gray],[0],None,[256],[0,256])
        p = h / max(1, np.sum(h))
        return float(-np.sum(p * np.log2(p + 1e-12)))
    e1, e2 = ent(qg), ent(rg)
    return float(np.clip(1.0 - abs(e1 - e2) / max(e1, e2, 1e-6), 0.0, 1.0))

def metric_edges(q: Image.Image, r: Image.Image) -> float:
    qg = cv2.cvtColor(np.array(q), cv2.COLOR_RGB2GRAY)
    rg = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2GRAY)
    e1 = cv2.Canny(qg, 100, 200).astype(np.float32).flatten()
    e2 = cv2.Canny(rg, 100, 200).astype(np.float32).flatten()
    if e1.std() < 1e-6 or e2.std() < 1e-6:
        return 0.0
    corr = np.corrcoef(e1, e2)[0,1]
    return float(np.clip((corr + 1.0)/2.0, 0.0, 1.0))

def metric_texture_glcm(q: Image.Image, r: Image.Image) -> float:
    qg = cv2.cvtColor(np.array(q), cv2.COLOR_RGB2GRAY)
    rg = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2GRAY)
    # quantize to 8-bit (already) and compute GLCM contrast
    g1 = graycomatrix(qg, [5], [0], symmetric=True, normed=True)
    g2 = graycomatrix(rg, [5], [0], symmetric=True, normed=True)
    c1 = float(graycoprops(g1, 'contrast')[0,0])
    c2 = float(graycoprops(g2, 'contrast')[0,0])
    return float(np.clip(1.0 - abs(c1 - c2) / max(c1, c2, 1e-6), 0.0, 1.0))

def metric_brightness(q: Image.Image, r: Image.Image) -> float:
    qg = cv2.cvtColor(np.array(q), cv2.COLOR_RGB2GRAY).astype(np.float32)
    rg = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.clip(1.0 - abs(qg.mean() - rg.mean())/255.0, 0.0, 1.0))

def metric_hue_similarity(q: Image.Image, r: Image.Image, bins: int, s_thr: float, v_thr: float) -> float:
    qhsv = cv2.cvtColor(np.array(q), cv2.COLOR_RGB2HSV)
    rhsv = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2HSV)
    # masks
    s_thr_255 = int(s_thr * 255.0)
    v_thr_255 = int(v_thr * 255.0)
    qmask = (qhsv[:,:,1] >= s_thr_255) & (qhsv[:,:,2] >= v_thr_255)
    rmask = (rhsv[:,:,1] >= s_thr_255) & (rhsv[:,:,2] >= v_thr_255)

    qh = qhsv[:,:,0][qmask]
    rh = rhsv[:,:,0][rmask]
    if qh.size == 0 or rh.size == 0:
        return 0.0

    # histograms over 0..180 (OpenCV hue range)
    hq, _ = np.histogram(qh, bins=bins, range=(0,180), density=True)
    hr, _ = np.histogram(rh, bins=bins, range=(0,180), density=True)

    # circularity handling: roll to best alignment and take max correlation -> similarity
    # normalize
    hq = hq.astype(np.float32); hr = hr.astype(np.float32)
    if np.linalg.norm(hq) < 1e-8 or np.linalg.norm(hr) < 1e-8:
        return 0.0
    best = -1.0
    for shift in range(bins):
        rolled = np.roll(hr, shift)
        corr = float(np.dot(hq, rolled) / (np.linalg.norm(hq)*np.linalg.norm(rolled)))
        best = max(best, corr)
    return float(np.clip((best + 1.0)/2.0, 0.0, 1.0))

# -----------------------------
# Metric dictionary for explainer
# -----------------------------
METRIC_TEXT = {
    "Structural Alignment": "Compares overall luminance, contrast, and local structure (SSIM). High values mean the two images share similar spatial composition and form.",
    "Color Histogram": "Looks at the distribution of colors across RGB channels (3D histogram). High values indicate similar overall color usage, not just a few matching swatches.",
    "Entropy Similarity": "Measures information density/complexity based on grayscale entropy. Similar entropy suggests comparable texture richness and tonal variability.",
    "Edge Complexity": "Compares edge patterns (Canny). Higher scores mean similar outline density and directional edge structure.",
    "Texture Correlation": "Uses GLCM contrast to compare micro-patterns. Higher values mean closer match in fine-grain surface qualities.",
    "Brightness Similarity": "Compares global brightness (mean intensity). High values indicate comparable exposure/lighting.",
    "Hue Distribution": "Compares dominant hue balance around the circular color wheel, masking low-saturation and very-dark pixels to avoid noise."
}

# -----------------------------
# Palettes
# -----------------------------
def extract_palette_kmeans(img: Image.Image, k: int) -> np.ndarray:
    arr = np.array(img.convert("RGB")).reshape(-1,3).astype(np.float32)
    if arr.shape[0] == 0:
        return np.zeros((k,3), dtype=np.uint8)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_  # float
    # order by cluster size
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    return np.clip(centers, 0, 255).astype(np.float32)

def rgb_to_lab_rows(rgb_rows: np.ndarray) -> np.ndarray:
    # rgb_rows: N x 3 in 0..255 float
    rgb_img = (rgb_rows.reshape(1, -1, 3) / 255.0).astype(np.float32)
    lab_img = rgb2lab(rgb_img)
    return lab_img.reshape(-1,3).astype(np.float32)

def lab_to_rgb_rows(lab_rows: np.ndarray) -> np.ndarray:
    lab_img = lab_rows.reshape(1, -1, 3).astype(np.float32)
    rgb_img = lab2rgb(lab_img)  # 0..1
    rgb_rows = (np.clip(rgb_img, 0, 1)*255.0).reshape(-1,3)
    return rgb_rows.astype(np.float32)

def palette_blended_midpoint(q_cols: np.ndarray, r_cols: np.ndarray, n: int) -> np.ndarray:
    # for each query color, find nearest ref in Lab and take Lab midpoint
    q_lab = rgb_to_lab_rows(q_cols)
    r_lab = rgb_to_lab_rows(r_cols)
    out = []
    for i in range(min(n, len(q_lab))):
        d = np.linalg.norm(r_lab - q_lab[i], axis=1)
        j = int(np.argmin(d))
        mid = 0.5*q_lab[i] + 0.5*r_lab[j]
        out.append(mid)
    if not out:
        return q_cols[:n].copy()
    out_lab = np.vstack(out)
    return lab_to_rgb_rows(out_lab)

def palette_shared_hue(q_cols: np.ndarray, r_cols: np.ndarray, n: int, bins: int) -> np.ndarray:
    # pick pairs whose hue angles are closest; return their simple RGB midpoint
    def rgb_to_hue(rgb):
        hsv = cv2.cvtColor(rgb.reshape(1,1,3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0,0]
        return float(hsv[0])  # 0..179
    out = []
    for i in range(min(n, len(q_cols))):
        qh = rgb_to_hue(q_cols[i])
        diffs = []
        for j in range(len(r_cols)):
            rh = rgb_to_hue(r_cols[j])
            d = min(abs(qh - rh), 180 - abs(qh - rh))
            diffs.append((d, j))
        if diffs:
            diffs.sort(key=lambda x: x[0])
            _, jbest = diffs[0]
            avg = 0.5*q_cols[i] + 0.5*r_cols[jbest]
            out.append(avg)
    if not out:
        return q_cols[:n].copy()
    return np.vstack(out)

def palette_weighted_hybrid(q_cols: np.ndarray, r_cols: np.ndarray, n: int, w: float) -> np.ndarray:
    # weighted blend in Lab to emphasize perceptual balance
    q_lab = rgb_to_lab_rows(q_cols)
    r_lab = rgb_to_lab_rows(r_cols)
    out = []
    m = min(n, len(q_lab), len(r_lab))
    if m == 0:
        return q_cols[:n].copy()
    for i in range(m):
        mix = w*q_lab[i] + (1.0 - w)*r_lab[i]
        out.append(mix)
    out_lab = np.vstack(out)
    return lab_to_rgb_rows(out_lab)

def plot_palette_squares(colors_rgb: np.ndarray, title: str) -> go.Figure:
    # colors_rgb: N x 3 (0..255 floats)
    N = colors_rgb.shape[0]
    hexes = ["#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2])) for c in colors_rgb]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(N)),
        y=[1]*N,
        marker=dict(color=["rgb({},{},{})".format(int(c[0]), int(c[1]), int(c[2])) for c in colors_rgb]),
        text=hexes,
        hovertext=hexes,
        hoverinfo="text",
        showlegend=False
    ))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(title=title, height=80, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# -----------------------------
# Metric explainer dropdown
# -----------------------------
with st.expander("Metric Explanations"):
    for name, text in METRIC_TEXT.items():
        st.markdown(f"**{name}** — {text}")

# -----------------------------
# Uploads
# -----------------------------
ref_zip = st.file_uploader("Upload a ZIP of Reference Images", type=["zip"])
qry_file = st.file_uploader("Upload a Query Image", type=["jpg","jpeg","png","bmp","tif","tiff","webp"])

if ref_zip and qry_file:
    ref_paths = extract_zip_images(ref_zip)
    if len(ref_paths) == 0:
        st.error("No valid reference images found. Make sure your ZIP contains standard image formats.")
        st.stop()

    query_img = read_image_safe(qry_file)
    if query_img is None:
        st.error("Could not read the query image.")
        st.stop()

    # Pre-extract query palette once
    q_palette = extract_palette_kmeans(query_img, num_colors)

    # Compute metrics per reference
    scored = []
    for p in ref_paths:
        img = read_image_safe(p)
        if img is None:
            continue
        img_resized = ensure_same_size(query_img, img, resize_refs)
        try:
            metrics = {
                "Structural Alignment": metric_ssim(query_img, img_resized),
                "Color Histogram":     metric_hist_rgb(query_img, img_resized, bins=8),
                "Entropy Similarity":  metric_entropy(query_img, img_resized),
                "Edge Complexity":     metric_edges(query_img, img_resized),
                "Texture Correlation": metric_texture_glcm(query_img, img_resized),
                "Brightness Similarity": metric_brightness(query_img, img_resized),
                "Hue Distribution":    metric_hue_similarity(query_img, img_resized, bins=hue_bins, s_thr=sat_thresh, v_thr=val_thresh),
            }
        except ValueError:
            # In case any metric still complains about size, enforce again:
            img_resized = img.resize(query_img.size)
            metrics = {
                "Structural Alignment": metric_ssim(query_img, img_resized),
                "Color Histogram":     metric_hist_rgb(query_img, img_resized, bins=8),
                "Entropy Similarity":  metric_entropy(query_img, img_resized),
                "Edge Complexity":     metric_edges(query_img, img_resized),
                "Texture Correlation": metric_texture_glcm(query_img, img_resized),
                "Brightness Similarity": metric_brightness(query_img, img_resized),
                "Hue Distribution":    metric_hue_similarity(query_img, img_resized, bins=hue_bins, s_thr=sat_thresh, v_thr=val_thresh),
            }

        overall = float(np.mean(list(metrics.values())))
        scored.append((p, img_resized, metrics, overall))

    if not scored:
        st.error("No valid comparisons could be computed.")
        st.stop()

    # sort and limit
    scored.sort(key=lambda x: x[3], reverse=True)
    results = scored[:top_k]

    st.subheader(f"Top {len(results)} Matches")

    # Render each match block
    for i, (path, ref_img_resized, metrics, overall) in enumerate(results, start=1):
        st.markdown(f"### Match {i}: {os.path.basename(path)}  —  Overall similarity {overall*100:.1f}%")

        c1, c2 = st.columns([2.5, 1.2], gap="large")

        with c1:
            st.image([query_img, ref_img_resized], caption=["Query Image", "Reference"], use_container_width=True)

            # Plotly horizontal bar with unique key
            names = list(metrics.keys())
            vals = [metrics[n] for n in names]
            bar_fig = go.Figure(go.Bar(
                x=vals, y=names, orientation='h',
                marker=dict(color=vals, colorscale='RdYlGn', cmin=0, cmax=1),
                text=[f"{v:.2f}" for v in vals], textposition="outside", hoverinfo="x+y"
            ))
            bar_fig.update_layout(
                xaxis=dict(range=[0,1], title="Similarity (0–1)"),
                yaxis=dict(title=""),
                height=260,
                margin=dict(l=80,r=40,t=10,b=10),
                showlegend=False,
                template="simple_white"
            )
            st.plotly_chart(bar_fig, use_container_width=True, key=f"bars_{i}")

            # Thin separator below the whole block
        with c2:
            st.markdown("#### Intersection Palettes")

            r_palette = extract_palette_kmeans(ref_img_resized, num_colors)

            # Make sure palettes have right length
            def pad(pal, n):
                if pal.shape[0] >= n: return pal[:n]
                if pal.shape[0] == 0: return np.tile(np.array([[128,128,128]], dtype=np.float32), (n,1))
                extra = np.repeat(pal[-1:,:], n - pal.shape[0], axis=0)
                return np.vstack([pal, extra])

            # 1) Blended Midpoint (Lab midpoint of nearest colors)
            blended = pad(palette_blended_midpoint(q_palette, r_palette, n=num_colors), num_colors)
            # 2) Shared Hue (closest hue pairs, RGB midpoint)
            shared  = pad(palette_shared_hue(q_palette, r_palette, n=num_colors, bins=hue_bins), num_colors)
            # 3) Weighted Hybrid (Lab weighted by sidebar weight)
            hybrid  = pad(palette_weighted_hybrid(q_palette, r_palette, n=num_colors, w=hybrid_weight), num_colors)

            st.plotly_chart(plot_palette_squares(blended, "Blended Midpoint"), use_container_width=True, key=f"pal_blend_{i}")
            st.caption("Average of nearest color pairs in perceptual Lab space. This captures the balanced midpoint of overlapping chromatic regions between the two images.")

            st.plotly_chart(plot_palette_squares(shared, "Shared Hue"), use_container_width=True, key=f"pal_shared_{i}")
            st.caption("Pairs colors whose hue angles are closest on the circular hue wheel and averages them. This emphasizes color families that both images strongly feature.")

            st.plotly_chart(plot_palette_squares(hybrid, "Weighted Hybrid"), use_container_width=True, key=f"pal_hybrid_{i}")
            st.caption("Perceptual blend in Lab space using the selected Query Weight. Higher weights keep the hybrid closer to the query’s palette; lower weights pull it toward the reference.")

        # thin horizontal rule to separate matches
        st.markdown("<hr style='height:1px;border:none;background:#DDD;'>", unsafe_allow_html=True)
else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

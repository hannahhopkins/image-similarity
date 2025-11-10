import os
import zipfile
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("Image Similarity Analyzer")
st.write(
    "Upload a ZIP of reference images and a query image. "
    "The app compares structure, color, texture, edges, entropy and hue distribution, "
    "and shows three distinct intersection palettes."
)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("User Controls")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 8)
resize_refs = st.sidebar.checkbox("Resize reference images to match query", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Hue Similarity Settings")
hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, help="Number of bins in the hue histogram (0–180° mapped into N bins).")
sat_thresh = st.sidebar.slider("Saturation mask threshold", 0.0, 1.0, 0.15, 0.01,
                               help="Pixels with saturation below this are ignored in Hue calculations.")
val_thresh = st.sidebar.slider("Value (brightness) mask threshold", 0.0, 1.0, 0.15, 0.01,
                               help="Pixels with value/brightness below this are ignored in Hue calculations.")

st.sidebar.markdown(
    "<small>"
    "<b>Hue bins</b>: resolution of the circular hue histogram. "
    "<b>Saturation mask</b>: ignores near-gray pixels. "
    "<b>Value mask</b>: ignores very dark pixels. "
    "</small>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Palette")
query_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.60, 0.05,
                                 help="How much the query palette dominates the weighted hybrid palette (the remainder is the reference).")
st.sidebar.markdown(
    "<small>"
    "The hybrid palette blends the nearest query/reference colors in Lab space using this weight and cluster sizes."
    "</small>",
    unsafe_allow_html=True
)

# ---------------------------
# Helpers
# ---------------------------
def open_rgb(img_path_or_file):
    """Open as RGB PIL.Image."""
    if isinstance(img_path_or_file, (str, Path)):
        img = Image.open(img_path_or_file)
    else:
        img = Image.open(img_path_or_file)
    return img.convert("RGB")

def kmeans_palette(pil_img, k):
    """Return (centers_rgb[K,3], counts[K]) ordered by descending count."""
    arr = np.array(pil_img.resize((200, 200))).reshape(-1, 3).astype(np.float32)
    k = int(max(1, min(k, len(arr)//20)))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    counts = counts[order]
    return centers, counts

def palette_image_squares(colors_rgb, square_size=40):
    """Create a horizontal strip of square swatches."""
    if len(colors_rgb) == 0:
        colors_rgb = [(128, 128, 128)]
    cols = len(colors_rgb)
    out = np.zeros((square_size, square_size*cols, 3), dtype=np.uint8)
    for i, c in enumerate(colors_rgb):
        out[:, i*square_size:(i+1)*square_size] = np.array(c, dtype=np.uint8)
    return Image.fromarray(out)

def to_gray(pil_img, size=None):
    if size is not None:
        pil_img = pil_img.resize(size)
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

def safe_norm01(x):
    return float(np.clip(x, 0.0, 1.0))

# ---------------------------
# Metrics
# ---------------------------
def structural_similarity(pil_q, pil_r):
    if pil_r.size != pil_q.size:
        pil_r = pil_r.resize(pil_q.size)
    gq = to_gray(pil_q)
    gr = to_gray(pil_r)
    dr = float(gr.max() - gr.min())
    if dr <= 0:
        dr = 1.0
    return safe_norm01(ssim(gq, gr, data_range=dr))

def color_hist_similarity(pil_q, pil_r, bins=8):
    """3D RGB histogram correlation mapped from [-1,1] → [0,1]."""
    if pil_r.size != pil_q.size:
        pil_r = pil_r.resize(pil_q.size)
    aq = np.array(pil_q)
    ar = np.array(pil_r)
    hq = cv2.calcHist([aq], [0,1,2], None, [bins,bins,bins], [0,256]*3)
    hr = cv2.calcHist([ar], [0,1,2], None, [bins,bins,bins], [0,256]*3)
    cv2.normalize(hq, hq)
    cv2.normalize(hr, hr)
    raw = cv2.compareHist(hq, hr, cv2.HISTCMP_CORREL)  # [-1,1]
    return float(np.clip((raw + 1.0)/2.0, 0.0, 1.0))

def entropy_similarity(pil_q, pil_r):
    """Shannon entropy similarity of grayscale histograms."""
    if pil_r.size != pil_q.size:
        pil_r = pil_r.resize(pil_q.size)
    gq = to_gray(pil_q)
    gr = to_gray(pil_r)
    hq = cv2.calcHist([gq], [0], None, [256], [0,256])
    hr = cv2.calcHist([gr], [0], None, [256], [0,256])
    hq = hq / (hq.sum() + 1e-8)
    hr = hr / (hr.sum() + 1e-8)
    eq = -np.sum(hq * np.log2(hq + 1e-12))
    er = -np.sum(hr * np.log2(hr + 1e-12))
    return safe_norm01(1.0 - abs(eq - er) / max(eq, er + 1e-8))

def edge_complexity_similarity(pil_q, pil_r):
    """Compare Canny edge density."""
    if pil_r.size != pil_q.size:
        pil_r = pil_r.resize(pil_q.size)
    gq = to_gray(pil_q)
    gr = to_gray(pil_r)
    e1 = cv2.Canny(gq, 100, 200)
    e2 = cv2.Canny(gr, 100, 200)
    dens1 = e1.mean()/255.0
    dens2 = e2.mean()/255.0
    return safe_norm01(1.0 - abs(dens1 - dens2))

def texture_correlation_similarity(pil_q, pil_r):
    """GLCM contrast match: closer contrast → higher similarity."""
    if pil_r.size != pil_q.size:
        pil_r = pil_r.resize(pil_q.size)
    gq = to_gray(pil_q)
    gr = to_gray(pil_r)
    try:
        g1 = graycomatrix(gq, [2], [0], symmetric=True, normed=True)
        g2 = graycomatrix(gr, [2], [0], symmetric=True, normed=True)
        c1 = float(graycoprops(g1, 'contrast')[0,0])
        c2 = float(graycoprops(g2, 'contrast')[0,0])
        if max(c1, c2) <= 1e-8:
            return 1.0
        return safe_norm01(1.0 - abs(c1 - c2) / max(c1, c2))
    except Exception:
        return 0.0

def hue_distribution_similarity(pil_q, pil_r, bins=36, s_min=0.15, v_min=0.15):
    """Circular hue histogram similarity with SV masking and rotation search."""
    if pil_r.size != pil_q.size:
        pil_r = pil_r.resize(pil_q.size)

    def hue_hist(pil_img):
        rgb = np.array(pil_img)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[:,:,0]          # 0..180
        s = hsv[:,:,1]/255.0    # 0..1
        v = hsv[:,:,2]/255.0
        mask = (s >= s_min) & (v >= v_min)
        if mask.sum() < 10:
            return np.zeros(bins, dtype=np.float32)
        hsel = h[mask].flatten()
        hist, _ = np.histogram(hsel, bins=bins, range=(0,180))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist

    hq = hue_hist(pil_q)
    hr = hue_hist(pil_r)

    if hq.sum() == 0 or hr.sum() == 0:
        return 0.0

    # circular rotation search: take the best correlation across rotations
    best = -1.0
    for shift in range(bins):
        hr_roll = np.roll(hr, shift)
        corr = np.dot(hq, hr_roll) / (np.linalg.norm(hq)*np.linalg.norm(hr_roll) + 1e-12)  # cosine sim in [-1,1]
        best = max(best, corr)
    return float(np.clip((best + 1.0)/2.0, 0.0, 1.0))

# ---------------------------
# Plotly bar chart
# ---------------------------
def metric_bar(metrics: dict):
    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        text=[f"{v:.2f}" for v in vals], textposition='outside',
        marker=dict(color=vals, colorscale='RdYlGn', cmin=0, cmax=1)
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1], title="Similarity (0–1)"),
        yaxis=dict(autorange="reversed"),
        height=28*len(names) + 60,
        margin=dict(l=80, r=20, t=10, b=10),
        template="simple_white",
        showlegend=False
    )
    return fig

# ---------------------------
# Intersection palettes
# ---------------------------
def rgb_to_lab(rgb_arr_uint8):
    """rgb_arr_uint8: (N,3) uint8 → Lab float."""
    rgb = (rgb_arr_uint8.reshape(-1,1,3).astype(np.float32))/255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1,3)

def lab_to_rgb(lab_arr):
    lab = lab_arr.reshape(-1,1,3).astype(np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).reshape(-1,3)
    return np.clip((rgb*255.0).round().astype(np.uint8), 0, 255)

def nearest_by_hue(src_colors, dst_colors):
    """Map each color in src to index of nearest hue in dst."""
    def hues(rgb_colors):
        hsv = cv2.cvtColor(rgb_colors.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1,3)
        return hsv[:,0].astype(np.float32)  # 0..180
    h_src = hues(src_colors)
    h_dst = hues(dst_colors)
    idxs = []
    for h in h_src:
        diffs = np.minimum(np.abs(h_dst-h), 180-np.abs(h_dst-h))
        idxs.append(int(np.argmin(diffs)))
    return np.array(idxs, dtype=int)

def palette_blended_midpoint(q_centers, r_centers, n_out):
    """Lab midpoint of nearest hues (query → reference)."""
    if len(q_centers)==0:
        return []
    match = nearest_by_hue(q_centers, r_centers) if len(r_centers) else np.zeros(len(q_centers), dtype=int)
    q_lab = rgb_to_lab(q_centers)
    r_lab = rgb_to_lab(r_centers[match] if len(r_centers) else q_centers)
    mid = (q_lab + r_lab)/2.0
    rgb = lab_to_rgb(mid)
    return rgb[:n_out].tolist()

def palette_shared_hues(q_centers, r_centers, bins, s_min, v_min, pil_q, pil_r, n_out):
    """
    Build hue histograms with masks, find overlapping bins, then pick representative colors
    from the palettes that fall into those bins. Produces fewer but 'shared' hues.
    """
    def hue_hist_bins(pil_img):
        rgb = np.array(pil_img)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[:,:,0]
        s = hsv[:,:,1]/255.0
        v = hsv[:,:,2]/255.0
        mask = (s >= s_min) & (v >= v_min)
        if mask.sum() < 10:
            return np.zeros(bins, dtype=np.float32)
        hist, _ = np.histogram(h[mask], bins=bins, range=(0,180))
        return hist.astype(np.float32)

    hq = hue_hist_bins(pil_q)
    hr = hue_hist_bins(pil_r)
    overlap = (hq>0) & (hr>0)
    if overlap.sum() == 0:
        # fall back to simple average if nothing overlaps
        return ((q_centers.astype(np.float32) + r_centers.astype(np.float32))/2.0)[:n_out].astype(np.uint8).tolist()

    # bin centers in degrees
    bin_edges = np.linspace(0,180,bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    shared_centers = bin_centers[overlap]
    # pick nearest palette color in either palette to each shared hue center
    def pick_from_palette(palette_rgb, target_h):
        hsv = cv2.cvtColor(palette_rgb.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1,3)
        h = hsv[:,0].astype(np.float32)
        diffs = np.minimum(np.abs(h-target_h), 180-np.abs(h-target_h))
        return palette_rgb[int(np.argmin(diffs))]
    out = []
    for h in shared_centers:
        qc = pick_from_palette(q_centers, h)
        rc = pick_from_palette(r_centers, h)
        out.append(((qc.astype(np.float32)+rc.astype(np.float32))/2.0).astype(np.uint8))
    # limit / pad
    out = out[:n_out]
    while len(out) < n_out:
        out.append(q_centers[0] if len(q_centers) else np.array([128,128,128], dtype=np.uint8))
    return np.stack(out, axis=0).tolist()

def palette_weighted_hybrid(q_centers, q_counts, r_centers, r_counts, query_w, n_out):
    """
    Weighted Lab blend using query_w (0..1) and cluster sizes as secondary weights.
    """
    if len(q_centers)==0:
        return []
    match = nearest_by_hue(q_centers, r_centers) if len(r_centers) else np.zeros(len(q_centers), dtype=int)
    q_lab = rgb_to_lab(q_centers)
    r_lab = rgb_to_lab(r_centers[match] if len(r_centers) else q_centers)
    qc = q_counts[:len(q_centers)].astype(np.float32) if len(q_counts) else np.ones(len(q_centers), dtype=np.float32)
    rc = r_counts[match].astype(np.float32) if len(r_counts) and len(r_centers) else np.ones(len(q_centers), dtype=np.float32)
    # normalize cluster weights
    qc = qc / (qc.sum() + 1e-8)
    rc = rc / (rc.sum() + 1e-8)
    # blend
    wq = query_w
    wr = 1.0 - query_w
    blended = q_lab * (wq*qc)[:,None] + r_lab * (wr*rc)[:,None]
    # rescale to remove magnitude shrinkage (optional simple renorm)
    # here we simply convert back to rgb
    rgb = lab_to_rgb(blended)
    return rgb[:n_out].tolist()

# ---------------------------
# File upload
# ---------------------------
uploaded_zip = st.file_uploader("Upload a ZIP of Reference Images", type=["zip"])
query_image = st.file_uploader("Upload a Query Image", type=["jpg","jpeg","png"])

if uploaded_zip and query_image:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_zip, "r") as zf:
            zf.extractall(tmp_dir)

        ref_paths = []
        for root, _, files in os.walk(tmp_dir):
            if "__MACOSX" in root:
                continue
            for f in files:
                if f.startswith("._"):
                    continue
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    ref_paths.append(os.path.join(root, f))

        if not ref_paths:
            st.error("No valid reference images found in the ZIP (JPG/PNG).")
            st.stop()

        query_img = open_rgb(query_image)
        results = []

        # Compute metrics per reference
        for p in ref_paths:
            try:
                ref_img = open_rgb(p)
                if resize_refs:
                    ref_img = ref_img.resize(query_img.size)

                metrics = {
                    "Structural Alignment": structural_similarity(query_img, ref_img),
                    "Color Histogram":    color_hist_similarity(query_img, ref_img, bins=8),
                    "Entropy Similarity": entropy_similarity(query_img, ref_img),
                    "Edge Complexity":    edge_complexity_similarity(query_img, ref_img),
                    "Texture Correlation":texture_correlation_similarity(query_img, ref_img),
                    "Hue Distribution":   hue_distribution_similarity(
                        query_img, ref_img,
                        bins=hue_bins, s_min=sat_thresh, v_min=val_thresh
                    )
                }
                score = float(np.mean(list(metrics.values())))
                results.append((p, ref_img, metrics, score))
            except Exception as e:
                st.warning(f"Skipped {p}: {e}")

        if not results:
            st.error("No valid comparisons could be made.")
            st.stop()

        # Sort by average similarity
        results.sort(key=lambda x: x[3], reverse=True)
        top_results = results[:top_k]

        st.subheader(f"Top {len(top_results)} Matches")

        for i, (ref_path, ref_img, metrics, score) in enumerate(top_results, start=1):
            colL, colR = st.columns([2.4, 1.2], gap="large")

            with colL:
                st.markdown(f"### Match {i}: {os.path.basename(ref_path)} — Overall {score:.2f}")
                st.image([query_img, ref_img], caption=["Query", "Reference"], use_container_width=True)
                st.plotly_chart(metric_bar(metrics), use_container_width=True)

                # concise metric explanations
                expl = {
                    "Structural Alignment": "Luminance/contrast/structure agreement (SSIM). Higher means more similar layout and forms.",
                    "Color Histogram": "3D RGB histogram correlation; similar overall color distributions score higher.",
                    "Entropy Similarity": "Similarity of image information content/complexity via Shannon entropy.",
                    "Edge Complexity": "Compares the density of detected edges (Canny)—how visually structured each image is.",
                    "Texture Correlation": "GLCM contrast match—closer micro-pattern contrast yields higher similarity.",
                    "Hue Distribution": "Circular hue histogram match with S/V masking and rotation alignment."
                }
                for k in metrics:
                    st.markdown(f"**{k} — {metrics[k]:.2f}**  \n{expl[k]}")

            with colR:
                st.markdown("#### Intersection Palettes")

                # palettes for query & ref
                q_centers, q_counts = kmeans_palette(query_img, num_colors)
                r_centers, r_counts = kmeans_palette(ref_img,   num_colors)

                # 1) Blended Midpoint (Lab)
                pal_mid = palette_blended_midpoint(q_centers, r_centers, n_out=min(num_colors, 6))
                st.image(palette_image_squares(pal_mid), caption="Blended Midpoint")
                st.caption("A perceptual (Lab) midpoint between nearest query/reference hues — shows balanced shared tonality.")

                # 2) Shared Hues Only
                pal_shared = palette_shared_hues(
                    q_centers.astype(np.uint8),
                    r_centers.astype(np.uint8),
                    bins=hue_bins, s_min=sat_thresh, v_min=val_thresh,
                    pil_q=query_img, pil_r=ref_img, n_out=min(num_colors, 6)
                )
                st.image(palette_image_squares(pal_shared), caption="Shared Hues")
                st.caption("Only hues present in both images (by masked hue hist overlap), averaged for a concise common palette.")

                # 3) Weighted Hybrid (Lab)
                pal_hybrid = palette_weighted_hybrid(
                    q_centers.astype(np.uint8), q_counts,
                    r_centers.astype(np.uint8), r_counts,
                    query_w=float(query_weight), n_out=min(num_colors, 6)
                )
                st.image(palette_image_squares(pal_hybrid), caption="Weighted Hybrid")
                st.caption("Nearest hues blended in Lab using cluster sizes and your Hybrid 'Query weight' (sidebar).")

else:
    st.info("Upload your ZIP of reference images and a query image to begin.")

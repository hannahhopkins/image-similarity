import os, zipfile, tempfile
from pathlib import Path
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

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide")
st.title("Image Similarity Analyzer")


# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
st.sidebar.header("Settings")

top_k = st.sidebar.slider("Number of matches to display", 1, 12, 5)
num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 6)
resize_refs = st.sidebar.checkbox("Resize reference images to match query for metrics", value=True)

st.sidebar.subheader("Hue Similarity Settings")

hue_bins = st.sidebar.slider(
    "Hue bins (granularity)",
    12, 72, 36, 6,
    help="Controls how finely the circular color wheel is divided when comparing hue distributions. "
         "Higher values produce finer hue distinctions; lower values group hues more broadly."
)

sat_thresh = st.sidebar.slider(
    "Saturation mask threshold (0–1)",
    0.0, 1.0, 0.10, 0.01,
    help="Pixels with saturation below this value are treated as near-neutral (gray) and excluded from hue similarity."
)

val_thresh = st.sidebar.slider(
    "Value mask threshold (0–1)",
    0.0, 1.0, 0.08, 0.01,
    help="Pixels darker than this level are excluded from hue similarity so shadows do not distort color metrics."
)

st.sidebar.caption(
    "Hue similarity compares dominant color families while ignoring neutral or extremely dark pixel regions."
)

st.sidebar.subheader("Hybrid Palette")

hybrid_weight = st.sidebar.slider(
    "Query weight (Hybrid Palette)",
    0.0, 1.0, 0.6, 0.05,
    help="Controls how strongly the query palette pulls the hybrid palette. "
         "Higher values bias toward the query, lower values bias toward the reference."
)


# ------------------------------------------------------------
# Metric Explanation Dictionary
# ------------------------------------------------------------
METRIC_TEXT = {
    "Structural Alignment": "Compares luminance and structural patterns using SSIM. Higher values indicate similar tone distribution and spatial form.",
    "Color Histogram": "Compares the distribution of colors across the RGB cube. High values indicate similar global color usage.",
    "Entropy Similarity": "Compares informational complexity across textures. High values indicate similar detail density.",
    "Edge Complexity": "Compares contours detected by Canny edge detection. High values indicate similar outline density and edge structure.",
    "Texture Correlation": "Compares micro-pattern textures using GLCM contrast. High values indicate similar surface patterning.",
    "Brightness Similarity": "Compares overall luminance means. High values indicate similar lighting and exposure.",
    "Hue Distribution": "Compares which hue families are dominant across the circular color wheel, masking grayscale and shadow regions."
}


# ------------------------------------------------------------
# File Handling
# ------------------------------------------------------------
def read_image_safe(fp):
    try:
        return Image.open(fp).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def extract_zip_images(uploaded_zip):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(tmp)

    image_paths = []
    for root, _, files in os.walk(tmp):
        if "__MACOSX" in root:
            continue
        for f in files:
            fn = f.lower()
            if fn.startswith("._"):
                continue
            if fn.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
                image_paths.append(str(Path(root) / f))
    return image_paths


def ensure_same_size(query_img, ref_img, do_resize=True):
    return ref_img.resize(query_img.size, Image.BILINEAR) if do_resize else ref_img


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def metric_ssim_(q, r):
    qg = cv2.cvtColor(np.array(q), cv2.COLOR_RGB2GRAY)
    rg = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2GRAY)
    dr = max(float(rg.max() - rg.min()), 1e-6)
    return float(np.clip(ssim(qg, rg, data_range=dr), 0, 1))


def metric_hist_rgb(q, r, bins=8):
    h1 = cv2.calcHist([np.array(q)], [0,1,2], None, [bins,bins,bins], [0,256,0,256,0,256])
    h2 = cv2.calcHist([np.array(r)], [0,1,2], None, [bins,bins,bins], [0,256,0,256,0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(np.clip((cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) + 1)/2, 0, 1))


def metric_entropy(q, r):
    def e(img):
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([g], [0], None, [256], [0,256])
        p = hist/np.sum(hist)
        return float(-np.sum(p*np.log2(p+1e-12)))
    e_q, e_r = e(q), e(r)
    return float(np.clip(1 - abs(e_q - e_r)/max(e_q, e_r, 1e-6), 0, 1))


def metric_edges(q, r):
    def ed(img):
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        return cv2.Canny(g,100,200).astype(np.float32).mean()/255.0
    return float(np.clip(1 - abs(ed(q) - ed(r)), 0, 1))


def metric_texture_glcm(q, r):
    def tex(img):
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        gl = graycomatrix(g, [5], [0], normed=True)
        return float(graycoprops(gl, 'contrast')[0,0])
    t1, t2 = tex(q), tex(r)
    return float(np.clip(1 - abs(t1 - t2)/max(t1, t2, 1e-6), 0, 1))


def metric_brightness(q, r):
    qg = cv2.cvtColor(np.array(q), cv2.COLOR_RGB2GRAY).mean()
    rg = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2GRAY).mean()
    return float(np.clip(1 - abs(qg - rg)/255.0, 0, 1))


def metric_hue_dist(q, r, bins, s_thr, v_thr):
    def mask_hsv(img):
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        m = (hsv[:,:,1] >= int(s_thr*255)) & (hsv[:,:,2] >= int(v_thr*255))
        return hsv[:,:,0][m]
    qh, rh = mask_hsv(q), mask_hsv(r)
    if qh.size == 0 or rh.size == 0:
        return 0.0
    hq, _ = np.histogram(qh, bins=bins, range=(0,180), density=True)
    hr, _ = np.histogram(rh, bins=bins, range=(0,180), density=True)
    hq, hr = hq.astype(np.float32), hr.astype(np.float32)
    best = max(np.dot(hq, np.roll(hr, s)) for s in range(bins))
    return float(np.clip((best+1)/2, 0, 1))


# ------------------------------------------------------------
# Palette Calculation
# ------------------------------------------------------------
def palette_kmeans(img, k):
    arr = np.array(img).reshape(-1,3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=6, random_state=0).fit(arr)
    centers = km.cluster_centers_
    return np.clip(centers,0,255).astype(np.float32)

def rgba(colors):
    return [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors]

def plot_palette(colors, title):
    hexes = ["#{:02x}{:02x}{:02x}".format(int(c[0]),int(c[1]),int(c[2])) for c in colors]
    fig = go.Figure(go.Bar(
        x=list(range(len(colors))), y=[1]*len(colors),
        marker=dict(color=rgba(colors)),
        hovertext=hexes, hoverinfo="text"
    ))
    fig.update_layout(title=title, height=80, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10,r=10,t=30,b=10))
    return fig

def palette_mid(q, r, n):
    q_lab, r_lab = rgb2lab(q/255.0), rgb2lab(r/255.0)
    out=[]
    for i in range(min(n,len(q_lab))):
        j = np.argmin(np.linalg.norm(r_lab - q_lab[i], axis=1))
        out.append((q_lab[i]+r_lab[j])/2)
    return np.clip(lab2rgb(np.array(out))*255,0,255)

def palette_shared(q, r, n):
    def hue(c):
        return cv2.cvtColor(np.array([[c]],dtype=np.uint8), cv2.COLOR_RGB2HSV)[0,0,0]
    out=[]
    for qc in q:
        diffs=[(min(abs(hue(qc)-hue(rc)), 180-abs(hue(qc)-hue(rc))),rc) for rc in r]
        diffs.sort(key=lambda x:x[0])
        out.append((qc+diffs[0][1])/2)
        if len(out)==n: break
    return np.array(out)

def palette_hybrid(q, r, n, w):
    q_lab, r_lab = rgb2lab(q/255.0), rgb2lab(r/255.0)
    m = min(n,len(q_lab),len(r_lab))
    mix = w*q_lab[:m] + (1-w)*r_lab[:m]
    return np.clip(lab2rgb(mix)*255,0,255)


# ------------------------------------------------------------
# Metric Explanations
# ------------------------------------------------------------
with st.expander("Metric Explanations"):
    for name, desc in METRIC_TEXT.items():
        st.markdown(f"**{name}** — {desc}")


# ------------------------------------------------------------
# Uploads
# ------------------------------------------------------------
ref_zip = st.file_uploader("Upload Reference Images (ZIP)", type=["zip"])
qry = st.file_uploader("Upload Query Image", type=["jpg","jpeg","png","bmp","webp","tif","tiff"])

if ref_zip and qry:
    ref_paths = extract_zip_images(ref_zip)
    if not ref_paths:
        st.error("No valid images found in ZIP.")
        st.stop()

    query_img = read_image_safe(qry)
    if query_img is None:
        st.error("Could not read query image.")
        st.stop()

    q_cols = palette_kmeans(query_img, num_colors)

    scored=[]
    for p in ref_paths:
        ref = read_image_safe(p)
        if ref is None:
            continue
        ref_r = ensure_same_size(query_img, ref, resize_refs)

        metrics = {
            "Structural Alignment": metric_ssim_(query_img, ref_r),
            "Color Histogram": metric_hist_rgb(query_img, ref_r),
            "Entropy Similarity": metric_entropy(query_img, ref_r),
            "Edge Complexity": metric_edges(query_img, ref_r),
            "Texture Correlation": metric_texture_glcm(query_img, ref_r),
            "Brightness Similarity": metric_brightness(query_img, ref_r),
            "Hue Distribution": metric_hue_dist(query_img, ref_r, hue_bins, sat_thresh, val_thresh),
        }
        scored.append((p, ref_r, metrics, np.mean(list(metrics.values()))))

    scored.sort(key=lambda x:x[3], reverse=True)
    scored = scored[:top_k]

    for i,(path, ref_img, metrics, score) in enumerate(scored,1):
        st.markdown(f"### Match {i}: {os.path.basename(path)} — {score*100:.1f}%")
        c1,c2 = st.columns([2.8,1.2],gap="large")

        with c1:
            st.image([query_img, ref_img], caption=["Query","Reference"], use_container_width=True)

            names=list(metrics.keys())
            vals=[metrics[n] for n in names]
            fig = go.Figure(go.Bar(
                x=vals, y=names, orientation='h',
                marker=dict(color=vals, colorscale='RdYlGn', cmin=0, cmax=1),
                text=[f"{v:.2f}" for v in vals], textposition="outside"
            ))
            fig.update_layout(xaxis=dict(range=[0,1]), margin=dict(l=80,r=40,t=10,b=10), height=260)
            st.plotly_chart(fig, use_container_width=True, key=f"bar_{i}")

        with c2:
            r_cols = palette_kmeans(ref_img, num_colors)
            mid = palette_mid(q_cols, r_cols, num_colors)
            shared = palette_shared(q_cols, r_cols, num_colors)
            hybrid = palette_hybrid(q_cols, r_cols, num_colors, hybrid_weight)

            st.plotly_chart(plot_palette(mid,"Blended Midpoint"), use_container_width=True, key=f"pal_blend_{i}")
            st.caption("Perceptual midpoint of nearest color pairs in Lab space.")

            st.plotly_chart(plot_palette(shared,"Shared Hue"), use_container_width=True, key=f"pal_shared_{i}")
            st.caption("Pairs colors closest in hue angle and averages them to show shared hue families.")

            st.plotly_chart(plot_palette(hybrid,"Weighted Hybrid"), use_container_width=True, key=f"pal_hybrid_{i}")
            st.caption("Weighted perceptual blend based on the query/reference balance selected.")

        st.markdown("<hr style='border:none;border-top:1px solid #ccc;margin:1rem 0;'>", unsafe_allow_html=True)
        

#!/usr/bin/env python3
# app.py
# Image Similarity Analyzer

import os
import zipfile
import tempfile
from pathlib import Path
from io import BytesIO
from typing import List, Tuple

import streamlit as st
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import faiss
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2lab, rgb2hsv
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from scipy.fft import fft2
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --------------------------
# Page & UI defaults
# --------------------------
st.set_page_config(page_title="Image Similarity", layout="wide")
st.title("Image Similarity")
st.write("Upload a ZIP of reference images and a query image. The app returns top matches, metric breakdowns, a Plotly radar (composite signature), "
         "and three intersection palettes for the top match (optionally for others).")

# --------------------------
# Sidebar options
# --------------------------
with st.sidebar:
    st.header("Options")
    TOP_N = st.selectbox("Top matches to display", [3, 5, 10], index=1)
    N_PALETTE = st.slider("Extracted palette colors (k-means)", 5, 20, 10, 1)
    N_INTERSECT = st.slider("Intersection palette size", 3, 7, 5, 1)
    SHOW_ALL_INTERSECT = st.checkbox("Automatically compute intersection palettes for all matches?", value=False)
    THUMBNAIL_MAX = st.slider("Max image dimension when processing (px)", 256, 1024, 512, 64)
    st.markdown("---")
    st.write("Advanced: tune palette size, intersection size, or downscale for speed.")

# --------------------------
# Utilities: model loading
# --------------------------
@st.cache_resource
def load_resnet():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess

model, preprocess = load_resnet()

def extract_deep_features(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze().numpy().astype("float32")

# --------------------------
# Safe image helpers
# --------------------------
def safe_open(path_or_file) -> Image.Image | None:
    try:
        if isinstance(path_or_file, (str, Path)):
            img = Image.open(path_or_file)
        else:
            img = Image.open(path_or_file)  # file-like
        img = img.convert("RGB")
        img.thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX))
        return img
    except UnidentifiedImageError:
        st.warning(f"Skipping {getattr(path_or_file, 'name', path_or_file)} — not a valid image.")
        return None
    except Exception as e:
        st.warning(f"Skipping {getattr(path_or_file, 'name', path_or_file)} — {e}")
        return None

# --------------------------
# ZIP extraction (recursive)
# --------------------------
def extract_zip_to_temp(uploaded_zip) -> List[str]:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as z:
        z.extractall(temp_dir)
    image_files = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().startswith('.'):
                continue
            if Path(f).suffix.lower() in valid_ext:
                image_files.append(os.path.join(root, f))
    if not image_files:
        st.warning("No valid images found in the uploaded ZIP.")
    else:
        st.success(f"Found {len(image_files)} images in the ZIP.")
    return image_files

# --------------------------
# Palette extraction (k-means)
# returns centers as list of RGB tuples and counts
# --------------------------
def extract_palette_kmeans(img: Image.Image, n_colors: int = 10) -> Tuple[List[Tuple[int,int,int]], List[int]]:
    arr = np.array(img.convert("RGB").resize((200,200))).reshape(-1, 3).astype(np.float32)
    if len(arr) == 0:
        return [], []
    k = min(n_colors, max(1, len(arr)//50))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers_ordered = [tuple(map(int, centers[i])) for i in order]
    counts_ordered = [int(counts[i]) for i in order]
    # pad if fewer than requested
    while len(centers_ordered) < n_colors:
        centers_ordered.append(tuple(map(int, arr.mean(axis=0))))
        counts_ordered.append(0)
    return centers_ordered[:n_colors], counts_ordered[:n_colors]

# --------------------------
# Color space helpers
# --------------------------
def rgb_to_lab(rgb: Tuple[int,int,int]) -> np.ndarray:
    rgb_arr = np.array(rgb, dtype=float).reshape(1,1,3)/255.0
    lab = rgb2lab(rgb_arr)
    return lab.reshape(3,)

def lab_to_rgb(lab: np.ndarray) -> Tuple[int,int,int]:
    # lab: (3,)
    lab_arr = lab.reshape(1,1,3)
    # convert back by approximate inverse using skimage? skimage does rgb2lab only.
    # We'll use a small trick: convert normalized lab to rgb via skimage if available.
    # Using skimage doesn't provide direct lab->rgb; but we can do conversion approximate by reversing rgb2lab isn't directly invertible here.
    # Instead, use colors in rgb mixing domain: we will store centers in rgb and do blending in L*a*b by converting centers to lab and back by solving via linear methods.
    # Simpler: since we only blend small changes, convert centers to lab, blend, and then convert by using `skimage.color.lab2rgb`.
    from skimage.color import lab2rgb
    rgb = lab2rgb(lab_arr)
    rgb255 = np.clip((rgb*255).astype(int).reshape(3,), 0, 255)
    return tuple(map(int, rgb255))

def lab_array_from_rgb_list(rgb_list: List[Tuple[int,int,int]]) -> np.ndarray:
    arr = np.array(rgb_list, dtype=float).reshape(-1,1,3)/255.0
    lab = rgb2lab(arr)
    return lab.reshape(len(rgb_list),3)

# --------------------------
# Metrics computations
# All metrics return normalized similarity in [0,1]
# --------------------------
def ciede2000_similarity(img1: Image.Image, img2: Image.Image) -> float:
    # mean deltaE across resized pixels
    a = np.array(img1.resize((128,128))).astype(float)
    b = np.array(img2.resize((128,128))).astype(float)
    lab1 = rgb2lab(a/255.0)
    lab2 = rgb2lab(b/255.0)
    # deltaE per-pixel approximate via Euclidean in Lab (not exact ΔE2000 but reasonable)
    delta = np.linalg.norm(lab1 - lab2, axis=2)
    mean_de = np.nanmean(delta)
    sim = max(0.0, min(1.0, 1.0 - (mean_de/60.0)))
    return float(sim)

def histogram_correlation(img1: Image.Image, img2: Image.Image, bins=8) -> float:
    a = np.array(img1.resize((224,224)))
    b = np.array(img2.resize((224,224)))
    h1 = cv2.calcHist([a], [0,1,2], None, [bins]*3, [0,256]*3)
    h2 = cv2.calcHist([b], [0,1,2], None, [bins]*3, [0,256]*3)
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    val = cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, val)))

def hue_similarity(img1: Image.Image, img2: Image.Image, bins=36) -> float:
    h1 = rgb2hsv(np.array(img1.resize((224,224)))/255.0)[:,:,0].flatten()
    h2 = rgb2hsv(np.array(img2.resize((224,224)))/255.0)[:,:,0].flatten()
    hh1,_ = np.histogram(h1, bins=bins, range=(0,1), density=True)
    hh2,_ = np.histogram(h2, bins=bins, range=(0,1), density=True)
    denom = (np.linalg.norm(hh1)*np.linalg.norm(hh2)+1e-8)
    return float(max(0.0, min(1.0, np.dot(hh1, hh2)/denom)))

def texture_glcm_energy(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(np.uint8)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(np.uint8)
    try:
        g1 = graycomatrix(a, [1], [0], symmetric=True, normed=True)
        g2 = graycomatrix(b, [1], [0], symmetric=True, normed=True)
        e1 = graycoprops(g1, 'energy')[0,0]
        e2 = graycoprops(g2, 'energy')[0,0]
        sim = 1.0 - abs(e1 - e2)
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0

def brightness_similarity(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(float)
    sim = 1.0 - abs(a.mean() - b.mean()) / 255.0
    return float(max(0.0, min(1.0, sim)))

def edge_sobel_similarity(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(float)
    e1 = cv2.Sobel(a, cv2.CV_64F, 1, 1, ksize=3).flatten()
    e2 = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3).flatten()
    if np.std(e1) < 1e-8 or np.std(e2) < 1e-8:
        return 0.0
    corr = pearsonr(e1, e2)[0]
    return float(max(0.0, min(1.0, (corr + 1.0) / 2.0)))

def pattern_fft_similarity(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((256,256))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((256,256))).astype(float)
    f1 = np.log1p(np.abs(fft2(a))).flatten()
    f2 = np.log1p(np.abs(fft2(b))).flatten()
    if np.std(f1) < 1e-8 or np.std(f2) < 1e-8:
        return 0.0
    corr = pearsonr(f1, f2)[0]
    return float(max(0.0, min(1.0, (corr + 1.0) / 2.0)))

def entropy_similarity(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(np.uint8)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(np.uint8)
    e1 = shannon_entropy(a)
    e2 = shannon_entropy(b)
    # entropy typically ranges and we'll normalize roughly by 8 bits
    sim = 1.0 - abs(e1 - e2) / 8.0
    return float(max(0.0, min(1.0, sim)))

def ssim_similarity(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(float)
    # ensure data_range present
    dr = b.max() - b.min()
    if dr <= 0:
        dr = 1.0
    score = ssim(a, b, data_range=dr)
    return float(max(0.0, min(1.0, score)))

# --------------------------
# Metric info: name -> (definition, interpret lambda)
# --------------------------
METRIC_INFO = {
    "SSIM": {
        "definition": "Structural Similarity Index — compares luminance, contrast and local structure between images.",
        "interpret": lambda v: (
            "High structural alignment — overall composition and spatial arrangement closely match."
            if v > 0.85 else
            "Moderate structural correspondence — some shared layout or form."
            if v > 0.6 else
            "Low structural similarity — different composition or form."
        )
    },
    "Texture": {
        "definition": "GLCM energy — measures micro-pattern energy and texture regularity.",
        "interpret": lambda v: (
            "Textures are similar in granularity and surface rhythm."
            if v > 0.8 else
            "Some shared textural structure."
            if v > 0.55 else
            "Textures differ noticeably (smooth vs coarse)."
        )
    },
    "Edges": {
        "definition": "Sobel edge correlation — alignment of contours and directional structure.",
        "interpret": lambda v: (
            "Strong contour and outline alignment."
            if v > 0.8 else
            "Partial edge alignment."
            if v > 0.55 else
            "Different edge structures or object outlines."
        )
    },
    "Brightness": {
        "definition": "Mean luminance comparison — global lightness/brightness of the images.",
        "interpret": lambda v: (
            "Similar brightness and tonal exposure."
            if v > 0.85 else
            "Moderate proximity in overall lightness."
            if v > 0.6 else
            "Different lighting/exposure levels."
        )
    },
    "Histogram": {
        "definition": "Color histogram correlation — compares distribution of RGB palettes.",
        "interpret": lambda v: (
            "Strong palette overlap across channels."
            if v > 0.85 else
            "Moderate palette similarity."
            if v > 0.6 else
            "Different color distributions."
        )
    },
    "Hue": {
        "definition": "Hue distribution similarity in HSV space — compares dominant tones.",
        "interpret": lambda v: (
            "Dominant hue families align."
            if v > 0.8 else
            "Some shared tonal families."
            if v > 0.6 else
            "Distinct dominant hues."
        )
    },
    "Pattern": {
        "definition": "FFT pattern correlation — compares spatial frequency content and repetition.",
        "interpret": lambda v: (
            "Matching repetitive structures and frequency rhythms."
            if v > 0.8 else
            "Partial pattern similarity."
            if v > 0.6 else
            "Different spatial frequency content."
        )
    },
    "Entropy": {
        "definition": "Shannon entropy — measures visual information density / complexity.",
        "interpret": lambda v: (
            "Similar information density and visual complexity."
            if v > 0.8 else
            "Moderate similarity in complexity."
            if v > 0.6 else
            "Different levels of detail or noise."
        )
    },
    "CIEDE2000": {
        "definition": "Perceptual color distance measured in Lab space (approx.), lower is more similar.",
        "interpret": lambda v: (
            "Perceptual color match is strong."
            if v > 0.85 else
            "Moderate perceptual color similarity."
            if v > 0.6 else
            "Perceptual color differences are noticeable."
        )
    }
}

# --------------------------
# Intersection palettes functions
# returns list of RGB tuples length n_intersect
# --------------------------
def blended_midpoint_palette(q_colors: List[Tuple[int,int,int]],
                             r_colors: List[Tuple[int,int,int]],
                             n_out: int = 5) -> List[Tuple[int,int,int]]:
    # For each query color (top n_out), find nearest ref color in Lab and compute midpoint in Lab, convert to RGB
    q_lab = lab_array_from_rgb_list(q_colors)
    r_lab = lab_array_from_rgb_list(r_colors)
    out = []
    # take top n_out from query centers
    for i in range(min(n_out, len(q_colors))):
        ql = q_lab[i]
        # find nearest in r_lab
        dists = np.linalg.norm(r_lab - ql, axis=1)
        j = np.argmin(dists)
        midpoint = (ql + r_lab[j]) / 2.0
        # convert midpoint back to RGB via skimage.lab2rgb
        from skimage.color import lab2rgb
        rgb = lab2rgb(midpoint.reshape(1,1,3)).reshape(3,) * 255.0
        out.append(tuple(int(np.clip(x,0,255)) for x in rgb))
    # pad if needed
    while len(out) < n_out:
        out.append(q_colors[0])
    return out

def shared_hue_palette(q_colors: List[Tuple[int,int,int]],
                       r_colors: List[Tuple[int,int,int]],
                       n_out: int = 5) -> List[Tuple[int,int,int]]:
    # Convert centers to HSV hue and find closest-hue pairs
    q_hsv = [rgb2hsv(np.array([[np.array(c)/255.0]]))[0,0,0] for c in q_colors]
    r_hsv = [rgb2hsv(np.array([[np.array(c)/255.0]]))[0,0,0] for c in r_colors]
    pairs = []
    for i,qh in enumerate(q_hsv):
        diffs = [min(abs(qh-rh), 1-abs(qh-rh)) for rh in r_hsv]
        j = int(np.argmin(diffs))
        pairs.append((i,j,diffs[j]))
    pairs_sorted = sorted(pairs, key=lambda x: x[2])[:n_out]
    out = []
    for i,j,_ in pairs_sorted:
        # average RGB of pair
        qc = np.array(q_colors[i]).astype(float)
        rc = np.array(r_colors[j]).astype(float)
        avg = ((qc + rc)/2.0).astype(int)
        out.append(tuple(np.clip(avg, 0, 255)))
    # pad
    while len(out) < n_out:
        out.append(q_colors[0])
    return out

def weighted_hybrid_palette(q_colors: List[Tuple[int,int,int]], q_counts: List[int],
                            r_colors: List[Tuple[int,int,int]], r_counts: List[int],
                            n_out: int = 5) -> List[Tuple[int,int,int]]:
    # weight by cluster sizes, match by nearest Lab and produce weighted average in Lab
    q_lab = lab_array_from_rgb_list(q_colors)
    r_lab = lab_array_from_rgb_list(r_colors)
    out = []
    for i in range(min(n_out, len(q_colors))):
        ql = q_lab[i]
        # compute weights based on counts
        qcount = q_counts[i] if i < len(q_counts) else 1
        dists = np.linalg.norm(r_lab - ql, axis=1)
        j = int(np.argmin(dists))
        rcount = r_counts[j] if j < len(r_counts) else 1
        total = qcount + rcount
        weighted = (ql * qcount + r_lab[j] * rcount) / max(total,1)
        from skimage.color import lab2rgb
        rgb = lab2rgb(weighted.reshape(1,1,3)).reshape(3,) * 255.0
        out.append(tuple(int(np.clip(x,0,255)) for x in rgb))
    while len(out) < n_out:
        out.append(q_colors[0])
    return out

# --------------------------
# Plotting helpers
# --------------------------
def palette_image_from_colors(colors: List[Tuple[int,int,int]], hex_labels: bool = True, height=40):
    n = len(colors)
    fig, ax = plt.subplots(figsize=(6, 0.4), dpi=100)
    for i,c in enumerate(colors):
        ax.add_patch(plt.Rectangle((i/n, 0), 1/n, 1, color=np.array(c)/255.0))
        if hex_labels:
            hexv = "#{:02x}{:02x}{:02x}".format(*c)
            ax.text((i+0.5)/n, 0.5, hexv, ha='center', va='center', fontsize=7,
                    color='white' if (0.299*c[0]+0.587*c[1]+0.114*c[2]) < 150 else 'black',
                    transform=ax.transAxes)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis('off')
    buf = BytesIO()
    plt.tight_layout(pad=0.2)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return buf

def plotly_radar(metrics_query: dict, metrics_match: dict, metric_order: List[str], match_label: str = "Match"):
    q_vals = [metrics_query.get(m, 0.0) for m in metric_order]
    m_vals = [metrics_match.get(m, 0.0) for m in metric_order]
    # ensure closed polygon
    categories = metric_order + [metric_order[0]]
    q_plot = q_vals + [q_vals[0]]
    m_plot = m_vals + [m_vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=q_plot, theta=categories, fill='toself', name='Query', line=dict(color='gray')))
    fig.add_trace(go.Scatterpolar(r=m_plot, theta=categories, fill='toself', name=match_label, line=dict(color='crimson')))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), showlegend=True, margin=dict(l=10,r=10,t=20,b=10))
    return fig

# --------------------------
# Main UI: uploads
# --------------------------
st.subheader("1) Upload reference images (ZIP) and a query image")
ref_zip = st.file_uploader("Reference folder ZIP", type=["zip"])
query_file = st.file_uploader("Query image", type=["jpg","jpeg","png","bmp","tiff","webp"])

if ref_zip and query_file:
    st.write("✅ ZIP uploaded:", getattr(ref_zip, "name", "uploaded_zip"))
    st.write("✅ Query image uploaded:", getattr(query_file, "name", "query"))
    ref_files = extract_zip_to_temp(ref_zip)
    st.write(f"Found {len(ref_files)} candidate images in ZIP.")

    query_img = safe_open(query_file)
    if query_img is None:
        st.error("Could not open query image.")
        st.stop()

    # Process reference images to extract deep features (batch)
    st.info("Extracting deep features from reference images — this may take a moment.")
    features = []
    processed_paths = []
    progress = st.progress(0)
    for i, p in enumerate(ref_files, start=1):
        img = safe_open(p)
        if img is None:
            progress.progress(i/len(ref_files))
            continue
        try:
            feat = extract_deep_features(img)
            features.append(feat)
            processed_paths.append(str(p))
        except Exception as e:
            st.warning(f"Skipping {p}: {e}")
        progress.progress(i/len(ref_files))
    progress.empty()
    st.success(f"Processed {len(processed_paths)} reference images.")

    if len(processed_paths) == 0:
        st.error("No reference images processed successfully.")
        st.stop()

    # FAISS index and search
    feats = np.array(features).astype("float32")
    index = faiss.IndexFlatL2(feats.shape[1])
    index.add(feats)
    q_feat = extract_deep_features(query_img)
    k = min(TOP_N, len(processed_paths))
    D, I = index.search(np.array([q_feat]), k=k)

    st.subheader("Results")
    st.image(query_img, caption="Query Image", use_container_width=True)

    # compute metrics for each top match (only computing full metrics for top matches)
    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        ref_path = processed_paths[idx]
        ref_img = safe_open(ref_path)
        if ref_img is None:
            continue

        # ensure same size
        ref_img_resized = ref_img.resize(query_img.size)

        # compute metrics set
        metrics = {}
        metrics["SSIM"] = ssim_similarity(query_img, ref_img_resized)
        metrics["Texture"] = texture_glcm_energy(query_img, ref_img_resized)
        metrics["Edges"] = edge_sobel_similarity(query_img, ref_img_resized)
        metrics["Brightness"] = brightness_similarity(query_img, ref_img_resized)
        metrics["Histogram"] = histogram_correlation(query_img, ref_img_resized)
        metrics["Hue"] = hue_similarity(query_img, ref_img_resized)
        metrics["Pattern"] = pattern_fft_similarity(query_img, ref_img_resized)
        metrics["Entropy"] = entropy_similarity(query_img, ref_img_resized)
        metrics["CIEDE2000"] = ciede2000_similarity(query_img, ref_img_resized)

        overall = float(np.mean(list(metrics.values()))) * 100.0

        results.append({
            "rank": rank,
            "path": ref_path,
            "img": ref_img_resized,
            "metrics": metrics,
            "distance": float(dist),
            "score": overall
        })

    if len(results) == 0:
        st.error("No matches could be computed.")
        st.stop()

    # Show top results
    metric_order = ["SSIM", "Texture", "Edges", "Brightness", "Histogram", "Hue", "Pattern", "Entropy", "CIEDE2000"]
    for r in results:
        rank = r["rank"]
        ref_path = r["path"]
        ref_img = r["img"]
        metrics = r["metrics"]
        score = r["score"]
        st.markdown(f"### Match {rank}: **{os.path.basename(ref_path)}** — Overall: **{score:.1f}%** (distance {r['distance']:.4f})")
        cols = st.columns([1,1])
        with cols[0]:
            st.image(query_img, caption="Query", use_container_width=True)
        with cols[1]:
            st.image(ref_img, caption=f"Match {rank}", use_container_width=True)

        # small metric bar chart + descriptions in two columns
        metric_scores_small = {k: metrics[k] for k in metric_order}
        # compact chart left, descriptions right
        col_chart, col_desc = st.columns([1,1.6])
        with col_chart:
            fig, ax = plt.subplots(figsize=(3, 0.9))
            names = list(metric_scores_small.keys())
            vals = [metric_scores_small[n] for n in names]
            y_pos = np.arange(len(names))
            ax.barh(y_pos, vals, color="#888888", height=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlim(0,1)
            ax.invert_yaxis()
            for i,v in enumerate(vals):
                ax.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=8)
            ax.set_xlabel("Similarity", fontsize=8)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig, use_container_width=True)

            # Plotly radar
            radar_fig = plotly_radar({k:metrics[k] for k in metric_order}, metrics, metric_order, match_label=f"Match {rank}")
            st.plotly_chart(radar_fig, use_container_width=True)

        with col_desc:
            st.markdown("#### Metric definitions & interpretation")
            for name in metric_order:
                info = METRIC_INFO.get(name, None)
                val = metrics.get(name, 0.0)
                if info:
                    st.markdown(f"**{name}** — {info['definition']}")
                    st.caption(info['interpret'](val))
                else:
                    st.markdown(f"**{name}** — value: `{val:.2f}`")
            # Intersection palette toggle for non-top1
            if rank == 1:
                show_intersection = True
            else:
                if SHOW_ALL_INTERSECT:
                    show_intersection = True
                else:
                    show_intersection = st.checkbox(f"Show intersection palettes for Match {rank}?", key=f"inter_{rank}")

        # For the first match, always compute three intersection palettes and display with descriptions
        if show_intersection:
            # extract kmeans palettes for query and ref
            q_centers, q_counts = extract_palette_kmeans(query_img, N_PALETTE)
            r_centers, r_counts = extract_palette_kmeans(ref_img, N_PALETTE)
            # intersection palettes of size N_INTERSECT
            blended = blended_midpoint_palette(q_centers, r_centers, n_out=N_INTERSECT)
            shared = shared_hue_palette(q_centers, r_centers, n_out=N_INTERSECT)
            hybrid = weighted_hybrid_palette(q_centers, q_counts, r_centers, r_counts, n_out=N_INTERSECT)

            st.markdown("**Intersection Palettes**")
            # show three palettes horizontally
            pcol1, pcol2, pcol3 = st.columns([1,1,1])
            with pcol1:
                buf = palette_image_from_colors(blended, hex_labels=True)
                st.image(buf, caption=f"Blended Midpoint ({N_INTERSECT} colors) — Balanced blend of shared chromatic tendencies.")
            with pcol2:
                buf = palette_image_from_colors(shared, hex_labels=True)
                st.image(buf, caption=f"Shared Hues ({N_INTERSECT} colors) — Hues common to both images.")
            with pcol3:
                buf = palette_image_from_colors(hybrid, hex_labels=True)
                st.image(buf, caption=f"Weighted Hybrid ({N_INTERSECT} colors) — Weighted by cluster prominence (dynamic energy).")

        st.markdown("---")

else:
    st.info("Upload a ZIP of reference images and a query image to get started.")

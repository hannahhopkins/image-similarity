#!/usr/bin/env python3
# app.py — Image Similarity Analyzer with color sub-metrics + palettes (10 colors)

import os
import zipfile
import tempfile
from pathlib import Path
from io import BytesIO
from typing import List

import streamlit as st
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import faiss
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2lab, deltaE_ciede2000, rgb2hsv
from scipy.fft import fft2
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import matplotlib

# Use Agg backend for environments without display
matplotlib.use("Agg")

st.set_page_config(page_title="Image Similarity — Color + Metrics", layout="wide")
st.title("🔍 Image Similarity — Color Sub-metrics + Compact Palettes")
st.write(
    "Upload a ZIP of reference images and a query image. "
    "Choose top-N matches. Each match displays technical metrics and a compact 10-color palette "
    "with hex labels overlaid."
)

# ---------------------------
# Model loading & preprocessing
# ---------------------------
@st.cache_resource
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess

model, preprocess = load_model()

def extract_deep_features(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze().numpy().astype("float32")

# ---------------------------
# Safe image helpers
# ---------------------------
def safe_open_image(path_or_file) -> Image.Image | None:
    try:
        if isinstance(path_or_file, (str, Path)):
            img = Image.open(path_or_file)
        else:
            img = Image.open(path_or_file)  # file-like
        img = img.convert("RGB")
        img.thumbnail((1024, 1024))  # keep reasonable memory usage
        return img
    except UnidentifiedImageError:
        st.warning(f"Skipping {getattr(path_or_file, 'name', path_or_file)} — not a valid image.")
        return None
    except Exception as e:
        st.warning(f"Skipping {getattr(path_or_file, 'name', path_or_file)} — {e}")
        return None

# ---------------------------
# Color sub-metrics
# ---------------------------
def ciede2000_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Return normalized perceptual similarity in [0,1] using deltaE (smaller deltaE => higher similarity)."""
    a1 = np.array(img1.resize((128,128))).astype(float)
    a2 = np.array(img2.resize((128,128))).astype(float)
    lab1 = rgb2lab(a1 / 255.0)
    lab2 = rgb2lab(a2 / 255.0)
    # compute mean deltaE across pixels
    delta = deltaE_ciede2000(lab1, lab2)
    mean_de = np.nanmean(delta)
    # map mean_de to similarity: 0 -> 1, large -> 0; typical max perceptual ~ 100
    sim = max(0.0, min(1.0, 1.0 - (mean_de / 60.0)))  # scale factor 60 chosen empirically
    return float(sim)

def histogram_correlation(img1: Image.Image, img2: Image.Image, bins=8) -> float:
    a = np.array(img1.resize((224,224)))
    b = np.array(img2.resize((224,224)))
    hist1 = cv2.calcHist([a], [0,1,2], None, [bins]*3, [0,256]*3)
    hist2 = cv2.calcHist([b], [0,1,2], None, [bins]*3, [0,256]*3)
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    val = cv2.compareHist(hist1.astype('float32'), hist2.astype('float32'), cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, val)))

def hue_similarity(img1: Image.Image, img2: Image.Image, bins=36) -> float:
    hsv1 = rgb2hsv(np.array(img1.resize((224,224))) / 255.0)
    hsv2 = rgb2hsv(np.array(img2.resize((224,224))) / 255.0)
    h1 = np.histogram(hsv1[:,:,0].flatten(), bins=bins, range=(0,1), density=True)[0]
    h2 = np.histogram(hsv2[:,:,0].flatten(), bins=bins, range=(0,1), density=True)[0]
    denom = (np.linalg.norm(h1)*np.linalg.norm(h2) + 1e-8)
    sim = np.dot(h1, h2) / denom
    return float(max(0.0, min(1.0, sim)))

# ---------------------------
# Other visual metrics
# ---------------------------
def texture_glcm_energy(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((224,224)))
    b = np.array(ImageOps.grayscale(img2).resize((224,224)))
    try:
        glcm1 = graycomatrix(a, [1], [0], symmetric=True, normed=True)
        glcm2 = graycomatrix(b, [1], [0], symmetric=True, normed=True)
        e1 = graycoprops(glcm1, 'energy')[0,0]
        e2 = graycoprops(glcm2, 'energy')[0,0]
        sim = 1.0 - abs(e1 - e2)
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0

def brightness_similarity(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(float)
    sim = 1.0 - abs(a.mean() - b.mean()) / 255.0
    return float(max(0.0, min(1.0, sim)))

def edge_sobel_correlation(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(float)
    e1 = cv2.Sobel(a, cv2.CV_64F, 1, 1, ksize=3).flatten()
    e2 = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3).flatten()
    if np.std(e1) < 1e-8 or np.std(e2) < 1e-8:
        return 0.0
    corr = pearsonr(e1, e2)[0]
    return float(max(0.0, min(1.0, (corr + 1.0) / 2.0)))

def pattern_fft_correlation(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(ImageOps.grayscale(img1).resize((256,256))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((256,256))).astype(float)
    f1 = np.log1p(np.abs(fft2(a))).flatten()
    f2 = np.log1p(np.abs(fft2(b))).flatten()
    if np.std(f1) < 1e-8 or np.std(f2) < 1e-8:
        return 0.0
    corr = pearsonr(f1, f2)[0]
    return float(max(0.0, min(1.0, (corr + 1.0) / 2.0)))

# ---------------------------
# Palette extraction + plotting (10 colors, compact bars with hex labels)
# ---------------------------
@st.cache_data(show_spinner=False)
def extract_palette(img: Image.Image, n_colors: int = 10) -> List[tuple]:
    """Return list of RGB tuples (0-255) representing palette (ordered by cluster size)."""
    arr = np.array(img.convert("RGB").resize((200, 200))).reshape(-1, 3).astype(np.float32)
    # Remove near-transparent / degenerate if any
    # KMeans from sklearn
    try:
        km = KMeans(n_clusters=min(n_colors, len(arr)//50 or 1), n_init=10, random_state=1)
        labels = km.fit_predict(arr)
        centers = km.cluster_centers_.astype(int)
        # order by cluster frequency
        counts = np.bincount(labels)
        order = np.argsort(-counts)
        ordered = [tuple(map(int, centers[i])) for i in order]
        # if fewer than requested, pad with background average
        while len(ordered) < n_colors:
            ordered.append(tuple(map(int, arr.mean(axis=0))))
        return ordered[:n_colors]
    except Exception:
        # fallback: simple histogram-based palette
        vals, counts = np.unique(arr.reshape(-1,3), axis=0, return_counts=True)
        if len(vals) == 0:
            return [(128,128,128)] * n_colors
        order = np.argsort(-counts)
        colors = [tuple(map(int, vals[i])) for i in order[:n_colors]]
        while len(colors) < n_colors:
            colors.append(colors[-1])
        return colors[:n_colors]

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def plot_palettes(query_img: Image.Image, ref_img: Image.Image, n_colors: int = 10):
    q_colors = extract_palette(query_img, n_colors)
    r_colors = extract_palette(ref_img, n_colors)
    # create compact horizontal palette image (two rows)
    fig, axes = plt.subplots(2, 1, figsize=(6, 1.0), dpi=100)
    for ax, colors, title in zip(axes, [q_colors, r_colors], ["Query palette", "Reference palette"]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        total = len(colors)
        for i, c in enumerate(colors):
            rect = patches.Rectangle((i/total, 0), 1/total, 1, facecolor=np.array(c)/255.0, transform=ax.transAxes)
            ax.add_patch(rect)
            # overlay hex
            hexc = rgb_to_hex(c)
            ax.text((i+0.5)/total, 0.5, hexc, ha='center', va='center',
                    fontsize=6, color='white' if (0.299*c[0]+0.587*c[1]+0.114*c[2]) < 150 else 'black',
                    transform=ax.transAxes)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_title(title, fontsize=8, pad=2)
    plt.tight_layout(pad=0.2)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------------------
# Main UI
# ---------------------------
st.sidebar.header("Options")
top_n = st.sidebar.selectbox("Top matches to display", options=[3,5,10], index=1)
n_palette = st.sidebar.slider("Palette colors (compact)", min_value=5, max_value=20, value=10, step=1)

st.sidebar.markdown("**Performance**")
thumbnail_max = st.sidebar.slider("Downscale before processing (pixels max)", 256, 1024, 512, step=64)

st.subheader("1) Upload a ZIP of reference images")
ref_zip = st.file_uploader("ZIP file containing reference images (folders allowed)", type=["zip"])
ref_folder = None
ref_image_paths = []

if ref_zip:
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(ref_zip, "r") as z:
        z.extractall(tmpdir)
    # recursively collect images
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    for p in Path(tmpdir).rglob("*"):
        if p.suffix.lower() in valid_ext and not p.name.startswith("."):
            ref_image_paths.append(p)
    if len(ref_image_paths) == 0:
        st.error("No valid images found inside the ZIP.")
    else:
        st.success(f"Found {len(ref_image_paths)} images in ZIP.")

st.subheader("2) Upload query image")
query_file = st.file_uploader("Query image", type=["jpg","jpeg","png","bmp","tiff","webp"])

if ref_image_paths and query_file:
    # Process reference images robustly and show progress
    st.info("Processing reference images (extracting deep features). This may take a minute.")
    ref_features = []
    processed_paths = []
    progress = st.progress(0)
    total = len(ref_image_paths)
    for i, p in enumerate(ref_image_paths, start=1):
        img = safe_open_image(p)
        if img is None:
            progress.progress(i/total)
            continue
        # optionally downscale for speed
        img.thumbnail((thumbnail_max, thumbnail_max))
        try:
            feat = extract_deep_features(img)
            ref_features.append(feat)
            processed_paths.append(str(p))
        except Exception as e:
            st.warning(f"Skipping {p.name}: {e}")
        progress.progress(i/total)
    progress.empty()
    st.success(f"Processed {len(processed_paths)} / {total} images.")

    if len(processed_paths) == 0:
        st.error("No reference images were successfully processed.")
        st.stop()

    # Build FAISS index
    feats = np.array(ref_features).astype("float32")
    index = faiss.IndexFlatL2(feats.shape[1])
    index.add(feats)

    # open query image
    q_img = safe_open_image(query_file)
    q_img.thumbnail((thumbnail_max, thumbnail_max))
    q_feat = extract_deep_features(q_img)

    # search
    k = min(top_n, len(processed_paths))
    D, I = index.search(np.array([q_feat]), k=k)

    st.subheader("Results")
    st.image(q_img, caption="Query Image", use_container_width=True)

    # weights for overall score (rebalanced)
    # color submetrics share a combined weight but show individually
    WEIGHTS = {
        "ciede": 0.12,
        "hist": 0.10,
        "hue": 0.08,
        "texture": 0.20,
        "brightness": 0.10,
        "edges": 0.20,
        "pattern": 0.20
    }
    # normalize weights (just in case)
    s = sum(WEIGHTS.values())
    for k_ in WEIGHTS: WEIGHTS[k_] /= s

    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        ref_path = processed_paths[idx]
        ref_img = safe_open_image(ref_path)
        if ref_img is None:
            continue

        # compute all metrics
        ciede = ciede2000_similarity(q_img, ref_img)
        hist_corr = histogram_correlation(q_img, ref_img)
        hue_sim = hue_similarity(q_img, ref_img)
        texture = texture_glcm_energy(q_img, ref_img)
        bright = brightness_similarity(q_img, ref_img)
        edges = edge_sobel_correlation(q_img, ref_img)
        pattern = pattern_fft_correlation(q_img, ref_img)

        # clamp metrics to [0,1]
        metrics = {
            "CIEDE2000": float(np.clip(ciede, 0.0, 1.0)),
            "Histogram": float(np.clip(hist_corr, 0.0, 1.0)),
            "Hue": float(np.clip(hue_sim, 0.0, 1.0)),
            "Texture": float(np.clip(texture, 0.0, 1.0)),
            "Brightness": float(np.clip(bright, 0.0, 1.0)),
            "Edges": float(np.clip(edges, 0.0, 1.0)),
            "Pattern": float(np.clip(pattern, 0.0, 1.0))
        }

        # overall weighted score
        overall = (
            metrics["CIEDE2000"] * WEIGHTS["ciede"] +
            metrics["Histogram"] * WEIGHTS["hist"] +
            metrics["Hue"] * WEIGHTS["hue"] +
            metrics["Texture"] * WEIGHTS["texture"] +
            metrics["Brightness"] * WEIGHTS["brightness"] +
            metrics["Edges"] * WEIGHTS["edges"] +
            metrics["Pattern"] * WEIGHTS["pattern"]
        ) * 100.0

        # technical explanation (mention top 2 metrics)
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = sorted_metrics[0], sorted_metrics[1]
        explanation = (
            f"Technical summary: high similarity in {top1[0]} ({top1[1]:.2f}) and {top2[0]} ({top2[1]:.2f}). "
            f"Overall score: {overall:.1f}% (distance {dist:.4f})."
        )

        # Layout: image pair (query left, reference right), then compact palettes, then metric bars + explanation
        st.markdown(f"### Match {rank} — Overall: **{overall:.1f}%**")
        cols = st.columns([1,1])
        with cols[0]:
            st.image(q_img, caption="Query", use_container_width=True)
        with cols[1]:
            st.image(ref_img, caption=os.path.basename(ref_path), use_container_width=True)

        # palettes directly below image pair
        pal_buf = plot_palettes(q_img, ref_img, n_colors=n_palette)
        st.image(pal_buf, use_container_width=True)

        # metric bar chart (compact) using matplotlib
        fig, ax = plt.subplots(figsize=(6, 1.6), dpi=100)
        names = list(metrics.keys())
        vals = [metrics[n] for n in names]
        bars = ax.barh(range(len(names)), vals, color="#888888", height=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Similarity", fontsize=9)
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{vals[i]:.2f}", va='center', fontsize=8)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig)

        st.markdown(f"{explanation}")

else:
    st.info("Upload a reference ZIP (left) and a query image (above) to run the analysis.")

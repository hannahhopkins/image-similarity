#!/usr/bin/env python3
# app.py
# Streamlit Image Similarity Analyzer

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
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Image Similarity", layout="wide")
st.title("Image Similarity")
st.write("Upload a ZIP of reference images and a query image. Results show metric bars (Plotly), radar, palettes and intersection palettes.")

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Options")
    TOP_N = st.selectbox("Top matches to display", [3, 5, 10], index=1)
    N_PALETTE = st.slider("Palette colors (k-means)", 5, 20, 10, 1)
    N_INTERSECT = st.slider("Intersection palette size", 3, 7, 5, 1)
    SHOW_ALL_INTERSECT = st.checkbox("Auto compute intersection palettes for all matches", value=False)
    THUMBNAIL_MAX = st.slider("Downscale images before processing (px)", 256, 1024, 512, 64)
    st.markdown("---")
    st.caption("Plotly bar chart shows metric name, value, definition, and interpretation on hover.")

# -------------------------
# Model loading
# -------------------------
@st.cache_resource
def load_resnet():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier head
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

# -------------------------
# Helpers: safe open, zip extract
# -------------------------
def safe_open(path_or_file) -> Image.Image | None:
    try:
        if isinstance(path_or_file, (str, Path)):
            img = Image.open(path_or_file)
        else:
            img = Image.open(path_or_file)
        img = img.convert("RGB")
        img.thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX))
        return img
    except UnidentifiedImageError:
        st.warning(f"Skipping {getattr(path_or_file, 'name', path_or_file)} — not a valid image.")
        return None
    except Exception as e:
        st.warning(f"Skipping {getattr(path_or_file, 'name', path_or_file)} — {e}")
        return None

def extract_zip_to_temp(uploaded_zip) -> List[str]:
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as z:
        z.extractall(tmp)
    image_files = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.startswith("."):
                continue
            if Path(f).suffix.lower() in valid_ext:
                image_files.append(os.path.join(root, f))
    if not image_files:
        st.warning("No valid images found in uploaded ZIP.")
    else:
        st.success(f"Found {len(image_files)} images in ZIP.")
    return image_files

# -------------------------
# Palette extraction (k-means)
# -------------------------
def extract_palette_kmeans(img: Image.Image, n_colors: int = 10) -> Tuple[List[Tuple[int,int,int]], List[int]]:
    arr = np.array(img.convert("RGB").resize((200,200))).reshape(-1, 3).astype(np.float32)
    if arr.shape[0] == 0:
        return [], []
    k = min(n_colors, max(1, arr.shape[0] // 50))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers_ordered = [tuple(map(int, centers[i])) for i in order]
    counts_ordered = [int(counts[i]) for i in order]
    while len(centers_ordered) < n_colors:
        centers_ordered.append(tuple(map(int, arr.mean(axis=0))))
        counts_ordered.append(0)
    return centers_ordered[:n_colors], counts_ordered[:n_colors]

# -------------------------
# Color helpers (Lab conversions)
# -------------------------
def lab_array_from_rgb_list(rgb_list: List[Tuple[int,int,int]]) -> np.ndarray:
    arr = np.array(rgb_list, dtype=float).reshape(-1,1,3)/255.0
    from skimage.color import rgb2lab
    lab = rgb2lab(arr)
    return lab.reshape(len(rgb_list), 3)

# -------------------------
# Intersection palettes
# -------------------------
def blended_midpoint_palette(q_colors, r_colors, n_out=5):
    q_lab = lab_array_from_rgb_list(q_colors)
    r_lab = lab_array_from_rgb_list(r_colors)
    out = []
    from skimage.color import lab2rgb
    for i in range(min(n_out, len(q_colors))):
        ql = q_lab[i]
        dists = np.linalg.norm(r_lab - ql, axis=1)
        if len(dists)==0:
            out.append(q_colors[i])
            continue
        j = np.argmin(dists)
        mid = (ql + r_lab[j])/2.0
        rgb = lab2rgb(mid.reshape(1,1,3)).reshape(3,) * 255.0
        out.append(tuple(int(np.clip(x,0,255)) for x in rgb))
    while len(out) < n_out:
        out.append(q_colors[0])
    return out

def shared_hue_palette(q_colors, r_colors, n_out=5):
    from skimage.color import rgb2hsv
    q_hues = [rgb2hsv(np.array(c).reshape(1,1,3)/255.0)[0,0,0] for c in q_colors]
    r_hues = [rgb2hsv(np.array(c).reshape(1,1,3)/255.0)[0,0,0] for c in r_colors]
    pairs = []
    for i,qh in enumerate(q_hues):
        diffs = [min(abs(qh-rh), 1-abs(qh-rh)) for rh in r_hues] if len(r_hues)>0 else [1.0]
        j = int(np.argmin(diffs))
        pairs.append((i,j,diffs[j]))
    pairs_sorted = sorted(pairs, key=lambda x: x[2])[:n_out]
    out = []
    for i,j,_ in pairs_sorted:
        qc = np.array(q_colors[i]).astype(float)
        rc = np.array(r_colors[j]).astype(float) if j < len(r_colors) else qc
        avg = ((qc + rc)/2.0).astype(int)
        out.append(tuple(np.clip(avg,0,255)))
    while len(out) < n_out:
        out.append(q_colors[0])
    return out

def weighted_hybrid_palette(q_colors, q_counts, r_colors, r_counts, n_out=5):
    q_lab = lab_array_from_rgb_list(q_colors)
    r_lab = lab_array_from_rgb_list(r_colors)
    out = []
    from skimage.color import lab2rgb
    for i in range(min(n_out, len(q_colors))):
        ql = q_lab[i]
        qcount = q_counts[i] if i < len(q_counts) else 1
        if len(r_lab)==0:
            rgb = lab2rgb(ql.reshape(1,1,3)).reshape(3,) * 255.0
            out.append(tuple(int(np.clip(x,0,255)) for x in rgb))
            continue
        dists = np.linalg.norm(r_lab - ql, axis=1)
        j = int(np.argmin(dists))
        rcount = r_counts[j] if j < len(r_counts) else 1
        total = qcount + rcount
        weighted = (ql * qcount + r_lab[j] * rcount) / max(total,1)
        rgb = lab2rgb(weighted.reshape(1,1,3)).reshape(3,) * 255.0
        out.append(tuple(int(np.clip(x,0,255)) for x in rgb))
    while len(out) < n_out:
        out.append(q_colors[0])
    return out

# -------------------------
# Palette plotting (matplotlib -> buffer)
# -------------------------
def palette_image_from_colors(colors: List[Tuple[int,int,int]], hex_labels=True, height=40):
    n = len(colors)
    fig, ax = plt.subplots(figsize=(6, 0.4), dpi=100)
    for i,c in enumerate(colors):
        ax.add_patch(plt.Rectangle((i/n,0), 1/n, 1, color=np.array(c)/255.0))
        if hex_labels:
            hexv = "#{:02x}{:02x}{:02x}".format(*c)
            ax.text((i+0.5)/n, 0.5, hexv, ha='center', va='center', fontsize=7,
                    color='white' if (0.299*c[0]+0.587*c[1]+0.114*c[2]) < 150 else 'black',
                    transform=ax.transAxes)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    buf = BytesIO()
    plt.tight_layout(pad=0.2)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------------
# Metric computations (normalized to 0..1)
# -------------------------
def ciede2000_similarity(img1, img2):
    a = np.array(img1.resize((128,128))).astype(float)
    b = np.array(img2.resize((128,128))).astype(float)
    lab1 = rgb2lab(a/255.0)
    lab2 = rgb2lab(b/255.0)
    delta = np.linalg.norm(lab1 - lab2, axis=2)
    mean_de = np.nanmean(delta)
    sim = max(0.0, min(1.0, 1.0 - (mean_de/60.0)))
    return float(sim)

def histogram_correlation(img1, img2, bins=8):
    a = np.array(img1.resize((224,224)))
    b = np.array(img2.resize((224,224)))
    h1 = cv2.calcHist([a], [0,1,2], None, [bins]*3, [0,256]*3)
    h2 = cv2.calcHist([b], [0,1,2], None, [bins]*3, [0,256]*3)
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    val = cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, val)))

def hue_similarity(img1, img2, bins=36):
    h1 = rgb2hsv(np.array(img1.resize((224,224)))/255.0)[:,:,0].flatten()
    h2 = rgb2hsv(np.array(img2.resize((224,224)))/255.0)[:,:,0].flatten()
    hh1,_ = np.histogram(h1, bins=bins, range=(0,1), density=True)
    hh2,_ = np.histogram(h2, bins=bins, range=(0,1), density=True)
    denom = (np.linalg.norm(hh1)*np.linalg.norm(hh2)+1e-8)
    return float(max(0.0, min(1.0, np.dot(hh1, hh2)/denom)))

def texture_glcm_energy(img1, img2):
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

def brightness_similarity(img1, img2):
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(float)
    sim = 1.0 - abs(a.mean() - b.mean()) / 255.0
    return float(max(0.0, min(1.0, sim)))

def edge_sobel_similarity(img1, img2):
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(float)
    e1 = cv2.Sobel(a, cv2.CV_64F, 1, 1, ksize=3).flatten()
    e2 = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3).flatten()
    if np.std(e1) < 1e-8 or np.std(e2) < 1e-8:
        return 0.0
    corr = pearsonr(e1, e2)[0]
    return float(max(0.0, min(1.0, (corr + 1.0)/2.0)))

def pattern_fft_similarity(img1, img2):
    a = np.array(ImageOps.grayscale(img1).resize((256,256))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((256,256))).astype(float)
    f1 = np.log1p(np.abs(fft2(a))).flatten()
    f2 = np.log1p(np.abs(fft2(b))).flatten()
    if np.std(f1) < 1e-8 or np.std(f2) < 1e-8:
        return 0.0
    corr = pearsonr(f1, f2)[0]
    return float(max(0.0, min(1.0, (corr + 1.0)/2.0)))

def entropy_similarity(img1, img2):
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(np.uint8)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(np.uint8)
    e1 = shannon_entropy(a)
    e2 = shannon_entropy(b)
    sim = 1.0 - abs(e1 - e2) / 8.0
    return float(max(0.0, min(1.0, sim)))

def ssim_similarity(img1, img2):
    a = np.array(ImageOps.grayscale(img1).resize((224,224))).astype(float)
    b = np.array(ImageOps.grayscale(img2).resize((224,224))).astype(float)
    dr = b.max() - b.min()
    if dr <= 0:
        dr = 1.0
    score = ssim(a, b, data_range=dr)
    return float(max(0.0, min(1.0, score)))

# -------------------------
# Metric info dictionary (definition + interpret)
# -------------------------
METRIC_INFO = {
    "SSIM": {
        "definition": "Structural Similarity Index — compares luminance, contrast and local structure.",
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
        "definition": "Mean luminance comparison — global brightness and exposure similarity.",
        "interpret": lambda v: (
            "Similar brightness and tonal exposure."
            if v > 0.85 else
            "Moderate proximity in overall lightness."
            if v > 0.6 else
            "Different lighting/exposure levels."
        )
    },
    "Histogram": {
        "definition": "Color histogram correlation — compares RGB distribution across channels.",
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
        "definition": "Perceptual color distance approximated in Lab space (lower deltaE implies closer).",
        "interpret": lambda v: (
            "Perceptual color match is strong."
            if v > 0.85 else
            "Moderate perceptual color similarity."
            if v > 0.6 else
            "Perceptual color differences are noticeable."
        )
    }
}

# -------------------------
# Plotly bar: clean horizontal with tooltip and color scale
# -------------------------
def plotly_metric_bar(metrics: dict, metric_order: List[str]):
    # Prepare DataFrame-like lists
    names = []
    vals = []
    texts = []
    for name in metric_order:
        v = float(metrics.get(name, 0.0))
        info = METRIC_INFO.get(name, {})
        defn = info.get("definition", "")
        interp = info.get("interpret", lambda x: "")(v)
        tooltip = f"<b>{name}</b><br>{defn}<br><i>{interp}</i><br>Score: {v:.2f}"
        names.append(name)
        vals.append(v)
        texts.append(tooltip)
    # Build plotly figure
    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation='h',
        text=[f"{v:.2f}" for v in vals],
        textposition='outside',
        marker=dict(color=vals, colorscale='RdYlGn', cmin=0, cmax=1, colorbar=dict(title="Similarity", x=1.02)),
        hovertemplate=texts
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1], title="Similarity", tickformat=".2f"),
        margin=dict(l=80, r=60, t=20, b=20),
        height= (30 * len(names)) + 60,
        yaxis=dict(autorange="reversed"),  # keep highest on top
    )
    return fig

# -------------------------
# Radar helper (Plotly)
# -------------------------
def plotly_radar(metrics_query: dict, metrics_match: dict, metric_order: List[str], match_label="Match"):
    q_vals = [metrics_query.get(m, 0.0) for m in metric_order]
    m_vals = [metrics_match.get(m, 0.0) for m in metric_order]
    categories = metric_order + [metric_order[0]]
    q_plot = q_vals + [q_vals[0]]
    m_plot = m_vals + [m_vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=q_plot, theta=categories, fill='toself', name='Query', line=dict(color='gray')))
    fig.add_trace(go.Scatterpolar(r=m_plot, theta=categories, fill='toself', name=match_label, line=dict(color='crimson')))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), showlegend=True, margin=dict(l=20,r=20,t=20,b=20), height=350)
    return fig

# -------------------------
# Main UI: upload + processing
# -------------------------
st.subheader("1) Upload")
ref_zip = st.file_uploader("Reference folder (ZIP)", type=["zip"])
query_file = st.file_uploader("Query image", type=["jpg","jpeg","png","bmp","tiff","webp"])

if ref_zip and query_file:
    st.write("✅ ZIP uploaded:", getattr(ref_zip,"name","reference.zip"))
    st.write("✅ Query image uploaded:", getattr(query_file,"name","query"))
    ref_paths = extract_zip_to_temp(ref_zip)
    st.write(f"Found {len(ref_paths)} candidate images in ZIP.")

    query_img = safe_open(query_file)
    if query_img is None:
        st.error("Could not open query image.")
        st.stop()

    # extract deep features for references
    st.info("Extracting features from reference images — this can take a moment.")
    feats = []
    processed_paths = []
    progress = st.progress(0)
    for i,p in enumerate(ref_paths, start=1):
        img = safe_open(p)
        if img is None:
            progress.progress(i/len(ref_paths))
            continue
        try:
            feat = extract_deep_features(img)
            feats.append(feat)
            processed_paths.append(p)
        except Exception as e:
            st.warning(f"Skipping {p}: {e}")
        progress.progress(i/len(ref_paths))
    progress.empty()
    st.success(f"Processed {len(processed_paths)} reference images.")

    if len(processed_paths) == 0:
        st.error("No reference images successfully processed.")
        st.stop()

    # FAISS search
    feats_np = np.array(feats).astype("float32")
    index = faiss.IndexFlatL2(feats_np.shape[1])
    index.add(feats_np)
    q_feat = extract_deep_features(query_img)
    k = min(TOP_N, len(processed_paths))
    D, I = index.search(np.array([q_feat]), k=k)

    st.subheader("Results")
    st.image(query_img, caption="Query Image", use_container_width=True)

    # compute metrics for each match and display in vertical flow
    metric_order = ["SSIM","Texture","Edges","Brightness","Histogram","Hue","Pattern","Entropy","CIEDE2000"]
    query_metrics_template = {}  # for radar: treat query compared to itself (all 1s)
    for m in metric_order:
        query_metrics_template[m] = 1.0

    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        ref_path = processed_paths[idx]
        ref_img = safe_open(ref_path)
        if ref_img is None:
            continue
        ref_img_resized = ref_img.resize(query_img.size)

        # compute metrics
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

        # UI: vertical match section
        st.markdown(f"## Match {rank}: **{os.path.basename(ref_path)}** — Overall {overall:.1f}% (distance {float(dist):.4f})")
        cols = st.columns([1,1])
        with cols[0]:
            st.image(query_img, caption="Query", use_container_width=True)
        with cols[1]:
            st.image(ref_img_resized, caption=f"Match {rank}", use_container_width=True)

        # Plotly bar chart (compact) with hover containing definition + interpretive text
        bar_fig = plotly_metric_bar(metrics, metric_order)
        st.plotly_chart(bar_fig, use_container_width=True)

        # radar (query vs match)
        radar_fig = plotly_radar(query_metrics_template, metrics, metric_order, match_label=f"Match {rank}")
        st.plotly_chart(radar_fig, use_container_width=True)

        # metric explanations section
        st.markdown("### Metric definitions & interpretations")
        for name in metric_order:
            info = METRIC_INFO.get(name, {})
            val = metrics.get(name, 0.0)
            defn = info.get("definition","")
            interp = info.get("interpret", lambda x: "")(val)
            st.markdown(f"**{name}** — {defn}")
            st.caption(f"{interp} (score: {val:.2f})")

        # intersection palettes: top match always; others optionally
        if rank == 1 or SHOW_ALL_INTERSECT:
            show_inter = True
        else:
            show_inter = st.checkbox(f"Show intersection palettes for Match {rank}?", key=f"inter_{rank}")

        if show_inter:
            # compute palettes for query & reference
            q_centers, q_counts = extract_palette_kmeans(query_img, N_PALETTE)
            r_centers, r_counts = extract_palette_kmeans(ref_img_resized, N_PALETTE)
            blended = blended_midpoint_palette(q_centers, r_centers, n_out=N_INTERSECT)
            shared = shared_hue_palette(q_centers, r_centers, n_out=N_INTERSECT)
            hybrid = weighted_hybrid_palette(q_centers, q_counts, r_centers, r_counts, n_out=N_INTERSECT)

            st.markdown("#### Intersection Palettes")
            p1, p2, p3 = st.columns([1,1,1])
            with p1:
                buf = palette_image_from_colors(blended, hex_labels=True)
                st.image(buf, caption=f"Blended Midpoint ({N_INTERSECT}) — Balanced blend of shared chromatic tendencies")
            with p2:
                buf = palette_image_from_colors(shared, hex_labels=True)
                st.image(buf, caption=f"Shared Hues ({N_INTERSECT}) — Common dominant tones across both palettes")
            with p3:
                buf = palette_image_from_colors(hybrid, hex_labels=True)
                st.image(buf, caption=f"Weighted Hybrid ({N_INTERSECT}) — Weighted by cluster prominence (dynamic energy)")

        st.markdown("---")

else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

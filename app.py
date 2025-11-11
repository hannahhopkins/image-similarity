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
# Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide")
st.title("Image Similarity Analyzer")


# -----------------------------------------------------------------------------
# Small Inline Hover Tooltip Icon
# -----------------------------------------------------------------------------
def info_icon(text):
    return (
        f"<span style='font-size:0.85em; cursor:help; color:#666;' "
        f"title='{text}'>ⓘ</span>"
    )


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Settings")

top_k = st.sidebar.slider("Number of matches to display", 1, 12, 5)
st.sidebar.caption("How many closest reference matches to show.")

num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 6)
st.sidebar.caption("Number of dominant colors extracted for palette visualization.")

st.sidebar.markdown("---")
st.sidebar.subheader("Hue Similarity Settings")

hue_bins = st.sidebar.slider("Hue bins (granularity)", 12, 72, 36, step=6)
sat_thresh = st.sidebar.slider("Saturation threshold (0–1)", 0.0, 1.0, 0.15, step=0.01)
val_thresh = st.sidebar.slider("Brightness threshold (0–1)", 0.0, 1.0, 0.15, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Palette Weight")
hybrid_query_weight = st.sidebar.slider("Query influence", 0.0, 1.0, 0.6, step=0.05)
st.sidebar.caption("0 = lean toward reference colors, 1 = lean toward query colors.")


# -----------------------------------------------------------------------------
# Image IO Helpers
# -----------------------------------------------------------------------------
def safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None


def iter_zip_images(folder):
    for root, _, files in os.walk(folder):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png"):
                yield os.path.join(root, f)


def analysis_resize(pil_img, target=512):
    w, h = pil_img.size
    scale = target / max(w, h)
    return pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)


# -----------------------------------------------------------------------------
# Palette Extraction
# -----------------------------------------------------------------------------
def kmeans_palette(pil_img, k):
    arr = np.array(pil_img)
    flat = arr.reshape(-1, 3).astype(np.float32)
    k_eff = min(k, max(1, flat.shape[0]//200))
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit(flat)
    centers = km.cluster_centers_
    labels = km.labels_
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    if centers.shape[0] < k:
        pad = np.tile(np.mean(flat, axis=0), (k - centers.shape[0], 1))
        centers = np.vstack([centers, pad])
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    return [tuple(map(int,c)) for c in centers[:k]]


# -----------------------------------------------------------------------------
# Hue Histogram Computation
# -----------------------------------------------------------------------------
def rgb_to_hsv01(arr_uint8):
    hsv = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,0] /= 179.0
    hsv[...,1] /= 255.0
    hsv[...,2] /= 255.0
    return hsv


def hue_histogram(pil_img, bins, s_thr, v_thr):
    arr = np.array(pil_img)
    hsv = rgb_to_hsv01(arr)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    mask = (S >= s_thr) & (V >= v_thr)
    if not np.any(mask):
        return np.ones(bins) / bins
    hist, _ = np.histogram(H[mask], bins=bins, range=(0,1))
    hist = hist.astype(np.float32)
    hist = hist / hist.sum() if hist.sum() > 0 else np.ones(bins)/bins
    return hist


# -----------------------------------------------------------------------------
# Similarity Metrics (Size Independent)
# -----------------------------------------------------------------------------
def structural_similarity_metric(a, b):
    a = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.array(b), cv2.COLOR_RGB2GRAY)
    dr = max(b.max() - b.min(), 1e-6)
    return float(np.clip(ssim(a, b, data_range=dr), 0, 1))


def color_hist_similarity(a, b):
    a = np.array(a); b = np.array(b)
    h1 = cv2.calcHist([a],[0,1,2],None,[8,8,8],[0,256]*3)
    h2 = cv2.calcHist([b],[0,1,2],None,[8,8,8],[0,256]*3)
    cv2.normalize(h1,h1); cv2.normalize(h2,h2)
    raw = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
    return float(np.clip((raw+1)/2, 0, 1))


def entropy_similarity(a, b):
    def ent(pil):
        g = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([g],[0],None,[256],[0,256]).astype(np.float32)
        p = hist / (hist.sum()+1e-8)
        return float(-np.sum(p*np.log2(p+1e-12)))
    e1, e2 = ent(a), ent(b)
    return float(np.clip(1 - abs(e1-e2)/max(e1,e2,1e-6),0,1))


def edge_complexity_similarity(a, b):
    def ed(pil):
        g = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g,100,200)
        return float(e.mean()/255.0)
    return float(np.clip(1 - abs(ed(a)-ed(b)), 0, 1))


def texture_correlation_similarity(a, b):
    def tex(pil):
        g = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
        gl = graycomatrix(g, [2], [0], symmetric=True, normed=True)
        return float(graycoprops(gl,'energy')[0,0])
    e1, e2 = tex(a), tex(b)
    return float(np.clip(1 - abs(e1-e2)/max(e1,e2,1e-6), 0, 1))


def hue_distribution_similarity(a, b, bins, s_thr, v_thr):
    h1 = hue_histogram(a, bins, s_thr, v_thr)
    h2 = hue_histogram(b, bins, s_thr, v_thr)
    denom = (np.linalg.norm(h1)*np.linalg.norm(h2))+1e-10
    return float(np.dot(h1,h2)/denom)


# -----------------------------------------------------------------------------
# Palette Intersection Models
# -----------------------------------------------------------------------------
def lab_midpoint_palette(q_cols, r_cols, n):
    out=[]
    for qc in q_cols[:n]:
        rc = min(r_cols, key=lambda x: np.linalg.norm(np.array(x)-np.array(qc)))
        qlab=cv2.cvtColor(np.uint8([[qc]]),cv2.COLOR_RGB2LAB).astype(np.float32)
        rlab=cv2.cvtColor(np.uint8([[rc]]),cv2.COLOR_RGB2LAB).astype(np.float32)
        mid=(qlab+rlab)/2
        rgb=cv2.cvtColor(mid.astype(np.uint8),cv2.COLOR_LAB2RGB)[0,0]
        out.append(tuple(int(x) for x in rgb))
    return out


def shared_hue_palette(q_img, r_img, n, bins, s_thr, v_thr):
    hq = hue_histogram(q_img, bins, s_thr, v_thr)
    hr = hue_histogram(r_img, bins, s_thr, v_thr)
    score = hq * hr + 1e-9
    centers = (np.linspace(0,1,bins+1)[:-1] + np.linspace(0,1,bins+1)[1:]) / 2
    best_idx = np.argsort(-score)[:n]
    out=[]
    for ci in best_idx:
        hc = centers[ci]
        hsv = np.array([[[hc*179.0, 200, 200]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0,0]
        out.append(tuple(int(x) for x in rgb))
    while len(out)<n:
        out.append(out[-1] if out else (128,128,128))
    return out


def weighted_hybrid_palette(q_cols, r_cols, n, w):
    out=[]
    for qc in q_cols[:n]:
        rc = min(r_cols, key=lambda x: np.linalg.norm(np.array(x)-np.array(qc)))
        qlab=cv2.cvtColor(np.uint8([[qc]]),cv2.COLOR_RGB2LAB).astype(np.float32)
        rlab=cv2.cvtColor(np.uint8([[rc]]),cv2.COLOR_RGB2LAB).astype(np.float32)
        blend=qlab*w+rlab*(1-w)
        rgb=cv2.cvtColor(blend.astype(np.uint8),cv2.COLOR_LAB2RGB)[0,0]
        out.append(tuple(int(x) for x in rgb))
    return out


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plotly_palette(colors, key):
    fig = go.Figure()
    for i,c in enumerate(colors):
        hexv=f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        fig.add_shape(type="rect",x0=i,x1=i+1,y0=0,y1=1,fillcolor=hexv,line=dict(width=0))
        fig.add_trace(go.Scatter(x=[i+0.5],y=[0.5],mode="markers",
                                 marker=dict(opacity=0),hovertext=[hexv],hoverinfo="text"))
    fig.update_layout(
        xaxis=dict(visible=False),yaxis=dict(visible=False),
        height=60, margin=dict(l=0,r=0,t=2,b=2)
    )
    st.plotly_chart(fig,use_container_width=True,key=key)


def metric_bar_chart(metrics, key):
    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]

    metric_info = {
        "Structural Alignment": "Similarity in tonal structure & spatial composition.",
        "Color Histogram": "Similarity in global color distribution.",
        "Entropy Similarity": "Similarity in visual detail density.",
        "Edge Complexity": "Similarity in sharpness and contour frequency.",
        "Texture Correlation": "Similarity in surface micro-patterns.",
        "Hue Distribution": "Similarity in dominant hue families once shadows & neutrals are excluded."
    }

    hover = [metric_info[n] for n in names]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=1),
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        hovertext=hover, hoverinfo="text"
    ))

    fig.update_layout(
        xaxis=dict(range=[0,1], title="Similarity (0–1)"),
        yaxis=dict(autorange="reversed"),
        height=max(240, 34*len(names)),
        margin=dict(l=80,r=20,t=10,b=10),
        template="simple_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# -----------------------------------------------------------------------------
# File Upload
# -----------------------------------------------------------------------------
uploaded_zip = st.file_uploader("Upload Reference Images (ZIP)", type=["zip"])
query_file = st.file_uploader("Upload Query Image", type=["jpg","jpeg","png"])


# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------
if uploaded_zip and query_file:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(uploaded_zip,"r") as z:
            z.extractall(tmp)

        ref_paths = list(iter_zip_images(tmp))
        if not ref_paths:
            st.error("No valid reference images found.")
            st.stop()

        query_img = safe_open_image(query_file)
        if query_img is None:
            st.error("Could not open query image.")
            st.stop()

        # Create analysis copies for structural metrics only
        query_analysis = analysis_resize(query_img, 512)

        results=[]
        for p in ref_paths:
            ref = safe_open_image(p)
            if ref is None: 
                continue

            # Resized copy for structural comparisons
            ref_analysis = analysis_resize(ref, 512)

            # All structural metrics use the resized versions
            qa, ra = query_analysis, ref_analysis

            metrics = {
                "Structural Alignment": structural_similarity_metric(qa, ra),
                "Color Histogram": color_hist_similarity(query_img, ref),
                "Entropy Similarity": entropy_similarity(query_img, ref),
                "Edge Complexity": edge_complexity_similarity(qa, ra),
                "Texture Correlation": texture_correlation_similarity(qa, ra),
                "Hue Distribution": hue_distribution_similarity(query_img, ref, hue_bins, sat_thresh, val_thresh),
            }

            score = float(np.mean(list(metrics.values())))
            results.append((p, ref, metrics, score))

        results.sort(key=lambda x:x[3], reverse=True)
        top = results[:top_k]

        st.subheader("Top Matches")

        for idx, (path, ref, metrics, score) in enumerate(top,1):
            c1, c2 = st.columns([2.6, 1.4])

            with c1:
                st.markdown(f"### Match {idx}: {os.path.basename(path)} — {score:.2f}")
                st.image([query_img, ref], caption=["Query", "Reference"], use_container_width=True)
                metric_bar_chart(metrics, f"bar_{idx}")

                with st.expander("Metric Explanations", expanded=False):
                    st.markdown(
                        """
                        Structural Alignment  
                        Measures how similarly the images distribute tonal contrast and spatial composition.

                        Color Histogram  
                        Compares the relative presence of color values throughout the image.

                        Entropy Similarity  
                        Indicates whether the images have comparable levels of visual complexity.

                        Edge Complexity  
                        Compares the density and sharpness of contour boundaries.

                        Texture Correlation  
                        Measures similarity of fine-scale surface structure and patterning.

                        Hue Distribution  
                        Compares dominant hue families after excluding neutrals and low-light regions.
                        """
                    )

            with c2:
                q_cols = kmeans_palette(query_img, num_colors)
                r_cols = kmeans_palette(ref, num_colors)

                mid = lab_midpoint_palette(q_cols, r_cols, min(num_colors,6))
                shared = shared_hue_palette(query_img, ref, min(num_colors,6), hue_bins, sat_thresh, val_thresh)
                hybrid = weighted_hybrid_palette(q_cols, r_cols, min(num_colors,6), hybrid_query_weight)

                st.markdown(
                    f"Blended Midpoint {info_icon('Perceptual mid-blend where the two palettes converge.')}",
                    unsafe_allow_html=True
                )
                plotly_palette(mid, f"mid_{idx}")

                st.markdown(
                    f"Shared Hue Regions {info_icon('Hue families strongly expressed in both images.')}",
                    unsafe_allow_html=True
                )
                plotly_palette(shared, f"shared_{idx}")

                st.markdown(
                    f"Weighted Hybrid {info_icon('Blends palettes according to the query weight slider.')}",
                    unsafe_allow_html=True
                )
                plotly_palette(hybrid, f"hyb_{idx}")

            st.markdown(
                "<hr style='margin-top:2rem;margin-bottom:2rem;border:none;border-top:1px solid #888;'/>",
                unsafe_allow_html=True
            )

else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

import os
import zipfile
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import cv2
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, lab2rgb
import plotly.graph_objects as go

st.set_page_config(
    page_title="Image Similarity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Image Similarity Analyzer")

st.write(
    "Upload a ZIP of reference images and a query image. This tool compares structural, color, texture, "
    "edge, entropy, and hue distribution similarity, and generates interaction palettes to summarize "
    "shared and blended chromatic qualities."
)

# ---------------- Sidebar Controls ---------------- #
st.sidebar.header("Display & Matching Settings")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
st.sidebar.caption("Controls how many closest matches to show.")

num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 7)
st.sidebar.caption("Number of representative colors extracted per image using k-means clustering.")

resize_refs = st.sidebar.checkbox("Resize reference images to match query", True)
st.sidebar.caption("Ensures same dimensions for metrics like SSIM.")

st.sidebar.markdown("---")
st.sidebar.subheader("Hue Similarity Settings")

hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, step=6)
st.sidebar.caption("Number of segments used to compare hue distributions.")

sat_thresh = st.sidebar.slider("Saturation mask threshold", 0, 255, 30, step=5)
st.sidebar.caption("Pixels below this saturation are ignored in hue metrics.")

val_thresh = st.sidebar.slider("Value mask threshold", 0, 255, 30, step=5)
st.sidebar.caption("Pixels below this brightness are ignored in hue metrics.")

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Palette")

q_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.6, 0.05)
st.sidebar.caption("Weight between query and reference palettes in hybrid blending.")

# ---------------- Safe image open ---------------- #
def safe_open_image(p):
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

# ---------------- Palette Extraction ---------------- #
def kmeans_palette(pil_img, k):
    arr = np.array(pil_img.convert("RGB"))
    flat = arr.reshape(-1, 3).astype(np.float32)
    k_eff = min(k, max(1, flat.shape[0] // 50))
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=0)
    labels = km.fit_predict(flat)
    centers = km.cluster_centers_
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    if centers.shape[0] < k:
        pad = np.tile(np.mean(flat, axis=0, keepdims=True), (k - centers.shape[0], 1))
        centers = np.vstack([centers, pad])
    centers = np.clip(centers[:k], 0, 255).astype(np.uint8)
    return centers

# ---------------- Metrics ---------------- #
def structural_similarity_metric(a, b):
    a = np.array(a.convert("L"))
    b = np.array(b.convert("L"))
    dr = max(b.max() - b.min(), 1)
    return float(np.clip(ssim(a, b, data_range=dr), 0, 1))

def histogram_rgb_correlation(a, b):
    a = np.array(a)
    b = np.array(b)
    h1 = cv2.calcHist([a], [0,1,2], None, [8,8,8], [0,256]*3)
    h2 = cv2.calcHist([b], [0,1,2], None, [8,8,8], [0,256]*3)
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    raw = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return float(np.clip((raw + 1) / 2, 0, 1))

def entropy_similarity(a, b):
    a = np.array(a.convert("L"))
    b = np.array(b.convert("L"))
    pa = cv2.calcHist([a],[0],None,[256],[0,256])
    pb = cv2.calcHist([b],[0],None,[256],[0,256])
    pa = pa / (pa.sum() + 1e-8)
    pb = pb / (pb.sum() + 1e-8)
    ea = -np.sum(pa*np.log2(pa+1e-12))
    eb = -np.sum(pb*np.log2(pb+1e-12))
    if max(ea,eb) <= 0:
        return 0.5
    return float(np.clip(1 - abs(ea-eb) / max(ea,eb), 0, 1))

def edge_density_similarity(a, b):
    a = cv2.Canny(np.array(a.convert("L")), 100, 200)
    b = cv2.Canny(np.array(b.convert("L")), 100, 200)
    return float(np.clip(1 - abs(a.mean()-b.mean())/255, 0, 1))

def texture_glcm_similarity(a, b):
    a = np.array(a.convert("L"))
    b = np.array(b.convert("L"))
    g1 = graycomatrix(a, [5], [0], symmetric=True, normed=True)
    g2 = graycomatrix(b, [5], [0], symmetric=True, normed=True)
    c1 = graycoprops(g1,'contrast')[0,0]
    c2 = graycoprops(g2,'contrast')[0,0]
    denom = max(c1,c2,1e-6)
    return float(np.clip(1 - abs(c1-c2)/denom, 0, 1))

def hue_distribution_similarity(a, b, bins, s_thr, v_thr):
    a = np.array(a)
    b = np.array(b)
    ha = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    hb = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
    mask_a = (ha[...,1] >= s_thr) & (ha[...,2] >= v_thr)
    mask_b = (hb[...,1] >= s_thr) & (hb[...,2] >= v_thr)
    ha = ha[...,0][mask_a]
    hb = hb[...,0][mask_b]
    if ha.size==0 or hb.size==0:
        return 0.5
    h1,_ = np.histogram(ha, bins=bins, range=(0,180))
    h2,_ = np.histogram(hb, bins=bins, range=(0,180))
    h1 = h1 / max(h1.sum(),1)
    h2 = h2 / max(h2.sum(),1)
    denom = (np.linalg.norm(h1)*np.linalg.norm(h2))+1e-12
    return float(np.dot(h1,h2)/denom)

# ---------------- Intersection Palettes ---------------- #
def nearest_lab_index(lab_arr, color_lab):
    d = np.linalg.norm(lab_arr - color_lab, axis=1)
    return np.argmin(d)

def blended_midpoint_palette(q_cols, r_cols, k):
    q_lab = rgb2lab(q_cols.reshape(-1,1,3)/255).reshape(-1,3)
    r_lab = rgb2lab(r_cols.reshape(-1,1,3)/255).reshape(-1,3)
    out=[]
    for i in range(min(k,len(q_lab))):
        j = nearest_lab_index(r_lab, q_lab[i])
        mid=(q_lab[i]+r_lab[j])/2
        out.append(lab2rgb(mid.reshape(1,1,3)).reshape(3,)*255)
    while len(out)<k: out.append(q_cols[0])
    return np.clip(np.array(out),0,255).astype(np.uint8)

def shared_hue_intersection_palette(q_cols, r_cols, k, bins, s_thr, v_thr):
    def valid_bin(c):
        h,s,v = cv2.cvtColor(c.reshape(1,1,3), cv2.COLOR_RGB2HSV)[0,0]
        if s < s_thr or v < v_thr:
            return None
        return min(bins-1, int(h/(180/bins)))
    qb, rb = {},{}
    for c in q_cols:
        idx = valid_bin(c)
        if idx is not None: qb.setdefault(idx,[]).append(c)
    for c in r_cols:
        idx = valid_bin(c)
        if idx is not None: rb.setdefault(idx,[]).append(c)
    shared = sorted(set(qb.keys()) & set(rb.keys()))
    out=[]
    for idx in shared:
        qc=np.mean(qb[idx],axis=0)
        rc=np.mean(rb[idx],axis=0)
        out.append((qc+rc)/2)
    if len(out)<k:
        return blended_midpoint_palette(q_cols,r_cols,k)
    return np.clip(np.array(out[:k]),0,255).astype(np.uint8)

def weighted_hybrid_palette(q_cols, r_cols, k, w):
    q_lab = rgb2lab(q_cols.reshape(-1,1,3)/255).reshape(-1,3)
    r_lab = rgb2lab(r_cols.reshape(-1,1,3)/255).reshape(-1,3)
    out=[]
    for i in range(min(k,len(q_lab))):
        j = nearest_lab_index(r_lab, q_lab[i])
        blend = w*q_lab[i] + (1-w)*r_lab[j]
        out.append(lab2rgb(blend.reshape(1,1,3)).reshape(3,)*255)
    while len(out)<k: out.append(q_cols[0])
    return np.clip(np.array(out),0,255).astype(np.uint8)

# ---------------- Plotly palette display ---------------- #
def plotly_palette(colors):
    if colors.size==0:
        return go.Figure()

    hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in colors]

    fig = go.Figure(
        go.Bar(
            x=[1]*len(colors),
            y=[1]*len(colors),
            marker=dict(color=hex_colors),
            hovertext=hex_colors,
            hoverinfo="text",
            orientation="h"
        )
    )
    fig.update_layout(
        height=60,
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        template="simple_white"
    )
    return fig

# ---------------- UI ---------------- #
uploaded_zip = st.file_uploader("Upload Reference Images (ZIP)", type=["zip"])
query_file = st.file_uploader("Upload Query Image", type=["jpg","jpeg","png"])

if uploaded_zip and query_file:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(uploaded_zip) as z:
            z.extractall(tmp)

        ref_paths=[]
        for root,_,files in os.walk(tmp):
            if "__MACOSX" in root: continue
            for f in files:
                if f.startswith("._"): continue
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    ref_paths.append(os.path.join(root,f))

        if not ref_paths:
            st.error("No valid images found in ZIP.")
            st.stop()

        query = safe_open_image(query_file)
        if query is None:
            st.error("Could not open query image.")
            st.stop()

        results=[]
        for p in ref_paths:
            img = safe_open_image(p)
            if img is None: continue
            if resize_refs: img = img.resize(query.size)

            metrics={
                "Structural Alignment": structural_similarity_metric(query,img),
                "Color Histogram": histogram_rgb_correlation(query,img),
                "Entropy Similarity": entropy_similarity(query,img),
                "Edge Complexity": edge_density_similarity(query,img),
                "Texture Correlation": texture_glcm_similarity(query,img),
                "Hue Distribution": hue_distribution_similarity(query,img,hue_bins,sat_thresh,val_thresh),
            }
            score = float(np.mean(list(metrics.values())))
            results.append((p,img,metrics,score))

        if not results:
            st.error("No valid comparisons produced.")
            st.stop()

        results.sort(key=lambda x: x[3], reverse=True)
        top = results[:top_k]

        st.subheader(f"Top {len(top)} Matches")

        for rank,(path,img,metrics,score) in enumerate(top,1):
            st.markdown(f"### Match {rank}: {os.path.basename(path)} — Overall {score:.2f}")

            col1,col2=st.columns([2.5,1.3])

            with col1:
                st.image([query,img], caption=["Query","Reference"], use_container_width=True)

                st.plotly_chart(
                    go.Figure(go.Bar(
                        x=list(metrics.values()),
                        y=list(metrics.keys()),
                        orientation="h",
                        text=[f"{v:.2f}" for v in metrics.values()],
                        textposition="outside",
                        marker=dict(color=list(metrics.values()), colorscale="RdYlGn", cmin=0, cmax=1)
                    )).update_layout(
                        xaxis=dict(range=[0,1], title="Similarity (0–1)"),
                        yaxis=dict(autorange="reversed"),
                        height=260,
                        margin=dict(l=100,r=40,t=20,b=20),
                        showlegend=False
                    ),
                    use_container_width=True
                )

                with st.expander("Metric Explanations."):
                    st.markdown("**Structural Alignment** — SSIM-based similarity of luminance/contrast/structure.")
                    st.markdown("**Color Histogram** — Comparison of overall RGB distribution shape.")
                    st.markdown("**Entropy Similarity** — Compares information complexity via luminance entropy.")
                    st.markdown("**Edge Complexity** — Compares contour/outline activity via Canny edge density.")
                    st.markdown("**Texture Correlation** — Compares micro-pattern structure using GLCM contrast.")
                    st.markdown("**Hue Distribution** — Compares masked hue histograms to assess dominant color families.")

            with col2:
                q_cols = kmeans_palette(query, num_colors)
                r_cols = kmeans_palette(img, num_colors)

                mid = blended_midpoint_palette(q_cols, r_cols, min(5,num_colors))
                inter = shared_hue_intersection_palette(q_cols, r_cols, min(5,num_colors), hue_bins, sat_thresh, val_thresh)
                hybrid = weighted_hybrid_palette(q_cols, r_cols, min(5,num_colors), q_weight)

                st.markdown("Blended Midpoint")
                st.plotly_chart(plotly_palette(mid), use_container_width=True)

                st.markdown("Shared Hue Intersection")
                st.plotly_chart(plotly_palette(inter), use_container_width=True)

                st.markdown("Weighted Hybrid")
                st.plotly_chart(plotly_palette(hybrid), use_container_width=True)

else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

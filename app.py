import os
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# --------------------------------------------------------
# Streamlit page setup
# --------------------------------------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("Image Similarity Analyzer")
st.write(
    "Upload a ZIP of reference images and a query image. The application compares visual structure, color environments, "
    "surface qualities, and generates palette intersection analyses."
)

# --------------------------------------------------------
# Sidebar Controls
# --------------------------------------------------------
st.sidebar.header("Display Controls")
top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
num_colors = st.sidebar.slider("Palette size (per image)", 3, 12, 6)
resize_refs = st.sidebar.checkbox("Resize reference images to match query", True)

st.sidebar.markdown("---")
st.sidebar.subheader("Hue Similarity Parameters")

hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, step=6)
st.sidebar.caption(
    "Higher = more precise color family comparison; lower = broader categories."
)

sat_thresh = st.sidebar.slider("Saturation threshold", 0.0, 1.0, 0.15, 0.01)
st.sidebar.caption("Low-saturation (grayish) pixels are ignored when comparing hue emphasis.")

val_thresh = st.sidebar.slider("Value threshold", 0.0, 1.0, 0.15, 0.01)
st.sidebar.caption("Very dark pixels are excluded, since hue is not visually interpretable there.")

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Palette")
hybrid_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.6, 0.05)
st.sidebar.caption(
    "Controls how strongly the hybrid palette stays near the query image vs. shifts toward the reference."
)

# --------------------------------------------------------
# Safe Image Loader
# --------------------------------------------------------
def load_rgb(obj):
    try:
        return Image.open(obj).convert("RGB")
    except:
        return None

# --------------------------------------------------------
# ZIP Extraction with MacOS Junk Filter
# --------------------------------------------------------
def load_reference_images(uploaded_zip):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(tmp)
    paths = []
    for root, _, files in os.walk(tmp):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, f))
    return paths

# --------------------------------------------------------
# Correct Lab Conversion
# --------------------------------------------------------
def to_lab(rgb_arr):
    rgb = (rgb_arr.astype(np.float32)/255.0).reshape(-1,1,3)
    return rgb2lab(rgb).reshape(-1,3).astype(np.float32)

def to_rgb(lab_arr):
    lab = lab_arr.reshape(-1,1,3).astype(np.float32)
    rgb = lab2rgb(lab).reshape(-1,3)
    return np.clip(rgb*255.0, 0, 255).astype(np.uint8)

# --------------------------------------------------------
# Palette Extraction (returns both colors and counts)
# --------------------------------------------------------
def extract_palette_kmeans(img, k):
    arr = np.array(img)
    small = cv2.resize(arr,(200,200))
    pts = small.reshape(-1,3).astype(np.float32)
    k = min(k, max(1, pts.shape[0]//50))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(pts)
    centers = km.cluster_centers_.astype(np.uint8)
    counts = np.bincount(km.labels_)
    order = np.argsort(-counts)  # sort by dominant clusters
    return centers[order], counts[order]

def palette_strip(colors, sq=40):
    if colors.size == 0:
        return Image.new("RGB",(sq, sq),(0,0,0))
    n = len(colors)
    canvas = np.zeros((sq, sq*n, 3), dtype=np.uint8)
    for i,c in enumerate(colors):
        canvas[:,i*sq:(i+1)*sq] = c
    return Image.fromarray(canvas)

# --------------------------------------------------------
# Shared + Unique Hue Palettes (Option B)
# --------------------------------------------------------
def shared_and_unique_hue_palettes(q_colors, q_counts, r_colors, r_counts, bins):
    if len(q_colors)==0 or len(r_colors)==0:
        return q_colors, q_colors, r_colors

    hsv_q = cv2.cvtColor(q_colors.reshape(1,-1,3),cv2.COLOR_RGB2HSV).reshape(-1,3)
    hsv_r = cv2.cvtColor(r_colors.reshape(1,-1,3),cv2.COLOR_RGB2HSV).reshape(-1,3)

    hq,_ = np.histogram(hsv_q[:,0], bins=bins, range=(0,180))
    hr,_ = np.histogram(hsv_r[:,0], bins=bins, range=(0,180))

    shared_strength = hq * hr
    query_strength = np.clip(hq - hr, 0, None)
    reference_strength = np.clip(hr - hq, 0, None)

    def pick(strength, colors, counts):
        if strength.sum()==0:
            return np.zeros((0,3), dtype=np.uint8)
        ordered = np.argsort(-strength)
        chosen = []
        for b in ordered:
            center = (b+0.5)*(180/bins)
            diffs = np.abs(hsv_q[:,0] - center)
            chosen.append(colors[np.argmin(diffs)])
            if len(chosen)>=len(colors):
                break
        chosen = np.array(chosen)
        # Sort chosen by cluster frequency:
        chosen_order = np.argsort(-counts[:len(chosen)])
        return chosen[chosen_order]

    shared = pick(shared_strength, q_colors, q_counts)
    query_u = pick(query_strength, q_colors, q_counts)
    reference_u = pick(reference_strength, r_colors, r_counts)

    return shared, query_u, reference_u

# --------------------------------------------------------
# Other Palettes (with frequency sorting preserved)
# --------------------------------------------------------
def blended_midpoint_palette(q_colors, r_colors):
    q_lab = to_lab(q_colors)
    r_lab = to_lab(r_colors)
    out=[]
    for q in q_lab:
        j=np.argmin(np.linalg.norm(r_lab-q,axis=1))
        out.append((q+r_lab[j])/2)
    return to_rgb(np.vstack(out))

def weighted_hybrid_palette(q_colors, r_colors, w):
    w=float(np.clip(w,0,1))
    q_lab = to_lab(q_colors)
    r_lab = to_lab(r_colors)
    out=[]
    for q in q_lab:
        j=np.argmin(np.linalg.norm(r_lab-q,axis=1))
        out.append(q*w + r_lab[j]*(1-w))
    return to_rgb(np.vstack(out))

# --------------------------------------------------------
# Metrics
# --------------------------------------------------------
def structural(img1,img2):
    dr=float(max(1,img2.max()-img2.min()))
    return float(np.clip(ssim(img1,img2,data_range=dr),0,1))

def hist_color(img1,img2,bins=8):
    h1=cv2.calcHist([img1],[0,1,2],None,[bins]*3,[0,256]*3)
    h2=cv2.calcHist([img2],[0,1,2],None,[bins]*3,[0,256]*3)
    cv2.normalize(h1,h1); cv2.normalize(h2,h2)
    raw=cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)
    return float(np.clip((raw+1)/2,0,1))

def entropy_sim(g1,g2):
    h1=cv2.calcHist([g1],[0],None,[256],[0,256]);h1/=max(1,h1.sum())
    h2=cv2.calcHist([g2],[0],None,[256],[0,256]);h2/=max(1,h2.sum())
    e1=float(-np.sum(h1*np.log2(h1+1e-12)))
    e2=float(-np.sum(h2*np.log2(h2+1e-12)))
    return float(np.clip(1-abs(e1-e2)/max(e1,e2,1e-12),0,1))

def edges(g1,g2):
    e1=cv2.Canny(g1,100,200).mean()/255
    e2=cv2.Canny(g2,100,200).mean()/255
    return float(np.clip(1-abs(e1-e2),0,1))

def texture(g1,g2):
    try:
        c1=float(graycoprops(graycomatrix(g1,[5],[0],normed=True,symmetric=True),"contrast")[0,0])
        c2=float(graycoprops(graycomatrix(g2,[5],[0],normed=True,symmetric=True),"contrast")[0,0])
        return float(np.clip(1-abs(c1-c2)/max(c1,c2,1e-12),0,1))
    except:
        return 0.5

def hue_similarity(img1,img2,bins,s_min,v_min):
    hsv1=cv2.cvtColor(img1,cv2.COLOR_RGB2HSV)
    hsv2=cv2.cvtColor(img2,cv2.COLOR_RGB2HSV)
    mask1=(hsv1[:,:,1]/255>=s_min)&(hsv1[:,:,2]/255>=v_min)
    mask2=(hsv2[:,:,1]/255>=s_min)&(hsv2[:,:,2]/255>=v_min)
    h1=hsv1[:,:,0][mask1] if mask1.any() else hsv1[:,:,0].flatten()
    h2=hsv2[:,:,0][mask2] if mask2.any() else hsv2[:,:,0].flatten()
    hist1,_=np.histogram(h1,bins=bins,range=(0,180))
    hist2,_=np.histogram(h2,bins=bins,range=(0,180))
    if hist1.sum()==0 or hist2.sum()==0:
        return 0.0
    v1=hist1/hist1.sum(); v2=hist2/hist2.sum()
    sim=float(np.dot(v1,v2)/((np.linalg.norm(v1)*np.linalg.norm(v2))+1e-12))
    return float(np.clip(sim,0,1))

METRIC_EXPLAIN = {
    "Structural Alignment": "Similar spatial layout and balance of tonal regions.",
    "Color Histogram": "Overall color distribution and mood similarity.",
    "Entropy Similarity": "Similarity in visual complexity or simplicity.",
    "Edge Complexity": "Similarity in contour sharpness and boundary contrast.",
    "Texture Correlation": "Similarity in fine surface patterning / granularity.",
    "Hue Distribution": "Similarity in color family emphasis across the hue wheel."
}

def plot_metrics(d):
    names=list(d.keys()); vals=[d[k] for k in names]
    fig=go.Figure(go.Bar(x=vals, y=names, orientation="h",
                         text=[f"{v:.2f}" for v in vals], textposition="outside",
                         marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=1)))
    fig.update_layout(height=260,margin=dict(l=80,r=20,t=10,b=10),template="simple_white",
                      xaxis=dict(range=[0,1],title="Similarity"),
                      yaxis=dict(autorange="reversed"))
    return fig

# --------------------------------------------------------
# File Inputs
# --------------------------------------------------------
ref_zip = st.file_uploader("Upload ZIP of Reference Images", type=["zip"])
query_file = st.file_uploader("Upload Query Image", type=["jpg","jpeg","png"])

# --------------------------------------------------------
# Processing
# --------------------------------------------------------
if ref_zip and query_file:
    ref_paths = load_reference_images(ref_zip)
    if not ref_paths:
        st.error("No valid reference images found.")
        st.stop()

    qimg = load_rgb(query_file)
    if qimg is None:
        st.error("Could not open query image.")
        st.stop()

    qrgb = np.array(qimg)
    qgray = cv2.cvtColor(qrgb, cv2.COLOR_RGB2GRAY)

    results=[]
    for p in ref_paths:
        rimg = load_rgb(p)
        if rimg is None:
            continue
        if resize_refs:
            rimg = rimg.resize(qimg.size)
        rrgb=np.array(rimg)
        rgray=cv2.cvtColor(rrgb,cv2.COLOR_RGB2GRAY)

        metrics = {
            "Structural Alignment": structural(qgray,rgray),
            "Color Histogram": hist_color(qrgb,rrgb),
            "Entropy Similarity": entropy_sim(qgray,rgray),
            "Edge Complexity": edges(qgray,rgray),
            "Texture Correlation": texture(qgray,rgray),
            "Hue Distribution": hue_similarity(qrgb,rrgb,hue_bins,sat_thresh,val_thresh)
        }
        score=float(np.mean(list(metrics.values())))
        results.append((p,rimg,metrics,score))

    results.sort(key=lambda x:x[3],reverse=True)
    top=results[:top_k]

    for idx,(path,rimg,metrics,score) in enumerate(top, start=1):
        c1,c2 = st.columns([2.5,1.4])

        with c1:
            st.markdown(f"### Match {idx}: {Path(path).name} — {score:.2f}")
            st.image([qimg,rimg], caption=["Query","Reference"], use_container_width=True)
            st.plotly_chart(plot_metrics(metrics), use_container_width=True)
            with st.expander("Metric Explanations"):
                for k,v in metrics.items():
                    st.markdown(f"**{k} ({v:.2f})**")
                    st.caption(METRIC_EXPLAIN[k])

        with c2:
            st.markdown("#### Intersection Palettes")

            q_cols, q_counts = extract_palette_kmeans(qimg, num_colors)
            r_cols, r_counts = extract_palette_kmeans(rimg, num_colors)

            # Shared + Unique hue palettes, sorted by frequency
            shared_pal, query_pal, ref_pal = shared_and_unique_hue_palettes(q_cols, q_counts, r_cols, r_counts, hue_bins)

            st.markdown("**Shared Hue Palette**")
            st.image(palette_strip(shared_pal), use_container_width=True)
            st.caption("Hue families strongly emphasized in both images.")

            st.markdown("**Query-Dominant Hue Palette**")
            st.image(palette_strip(query_pal), use_container_width=True)
            st.caption("Hue families more strongly emphasized in the query image.")

            st.markdown("**Reference-Dominant Hue Palette**")
            st.image(palette_strip(ref_pal), use_container_width=True)
            st.caption("Hue families more strongly emphasized in the reference image.")

            st.markdown("**Weighted Hybrid Palette**")
            pal_hybrid = weighted_hybrid_palette(q_cols, r_cols, hybrid_weight)
            st.image(palette_strip(pal_hybrid), use_container_width=True)
            st.caption("Per-cluster blend in perceptual space, controlled by sidebar weight.")

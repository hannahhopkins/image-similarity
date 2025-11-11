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

st.write(
    "Upload a ZIP of reference images and a query image. "
    "This tool compares them on structure, color, texture, edge, entropy, and hue metrics, "
    "and generates three intersection palettes that show chromatic relationships."
)

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Settings")

top_k = st.sidebar.slider("Number of matches to display", 1, 10, 5)
st.sidebar.caption("How many closest reference matches to show.")

num_colors = st.sidebar.slider("Palette size (colors per image)", 3, 12, 6)
st.sidebar.caption("Number of dominant colors extracted from each image via clustering.")

# Resize strategy selector
resize_mode = st.sidebar.selectbox(
    "Resize Strategy",
    [
        "Resize reference images to match query",
        "Resize query to match reference images",
        "Resize both to a neutral analysis size (512px long edge)"
    ],
    index=0
)
st.sidebar.caption(
    "Structural metrics require images to share dimensions. "
    "Resizing method influences edge, texture, and SSIM similarity."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hue Similarity Settings")

hue_bins = st.sidebar.slider("Hue bins", 12, 72, 36, step=6)
st.sidebar.caption("Larger = more precise hue comparison. Smaller = broader grouping of hue families.")

sat_thresh = st.sidebar.slider("Saturation mask threshold", 0.0, 1.0, 0.15, step=0.01)
st.sidebar.caption("Pixels below this saturation are treated as neutral and excluded from hue histograms.")

val_thresh = st.sidebar.slider("Value mask threshold", 0.0, 1.0, 0.15, step=0.01)
st.sidebar.caption("Pixels below this brightness are excluded to prevent noise in hue similarity.")

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Palette")

hybrid_query_weight = st.sidebar.slider("Hybrid palette: query weight", 0.0, 1.0, 0.6, step=0.05)
st.sidebar.caption("Controls how strongly the query palette influences the Hybrid palette.")


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def safe_open_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        return None

def iter_zip_images(extract_dir: str):
    for root, _, files in os.walk(extract_dir):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            ext = Path(f).suffix.lower()
            if ext in (".jpg", ".jpeg", ".png"):
                yield os.path.join(root, f)

def resize_to_long_edge(pil_img, target=512):
    w, h = pil_img.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    return pil_img.resize((nw, nh), Image.LANCZOS)

# -----------------------------------------------------------------------------
# Palette Extraction and Color / Hue Helpers
# -----------------------------------------------------------------------------

def kmeans_palette(pil_img, k):
    arr = np.array(pil_img)
    flat = arr.reshape(-1, 3).astype(np.float32)
    k_eff = min(k, max(1, flat.shape[0] // 200))
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=0)
    labels = km.fit_predict(flat)
    centers = km.cluster_centers_
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    centers = centers[order]
    if centers.shape[0] < k:
        pad = np.tile(np.mean(flat, axis=0, keepdims=True), (k - centers.shape[0], 1))
        centers = np.vstack([centers, pad])
    return [tuple(map(int, np.clip(c, 0, 255))) for c in centers[:k]]

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
    h = H[mask]
    hist, _ = np.histogram(h, bins=bins, range=(0,1))
    hist = hist.astype(np.float32)
    hist = hist / hist.sum() if hist.sum() > 0 else np.ones(bins)/bins
    return hist

# -----------------------------------------------------------------------------
# Similarity Metrics
# -----------------------------------------------------------------------------

def structural_similarity_metric(a, b):
    a = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.array(b), cv2.COLOR_RGB2GRAY)
    dr = max(b.max() - b.min(), 1e-6)
    return float(np.clip(ssim(a, b, data_range=dr), 0, 1))

def color_hist_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    h1 = cv2.calcHist([a],[0,1,2],None,[8,8,8],[0,256]*3)
    h2 = cv2.calcHist([b],[0,1,2],None,[8,8,8],[0,256]*3)
    cv2.normalize(h1,h1)
    cv2.normalize(h2,h2)
    raw = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
    return float(np.clip((raw+1)/2, 0, 1))

def entropy_similarity(a, b):
    def entropy_gray(img):
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([g],[0],None,[256],[0,256]).astype(np.float32)
        p = hist / (hist.sum() + 1e-8)
        return float(-np.sum(p*np.log2(p+1e-12)))
    e1, e2 = entropy_gray(a), entropy_gray(b)
    return float(np.clip(1 - abs(e1-e2) / max(e1,e2,1e-6), 0, 1))

def edge_complexity_similarity(a, b):
    def edge_density(pil):
        g = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g, 100, 200)
        return float(e.mean()/255.0)
    d1, d2 = edge_density(a), edge_density(b)
    return float(np.clip(1 - abs(d1-d2), 0, 1))

def texture_correlation_similarity(a, b):
    def glcm_energy(gray):
        gl = graycomatrix(gray, [2], [0], symmetric=True, normed=True)
        return float(graycoprops(gl,'energy')[0,0])
    ga = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(np.array(b), cv2.COLOR_RGB2GRAY)
    e1, e2 = glcm_energy(ga), glcm_energy(gb)
    return float(np.clip(1 - abs(e1-e2)/max(e1,e2,1e-6), 0, 1))

def hue_distribution_similarity(a, b, bins, s_thr, v_thr):
    h1 = hue_histogram(a, bins, s_thr, v_thr)
    h2 = hue_histogram(b, bins, s_thr, v_thr)
    denom = (np.linalg.norm(h1)*np.linalg.norm(h2))+1e-10
    return float(np.dot(h1, h2) / denom)

# -----------------------------------------------------------------------------
# Palette Construction (Distinct)
# -----------------------------------------------------------------------------

def lab_midpoint_palette(q_cols, r_cols, n):
    if not q_cols or not r_cols:
        return q_cols[:n]
    q = np.array(q_cols, dtype=np.uint8).reshape(-1,1,3)
    r = np.array(r_cols, dtype=np.uint8).reshape(-1,1,3)
    q_hsv = cv2.cvtColor(q, cv2.COLOR_RGB2HSV).reshape(-1,3)
    r_hsv = cv2.cvtColor(r, cv2.COLOR_RGB2HSV).reshape(-1,3)
    q_lab = cv2.cvtColor(q, cv2.COLOR_RGB2LAB).reshape(-1,3).astype(np.float32)
    r_lab = cv2.cvtColor(r, cv2.COLOR_RGB2LAB).reshape(-1,3).astype(np.float32)

    out=[]
    for i in range(min(n, len(q_lab))):
        qh = q_hsv[i,0]/179.0
        diffs=[(min(abs(qh - (r_hsv[j,0]/179.0)),1-abs(qh - (r_hsv[j,0]/179.0))), j)
               for j in range(len(r_hsv))]
        j=min(diffs,key=lambda t:t[0])[1]
        mid=(q_lab[i]+r_lab[j])/2
        rgb=cv2.cvtColor(mid.reshape(1,1,3).astype(np.uint8),cv2.COLOR_LAB2RGB).reshape(3,)
        out.append(tuple(int(x) for x in rgb))
    while len(out)<n: out.append(q_cols[0])
    return out

def shared_hue_palette(query_img, ref_img, n, bins, s_thr, v_thr):
    arr_q=np.array(query_img); arr_r=np.array(ref_img)
    hsv_q=rgb_to_hsv01(arr_q); hsv_r=rgb_to_hsv01(arr_r)
    Hq,Sq,Vq=hsv_q[...,0],hsv_q[...,1],hsv_q[...,2]
    Hr,Sr,Vr=hsv_r[...,0],hsv_r[...,1],hsv_r[...,2]
    mq=(Sq>=s_thr)&(Vq>=v_thr); mr=(Sr>=s_thr)&(Vr>=v_thr)
    if not np.any(mq) or not np.any(mr):
        return kmeans_palette(query_img,n)

    hq=Hq[mq]; hr=Hr[mr]
    hist_q,_=np.histogram(hq,bins=bins,range=(0,1))
    hist_r,_=np.histogram(hr,bins=bins,range=(0,1))
    centers=(np.linspace(0,1,bins+1)[:-1]+np.linspace(0,1,bins+1)[1:])/2
    shared_idx=np.argsort(-(hist_q*hist_r+1e-9))[:n]

    out=[]
    for ci in shared_idx:
        hc=centers[ci]
        # pick representative S,V from each
        def sample_sv(H,S,V,mask):
            Hm=H[mask]
            if Hm.size==0: return 0.5,0.5
            idx=np.argmin(np.minimum(np.abs(Hm-hc),1-np.abs(Hm-hc)))
            Sm=S[mask][idx]; Vm=V[mask][idx]
            return float(Sm),float(Vm)
        s_q,v_q=sample_sv(Hq,Sq,Vq,mq)
        s_r,v_r=sample_sv(Hr,Sr,Vr,mr)
        s_mid=(s_q+s_r)/2; v_mid=(v_q+v_r)/2
        hsv=np.array([[[hc*179.0,s_mid*255.0,v_mid*255.0]]],dtype=np.uint8)
        rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB).reshape(3,)
        out.append(tuple(int(x) for x in rgb))
    while len(out)<n:
        out.append(out[-1] if out else (128,128,128))
    return out

def weighted_hybrid_palette(q_cols, r_cols, n, w):
    if not q_cols:
        return []
    if not r_cols:
        return q_cols[:n]
    q=np.array(q_cols,dtype=np.uint8).reshape(-1,1,3)
    r=np.array(r_cols,dtype=np.uint8).reshape(-1,1,3)
    q_hsv=cv2.cvtColor(q,cv2.COLOR_RGB2HSV).reshape(-1,3)
    r_hsv=cv2.cvtColor(r,cv2.COLOR_RGB2HSV).reshape(-1,3)
    q_lab=cv2.cvtColor(q,cv2.COLOR_RGB2LAB).reshape(-1,3).astype(np.float32)
    r_lab=cv2.cvtColor(r,cv2.COLOR_RGB2LAB).reshape(-1,3).astype(np.float32)
    out=[]
    for i in range(min(n,len(q_lab))):
        qh=q_hsv[i,0]/179.0
        diffs=[(min(abs(qh - (r_hsv[j,0]/179.0)),1-abs(qh - (r_hsv[j,0]/179.0))), j)
               for j in range(len(r_hsv))]
        j=min(diffs,key=lambda t:t[0])[1]
        blend=q_lab[i]*w+r_lab[j]*(1-w)
        rgb=cv2.cvtColor(blend.reshape(1,1,3).astype(np.uint8),cv2.COLOR_LAB2RGB).reshape(3,)
        out.append(tuple(int(x) for x in rgb))
    while len(out)<n:
        out.append(q_cols[0])
    return out

# -----------------------------------------------------------------------------
# Plotly Palette Display
# -----------------------------------------------------------------------------

def plotly_palette(colors, key, title=None):
    fig = go.Figure()
    shapes=[]
    xs=[]
    ys=[]
    hovers=[]
    for i,c in enumerate(colors):
        hexv=f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        shapes.append(dict(type="rect",x0=i,x1=i+1,y0=0,y1=1,fillcolor=hexv,line=dict(width=0)))
        xs.append(i+0.5); ys.append(0.5); hovers.append(hexv)
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers",marker=dict(opacity=0),
                             hovertext=hovers,hoverinfo="text",showlegend=False))
    fig.update_layout(
        shapes=shapes,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0,r=0,t=20 if title else 4,b=4),
        height=60,
        title=title
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# -----------------------------------------------------------------------------
# Metric Bar Chart
# -----------------------------------------------------------------------------

def metric_bar_chart(metrics, key):
    names=list(metrics.keys())
    vals=[float(metrics[k]) for k in names]
    fig=go.Figure(go.Bar(
        x=vals,y=names,orientation='h',
        marker=dict(color=vals,colorscale="RdYlGn",cmin=0,cmax=1),
        text=[f"{v:.2f}" for v in vals],textposition="outside",hoverinfo="skip"
    ))
    fig.update_layout(
        xaxis=dict(range=[0,1],title="Similarity (0–1)"),
        yaxis=dict(title=""),
        height=max(220, 28*len(names)+60),
        margin=dict(l=80,r=20,t=10,b=10),
        template="simple_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# -----------------------------------------------------------------------------
# Upload UI
# -----------------------------------------------------------------------------
uploaded_zip = st.file_uploader("Upload Reference Images (ZIP)", type=["zip"])
query_file = st.file_uploader("Upload Query Image", type=["jpg","jpeg","png"])

# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
if uploaded_zip and query_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmp_dir)

        ref_paths=[p for p in iter_zip_images(tmp_dir)]
        if not ref_paths:
            st.error("No valid reference images found.")
            st.stop()

        query_img = safe_open_image(query_file)
        if query_img is None:
            st.error("Could not open query image.")
            st.stop()

        # Determine final comparison dimensions based on resize mode:
        if resize_mode == "Resize reference images to match query":
            def prep(img): return img.resize(query_img.size, Image.LANCZOS)

        elif resize_mode == "Resize query to match reference images":
            def prep(img): return img  # reference images remain native size
            # Resize query to first reference
            first = safe_open_image(ref_paths[0])
            if first:
                query_img = query_img.resize(first.size, Image.LANCZOS)
            def prep(img): return img

        else: # Resize both to 512px long edge
            query_img = resize_to_long_edge(query_img, 512)
            def prep(img): return resize_to_long_edge(img, 512)

        # Compute similarity
        results=[]
        for p in ref_paths:
            img=safe_open_image(p)
            if img is None: continue
            img_use=prep(img)

            metrics={
                "Structural Alignment": structural_similarity_metric(query_img, img_use),
                "Color Histogram":     color_hist_similarity(query_img, img_use),
                "Entropy Similarity":  entropy_similarity(query_img, img_use),
                "Edge Complexity":     edge_complexity_similarity(query_img, img_use),
                "Texture Correlation": texture_correlation_similarity(query_img, img_use),
                "Hue Distribution":    hue_distribution_similarity(query_img, img_use, hue_bins, sat_thresh, val_thresh),
            }
            score=float(np.mean(list(metrics.values())))
            results.append((p, img_use, metrics, score))

        results.sort(key=lambda x:x[3], reverse=True)
        top = results[:top_k]

        st.subheader(f"Top {len(top)} Matches")

        for idx,(path,img_use,metrics,score) in enumerate(top,1):
            c1,c2 = st.columns([2.6,1.4])
            with c1:
                st.markdown(f"### Match {idx}: {os.path.basename(path)} — {score:.2f}")
                st.image([query_img, img_use], caption=["Query","Reference"], use_container_width=True)
                metric_bar_chart(metrics, key=f"bar_{idx}")
                with st.expander("Metric Explanations", expanded=False):
                    st.markdown(
                        "- **Structural Alignment**: Measures similarity in luminance and spatial structure.\n"
                        "- **Color Histogram**: Measures similarity of overall color distribution.\n"
                        "- **Entropy Similarity**: Compares tonal complexity and detail density.\n"
                        "- **Edge Complexity**: Compares the density of contour and edge structures.\n"
                        "- **Texture Correlation**: Compares surface texture via GLCM energy patterns.\n"
                        "- **Hue Distribution**: Compares dominant hue families after neutral/low-light masking."
                    )

            with c2:
                q_cols = kmeans_palette(query_img, num_colors)
                r_cols = kmeans_palette(img_use, num_colors)

                mid  = lab_midpoint_palette(q_cols, r_cols, min(num_colors,6))
                shared = shared_hue_palette(query_img, img_use, min(num_colors,6), hue_bins, sat_thresh, val_thresh)
                hybrid = weighted_hybrid_palette(q_cols, r_cols, min(num_colors,6), hybrid_query_weight)

                st.markdown("Intersection Palettes")
                plotly_palette(mid, key=f"mid_{idx}", title="Blended Midpoint")
                plotly_palette(shared, key=f"shared_{idx}", title="Shared Hue")
                plotly_palette(hybrid, key=f"hyb_{idx}", title="Weighted Hybrid")

            # ---- Separator ----
            st.markdown(
                "<hr style='margin-top:2rem; margin-bottom:2rem; border:none; border-top:1px solid #888;'/>",
                unsafe_allow_html=True
            )

else:
    st.info("Upload a ZIP of reference images and a query image to begin.")

import streamlit as st
import os
import zipfile
import tempfile
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import faiss
from skimage.feature import graycomatrix, graycoprops
from scipy.fftpack import fft2
from scipy.stats import pearsonr
import cv2
import matplotlib.pyplot as plt

# --- Page setup ---
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide")
st.title("🔍 Image Similarity Analyzer — Technical Metrics Edition")
st.write(
    "Upload a **ZIP folder** of reference images and a **query image**. "
    "The app finds the 5 most visually similar images using deep features "
    "and interpretable visual metrics."
)

# --- Model setup ---
@st.cache_resource
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess

model, preprocess = load_model()

def extract_features(image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze().numpy().astype("float32")

# --- Helper: safe image open ---
def safe_open_image(path):
    try:
        image = Image.open(path)
        image = image.convert("RGB")
        image.thumbnail((512, 512))  # downscale large images for performance
        return image
    except UnidentifiedImageError:
        st.warning(f"⚠️ Skipping {os.path.basename(path)}: not a valid image.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Skipping {os.path.basename(path)}: {e}")
        return None

# --- Image analysis metrics ---
def analyze_similarity_metrics(img1, img2):
    img1_gray = ImageOps.grayscale(img1).resize((224, 224))
    img2_gray = ImageOps.grayscale(img2).resize((224, 224))
    a = np.array(img1_gray)
    b = np.array(img2_gray)

    # Color similarity
    c1 = np.mean(np.array(img1).reshape(-1, 3), axis=0)
    c2 = np.mean(np.array(img2).reshape(-1, 3), axis=0)
    color_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-6)

    # Texture similarity (GLCM energy)
    glcm1 = graycomatrix(a, [1], [0], symmetric=True, normed=True)
    glcm2 = graycomatrix(b, [1], [0], symmetric=True, normed=True)
    tex_sim = 1 - abs(graycoprops(glcm1, "energy")[0, 0] - graycoprops(glcm2, "energy")[0, 0])

    # Brightness similarity
    bright_sim = 1 - abs(a.mean() - b.mean()) / 255.0

    # Edge similarity
    edge1 = cv2.Sobel(a, cv2.CV_64F, 1, 1, ksize=3)
    edge2 = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3)
    edge_sim = pearsonr(edge1.flatten(), edge2.flatten())[0] if np.std(edge1) and np.std(edge2) else 0

    # Pattern similarity
    fft1 = np.abs(fft2(a))
    fft2v = np.abs(fft2(b))
    fft_sim = pearsonr(fft1.flatten(), fft2v.flatten())[0] if np.std(fft1) and np.std(fft2v) else 0

    # Normalize metrics to 0–1
    def norm(v): return (v + 1) / 2 if isinstance(v, (int, float)) else 0.5
    metrics = {
        "color": norm(color_sim),
        "texture": norm(tex_sim),
        "brightness": norm(bright_sim),
        "edges": norm(edge_sim),
        "pattern": norm(fft_sim)
    }
    return metrics

def describe_metrics(metrics):
    """Generate concise technical-style descriptions."""
    key_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:2]
    descriptors = {
        "color": "shared chromatic distribution and hue alignment",
        "texture": "parallel microstructural textures and fine-grain consistency",
        "brightness": "balanced luminance and tonal mapping",
        "edges": "comparable structural rhythm and directional contour density",
        "pattern": "aligned frequency spectra and repeating compositional geometry"
    }
    desc = " and ".join(descriptors[k] for k, _ in key_metrics)
    return f"Technical analysis indicates {desc}."

def plot_metrics(metrics):
    """Compact minimalist bar chart visualization of metrics."""
    fig, ax = plt.subplots(figsize=(4.8, 1.8))
    names = list(metrics.keys())
    values = [max(0, v) for v in metrics.values()]
    bars = ax.barh(names, values, color="#BBBBBB", edgecolor="#444444", height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Similarity", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Metric Breakdown", fontsize=9, pad=5)
    ax.grid(False)
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{values[i]:.2f}", va='center', fontsize=8, color="#333333")
    plt.tight_layout(pad=0.4)
    return fig

# --- Uploads ---
st.subheader("📁 Upload Reference Image Folder (ZIP)")
ref_zip = st.file_uploader("Upload a ZIP file containing reference images", type=["zip"])

ref_folder = None
if ref_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(ref_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    ref_folder = temp_dir

    # Collect only valid image files recursively
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    ref_image_files = [
        p for p in Path(ref_folder).rglob("*")
        if p.suffix.lower() in valid_ext and not p.name.startswith(".")
    ]

    if not ref_image_files:
        st.error("No valid images found in your ZIP. Make sure it contains JPG, PNG, BMP, or TIFF files.")
        st.stop()
    else:
        st.success(f"✅ Found {len(ref_image_files)} candidate images.")
else:
    st.info("Please upload a ZIP file to continue.")

st.subheader("🖼️ Upload Query Image")
query_file = st.file_uploader("Upload a single query image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if ref_zip and query_file:
    ref_features = []
    ref_paths = []

    total_images = len(ref_image_files)
    progress = st.progress(0)
    status = st.empty()
    st.write("Extracting visual features from reference images...")

    for i, filepath in enumerate(ref_image_files, start=1):
        image = safe_open_image(filepath)
        if image is None:
            continue
        try:
            feat = extract_features(image)
            ref_features.append(feat)
            ref_paths.append(str(filepath))
        except Exception as e:
            st.warning(f"⚠️ Could not process {filepath.name}: {e}")
        progress.progress(i / total_images)
        status.text(f"Processed {i}/{total_images} images")

    progress.empty()
    status.empty()

    if len(ref_features) == 0:
        st.error("No valid reference images were successfully processed.")
        st.stop()

    ref_features = np.array(ref_features)
    index = faiss.IndexFlatL2(ref_features.shape[1])
    index.add(ref_features)

    query_img = safe_open_image(query_file)
    query_feat = extract_features(query_img)
    D, I = index.search(np.array([query_feat]), k=5)

    st.subheader("🎯 Top 5 Technically Similar Images")
    st.image(query_img, caption="Query Image", use_container_width=True)

    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        sim_img = safe_open_image(ref_paths[idx])
        if sim_img is None:
            continue
        metrics = analyze_similarity_metrics(query_img, sim_img)
        explanation = describe_metrics(metrics)
        fig = plot_metrics(metrics)

        st.markdown(f"### Match {rank} — Distance: `{dist:.2f}`")
        st.image(sim_img, caption=explanation, use_container_width=True)
        st.pyplot(fig)

else:
    st.info("Please upload both a reference ZIP and a query image to start the analysis.")

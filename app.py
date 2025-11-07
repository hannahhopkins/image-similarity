import streamlit as st
import os
import zipfile
import tempfile
from pathlib import Path
from PIL import Image, ImageOps
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
st.write("Upload a **ZIP folder** of reference images and a **query image**. The app finds the 5 most visually similar images using deep features and interpretable visual metrics.")

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

# --- Image analysis metrics ---
def analyze_similarity_metrics(img1, img2):
    """Return 5 interpretable similarity metrics between two images."""
    img1_gray = ImageOps.grayscale(img1).resize((224, 224))
    img2_gray = ImageOps.grayscale(img2).resize((224, 224))
    a = np.array(img1_gray)
    b = np.array(img2_gray)

    # 1. Color palette similarity
    c1 = np.mean(np.array(img1).reshape(-1, 3), axis=0)
    c2 = np.mean(np.array(img2).reshape(-1, 3), axis=0)
    color_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-6)

    # 2. Texture similarity via GLCM energy
    glcm1 = graycomatrix(a, [1], [0], symmetric=True, normed=True)
    glcm2 = graycomatrix(b, [1], [0], symmetric=True, normed=True)
    tex_sim = 1 - abs(graycoprops(glcm1, "energy")[0, 0] - graycoprops(glcm2, "energy")[0, 0])

    # 3. Brightness / contrast proximity
    bright_sim = 1 - abs(a.mean() - b.mean()) / 255.0

    # 4. Edge density correlation
    edge1 = cv2.Sobel(a, cv2.CV_64F, 1, 1, ksize=3)
    edge2 = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3)
    edge_sim = pearsonr(edge1.flatten(), edge2.flatten())[0] if np.std(edge1) and np.std(edge2) else 0

    # 5. Pattern alignment (FFT)
    fft1 = np.abs(fft2(a))
    fft2v = np.abs(fft2(b))
    fft_sim = pearsonr(fft1.flatten(), fft2v.flatten())[0] if np.std(fft1) and np.std(fft2v) else 0

    return {
        "color": color_sim,
        "texture": tex_sim,
        "brightness": bright_sim,
        "edges": edge_sim,
        "pattern": fft_sim
    }

def describe_metrics(metrics):
    """Generate a technical text explanation from metric values."""
    key_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:2]
    descriptors = {
        "color": "shared dominant hues and tone distribution",
        "texture": "comparable surface granularity and microstructure",
        "brightness": "aligned tonal luminance and contrast levels",
        "edges": "similar compositional form and edge rhythm",
        "pattern": "parallel pattern repetition and spatial cadence"
    }
    desc = " and ".join(descriptors[k] for k, _ in key_metrics)
    return f"This image exhibits {desc} relative to the query image."

def plot_metrics(metrics):
    """Return a Matplotlib bar chart of the metric scores."""
    fig, ax = plt.subplots(figsize=(5, 2))
    names = list(metrics.keys())
    values = [max(0, v) for v in metrics.values()]  # clip negatives
    ax.barh(names, values)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Similarity")
    ax.set_title("Metric Breakdown")
    plt.tight_layout()
    return fig

# --- ZIP upload and reference extraction ---
st.subheader("📁 Upload Reference Image Folder (ZIP)")
ref_zip = st.file_uploader("Upload a ZIP file containing reference images", type=["zip"])

ref_folder = None
if ref_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(ref_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    ref_folder = temp_dir

    image_extensions = {".jpg", ".jpeg", ".png"}
    ref_image_files = [p for p in Path(ref_folder).rglob("*") if p.suffix.lower() in image_extensions]

    if not ref_image_files:
        st.error("No images found in the uploaded ZIP. Make sure it contains JPG or PNG files.")
        st.stop()
    else:
        st.success(f"✅ Found {len(ref_image_files)} images in your ZIP archive.")
else:
    st.info("Please upload a ZIP file to continue.")

# --- Query image upload ---
st.subheader("🖼️ Upload Query Image")
query_file = st.file_uploader("Upload a single query image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if ref_zip and query_file:
    ref_features = []
    ref_paths = []

    st.write("Extracting features from reference images...")
    progress = st.progress(0)
    for i, filepath in enumerate(ref_image_files):
        try:
            image = Image.open(filepath).convert("RGB")
            feat = extract_features(image)
            ref_features.append(feat)
            ref_paths.append(str(filepath))
        except Exception as e:
            st.warning(f"Skipping {filepath.name}: {e}")
        progress.progress((i + 1) / len(ref_image_files))
    progress.empty()

    if len(ref_features) == 0:
        st.error("No valid reference images found.")
        st.stop()

    ref_features = np.array(ref_features)
    index = faiss.IndexFlatL2(ref_features.shape[1])
    index.add(ref_features)

    query_img = Image.open(query_file).convert("RGB")
    query_feat = extract_features(query_img)
    D, I = index.search(np.array([query_feat]), k=5)

    st.subheader("🎯 Top 5 Technically Similar Images")
    st.image(query_img, caption="Query Image", use_container_width=True)

    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        sim_img = Image.open(ref_paths[idx]).convert("RGB")
        metrics = analyze_similarity_metrics(query_img, sim_img)
        explanation = describe_metrics(metrics)
        fig = plot_metrics(metrics)

        st.markdown(f"### Match {rank} — Distance: `{dist:.2f}`")
        st.image(sim_img, caption=explanation, use_container_width=True)
        st.pyplot(fig)

else:
    st.info("Please upload both a reference ZIP and a query image to start the analysis.")

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import faiss
import streamlit as st
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from scipy.fft import fft2, fftshift

# ---------------------------
# Streamlit setup
# ---------------------------
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide")
st.title("🔍 Image Similarity Analyzer")
st.caption("Upload a **query image** and analyze its similarity to a folder of reference images using deep features and visual metrics.")

# ---------------------------
# Model setup
# ---------------------------
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
    return features.squeeze().numpy()

# ---------------------------
# Similarity metrics
# ---------------------------
def color_distance_ciede(img1, img2):
    img1_lab, img2_lab = rgb2lab(img1 / 255.0), rgb2lab(img2 / 255.0)
    diff = np.mean(np.sqrt(np.sum((img1_lab - img2_lab) ** 2, axis=2)))
    return max(0.0, min(1.0, 1 - diff / 100))

def brightness_similarity(img1, img2):
    b1, b2 = np.mean(img1), np.mean(img2)
    return max(0.0, 1 - abs(b1 - b2) / 255)

def ssim_similarity(img1, img2):
    img1_gray, img2_gray = np.mean(img1, axis=2), np.mean(img2, axis=2)
    s = ssim(img1_gray, img2_gray, data_range=img1_gray.max() - img1_gray.min())
    return (s + 1) / 2

def fourier_similarity(img1, img2):
    img1_gray, img2_gray = np.mean(img1, axis=2), np.mean(img2, axis=2)
    f1, f2 = np.log1p(np.abs(fftshift(fft2(img1_gray)))), np.log1p(np.abs(fftshift(fft2(img2_gray))))
    diff = np.mean(np.abs(f1 - f2))
    return max(0.0, min(1.0, 1 - diff / np.max(f1)))

def entropy_similarity(img1, img2):
    def entropy(im):
        hist, _ = np.histogram(im, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    e1, e2 = entropy(np.mean(img1, axis=2)), entropy(np.mean(img2, axis=2))
    return max(0.0, min(1.0, 1 - abs(e1 - e2) / 8.0))

# ---------------------------
# File inputs
# ---------------------------
query_file = st.file_uploader("Upload a query image", type=["png", "jpg", "jpeg"])
ref_folder = st.text_input("Enter path to a folder of reference images", "")

if query_file and ref_folder:
    if not os.path.exists(ref_folder):
        st.error(f"Folder not found: {ref_folder}")
    else:
        query_img = Image.open(query_file).convert("RGB")
        query_arr = np.array(query_img)
        query_features = extract_features(query_img).astype('float32')

        st.image(query_img, caption="Query Image", use_container_width=True)

        # Load reference images and compute features
        features_list, image_paths = [], []
        with st.spinner("Extracting features from reference images..."):
            for filename in os.listdir(ref_folder):
                filepath = os.path.join(ref_folder, filename)
                try:
                    img = Image.open(filepath).convert("RGB")
                    features = extract_features(img)
                    features_list.append(features)
                    image_paths.append(filepath)
                except Exception:
                    pass

        if len(image_paths) == 0:
            st.error("No valid images found in the folder.")
        else:
            # FAISS index
            features_matrix = np.array(features_list).astype('float32')
            index = faiss.IndexFlatL2(features_matrix.shape[1])
            index.add(features_matrix)

            # Search
            D, I = index.search(np.array([query_features]), k=5)

            st.subheader("Top 5 Similar Images")
            for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
                ref_img = np.array(Image.open(image_paths[idx]).convert("RGB"))
                h, w = query_arr.shape[:2]
                ref_resized = np.array(Image.fromarray(ref_img).resize((w, h)))

                try:
                    texture_score = np.corrcoef(
                        np.mean(query_arr, axis=2).flatten(),
                        np.mean(ref_resized, axis=2).flatten()
                    )[0, 1]
                except Exception:
                    texture_score = 0.0

                metrics = {
                    "Color": color_distance_ciede(query_arr, ref_resized),
                    "Texture": max(0.0, min(1.0, (texture_score + 1) / 2)),
                    "Brightness": brightness_similarity(query_arr, ref_resized),
                    "Edges (SSIM)": ssim_similarity(query_arr, ref_resized),
                    "Patterns (Fourier)": fourier_similarity(query_arr, ref_resized),
                    "Entropy": entropy_similarity(query_arr, ref_resized)
                }

                overall_score = np.mean([
                    metrics["Color"] * 0.25,
                    metrics["Texture"] * 0.2,
                    metrics["Brightness"] * 0.1,
                    metrics["Edges (SSIM)"] * 0.2,
                    metrics["Patterns (Fourier)"] * 0.15,
                    metrics["Entropy"] * 0.1
                ]) * 100

                with st.container():
                    st.markdown(f"### Match {rank}: **{overall_score:.1f}% Similarity**")
                    st.image(ref_img, caption=os.path.basename(image_paths[idx]), use_container_width=True)

                    # Metric bars
                    for metric, value in metrics.items():
                        st.progress(value)
                        st.caption(f"**{metric}:** {value:.2f}")

                    # Technical summary
                    st.markdown(
                        f"<p style='font-size: 14px; color: #555;'>"
                        f"Color alignment: {metrics['Color']:.2f}, "
                        f"Texture coherence: {metrics['Texture']:.2f}, "
                        f"Edge correspondence: {metrics['Edges (SSIM)']:.2f}, "
                        f"Pattern structural overlap: {metrics['Patterns (Fourier)']:.2f}, "
                        f"Entropy similarity: {metrics['Entropy']:.2f}.<br>"
                        f"<em>Overall visual correspondence: {overall_score:.1f}%</em>"
                        f"</p>",
                        unsafe_allow_html=True
                    )

else:
    st.info("👆 Upload a query image and provide a valid reference folder path to begin.")

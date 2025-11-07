import streamlit as st
import os
import zipfile
import tempfile
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import faiss

# --- Page setup ---
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide")
st.title("🔍 Image Similarity Analyzer")
st.write("Upload a **ZIP folder** of reference images, then upload a query image to find the five most visually similar images.")

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

# --- Reference ZIP upload ---
st.subheader("📁 Upload Reference Image Folder (ZIP)")
ref_zip = st.file_uploader("Upload a ZIP file containing reference images", type=["zip"])

ref_folder = None
if ref_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(ref_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    ref_folder = temp_dir
    st.success(f"✅ Extracted {len(os.listdir(ref_folder))} images from ZIP.")
else:
    st.info("Please upload a ZIP file to continue.")

# --- Query image upload ---
st.subheader("🖼️ Upload Query Image")
query_file = st.file_uploader("Upload a single query image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if ref_folder and query_file:
    # Load reference images
    ref_features = []
    ref_paths = []
    for filename in os.listdir(ref_folder):
        filepath = os.path.join(ref_folder, filename)
        try:
            image = Image.open(filepath).convert("RGB")
            feat = extract_features(image)
            ref_features.append(feat)
            ref_paths.append(filepath)
        except Exception as e:
            st.warning(f"Skipping {filename}: {e}")

    if len(ref_features) == 0:
        st.error("No valid reference images found.")
        st.stop()

    ref_features = np.array(ref_features)
    index = faiss.IndexFlatL2(ref_features.shape[1])
    index.add(ref_features)

    # Query image
    query_img = Image.open(query_file).convert("RGB")
    query_feat = extract_features(query_img)
    D, I = index.search(np.array([query_feat]), k=5)

    st.subheader("🎯 Top 5 Similar Images")
    st.image(query_img, caption="Query Image", use_container_width=True)

    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        sim_img = Image.open(ref_paths[idx]).convert("RGB")
        explanation = f"""
        **Match {rank}**  
        Distance: `{dist:.2f}`  
        _This image shares key compositional patterns and visual texture characteristics._
        """
        st.image(sim_img, caption=explanation, use_container_width=True)

else:
    st.info("Please upload both a reference ZIP and a query image to start the analysis.")

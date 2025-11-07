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

# --- Image analysis metrics ---
def analyze_similarity_metrics(img1, img2):
    img1_gray = ImageOps.grayscale(img1).resize((224, 224))
    img2_gray = ImageOps.grayscale(img2).resize((224, 224))
    a = np.array(img1_gray)
    b = np.array(img2_gray)

    c1 = np.mean(np.array(img1).reshape(-1, 3), axis=0)
    c2 = np.mean(np.array(img2).reshape(-1, 3), axis=0)
    color_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-6)

    glcm1 = graycomatrix(a, [1], [0], symmetric=True, normed=True)
    glcm2 = graycomatrix(b, [1], [0], symmetric=True, normed=True)
    tex_sim = 1 - abs(graycoprops(glcm1,_

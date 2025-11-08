import os
import zipfile
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import pairwise_distances
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

# --- Page setup ---
st.set_page_config(page_title="Image Similarity Analyzer", layout="wide")
st.title("🎨 Image Similarity Analyzer")
st.markdown("Upload a reference folder (as ZIP) and a single query image to compare visual characteristics.")

# --- Utility functions ---
def extract_zip_to_temp(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    image_files = [
        os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return image_files

def compute_histogram_similarity(img1, img2):
    h1 = np.histogram(np.array(img1).ravel(), bins=256, range=(0, 255))[0]
    h2 = np.histogram(np.array(img2).ravel(), bins=256, range=(0, 255))[0]
    return 1 - pairwise_distances([h1], [h2], metric='cosine')[0][0]

def compute_texture_similarity(img1, img2):
    gray1, gray2 = rgb2gray(img1), rgb2gray(img2)
    glcm1 = graycomatrix((gray1 * 255).astype('uint8'), [1], [0], symmetric=True, normed=True)
    glcm2 = graycomatrix((gray2 * 255).astype('uint8'), [1], [0], symmetric=True, normed=True)
    t1 = graycoprops(glcm1, 'contrast')[0, 0]
    t2 = graycoprops(glcm2, 'contrast')[0, 0]
    return 1 - abs(t1 - t2) / max(t1, t2, 1e-5)

def compute_structure_similarity(img1, img2):
    gray1, gray2 = rgb2gray(img1), rgb2gray(img2)
    score, _ = ssim(gray1, gray2, full=True)
    return max(score, 0)

def compute_brightness_similarity(img1, img2):
    return 1 - abs(np.mean(np.array(img1)) - np.mean(np.array(img2))) / 255

def compute_color_harmony_similarity(img1, img2):
    avg1 = np.mean(np.array(img1).reshape(-1, 3), axis=0)
    avg2 = np.mean(np.array(img2).reshape(-1, 3), axis=0)
    c1 = convert_color(sRGBColor(*avg1/255), LabColor)
    c2 = convert_color(sRGBColor(*avg2/255), LabColor)
    delta_e = np.linalg.norm([c1.lab_l - c2.lab_l, c1.lab_a - c2.lab_a, c1.lab_b - c2.lab_b])
    return max(0, 1 - delta_e / 100)

def generate_color_palette(img, n_colors=10):
    arr = np.array(img).reshape(-1, 3)
    idx = np.random.choice(len(arr), size=min(10000, len(arr)), replace=False)
    sample = arr[idx]
    colors, counts = np.unique(sample, axis=0, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1][:n_colors]
    top_colors = colors[sorted_idx]
    bar = np.zeros((50, n_colors * 50, 3), dtype=np.uint8)
    for i, c in enumerate(top_colors):
        bar[:, i * 50:(i + 1) * 50] = c
    return bar, [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in top_colors]

def display_similarity_analysis(metric_scores, palette_img=None):
    metric_explanations = {
        "Color Histogram Match": {
            "desc": "Compares hue and intensity distribution between images.",
            "interpret": lambda v: (
                "High overlap in tonal structure." if v > 0.8 else
                "Moderate overlap in overall hue distribution." if v > 0.5 else
                "Distinct hue or brightness distribution."
            )
        },
        "Texture Entropy Similarity": {
            "desc": "Compares pixel variance and micro-patterning.",
            "interpret": lambda v: (
                "Strong similarity in surface detail." if v > 0.8 else
                "Some shared texture complexity." if v > 0.5 else
                "Distinct surface structure or pattern density."
            )
        },
        "Structural Pattern Consistency": {
            "desc": "Compares repeating forms and edge layout.",
            "interpret": lambda v: (
                "Strong geometric alignment." if v > 0.8 else
                "Moderate correspondence in visual rhythm." if v > 0.5 else
                "Divergent structural organization."
            )
        },
        "Mean Brightness Proximity": {
            "desc": "Measures tonal lightness similarity.",
            "interpret": lambda v: (
                "Nearly identical luminance range." if v > 0.8 else
                "Comparable exposure and tone balance." if v > 0.5 else
                "Different overall brightness."
            )
        },
        "Color Harmony Distance": {
            "desc": "Compares warm–cool tonal balance.",
            "interpret": lambda v: (
                "Very close chromatic temperature." if v > 0.8 else
                "Similar color warmth balance." if v > 0.5 else
                "Different overall tone temperature."
            )
        }
    }

    names = list(metric_scores.keys())
    values = [metric_scores[n] for n in names]
    col1, col2 = st.columns([1, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(3, len(names) * 0.35))
        bars = ax.barh(names, values)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        if palette_img is not None:
            st.image(palette_img, caption="Color Palette Comparison", use_container_width=True)

    with col2:
        st.markdown("### Technical Image Similarity Report")
        for name, value in metric_scores.items():
            info = metric_explanations.get(name, {})
            desc = info.get("desc", "No description available.")
            interpret = info.get("interpret", lambda v: "No interpretation available.")(value)
            st.markdown(f"**{name}** — {desc}")
            st.caption(f"**Analysis:** {interpret}")
            st.markdown(f"**Score:** `{value:.2f}`")
            st.divider()

# --- Streamlit UI ---
uploaded_zip = st.file_uploader("Upload ZIP of reference images", type="zip")
uploaded_query = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])

if uploaded_zip and uploaded_query:
    with st.spinner("Processing images..."):
            st.write("✅ ZIP uploaded:", uploaded_zip.name)
            st.write("✅ Query image uploaded:", uploaded_query.name)
        ref_images = extract_zip_to_temp(uploaded_zip)
        query_img = Image.open(uploaded_query).convert("RGB")
        st.write(f"📸 Found {len(ref_images)} reference images.")


        results = []
        for ref_path in ref_images:
            ref_img = Image.open(ref_path).convert("RGB").resize(query_img.size)

            metrics = {
                "Color Histogram Match": compute_histogram_similarity(query_img, ref_img),
                "Texture Entropy Similarity": compute_texture_similarity(query_img, ref_img),
                "Structural Pattern Consistency": compute_structure_similarity(query_img, ref_img),
                "Mean Brightness Proximity": compute_brightness_similarity(query_img, ref_img),
                "Color Harmony Distance": compute_color_harmony_similarity(query_img, ref_img)
            }

            avg_score = np.mean(list(metrics.values()))
            results.append((ref_path, metrics, avg_score))

        results = sorted(results, key=lambda x: x[2], reverse=True)[:5]

if not results:
st.error("No valid image comparisons found. Check that your ZIP contains only JPG or PNG images.")
else:
    st.subheader("Top 5 Similar Images")
    
st.subheader("Top 5 Similar Images")
        for i, (path, metrics, score) in enumerate(results, start=1):
            st.markdown(f"### {i}. {os.path.basename(path)} (Overall Similarity: `{score:.2f}`)")
            col1, col2 = st.columns(2)
            with col1:
                st.image(query_img, caption="Query Image", use_container_width=True)
            with col2:
                st.image(path, caption="Reference Image", use_container_width=True)

            bar1, labels1 = generate_color_palette(query_img)
            bar2, labels2 = generate_color_palette(Image.open(path))
            combined_palette = np.vstack([bar1, bar2])
            st.image(combined_palette, caption=f"Palette Comparison: {', '.join(labels1[:5])}", use_container_width=True)

            display_similarity_analysis(metrics, palette_img=combined_palette)
            st.divider()

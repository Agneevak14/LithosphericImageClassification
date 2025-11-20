# # app.py
# import cv2
# import numpy as np
# import streamlit as st
# from plotly import graph_objects as go

# # MUST be the first Streamlit call
# st.set_page_config(page_title="Interactive K-Means Segmentation", layout="wide")

# @st.cache_data(show_spinner=False)
# def load_image(path_or_bytes):
#     if isinstance(path_or_bytes, str):
#         bgr = cv2.imread(path_or_bytes)
#         if bgr is None:
#             raise FileNotFoundError(f"Could not load image at: {path_or_bytes}")
#     else:
#         file_bytes = np.asarray(bytearray(path_or_bytes.read()), dtype=np.uint8)
#         bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     return rgb

# def run_kmeans(image_rgb, k, attempts=10, max_iter=100, epsilon=0.2, sample_pixels=None, init="pp"):
#     H, W, _ = image_rgb.shape
#     pixels = image_rgb.reshape(-1, 3).astype(np.float32)

#     if sample_pixels is not None and sample_pixels < len(pixels):
#         idx = np.random.choice(len(pixels), size=sample_pixels, replace=False)
#         pixels_used = pixels[idx]
#     else:
#         pixels_used = pixels

#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
#     flags = cv2.KMEANS_PP_CENTERS if init == "pp" else cv2.KMEANS_RANDOM_CENTERS

#     compactness, labels, centers = cv2.kmeans(pixels_used, k, None, criteria, attempts, flags)

#     # If subsampled, label all pixels by nearest center
#     if sample_pixels is not None and sample_pixels < len(pixels):
#         dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
#         full_labels = np.argmin(dists, axis=1).astype(np.int32)
#     else:
#         full_labels = labels.flatten()

#     centers_u8 = np.uint8(centers)
#     seg_map = full_labels.reshape(H, W)
#     return seg_map, centers_u8, float(compactness)

# def contour_overlay(image_rgb, seg_map, centers):
#     overlay = image_rgb.copy()
#     for i in range(len(centers)):
#         mask = (seg_map == i).astype(np.uint8)
#         contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         color_bgr = tuple(int(c) for c in centers[i][::-1])  # centers are RGB
#         cv2.drawContours(overlay, contours, -1, color_bgr, 2)
#     return overlay

# def contours_on_white(image_rgb, seg_map, centers):
#     white = np.ones_like(image_rgb) * 255
#     for i in range(len(centers)):
#         mask = (seg_map == i).astype(np.uint8)
#         contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         color_bgr = tuple(int(c) for c in centers[i][::-1])  # centers are RGB
#         cv2.drawContours(white, contours, -1, color_bgr, 2)
#     return white

# # Sidebar controls
# st.sidebar.title("Controls")
# uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
# default_path = st.sidebar.text_input("...or use a file path", value="")
# use_default = st.sidebar.checkbox("Use file path above", value=False)

# k = st.sidebar.slider("Number of colors (k)", 1, 15, 5, 1)
# init_mode = st.sidebar.selectbox("Init mode", ["k-means++", "random"], index=0)
# attempts = st.sidebar.slider("Attempts", 1, 20, 10)
# max_iter = st.sidebar.slider("Max iterations", 10, 300, 100, 10)
# epsilon = st.sidebar.number_input("Epsilon (termination)", value=0.2, step=0.05, min_value=0.0)
# sample = st.sidebar.selectbox("Subsample pixels (speed-up)", ["None", "100k", "200k", "500k"], index=1)
# sample_map = {"None": None, "100k": 100_000, "200k": 200_000, "500k": 500_000}
# sample_pixels = sample_map[sample]

# view = st.sidebar.radio(
#     "View",
#     [
#         "Original",
#         "Segmented (flat colors)",
#         "Contours overlay",
#         "Overlay with alpha",
#         "Contours on white background"
#     ]
# )
# alpha = st.sidebar.slider("Overlay alpha", 0.0, 1.0, 0.5) if view == "Overlay with alpha" else 0.5

# # Load image
# if uploaded is not None and not use_default:
#     image_rgb = load_image(uploaded)
# elif use_default and default_path.strip():
#     image_rgb = load_image(default_path.strip())
# else:
#     st.info("Upload an image or provide a valid file path to begin.")
#     st.stop()

# st.write(f"**Image size:** {image_rgb.shape[1]}×{image_rgb.shape[0]}  |  **k = {k}**")

# # Run K-means
# seg_map, centers, compactness = run_kmeans(
#     image_rgb,
#     k=k,
#     attempts=attempts,
#     max_iter=max_iter,
#     epsilon=epsilon,
#     sample_pixels=sample_pixels,
#     init="pp" if init_mode == "k-means++" else "random",
# )

# # Build display image
# H, W, _ = image_rgb.shape
# segmented_flat = centers[seg_map].reshape(H, W, 3)

# if view == "Original":
#     disp = image_rgb
# elif view == "Segmented (flat colors)":
#     disp = segmented_flat
# elif view == "Contours overlay":
#     disp = contour_overlay(image_rgb, seg_map, centers)
# elif view == "Contours on white background":
#     disp = contours_on_white(image_rgb, seg_map, centers)
# else:
#     overlay = segmented_flat
#     disp = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)

# # Plot image (scrollZoom enabled)
# fig = go.Figure(go.Image(z=disp))
# fig.update_layout(
#     margin=dict(l=0, r=0, t=30, b=0),
#     title=f"K-Means (k={k}) | Compactness: {compactness:.2f}",
#     dragmode="pan",
# )
# st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# # Summary table
# st.subheader("Color Coverage")
# total = H * W
# rows = []
# for i in range(k):
#     pct = 100.0 * np.sum(seg_map == i) / total
#     rgb = tuple(int(v) for v in centers[i])
#     rows.append({"cluster": i + 1, "RGB center": rgb, "coverage %": round(pct, 2)})

# st.dataframe(rows, use_container_width=True)

# app.py
import cv2
import numpy as np
import streamlit as st
from plotly import graph_objects as go
import pandas as pd

# MUST be the first Streamlit call
st.set_page_config(page_title="Interactive K-Means Segmentation", layout="wide")

@st.cache_data(show_spinner=False)
def load_image(path_or_bytes):
    if isinstance(path_or_bytes, str):
        bgr = cv2.imread(path_or_bytes)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image at: {path_or_bytes}")
    else:
        file_bytes = np.asarray(bytearray(path_or_bytes.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def run_kmeans(image_rgb, k, attempts=10, max_iter=100, epsilon=0.2, sample_pixels=None, init="pp"):
    H, W, _ = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    if sample_pixels is not None and sample_pixels < len(pixels):
        idx = np.random.choice(len(pixels), size=sample_pixels, replace=False)
        pixels_used = pixels[idx]
    else:
        pixels_used = pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    flags = cv2.KMEANS_PP_CENTERS if init == "pp" else cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(pixels_used, k, None, criteria, attempts, flags)

    # If subsampled, label all pixels by nearest center
    if sample_pixels is not None and sample_pixels < len(pixels):
        dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        full_labels = np.argmin(dists, axis=1).astype(np.int32)
    else:
        full_labels = labels.flatten()

    centers_u8 = np.uint8(centers)
    seg_map = full_labels.reshape(H, W)
    return seg_map, centers_u8, float(compactness)

def contour_overlay(image_rgb, seg_map, centers):
    overlay = image_rgb.copy()
    for i in range(len(centers)):
        mask = (seg_map == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_bgr = tuple(int(c) for c in centers[i][::-1])  # centers are RGB
        cv2.drawContours(overlay, contours, -1, color_bgr, 2)
    return overlay

def contours_on_white(image_rgb, seg_map, centers):
    white = np.ones_like(image_rgb) * 255
    for i in range(len(centers)):
        mask = (seg_map == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_bgr = tuple(int(c) for c in centers[i][::-1])  # centers are RGB
        cv2.drawContours(white, contours, -1, color_bgr, 2)
    return white

# Sidebar controls
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
default_path = st.sidebar.text_input("...or use a file path", value="")
use_default = st.sidebar.checkbox("Use file path above", value=False)

k = st.sidebar.slider("Number of colors (k)", 1, 50, 5, 1)
init_mode = st.sidebar.selectbox("Init mode", ["k-means++", "random"], index=0)
attempts = st.sidebar.slider("Attempts", 1, 20, 1)
max_iter = st.sidebar.slider("Max iterations", 1, 30, 100, 1)
epsilon = st.sidebar.number_input("Epsilon (termination)", value=0.2, step=0.05, min_value=0.0)
sample = st.sidebar.selectbox("Subsample pixels (speed-up)", ["None", "100k", "200k", "500k"], index=1)
sample_map = {"None": None, "100k": 100_000, "200k": 200_000, "500k": 500_000}
sample_pixels = sample_map[sample]

view = st.sidebar.radio(
    "View",
    [
        "Original",
        "Segmented (flat colors)",
        "Contours overlay",
        "Overlay with alpha",
        "Contours on white background"
    ]
)
alpha = st.sidebar.slider("Overlay alpha", 0.0, 1.0, 0.5) if view == "Overlay with alpha" else 0.5

# Load image
if uploaded is not None and not use_default:
    image_rgb = load_image(uploaded)
elif use_default and default_path.strip():
    image_rgb = load_image(default_path.strip())
else:
    st.info("Upload an image or provide a valid file path to begin.")
    st.stop()

st.write(f"**Image size:** {image_rgb.shape[1]}×{image_rgb.shape[0]}  |  **k = {k}**")

# Run K-means
seg_map, centers, compactness = run_kmeans(
    image_rgb,
    k=k,
    attempts=attempts,
    max_iter=max_iter,
    epsilon=epsilon,
    sample_pixels=sample_pixels,
    init="pp" if init_mode == "k-means++" else "random",
)

# Build display image
H, W, _ = image_rgb.shape
segmented_flat = centers[seg_map].reshape(H, W, 3)

if view == "Original":
    disp = image_rgb
elif view == "Segmented (flat colors)":
    disp = segmented_flat
elif view == "Contours overlay":
    disp = contour_overlay(image_rgb, seg_map, centers)
elif view == "Contours on white background":
    disp = contours_on_white(image_rgb, seg_map, centers)
else:
    overlay = segmented_flat
    disp = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)

# Plot image (scrollZoom enabled)
fig = go.Figure(go.Image(z=disp))
fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),
    title=f"K-Means (k={k}) | Compactness: {compactness:.2f}",
    dragmode="pan",
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# Summary table with color swatches
st.subheader("Color Coverage")
total = H * W
rows = []
for i in range(k):
    pct = 100.0 * np.sum(seg_map == i) / total
    rgb = tuple(int(v) for v in centers[i])         # (R, G, B)
    hex_color = '#%02x%02x%02x' % rgb
    swatch_html = f'<div style="width:36px;height:20px;border-radius:4px;border:1px solid #ddd;background-color:{hex_color}"></div>'
    rows.append({
        "cluster": i + 1,
        "RGB center": swatch_html,                  # show color swatch instead of coordinates
        "coverage %": round(pct, 2)
    })

df = pd.DataFrame(rows)
st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)


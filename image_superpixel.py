# app.py (top of file)

import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go

# 👉 FIRST Streamlit call must be page_config
st.set_page_config(page_title="Interactive K-Means Segmentation", layout="wide")

# Try scikit-image; delay any Streamlit messages until after page_config
SKIMAGE_OK = True
SKIMAGE_ERR = None
try:
    from skimage.segmentation import slic, expand_labels
    from skimage.morphology import erosion, disk
except Exception as e:
    SKIMAGE_OK = False
    SKIMAGE_ERR = e

# Now it's safe to use st.* calls
if not SKIMAGE_OK:
    st.warning(
        "scikit-image not found. Please add `scikit-image` to your environment or requirements.txt. "
        "The Superpixel mode requires it.\n\n"
        f"(Import error: {SKIMAGE_ERR})"
    )



# =========================
# Utilities
# =========================
@st.cache_data(show_spinner=False)
def load_image(path_or_bytes):
    """Load image from path or uploaded file-like object. Returns RGB uint8 image."""
    if isinstance(path_or_bytes, str):
        bgr = cv2.imread(path_or_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image at: {path_or_bytes}")
    else:
        file_bytes = np.asarray(bytearray(path_or_bytes.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Uploaded file is not a valid image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def resize_image_nn(img, scale):
    """Nearest-neighbor resize by scale factor (0<scale<=1 for downscale, >1 for up)."""
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def run_kmeans_per_pixel(image_rgb, k, attempts=10, max_iter=100, epsilon=0.2, sample_pixels=None, init="pp", seed=0):
    """Classic per-pixel K-means in RGB space. Optionally subsample pixels for speed."""
    H, W, _ = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    # optional subsampling
    if sample_pixels is not None and sample_pixels < len(pixels):
        if seed is not None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(pixels), size=sample_pixels, replace=False)
        else:
            idx = np.random.choice(len(pixels), size=sample_pixels, replace=False)
        pixels_used = pixels[idx]
    else:
        pixels_used = pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iter), float(epsilon))
    flags = cv2.KMEANS_PP_CENTERS if init == "pp" else cv2.KMEANS_RANDOM_CENTERS

    compactness, labels_used, centers = cv2.kmeans(pixels_used, int(k), None, criteria, int(attempts), flags)

    # If we subsampled, assign all pixels to nearest center
    if pixels_used is not pixels:
        # Euclidean distance to centers
        dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        full_labels = np.argmin(dists, axis=1).astype(np.int32)
    else:
        full_labels = labels_used.flatten().astype(np.int32)

    centers_u8 = np.uint8(centers)
    seg_map = full_labels.reshape(H, W)
    return seg_map, centers_u8, float(compactness)

def compute_superpixel_means(image_rgb, sp_labels):
    """Compute mean RGB per superpixel label efficiently."""
    H, W = sp_labels.shape
    n_sp = int(sp_labels.max()) + 1
    flat_img = image_rgb.reshape(-1, 3).astype(np.float64)
    flat_lab = sp_labels.reshape(-1).astype(np.int64)

    sums = np.zeros((n_sp, 3), dtype=np.float64)
    counts = np.zeros(n_sp, dtype=np.int64)

    np.add.at(sums, flat_lab, flat_img)
    np.add.at(counts, flat_lab, 1)

    means = sums / np.maximum(counts[:, None], 1)
    return means.astype(np.float32)  # shape: (n_sp, 3)

def kmeans_on_features(features, k, attempts=10, max_iter=100, epsilon=0.2, init="pp"):
    """Run OpenCV K-means on feature matrix (N x D). Returns (labels, centers, compactness)."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iter), float(epsilon))
    flags = cv2.KMEANS_PP_CENTERS if init == "pp" else cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(features.astype(np.float32), int(k), None, criteria, int(attempts), flags)
    return labels.flatten().astype(np.int32), centers, float(compactness)

def superpixel_cluster(
    image_rgb,
    k,
    num_segments=400,
    compactness=10.0,
    slic_sigma=1.0,
    attempts=10,
    max_iter=100,
    epsilon=0.2,
    init="pp",
):
    """
    1) SLIC superpixels -> sp_labels (H,W)
    2) Mean RGB per superpixel -> sp_means (n_sp,3)
    3) K-means on sp_means -> cluster id per superpixel
    4) Broadcast to pixels -> cluster_map (H,W)
    """
    sp_labels = slic(
        image_rgb,
        n_segments=int(num_segments),
        compactness=float(compactness),
        sigma=float(slic_sigma),
        start_label=0
    ).astype(np.int32)

    sp_means = compute_superpixel_means(image_rgb, sp_labels)  # (n_sp,3)
    sp_to_cluster, centers, compactness_val = kmeans_on_features(
        sp_means, k, attempts=attempts, max_iter=max_iter, epsilon=epsilon, init=init
    )
    cluster_map = sp_to_cluster[sp_labels]  # (H,W)
    centers_u8 = np.uint8(centers[:, :3])   # Ensure (k,3) uint8 for coloring
    return cluster_map, centers_u8, compactness_val

def shrink_labels_per_id(label_map, shrink_px):
    """Per-label erosion. Returns a label map where each region is eroded."""
    if shrink_px <= 0:
        return label_map
    se = disk(int(shrink_px))
    out = np.zeros_like(label_map, dtype=label_map.dtype)
    for lab in np.unique(label_map):
        if lab == 0:
            continue
        mask = (label_map == lab).astype(np.uint8)
        er = erosion(mask, se)
        out[er.astype(bool)] = lab
    return out

def colorize_by_centers(label_map, centers):
    """Create flat-color visualization using centers[k,3] (RGB)."""
    H, W = label_map.shape
    safe_labels = np.clip(label_map, 0, len(centers) - 1)
    colored = centers[safe_labels].reshape(H, W, 3).astype(np.uint8)
    return colored

def draw_contours(image_rgb, label_map, centers):
    """Draw cluster contours on the original image using each cluster's color."""
    overlay = image_rgb.copy()
    k = centers.shape[0]
    for i in range(k):
        mask = (label_map == i).astype(np.uint8)
        if mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_bgr = tuple(int(c) for c in centers[i][::-1])  # centers are RGB, OpenCV expects BGR
        cv2.drawContours(overlay, contours, -1, color_bgr, 2)
    return overlay

def draw_contours_on_white(label_map, centers):
    """Draw cluster contours on a white canvas."""
    H, W = label_map.shape
    white = np.ones((H, W, 3), dtype=np.uint8) * 255
    k = centers.shape[0]
    for i in range(k):
        mask = (label_map == i).astype(np.uint8)
        if mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_bgr = tuple(int(c) for c in centers[i][::-1])
        cv2.drawContours(white, contours, -1, color_bgr, 2)
    return white

def to_png_bytes(image_rgb):
    """Encode RGB uint8 image to PNG bytes."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    return buf.tobytes()

def mask_to_png_bytes(label_map):
    """Save label map as 16-bit PNG (to preserve many labels)."""
    # Normalize label map to 16-bit range if needed
    if label_map.dtype != np.uint16:
        if label_map.max() <= np.iinfo(np.uint16).max:
            out = label_map.astype(np.uint16)
        else:
            raise ValueError("Too many labels to store in uint16 PNG.")
    else:
        out = label_map
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("PNG encoding (mask) failed.")
    return buf.tobytes()

def hex_from_rgb(rgb_tuple):
    r, g, b = (int(v) for v in rgb_tuple)
    return f"#{r:02x}{g:02x}{b:02x}"

# =========================
# Sidebar Controls
# =========================
st.sidebar.title("Controls")

# Load image
uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
default_path = st.sidebar.text_input("...or use a file path", value="")
use_default = st.sidebar.checkbox("Use file path above", value=False)

# Global controls
seed = st.sidebar.number_input("Random seed (reproducibility)", value=0, step=1, min_value=0)
if seed:
    np.random.seed(int(seed))

downscale = st.sidebar.slider("Downscale for speed (×)", 0.1, 1.0, 1.0, 0.05, help="Processes a smaller copy, then upsamples labels back.")
mode = st.sidebar.selectbox("Segmentation mode", ["Per-pixel K-means", "Superpixel clusters"], index=1)

# K-means controls
k = st.sidebar.slider("Number of clusters (k)", 1, 50, 5, 1)
init_mode = st.sidebar.selectbox("Init mode", ["k-means++", "random"], index=0)
attempts = st.sidebar.slider("Attempts", 1, 20, 10)
max_iter = st.sidebar.slider("Max iterations", 10, 300, 100, 10)
epsilon = st.sidebar.number_input("Epsilon (termination)", value=0.2, step=0.05, min_value=0.0)

# Per-pixel optional subsampling
if mode == "Per-pixel K-means":
    sample = st.sidebar.selectbox("Subsample pixels (speed-up)", ["None", "100k", "200k", "500k"], index=1)
    sample_map = {"None": None, "100k": 100_000, "200k": 200_000, "500k": 500_000}
    sample_pixels = sample_map[sample]
else:
    sample_pixels = None

# Superpixel parameters
if mode == "Superpixel clusters":
    num_segments = st.sidebar.slider("Superpixels (approx count)", 50, 2000, 400, 50)
    slic_compact = st.sidebar.slider("SLIC compactness (color vs space)", 1.0, 40.0, 10.0, 0.5)
    slic_sigma = st.sidebar.slider("SLIC smoothing (sigma)", 0.0, 5.0, 1.0, 0.5)
    shrink_px = st.sidebar.slider("Shrink regions (px)", 0, 50, 0, 1)
    grow_px = st.sidebar.slider("Grow regions (px)", 0, 50, 0, 1)
else:
    num_segments = slic_compact = slic_sigma = shrink_px = grow_px = 0

# View / overlay
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

# =========================
# Load image or stop
# =========================
if uploaded is not None and not use_default:
    image_rgb_full = load_image(uploaded)
elif use_default and default_path.strip():
    image_rgb_full = load_image(default_path.strip())
else:
    st.info("Upload an image or provide a valid file path to begin.")
    st.stop()

H_full, W_full, _ = image_rgb_full.shape

# Optionally downscale for processing speed
image_rgb_proc = resize_image_nn(image_rgb_full, downscale)
H, W, _ = image_rgb_proc.shape

st.write(f"**Image size:** {W_full}×{H_full} (processing at {W}×{H}, scale ×{downscale:.2f})  |  **k = {k}**")

# =========================
# Segmentation
# =========================
if mode == "Per-pixel K-means":
    seg_map_proc, centers, compactness = run_kmeans_per_pixel(
        image_rgb_proc,
        k=k,
        attempts=attempts,
        max_iter=max_iter,
        epsilon=epsilon,
        sample_pixels=sample_pixels,
        init="pp" if init_mode == "k-means++" else "random",
        seed=int(seed) if seed else None
    )
    # If we processed at lower resolution, upsample the label map back to full size (NN)
    if downscale != 1.0:
        seg_map_full = resize_image_nn(seg_map_proc.astype(np.int32), 1.0 / downscale).astype(np.int32)
    else:
        seg_map_full = seg_map_proc
    final_label_map = seg_map_full

else:  # Superpixel clusters
    # Safety if scikit-image is missing
    try:
        seg_map_proc, centers, compactness = superpixel_cluster(
            image_rgb_proc,
            k=k,
            num_segments=num_segments,
            compactness=slic_compact,
            slic_sigma=slic_sigma,
            attempts=attempts,
            max_iter=max_iter,
            epsilon=epsilon,
            init="pp" if init_mode == "k-means++" else "random",
        )
    except NameError:
        st.error("Superpixel mode requires scikit-image. Please install it and restart.")
        st.stop()

    # Apply shrink then grow on the processed-resolution label map
    if shrink_px > 0:
        seg_map_proc = shrink_labels_per_id(seg_map_proc, shrink_px)
        # optional small fill if everything disappears in thin regions
        seg_map_proc = expand_labels(seg_map_proc, distance=1)

    if grow_px > 0:
        seg_map_proc = expand_labels(seg_map_proc, distance=int(grow_px))

    # Upsample labels back to full size (nearest neighbor)
    if downscale != 1.0:
        seg_map_full = resize_image_nn(seg_map_proc.astype(np.int32), 1.0 / downscale).astype(np.int32)
    else:
        seg_map_full = seg_map_proc

    final_label_map = seg_map_full

# =========================
# Build Display Image
# =========================
segmented_flat = colorize_by_centers(final_label_map, centers)

if view == "Original":
    disp = image_rgb_full
elif view == "Segmented (flat colors)":
    disp = segmented_flat
elif view == "Contours overlay":
    disp = draw_contours(image_rgb_full, final_label_map, centers)
elif view == "Contours on white background":
    disp = draw_contours_on_white(final_label_map, centers)
else:  # Overlay with alpha
    overlay = segmented_flat
    disp = cv2.addWeighted(image_rgb_full, 1 - alpha, overlay, alpha, 0)

# =========================
# Plot original vs processed (side by side)
# =========================
col1, col2 = st.columns(2)

with col1:
    fig_orig = go.Figure(go.Image(z=image_rgb_full))
    fig_orig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title="Original",
        dragmode="pan",
    )
    st.plotly_chart(fig_orig, use_container_width=True, config={"scrollZoom": True})

with col2:
    fig_proc = go.Figure(go.Image(z=disp))
    fig_proc.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{view} | Mode: {mode} | k={k} | Compactness: {compactness:.2f}",
        dragmode="pan",
    )
    st.plotly_chart(fig_proc, use_container_width=True, config={"scrollZoom": True})


# =========================
# Summary table
# =========================
st.subheader("Color Coverage")
total = final_label_map.size
rows = []
for i in range(k):
    cnt = int(np.sum(final_label_map == i))
    pct = 100.0 * cnt / total if total > 0 else 0.0
    rgb = tuple(int(v) for v in centers[i, :3])  # (R,G,B)
    hex_color = hex_from_rgb(rgb)
    swatch = f'<div style="width:36px;height:20px;border-radius:4px;border:1px solid #ddd;background-color:{hex_color}"></div>'
    rows.append({
        "cluster": i + 1,
        "RGB center": f"{rgb} {swatch}",
        "coverage %": round(pct, 2),
        "#pixels": cnt
    })

df = pd.DataFrame(rows)
st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# =========================
# Downloads
# =========================
st.subheader("Downloads")

# Ensure label map fits in uint16 for PNG export
if final_label_map.max() > np.iinfo(np.uint16).max:
    st.warning("Too many unique labels to store in 16-bit PNG. Use .npy download for the mask.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    mask_npy = io.BytesIO()
    np.save(mask_npy, final_label_map.astype(np.int32))
    st.download_button("Mask (.npy)", data=mask_npy.getvalue(), file_name="segmentation_mask.npy", mime="application/octet-stream")

with col2:
    try:
        st.download_button("Mask (.png 16-bit)", data=mask_to_png_bytes(final_label_map), file_name="segmentation_mask.png", mime="image/png")
    except Exception as e:
        st.caption("Mask PNG export not available (too many labels or encoding issue).")

with col3:
    st.download_button("Segmented (flat).png", data=to_png_bytes(segmented_flat), file_name="segmented_flat.png", mime="image/png")

with col4:
    if view == "Overlay with alpha":
        overlay_png = to_png_bytes(disp)
    else:
        overlay_img = cv2.addWeighted(image_rgb_full, 1 - alpha, segmented_flat, alpha, 0)
        overlay_png = to_png_bytes(overlay_img)
    st.download_button("Overlay (alpha).png", data=overlay_png, file_name="overlay_alpha.png", mime="image/png")

# =========================
# Notes / Tips
# =========================
with st.expander("Help & Notes", expanded=False):
    st.markdown(
        """
- **Per-pixel K-means**: classic color quantization; can be slow on very large images unless you subsample.
- **Superpixel clusters**: first oversegments into coherent regions (SLIC), then clusters superpixel means; much faster on large images and gives chunkier regions.
- **Grow/Shrink** apply to the **label map** (region geometry). Order matters: shrink then grow can remove spurious filaments and smooth the shapes.
- **Downscale** speeds up processing; labels are upsampled back with nearest-neighbor to match the original size.
- **Compactness value shown in the title** is the K-means within-cluster sum of squares returned by OpenCV (lower is tighter clusters).
        """
    )

"""Interactive Unit 2 segmentation and edge-detection laboratory.

Syllabus: Unit 2 - image segmentation
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from numpy.typing import NDArray

from docuvision.segmentation import (
    adaptive_threshold,
    canny,
    connectivity_comparison,
    connectivity_difference,
    detect_lines,
    detect_points,
    dog_edges,
    gradient_edges,
    log_edges,
    otsu_cv,
    region_growing,
    sauvola,
    split_and_merge,
    watershed_split,
)
from docuvision.ui.common import gray, require_image, syllabus_info


def _labels_rgb(labels: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Map integer segmentation labels to a repeatable display palette.

    Args:
        labels: Two-dimensional integer component or region label map.

    Returns:
        Three-channel BGR color visualization.

    Syllabus: Unit 2 - segmentation visualization
    """
    values = np.asarray(labels, dtype=np.int32)
    scaled = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def render(image: NDArray[np.generic] | None) -> None:
    """Render edge operators and region/threshold segmentation examples.

    Args:
        image: Shared grayscale or BGR source image or None.

    Returns:
        None.

    Syllabus: Unit 2 - image segmentation
    """
    syllabus_info("gradient edges, point/line masks, region growing, connected components", "Lab 3")
    source = require_image(image)
    if source is None:
        return
    gray_image = gray(source)
    st.subheader("Edges and textbook masks")
    edge_method = st.selectbox(
        "Edge detector",
        ("Sobel", "Prewitt", "Roberts", "LoG", "Canny", "DoG", "Point", "Line"),
    )
    if edge_method in ("Sobel", "Prewitt", "Roberts"):
        gx, gy, magnitude, direction = gradient_edges(gray_image, edge_method.lower())
        columns = st.columns(4)
        for column, title, output in zip(
            columns,
            ("Gx", "Gy", "Magnitude", "Direction"),
            (gx, gy, magnitude, direction),
        ):
            column.image(output, caption=title, clamp=True, use_container_width=True)
    elif edge_method == "LoG":
        sigma = st.slider("LoG sigma", 0.5, 5.0, 1.4, 0.1)
        st.image(log_edges(gray_image, sigma), caption="Zero crossings", clamp=True, use_container_width=True)
    elif edge_method == "Canny":
        threshold_span = st.slider("Automatic threshold spread", 0.0, 0.8, 0.33, 0.01)
        st.image(canny(gray_image, threshold_span), caption="Canny edges", clamp=True, use_container_width=True)
    elif edge_method == "DoG":
        narrow = st.slider("Narrow sigma", 0.3, 3.0, 1.0, 0.1)
        st.image(dog_edges(gray_image, narrow, narrow * 1.6), caption="Difference of Gaussians", clamp=True, use_container_width=True)
    elif edge_method == "Point":
        threshold = st.slider("Point response threshold", 0, 500, 80)
        st.image(detect_points(gray_image, threshold), caption="Point detections", clamp=True, use_container_width=True)
    else:
        direction = st.selectbox(
            "Line direction",
            ("horizontal", "vertical", "diagonal_45", "diagonal_135"),
        )
        st.image(detect_lines(gray_image, direction), caption=f"{direction} line response", clamp=True, use_container_width=True)

    st.subheader("Threshold and region segmentation")
    threshold_method = st.selectbox("Threshold method", ("Otsu", "Adaptive", "Sauvola"), key="seg_thresh")
    if threshold_method == "Otsu":
        threshold = otsu_cv(gray_image)
        binary = np.where(gray_image > threshold, 255, 0).astype(np.uint8)
        st.caption(f"Otsu threshold: {threshold}")
    elif threshold_method == "Adaptive":
        block = min(31, max(3, min(gray_image.shape) | 1))
        block = block if block % 2 else block - 1
        binary = adaptive_threshold(gray_image, "gaussian", block, C=2.0)
    else:
        window = min(31, max(3, min(gray_image.shape) | 1))
        window = window if window % 2 else window - 1
        binary = sauvola(gray_image, window, 0.2)
    st.image(binary, caption=f"{threshold_method} threshold", clamp=True, use_container_width=True)
    left, right = st.columns(2)
    seed_x = left.number_input("Seed x", min_value=0, max_value=gray_image.shape[1] - 1, value=gray_image.shape[1] // 2)
    seed_y = right.number_input("Seed y", min_value=0, max_value=gray_image.shape[0] - 1, value=gray_image.shape[0] // 2)
    tolerance = st.slider("Region intensity tolerance", 0, 100, 10)
    connectivity = st.selectbox("Region connectivity", (4, 8), index=1)
    region = region_growing(gray_image, [(int(seed_y), int(seed_x))], tolerance, connectivity)
    st.image(region, caption="Region-growing mask", clamp=True, use_container_width=True)

    st.subheader("Split-and-merge and connectivity")
    if max(gray_image.shape) <= 256:
        split_size = max(2, min(16, min(gray_image.shape) // 8))
        if split_size % 2:
            split_size += 1
        labels = split_and_merge(gray_image, min_size=split_size, std_thr=12)
        st.image(_labels_rgb(labels), caption="Split-and-merge labels", use_container_width=True)
    else:
        st.info("Split-and-merge is limited to 256 px per side. Resize in the Scan tab first.")
    binary_foreground = np.where(gray_image < otsu_cv(gray_image), 255, 0).astype(np.uint8)
    labels4, labels8 = connectivity_comparison(binary_foreground)
    count4, count8 = connectivity_difference(binary_foreground)
    st.caption(f"Foreground components — 4-connected: {count4}, 8-connected: {count8}")
    label_columns = st.columns(2)
    label_columns[0].image(_labels_rgb(labels4), caption="4-connected labels", use_container_width=True)
    label_columns[1].image(_labels_rgb(labels8), caption="8-connected labels", use_container_width=True)
    if st.checkbox("Show watershed split"):
        watershed = watershed_split(binary_foreground)
        count, _ = cv2.connectedComponents((watershed > 0).astype(np.uint8))
        st.image(_labels_rgb(watershed), caption=f"Watershed labels ({max(0, count - 1)} objects)", use_container_width=True)

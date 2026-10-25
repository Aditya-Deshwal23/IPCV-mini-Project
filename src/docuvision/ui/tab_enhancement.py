"""Interactive Unit 1 image enhancement laboratory.

Syllabus: Unit 1 - image enhancement
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray

from docuvision.enhancement import (
    absdiff,
    add,
    add_gaussian_noise,
    bit_plane_slice,
    bitwise_xor,
    clahe,
    contrast_stretch_minmax,
    digital_negative,
    divide,
    downsample,
    equalize_cv,
    gaussian_filter,
    high_boost,
    histogram_match_scratch,
    log_transform,
    mean_filter,
    median_filter,
    multiply,
    order_statistic_filter,
    power_law,
    reduce_gray_levels,
    subtract,
    unsharp_mask,
)
from docuvision.evaluation.metrics import psnr
from docuvision.ui.common import gray, require_image, syllabus_info


def _safe_kernel(requested: int, image: NDArray[np.uint8]) -> int:
    """Choose a usable odd filter width without exceeding the image dimensions.

    Args:
        requested: Desired odd square filter width.
        image: Grayscale uint8 image used to bound the width.

    Returns:
        Odd width no larger than the smallest image side, or one for tiny input.

    Syllabus: Unit 1 - spatial filtering
    """
    maximum = min(image.shape)
    if maximum < 3:
        return 1
    return min(requested, maximum if maximum % 2 else maximum - 1)


def _histogram(image: NDArray[np.uint8]) -> pd.DataFrame:
    """Calculate intensity counts in the shape expected by Streamlit charts.

    Args:
        image: Two-dimensional uint8 grayscale image.

    Returns:
        DataFrame indexed by columns intensity and pixels.

    Syllabus: Unit 1 - image histograms
    """
    counts = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    return pd.DataFrame({"intensity": np.arange(256), "pixels": counts})


def render(image: NDArray[np.generic] | None) -> None:
    """Render enhancement operations, histograms, noise filters, and sampling demos.

    Args:
        image: Shared grayscale or BGR source image or None.

    Returns:
        None.

    Syllabus: Unit 1 - image enhancement
    """
    syllabus_info("point transforms, histograms, spatial filters, sampling", "Lab 2")
    source = require_image(image)
    if source is None:
        return
    gray_image = gray(source)
    operation = st.selectbox(
        "Operation",
        [
            "negative",
            "stretch",
            "log",
            "gamma",
            "bit-plane",
            "equalize",
            "CLAHE",
            "histogram match",
            "mean filter",
            "median filter",
            "order statistic",
            "unsharp",
            "high boost",
            "image arithmetic",
        ],
        key="enh_operation",
    )
    first, second = st.columns(2)
    gamma = st.slider("Gamma", 0.2, 3.0, 1.0, 0.1, key="enh_gamma")
    bit_plane = st.slider("Bit plane", 0, 7, 7, key="enh_bit")
    kernel_request = st.slider("Filter kernel", 3, 15, 3, 2, key="enh_kernel")
    kernel = _safe_kernel(kernel_request, gray_image)
    clip = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.25)
    amount = st.slider("Sharpen amount", 0.0, 3.0, 1.0, 0.1)
    if operation == "negative":
        result = digital_negative(gray_image)
    elif operation == "stretch":
        result = contrast_stretch_minmax(gray_image)
    elif operation == "log":
        result = log_transform(gray_image)
    elif operation == "gamma":
        result = power_law(gray_image, gamma)
    elif operation == "bit-plane":
        result = bit_plane_slice(gray_image, bit_plane)
    elif operation == "equalize":
        result = equalize_cv(gray_image)
    elif operation == "CLAHE":
        result = clahe(gray_image, clip, tile=8)
    elif operation == "histogram match":
        result = histogram_match_scratch(gray_image, equalize_cv(gray_image))
    elif operation == "mean filter":
        result = mean_filter(gray_image, kernel)
    elif operation == "median filter":
        result = median_filter(gray_image, kernel)
    elif operation == "order statistic":
        kind = st.selectbox("Statistic", ("min", "max", "midpoint", "alpha_trimmed"))
        result = order_statistic_filter(gray_image, kernel, kind)
    elif operation == "unsharp":
        result = unsharp_mask(gray_image, sigma=1.0, amount=amount)
    elif operation == "high boost":
        result = high_boost(gray_image, k=amount, sigma=1.0)
    else:
        arithmetic = st.selectbox(
            "Arithmetic operation",
            ("add", "subtract", "multiply", "divide", "absolute difference", "XOR"),
        )
        operand = st.slider("Constant image value", 1, 255, 32)
        operand_image = np.full_like(gray_image, operand)
        arithmetic_functions = {
            "add": add,
            "subtract": subtract,
            "multiply": multiply,
            "divide": divide,
            "absolute difference": absdiff,
            "XOR": bitwise_xor,
        }
        result = arithmetic_functions[arithmetic](gray_image, operand_image)
    with first:
        st.image(gray_image, caption="Input", clamp=True, use_container_width=True)
        st.line_chart(_histogram(gray_image).set_index("intensity"))
    with second:
        st.image(result, caption=operation, clamp=True, use_container_width=True)
        st.line_chart(_histogram(result).set_index("intensity"))

    st.subheader("Noise and filter comparison")
    sigma = st.slider("Gaussian noise sigma", 0.0, 40.0, 12.0, 1.0)
    noise_seed = st.number_input("Noise seed", 0, 1_000_000, 123, key="enh_noise_seed")
    noisy = add_gaussian_noise(gray_image, sigma, int(noise_seed))
    variants = {
        "noisy": noisy,
        "mean": mean_filter(noisy, kernel),
        "median": median_filter(noisy, kernel),
        "gaussian": gaussian_filter(noisy, kernel),
    }
    st.image(noisy, caption="Noisy input", clamp=True, use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            [{"filter": name, "PSNR (dB)": psnr(gray_image, output)} for name, output in variants.items()]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Sampling and quantization")
    factor = st.slider("Sampling interval", 1, 8, 2)
    levels = st.slider("Gray levels", 2, 256, 8, 2)
    sampled = downsample(gray_image, factor)
    quantized = reduce_gray_levels(gray_image, levels)
    sample_columns = st.columns(3)
    sample_columns[0].image(gray_image, caption="Original", clamp=True, use_container_width=True)
    sample_columns[1].image(sampled, caption=f"Every {factor} px", clamp=True, use_container_width=True)
    sample_columns[2].image(quantized, caption=f"{levels} levels", clamp=True, use_container_width=True)

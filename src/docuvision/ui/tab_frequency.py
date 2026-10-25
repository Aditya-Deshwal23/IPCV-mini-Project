"""Interactive frequency-domain and morphology laboratory.

Syllabus: Unit 3 - frequency processing and morphology
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from numpy.typing import NDArray

from docuvision.frequency import (
    apply_frequency_filter,
    butterworth_highpass,
    butterworth_lowpass,
    fft_spectrum,
    gaussian_highpass,
    gaussian_lowpass,
    ideal_highpass,
    ideal_lowpass,
    iwht2d,
    notch_reject,
    ringing_demo,
    wht2d,
    wht_basis_images,
)
from docuvision.morphology import (
    blackhat,
    closing,
    dilate,
    erode,
    gradient,
    opening,
    structuring_element,
    tophat,
)
from docuvision.ui.common import gray, require_image, syllabus_info


def render(image: NDArray[np.generic] | None) -> None:
    """Render Fourier spectrum, filters, Walsh basis, and morphology operations.

    Args:
        image: Shared grayscale or BGR source image or None.

    Returns:
        None.

    Syllabus: Unit 3 - frequency processing and morphology
    """
    syllabus_info("Fourier spectrum, LPF/HPF, periodic noise, Walsh-Hadamard, morphology", "Lab 4")
    source = require_image(image)
    if source is None:
        return
    gray_image = gray(source)
    st.subheader("Fourier spectrum and frequency filtering")
    _, magnitude, phase = fft_spectrum(gray_image)
    spectrum_columns = st.columns(3)
    spectrum_columns[0].image(gray_image, caption="Input", use_container_width=True)
    spectrum_columns[1].image(magnitude, caption="Centered log magnitude", use_container_width=True)
    phase_display = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_columns[2].image(phase_display, caption="Phase", use_container_width=True)
    mask_kind = st.selectbox("Filter family", ("ideal", "Butterworth", "Gaussian"))
    pass_kind = st.selectbox("Pass type", ("Lowpass", "Highpass"))
    cutoff_max = max(2, min(gray_image.shape))
    cutoff = st.slider("Cutoff D0", 1, cutoff_max, min(30, cutoff_max), key="freq_d0")
    order = st.slider("Butterworth order n", 1, 8, 2)
    lowpass = pass_kind == "Lowpass"
    if mask_kind == "ideal":
        mask = ideal_lowpass(gray_image.shape, cutoff) if lowpass else ideal_highpass(gray_image.shape, cutoff)
    elif mask_kind == "Butterworth":
        mask = butterworth_lowpass(gray_image.shape, cutoff, order) if lowpass else butterworth_highpass(gray_image.shape, cutoff, order)
    else:
        mask = gaussian_lowpass(gray_image.shape, cutoff) if lowpass else gaussian_highpass(gray_image.shape, cutoff)
    filtered = apply_frequency_filter(gray_image, mask)
    filtered_column, mask_column = st.columns(2)
    filtered_column.image(filtered, caption=f"{mask_kind} {pass_kind}", use_container_width=True)
    mask_column.image(np.clip(mask * 255, 0, 255).astype(np.uint8), caption="Centered mask", use_container_width=True)
    with st.expander("Ideal versus smooth lowpass ringing demo"):
        demo_height, demo_width = min(128, gray_image.shape[0]), min(128, gray_image.shape[1])
        results = ringing_demo((max(8, demo_height), max(8, demo_width)), cutoff)
        ring_columns = st.columns(3)
        for column, key in zip(ring_columns, ("ideal", "butterworth", "gaussian")):
            column.image(results[key], caption=key.title(), use_container_width=True)
    with st.expander("Notch reject periodic-noise demo"):
        height, width = 128, 128
        _, xx = np.mgrid[:height, :width]
        moire = np.clip(128 + 65 * np.cos(2 * np.pi * 12 * xx / width), 0, 255).astype(np.uint8)
        notch_centers = [(height // 2, width // 2 + 12)]
        notch_mask = notch_reject(moire.shape, notch_centers, radius=2, n=2)
        cleaned = apply_frequency_filter(moire, notch_mask)
        st.image([moire, cleaned], caption=["Synthetic moire", "After notch reject"], use_container_width=True)

    st.subheader("Walsh-Hadamard transform")
    basis = wht_basis_images(8)
    montage = np.zeros((32, 32), dtype=np.uint8)
    for index, basis_image in enumerate(basis[:16]):
        row, column = divmod(index, 4)
        montage[row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8] = basis_image
    st.image(montage, caption="First 16 sequency-ordered basis images", use_container_width=False)
    block = cv2.resize(gray_image, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float64)
    coefficients = wht2d(block)
    retained_count = st.slider("WHT coefficients retained", 1, 64, 16)
    retained = np.zeros_like(coefficients)
    indices = np.argsort(np.abs(coefficients).ravel())[::-1][:retained_count]
    retained.ravel()[indices] = coefficients.ravel()[indices]
    reconstructed = np.clip(np.rint(iwht2d(retained)), 0, 255).astype(np.uint8)
    st.image(reconstructed, caption="WHT reconstruction (largest-magnitude coefficients)", width=200)

    st.subheader("Morphology")
    operation = st.selectbox("Operation", ("erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat"))
    shape = st.selectbox("Structuring element", ("rect", "ellipse", "cross"))
    size_max = max(1, min(31, max(gray_image.shape)))
    if size_max == 1:
        kernel_size = 1
        st.caption("Element size: 1 pixel (the image is tiny).")
    else:
        kernel_size = st.slider("Element size", 1, size_max, min(5, size_max))
    kernel = structuring_element(shape, kernel_size)
    binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    operation_map = {
        "erosion": erode,
        "dilation": dilate,
        "opening": opening,
        "closing": closing,
        "gradient": gradient,
        "tophat": tophat,
        "blackhat": blackhat,
    }
    morphed = operation_map[operation](binary, kernel)
    st.image([binary, morphed], caption=["Otsu binary", operation.title()], use_container_width=True)

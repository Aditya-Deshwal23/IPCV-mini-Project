"""Interactive Unit 6 compression and coding laboratory.

Syllabus: Unit 6 - image compression
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray

from docuvision.compression import (
    average_code_length,
    block_transform_compress,
    compare_codecs,
    huffman_code_table,
    igs_quantize,
    rate_distortion_curve,
    shannon_entropy,
)
from docuvision.enhancement import reduce_gray_levels
from docuvision.ui.common import gray, require_image, syllabus_info


@st.cache_data(show_spinner=False)
def _cached_rate_distortion(
    image_bytes: bytes, shape: tuple[int, int], transform: str, qsteps: tuple[float, ...]
) -> pd.DataFrame:
    """Cache a rate-distortion measurement for one image and transform.

    Args:
        image_bytes: Row-major uint8 image contents.
        shape: Height and width used to restore the image array.
        transform: Block transform name, WHT or DCT.
        qsteps: Quantizer step sizes to measure.

    Returns:
        DataFrame of measured compression ratio, PSNR, and SSIM results.

    Syllabus: Unit 6 - rate-distortion analysis
    """
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape).copy()
    return rate_distortion_curve(
        image,
        transform,
        [{"qstep": value} for value in qsteps],
    )


@st.cache_data(show_spinner=False)
def _cached_compare(gray_bytes: bytes, binary_bytes: bytes, shape: tuple[int, int]) -> pd.DataFrame:
    """Cache real codec round trips and serialized byte measurements.

    Args:
        gray_bytes: Row-major grayscale uint8 image contents.
        binary_bytes: Row-major binary uint8 image contents.
        shape: Shared image height and width.

    Returns:
        DataFrame of codec sizes, ratios, bit rates, and round-trip losslessness.

    Syllabus: Unit 6 - codec comparison
    """
    gray_image = np.frombuffer(gray_bytes, dtype=np.uint8).reshape(shape).copy()
    binary_image = np.frombuffer(binary_bytes, dtype=np.uint8).reshape(shape).copy()
    return compare_codecs(gray_image, binary_image)


def _bounded_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Bound the demo image size to keep interactive codecs responsive.

    Args:
        image: Two-dimensional grayscale uint8 image.

    Returns:
        Copy or area-resized image with no side longer than 256 pixels.

    Syllabus: Unit 6 - codec comparison
    """
    height, width = image.shape
    scale = min(1.0, 256.0 / max(height, width))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def render(image: NDArray[np.generic] | None) -> None:
    """Render byte-level codec comparisons, entropy, IGS, WHT, and DCT.

    Args:
        image: Shared grayscale or BGR source image or None.

    Returns:
        None.

    Syllabus: Unit 6 - image compression
    """
    syllabus_info("RLE, Huffman, LZW, IGS quantization, transform coding, rate-distortion", "Lab 12")
    source = require_image(image)
    if source is None:
        return
    gray_image = _bounded_gray(gray(source))
    threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if st.checkbox("Use binary scan as the primary codec input", key="compression_binary"):
        primary = threshold
    else:
        primary = gray_image
    if st.button("Run byte-level codec comparison", type="primary", key="compression_run"):
        try:
            with st.spinner("Encoding and verifying actual byte streams..."):
                table = _cached_compare(gray_image.tobytes(), threshold.tobytes(), gray_image.shape)
            st.session_state["compression_table"] = table
        except (ValueError, cv2.error) as error:
            st.error(f"Codec comparison failed: {error}")
    st.caption(f"Codec input dimensions: {gray_image.shape[1]}×{gray_image.shape[0]} (bounded for responsive demos).")
    table = st.session_state.get("compression_table")
    if isinstance(table, pd.DataFrame):
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.bar_chart(table.set_index("codec")[["compressed_bytes"]])
    else:
        st.info("Run the comparison to measure encoded sizes and verify real lossless round trips.")

    st.subheader("Huffman source statistics")
    entropy = shannon_entropy(primary)
    average = average_code_length(primary)
    entropy_col, average_col = st.columns(2)
    entropy_col.metric("Shannon entropy", f"{entropy:.3f} bits/pixel")
    average_col.metric("Huffman average length", f"{average:.3f} bits/pixel")
    st.dataframe(huffman_code_table(primary), hide_index=True, use_container_width=True)

    st.subheader("IGS versus plain truncation")
    bits = st.slider("Retained gray bits", 1, 7, 4)
    igs = igs_quantize(gray_image, bits=bits)
    levels = 2**bits
    truncated = reduce_gray_levels(gray_image, levels)
    columns = st.columns(3)
    columns[0].image(gray_image, caption="Source", use_container_width=True)
    columns[1].image(truncated, caption="Uniform truncation", use_container_width=True)
    columns[2].image(igs, caption="IGS", use_container_width=True)
    st.caption(f"Distinct output levels · truncation: {len(np.unique(truncated))}, IGS: {len(np.unique(igs))}")

    st.subheader("WHT and DCT rate-distortion")
    qstep = st.slider("Live quantization step", 1, 32, 8)
    keep = st.slider("Live coefficients retained per block", 1, 64, 24)
    transform = st.selectbox("Live transform", ("wht", "dct"))
    encoded, reconstruction = block_transform_compress(primary, transform, keep=keep)
    st.caption(f"Live stream: {len(encoded):,} bytes including headers.")
    st.image(reconstruction, caption=f"{transform.upper()} reconstruction · top {keep} coefficients", use_container_width=True)
    if min(primary.shape) < 7:
        st.info("Rate-distortion SSIM requires an image at least 7×7; use a larger source.")
    else:
        qsteps = (float(qstep), float(qstep * 2), float(qstep * 4), float(qstep * 8))
        try:
            curves = {
                name: _cached_rate_distortion(primary.tobytes(), primary.shape, name, qsteps)
                for name in ("wht", "dct")
            }
            curve_frame = pd.concat(
                [frame.assign(transform=name.upper(), qstep=qsteps) for name, frame in curves.items()],
                ignore_index=True,
            )
            st.line_chart(curve_frame, x="CR", y="PSNR", color="transform")
            st.dataframe(curve_frame, hide_index=True, use_container_width=True)
        except (ValueError, cv2.error) as error:
            st.warning(f"Rate-distortion curve could not be computed: {error}")

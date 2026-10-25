"""Shared presentation helpers for DocuVision Streamlit tabs.

Syllabus: Unit 1 - image processing application workflow
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from numpy.typing import NDArray


def syllabus_info(topics: str, lab: str) -> None:
    """Show the required syllabus topic and lab experiment context.

    Args:
        topics: Comma-separated syllabus topics covered in this tab.
        lab: Laboratory experiment number or label.

    Returns:
        None.

    Syllabus: Unit 1 - image processing application workflow
    """
    st.info(f"**Syllabus topics covered:** {topics}  \n**Lab experiment #:** {lab}")


def require_image(image: NDArray[np.generic] | None) -> NDArray[np.uint8] | None:
    """Validate optional shared image state and return an uint8 grayscale/BGR array.

    Args:
        image: Potential grayscale, singleton-channel, or BGR source image.

    Returns:
        Validated uint8 image, or None when no source is selected.

    Syllabus: Unit 1 - image representation
    """
    if image is None:
        st.info("Choose an image in the sidebar to use this lab.")
        return None
    result = np.asarray(image)
    if (
        result.dtype != np.uint8
        or result.size == 0
        or result.ndim not in (2, 3)
        or (result.ndim == 3 and result.shape[2] not in (1, 3))
    ):
        st.error("The selected image must be nonempty uint8 grayscale or BGR.")
        return None
    return result


def gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a shared grayscale/BGR image to two-dimensional grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rgb(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Convert a grayscale/BGR/one-channel image to RGB for Streamlit display."""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.shape[2] == 1:
        return cv2.cvtColor(array[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGR2RGB)


def show_pair(
    before: NDArray[np.generic],
    after: NDArray[np.generic],
    before_caption: str = "Before",
    after_caption: str = "After",
) -> None:
    """Show two images side by side with safe grayscale/BGR conversion."""
    left, right = st.columns(2)
    left.image(rgb(before), caption=before_caption, use_container_width=True)
    right.image(rgb(after), caption=after_caption, use_container_width=True)

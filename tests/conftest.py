"""Deterministic synthetic images shared by the test suite.

Syllabus: Unit 1 - image representation and testing
"""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def synthetic_page() -> NDArray[np.uint8]:
    """Create a white page with several deterministic text lines.

    Args:
        None.

    Returns:
        BGR uint8 page image.

    Syllabus: Unit 1 - image representation and testing
    """
    page = np.full((360, 280, 3), 255, dtype=np.uint8)
    lines = [
        "DocuVision Project",
        "Document image processing",
        "Stable frame selection",
        "Enhancement and OCR",
        "Classical methods only",
    ]
    for line_index, text in enumerate(lines):
        y_position = 55 + line_index * 55
        cv2.putText(
            page,
            text,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return page


@pytest.fixture
def checkerboard() -> NDArray[np.uint8]:
    """Create a deterministic grayscale checkerboard image.

    Args:
        None.

    Returns:
        Grayscale uint8 checkerboard.

    Syllabus: Unit 1 - image representation and sampling
    """
    cell_size = 16
    rows, columns = np.indices((128, 128))
    pattern = ((rows // cell_size + columns // cell_size) % 2) * 255
    return pattern.astype(np.uint8)


@pytest.fixture
def textured_image() -> NDArray[np.uint8]:
    """Create a reproducible blurred random texture.

    Args:
        None.

    Returns:
        Grayscale uint8 textured image.

    Syllabus: Unit 2 - image enhancement test data
    """
    random_generator = np.random.default_rng(303)
    noise = random_generator.integers(0, 256, size=(96, 96), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


@pytest.fixture
def step_edge_image() -> NDArray[np.uint8]:
    """Create a grayscale image with a vertical intensity step.

    Args:
        None.

    Returns:
        Grayscale uint8 image with a crisp edge.

    Syllabus: Unit 1 - image intensity and edges
    """
    image = np.zeros((64, 96), dtype=np.uint8)
    image[:, 48:] = 255
    return image

"""Tests for image input, output, and representation helpers.

Syllabus: Unit 1 - image types and formats
"""

from pathlib import Path

import numpy as np
import pytest

from docuvision.io_utils import (
    bgr_to_rgb,
    ensure_uint8,
    image_info,
    load_image,
    resize_max_side,
    save_image,
    to_gray,
)


def test_load_and_save_image_support_unicode_path(
    tmp_path: Path, synthetic_page: np.ndarray
) -> None:
    """Round-trip a BGR image through a path containing Unicode characters.

    Args:
        tmp_path: Pytest-provided temporary directory.
        synthetic_page: Synthetic BGR document page fixture.

    Returns:
        None.

    Syllabus: Unit 1 - image types and formats
    """
    image_path = tmp_path / "文書" / "page.png"
    save_image(image_path, synthetic_page)

    loaded_from_path = load_image(image_path)
    loaded_from_bytes = load_image(image_path.read_bytes())

    assert loaded_from_path.dtype == np.uint8
    assert loaded_from_path.shape == synthetic_page.shape
    np.testing.assert_array_equal(loaded_from_path, synthetic_page)
    np.testing.assert_array_equal(loaded_from_bytes, synthetic_page)


def test_load_image_rejects_invalid_bytes() -> None:
    """Raise an explicit error when encoded bytes are not an image.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image types and formats
    """
    with pytest.raises(ValueError, match="could not be decoded"):
        load_image(b"not an encoded image")


def test_ensure_uint8_clips_and_does_not_mutate_input() -> None:
    """Clip numeric values into byte range without changing the input array.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image types and formats
    """
    source = np.array([[-1.0, 12.8, 300.0]], dtype=np.float32)
    original = source.copy()

    result = ensure_uint8(source)

    np.testing.assert_array_equal(result, np.array([[0, 12, 255]], dtype=np.uint8))
    np.testing.assert_array_equal(source, original)


def test_to_gray_accepts_gray_and_bgr_without_mutation(
    synthetic_page: np.ndarray,
) -> None:
    """Convert color to grayscale and copy already-grayscale data.

    Args:
        synthetic_page: Synthetic BGR document page fixture.

    Returns:
        None.

    Syllabus: Unit 1 - image types and formats
    """
    gray = to_gray(synthetic_page)
    gray_input = gray.copy()

    copied_gray = to_gray(gray_input)

    assert gray.ndim == 2
    np.testing.assert_array_equal(copied_gray, gray_input)
    assert not np.shares_memory(copied_gray, gray_input)


def test_resize_max_side_preserves_aspect_ratio_and_returns_copy(
    synthetic_page: np.ndarray,
) -> None:
    """Resize to a bounded maximum side, preserving input pixels.

    Args:
        synthetic_page: Synthetic BGR document page fixture.

    Returns:
        None.

    Syllabus: Unit 1 - image sampling and spatial resolution
    """
    original = synthetic_page.copy()

    resized = resize_max_side(synthetic_page, 140)
    unchanged_size = resize_max_side(synthetic_page, 400)

    assert max(resized.shape[:2]) == 140
    assert resized.shape[1] == 109
    np.testing.assert_array_equal(synthetic_page, original)
    assert not np.shares_memory(unchanged_size, synthetic_page)


def test_bgr_to_rgb_changes_channel_order() -> None:
    """Convert BGR channel values to RGB in a separate array.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image types and formats
    """
    bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)

    rgb = bgr_to_rgb(bgr)

    np.testing.assert_array_equal(rgb, np.array([[[3, 2, 1]]], dtype=np.uint8))
    np.testing.assert_array_equal(bgr, np.array([[[1, 2, 3]]], dtype=np.uint8))


def test_image_info_classifies_binary_gray_and_color() -> None:
    """Report image type and storage properties for common representations.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image types and formats
    """
    binary = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    gray = np.array([[0, 20], [80, 160]], dtype=np.uint8)
    color = np.zeros((2, 2, 3), dtype=np.uint8)

    binary_info = image_info(binary)
    gray_info = image_info(gray)
    color_info = image_info(color)

    assert binary_info["type"] == "binary"
    assert binary_info["bit_depth"] == 8
    assert binary_info["n_unique_levels"] == 2
    assert binary_info["size_bytes"] == binary.nbytes
    assert gray_info["type"] == "gray"
    assert color_info["type"] == "color"


def test_resize_max_side_rejects_non_positive_limit(
    synthetic_page: np.ndarray,
) -> None:
    """Reject invalid maximum dimensions explicitly.

    Args:
        synthetic_page: Synthetic BGR document page fixture.

    Returns:
        None.

    Syllabus: Unit 1 - image sampling and spatial resolution
    """
    with pytest.raises(ValueError, match="positive"):
        resize_max_side(synthetic_page, 0)

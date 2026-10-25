"""Tests for classical binary and grayscale morphology.

Syllabus: Unit 3 - mathematical morphology
"""

import cv2
import numpy as np
import pytest

from docuvision.morphology import (
    blackhat,
    closing,
    dilate,
    dilate_scratch,
    erode,
    erode_scratch,
    extract_lines,
    gradient,
    join_broken_strokes,
    opening,
    remove_specks,
    structuring_element,
    text_line_mask,
    tophat,
)


@pytest.mark.parametrize("kind", ["rect", "ellipse", "cross"])
@pytest.mark.parametrize("operation", ["erode", "dilate"])
def test_scratch_morphology_matches_opencv(
    kind: str, operation: str
) -> None:
    """Compare NumPy sliding-window morphology with OpenCV on gray and binary data.

    Args:
        kind: Structuring-element kind under test.
        operation: Erosion or dilation operation under test.

    Returns:
        None.

    Syllabus: Unit 3 - mathematical morphology
    """
    random = np.random.default_rng(981)
    grayscale = random.integers(0, 256, size=(17, 23), dtype=np.uint8)
    binary = np.where(grayscale > 120, 255, 0).astype(np.uint8)
    kernel = structuring_element(kind, (5, 3))
    scratch = erode_scratch if operation == "erode" else dilate_scratch
    reference = cv2.erode if operation == "erode" else cv2.dilate

    np.testing.assert_array_equal(scratch(grayscale, kernel), reference(grayscale, kernel))
    np.testing.assert_array_equal(scratch(binary, kernel), reference(binary, kernel))


def test_morphology_wrappers_and_document_applications() -> None:
    """Exercise morphology wrappers and the document cleanup helpers.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 3 - mathematical morphology
    """
    binary = np.zeros((31, 41), dtype=np.uint8)
    binary[10:13, 10:13] = 255
    binary[20:22, 20:30] = 255
    kernel = structuring_element("rect", 3)

    assert erode(binary, kernel).shape == binary.shape
    assert dilate(binary, kernel).shape == binary.shape
    assert opening(binary, kernel).shape == binary.shape
    assert closing(binary, kernel).shape == binary.shape
    assert gradient(binary, kernel).shape == binary.shape
    assert tophat(binary, kernel).shape == binary.shape
    assert blackhat(binary, kernel).shape == binary.shape
    assert remove_specks(binary, 3).sum() < binary.sum()

    broken = np.zeros((20, 30), dtype=np.uint8)
    broken[8:12, 5:12] = 255
    broken[8:12, 14:21] = 255
    assert np.count_nonzero(join_broken_strokes(broken, 5)) > np.count_nonzero(broken)

    table = np.zeros((24, 36), dtype=np.uint8)
    table[5, 3:33] = 255
    table[12:22, 18] = 255
    horizontal = extract_lines(table, "horizontal", 20)
    vertical = extract_lines(table, "vertical", 8)
    assert horizontal[5, 10] == 255 and horizontal[18, 18] == 0
    assert vertical[18, 18] == 255 and vertical[5, 10] == 0
    assert text_line_mask(broken, kw=9, kh=3).shape == broken.shape


def test_structuring_element_rejects_unknown_kind() -> None:
    """Reject unsupported element names instead of silently choosing a kernel.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 3 - mathematical morphology
    """
    with pytest.raises(ValueError, match="kind"):
        structuring_element("triangle", 3)  # type: ignore[arg-type]

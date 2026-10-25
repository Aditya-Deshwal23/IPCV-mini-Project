"""Tests for classical image and document segmentation algorithms.

Syllabus: Unit 2 - image segmentation
"""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from docuvision.segmentation import (
    adaptive_threshold,
    canny,
    connectivity_comparison,
    connectivity_difference,
    detect_lines,
    detect_points,
    distance_transform,
    dog_edges,
    draw_boxes,
    global_threshold,
    gradient_edges,
    graph_segmentation,
    label_components,
    laplacian_edges,
    log_edges,
    otsu_cv,
    otsu_scratch,
    pixel_distance,
    region_growing,
    sauvola,
    segment_text_blocks,
    split_and_merge,
    watershed_split,
)


def test_connectivity_changes_diagonal_foreground_labels() -> None:
    """Show diagonal neighbors form one 8-connected but two 4-connected regions.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - connected components
    """
    binary = np.eye(3, dtype=np.uint8) * 255

    labels4, labels8 = connectivity_comparison(binary)
    counts = connectivity_difference(binary)

    assert len(np.unique(labels4)) - 1 == 3
    assert len(np.unique(labels8)) - 1 == 1
    assert counts == (3, 1)
    labels, stats = label_components(binary, connectivity=8)
    assert labels.dtype == np.int32
    assert stats[-1, cv2.CC_STAT_AREA] == 3


def test_point_line_gradient_and_laplacian_edges(
    step_edge_image: NDArray[np.uint8],
) -> None:
    """Exercise textbook point/line masks and three gradient operators.

    Args:
        step_edge_image: Synthetic grayscale vertical step.

    Returns:
        None.

    Syllabus: Unit 2 - edge detection
    """
    points = np.zeros((9, 9), dtype=np.uint8)
    points[4, 4] = 255
    point_map = detect_points(points, thr=100)
    horizontal_map = detect_lines(np.tile(points[4:5, :], (9, 1)), "horizontal")

    assert point_map[4, 4] == 255
    assert horizontal_map.shape == points.shape
    for operator in ("sobel", "prewitt", "roberts"):
        gx, gy, magnitude, direction = gradient_edges(step_edge_image, operator)
        assert gx.shape == gy.shape == magnitude.shape == direction.shape
        assert np.max(magnitude) > 0
    assert laplacian_edges(step_edge_image).max() > 0
    assert log_edges(step_edge_image, sigma=1.0).max() == 255


def test_region_growing_recovers_synthetic_disc_exactly() -> None:
    """Recover a uniform binary disc from a seed with zero intensity tolerance.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - region-based segmentation
    """
    rows, columns = np.indices((41, 41))
    disc = (rows - 20) ** 2 + (columns - 20) ** 2 <= 10**2
    image = np.where(disc, 180, 20).astype(np.uint8)

    result = region_growing(image, seeds=[(20, 20)], tol=0, connectivity=8)

    np.testing.assert_array_equal(result != 0, disc)


def test_split_merge_and_classical_thresholds() -> None:
    """Check Otsu threshold selection and key binary segmentation methods.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 5 - threshold-based segmentation
    """
    bimodal = np.array([[20] * 10 + [220] * 10] * 8, dtype=np.uint8)

    threshold = otsu_scratch(bimodal)
    assert 20 <= threshold <= 220
    assert abs(threshold - otsu_cv(bimodal)) <= 1
    assert np.unique(global_threshold(bimodal, threshold)).tolist() == [0, 255]

    variable = np.tile(np.linspace(10, 240, 32, dtype=np.uint8), (24, 1))
    assert adaptive_threshold(variable, "mean", block=7, C=2).shape == variable.shape
    assert sauvola(variable, window=7, k=0.2).shape == variable.shape
    labels = split_and_merge(variable, min_size=4, std_thr=1.0)
    assert labels.shape == variable.shape
    assert labels.dtype == np.int32


def test_watershed_splits_two_overlapping_circles() -> None:
    """Split touching disks into two foreground watershed labels.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 5 - watershed segmentation
    """
    image = np.zeros((100, 120), dtype=np.uint8)
    cv2.circle(image, (48, 50), 25, 255, -1)
    cv2.circle(image, (72, 50), 25, 255, -1)

    labels = watershed_split(image)
    foreground_labels = np.unique(labels[image != 0])
    foreground_labels = foreground_labels[foreground_labels > 0]

    assert foreground_labels.size == 2


def test_distance_graph_and_dog_helpers(
    step_edge_image: NDArray[np.uint8],
) -> None:
    """Verify the distance metrics and return shapes of edge/graph methods.

    Args:
        step_edge_image: Synthetic grayscale vertical step.

    Returns:
        None.

    Syllabus: Unit 5 - distance measures and graph segmentation
    """
    assert pixel_distance((0, 0), (3, 4)) == pytest.approx(5)
    assert pixel_distance((0, 0), (3, 4), "cityblock") == pytest.approx(7)
    assert pixel_distance((0, 0), (3, 4), "chessboard") == pytest.approx(4)
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[4:11, 4:11] = 255
    assert distance_transform(mask).shape == mask.shape
    assert canny(step_edge_image).shape == step_edge_image.shape
    assert dog_edges(step_edge_image, s1=1, s2=2).shape == step_edge_image.shape
    assert graph_segmentation(step_edge_image, min_size=5).shape == step_edge_image.shape


def test_text_blocks_follow_paragraph_reading_order(
    synthetic_page: NDArray[np.uint8],
) -> None:
    """Find one line-level text box for each line in the synthetic document.

    Args:
        synthetic_page: BGR page fixture containing five separated text lines.

    Returns:
        None.

    Syllabus: Unit 3 - document layout segmentation
    """
    gray = cv2.cvtColor(synthetic_page, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    source = synthetic_page.copy()

    boxes = segment_text_blocks(binary)
    visualization = draw_boxes(synthetic_page, boxes)

    assert len(boxes) == 5
    assert [box[1] for box in boxes] == sorted(box[1] for box in boxes)
    assert visualization.shape == synthetic_page.shape
    assert np.any(np.all(visualization == (0, 255, 0), axis=2))
    np.testing.assert_array_equal(synthetic_page, source)

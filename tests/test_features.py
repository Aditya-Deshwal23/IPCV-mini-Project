"""Tests for classical feature detection, description, and matching.

Syllabus: Unit 6 - image features and matching
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from docuvision.features import (
    align_to_template,
    build_dog,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    build_scale_space,
    compute_hog,
    draw_keypoints,
    estimate_homography,
    find_dog_extrema,
    harris_cv,
    harris_scratch,
    hessian_blobs,
    match_features,
    non_max_suppression,
    orb_detect,
    reconstruct_from_laplacian,
    shi_tomasi,
    sift_detect,
)


def _top_points(response: NDArray[np.generic], count: int) -> NDArray[np.float32]:
    """Extract local maxima from the strongest image-response peaks.

    Args:
        response: Finite 2D corner response.
        count: Maximum number of returned points.

    Returns:
        Point coordinates in (x, y) order.

    Syllabus: Unit 6 - feature point selection
    """
    maximum = cv2.dilate(response.astype(np.float32), np.ones((7, 7), np.uint8))
    rows, columns = np.where((response == maximum) & (response > 0))
    scores = response[rows, columns]
    order = np.argsort(scores)[::-1][:count]
    return np.column_stack((columns[order], rows[order])).astype(np.float32)


def test_harris_and_non_max_suppression_find_square_corners() -> None:
    """Find the four corners of a filled white square on a black background.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 6 - Harris corner detection
    """
    image = np.zeros((80, 80), dtype=np.uint8)
    image[20:60, 20:60] = 255
    response = harris_scratch(image)

    points = non_max_suppression(response, thr_rel=0.01, min_dist=12)
    expected = np.array([[20, 20], [59, 20], [20, 59], [59, 59]], dtype=np.float32)

    assert points.shape[0] == 4
    for corner in expected:
        assert np.min(np.linalg.norm(points - corner, axis=1)) <= 2


def test_scratch_and_opencv_harris_top_corners_agree(
    checkerboard: NDArray[np.uint8],
) -> None:
    """Compare top Harris response locations on the same checkerboard pattern.

    Args:
        checkerboard: Deterministic 128x128 grayscale checkerboard.

    Returns:
        None.

    Syllabus: Unit 6 - Harris corner detection
    """
    scratch_points = _top_points(harris_scratch(checkerboard), 50)
    cv_points = _top_points(harris_cv(checkerboard), 50)

    matched = sum(
        np.min(np.linalg.norm(cv_points - point, axis=1)) <= 3
        for point in scratch_points
    )
    assert matched / 50 >= 0.70
    assert len(shi_tomasi(checkerboard, max_corners=50)) > 0


def test_laplacian_pyramid_reconstructs_odd_dimensions() -> None:
    """Reconstruct a seeded image accurately after multi-level pyramid coding.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 6 - image pyramids
    """
    random = np.random.default_rng(74)
    image = random.integers(0, 256, size=(77, 101), dtype=np.uint8)

    gaussian = build_gaussian_pyramid(image, levels=5)
    laplacian = build_laplacian_pyramid(image, levels=5)
    reconstructed = reconstruct_from_laplacian(laplacian)

    assert len(gaussian) == len(laplacian) == 5
    assert reconstructed.shape == image.shape
    assert np.max(np.abs(reconstructed - image.astype(np.float32))) < 1


def test_scale_space_dog_and_extrema_detect_known_peak() -> None:
    """Build scale-space and verify 26-neighbor DoG extrema coordinates.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 6 - scale-space feature detection
    """
    scale_space = build_scale_space(
        np.full((64, 64), 100, dtype=np.uint8), octaves=2, scales=4
    )
    dogs = build_dog(scale_space)
    synthetic_dog = [
        [np.zeros((9, 9), dtype=np.float32) for _ in range(3)]
    ]
    synthetic_dog[0][1][4, 5] = 10
    extrema = find_dog_extrema(synthetic_dog)

    assert len(scale_space) == 2
    assert all(len(octave) == 4 for octave in scale_space)
    assert all(len(octave) == 3 for octave in dogs)
    np.testing.assert_array_equal(extrema, [[5, 4, 0, 1]])


def test_sift_matches_rotated_scaled_seeded_texture_with_ransac() -> None:
    """Recover a similarity warp from SIFT matches on seeded image texture.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 6 - SIFT feature matching and geometric alignment
    """
    random = np.random.default_rng(2025)
    noise = random.integers(0, 256, size=(260, 260), dtype=np.uint8)
    image = cv2.GaussianBlur(noise, (3, 3), 0.55)
    center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    affine = cv2.getRotationMatrix2D(center, 30.0, 0.8)
    transformed = cv2.warpAffine(
        image,
        affine,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    keypoints1, descriptors1 = sift_detect(image)
    keypoints2, descriptors2 = sift_detect(transformed)
    assert descriptors1 is not None and descriptors2 is not None
    matches = match_features(descriptors1, descriptors2, ratio=0.75)
    homography, inliers = estimate_homography(keypoints1, keypoints2, matches)

    assert homography.shape == (3, 3)
    assert int(inliers.sum()) >= 20


def test_orb_draw_alignment_and_hog_helpers(
    textured_image: NDArray[np.uint8],
) -> None:
    """Exercise ORB, keypoint rendering, HOG, and identity template alignment.

    Args:
        textured_image: Seeded grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 6 - feature description and image registration
    """
    keypoints, descriptors = orb_detect(textured_image)
    rendered = draw_keypoints(textured_image, keypoints)
    descriptor, visualization = compute_hog(textured_image)

    assert descriptors is not None and descriptors.shape[0] == len(keypoints)
    assert rendered.shape == (*textured_image.shape, 3)
    assert descriptor.size > 0 and visualization.shape == textured_image.shape
    aligned, inliers = align_to_template(textured_image, textured_image)
    assert aligned.shape == (*textured_image.shape, 3)
    assert inliers >= 4


def test_hessian_blobs_returns_xy_sigma() -> None:
    """Detect blob-like structures and preserve the documented coordinate shape.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 6 - Hessian blob detection
    """
    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(image, (32, 32), 9, 255, -1)
    blobs = hessian_blobs(image, min_sigma=2, max_sigma=12, threshold=0.01)

    assert blobs.ndim == 2
    assert blobs.shape[1] == 3
    if len(blobs):
        assert np.all(blobs[:, 2] > 0)

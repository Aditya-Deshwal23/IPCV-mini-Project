"""Tests for spatial enhancement, filtering, and image arithmetic.

Syllabus: Unit 2 - image enhancement
"""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from docuvision.enhancement import (
    absdiff,
    add,
    add_gaussian_noise,
    add_salt_pepper,
    background_division,
    bit_plane_slice,
    bit_plane_stack,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    clahe,
    compute_histogram_scratch,
    contrast_stretch_minmax,
    contrast_stretch_piecewise,
    digital_negative,
    divide,
    downsample,
    dynamic_range_compression,
    equalize_cv,
    equalize_scratch,
    gaussian_filter,
    high_boost,
    histogram_match_scratch,
    histogram_stretch,
    laplacian_sharpen,
    log_transform,
    mean_filter,
    mean_filter_scratch,
    median_filter,
    median_filter_scratch,
    multiply,
    order_statistic_filter,
    power_law,
    reduce_gray_levels,
    subtract,
    threshold_global,
    unsharp_mask,
    weighted_blend,
)


def _psnr(reference: NDArray[np.uint8], image: NDArray[np.uint8]) -> float:
    """Calculate peak signal-to-noise ratio for test comparisons.

    Args:
        reference: Reference uint8 image.
        image: Image whose reconstruction error is measured.

    Returns:
        PSNR in decibels, or infinity for an exact match.

    Syllabus: Unit 2 - image quality measurement
    """
    error = np.mean(
        (reference.astype(np.float64) - image.astype(np.float64)) ** 2
    )
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / error))


def test_negative_twice_is_identity(textured_image: NDArray[np.uint8]) -> None:
    """Verify that applying the digital negative transform twice is identity.

    Args:
        textured_image: Seeded synthetic grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 2 - point processing
    """
    result = digital_negative(digital_negative(textured_image))

    np.testing.assert_array_equal(result, textured_image)


def test_bit_planes_recombine_exactly(checkerboard: NDArray[np.uint8]) -> None:
    """Verify all eight extracted bit planes reconstruct source intensities.

    Args:
        checkerboard: Synthetic grayscale checkerboard fixture.

    Returns:
        None.

    Syllabus: Unit 2 - image representation and bit-plane slicing
    """
    planes = bit_plane_stack(checkerboard)
    reconstructed = np.zeros_like(checkerboard)
    for plane_index, plane_image in enumerate(planes):
        reconstructed |= ((plane_image // 255) << plane_index).astype(np.uint8)

    assert len(planes) == 8
    np.testing.assert_array_equal(reconstructed, checkerboard)
    assert set(np.unique(bit_plane_slice(checkerboard, 7))) <= {0, 255}


def test_point_transforms_and_quantization() -> None:
    """Check range, endpoint, and level-count behavior of point transforms.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - point processing and quantization
    """
    image = np.array([[10, 50, 100, 200]], dtype=np.uint8)

    stretched = contrast_stretch_minmax(image)
    piecewise = contrast_stretch_piecewise(image, 50, 30, 100, 220)
    thresholded = threshold_global(image, 100)
    gamma_identity = power_law(image, gamma=1.0)
    reduced = reduce_gray_levels(image, levels=4)
    sampled = downsample(np.arange(36, dtype=np.uint8).reshape(6, 6), factor=2)

    np.testing.assert_array_equal(stretched, [[0, 54, 121, 255]])
    np.testing.assert_array_equal(piecewise, [[6, 30, 220, 243]])
    np.testing.assert_array_equal(thresholded, [[0, 0, 0, 255]])
    np.testing.assert_array_equal(gamma_identity, image)
    assert np.unique(reduced).size <= 4
    np.testing.assert_array_equal(sampled, [[0, 2, 4], [12, 14, 16], [24, 26, 28]])


def test_log_transform_defaults_to_full_output_range() -> None:
    """Scale an automatically normalized log transform to the full byte range.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - point processing
    """
    image = np.array([[0, 255]], dtype=np.uint8)

    transformed = log_transform(image)

    np.testing.assert_array_equal(transformed, [[0, 255]])


def test_histogram_scratch_and_histogram_stretch() -> None:
    """Count grayscale levels and stretch the occupied range.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - image histograms
    """
    image = np.array([[10, 10, 20, 40]], dtype=np.uint8)
    histogram = compute_histogram_scratch(image)
    stretched = histogram_stretch(image)

    assert histogram.shape == (256,)
    assert histogram[10] == 2
    assert histogram[20] == 1
    assert histogram[40] == 1
    np.testing.assert_array_equal(stretched, [[0, 0, 85, 255]])


def test_equalization_handles_constant_and_has_near_linear_cdf() -> None:
    """Keep constants stable and distribute a ramp approximately uniformly.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - histogram equalization
    """
    constant = np.full((16, 16), 73, dtype=np.uint8)
    ramp = np.tile(np.arange(256, dtype=np.uint8), (8, 1))

    np.testing.assert_array_equal(equalize_scratch(constant), constant)
    equalized = equalize_scratch(ramp)
    cumulative = compute_histogram_scratch(equalized).cumsum() / equalized.size
    occupied = np.flatnonzero(compute_histogram_scratch(equalized))
    ideal = (occupied + 1) / 256.0

    assert np.max(np.abs(cumulative[occupied] - ideal)) < 0.01


def test_scratch_histogram_equalization_matches_opencv(
    textured_image: NDArray[np.uint8],
) -> None:
    """Compare the NumPy CDF implementation with OpenCV within one level.

    Args:
        textured_image: Seeded synthetic grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 2 - histogram equalization
    """
    scratch = equalize_scratch(textured_image)
    opencv = equalize_cv(textured_image)

    assert np.max(np.abs(scratch.astype(np.int16) - opencv.astype(np.int16))) <= 1


def test_clahe_histogram_matching_and_spectrum_compression(
    textured_image: NDArray[np.uint8],
) -> None:
    """Exercise local equalization, histogram specification, and log display.

    Args:
        textured_image: Seeded synthetic grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 2 - histogram modelling and frequency-domain display
    """
    reference = np.full((32, 32), 160, dtype=np.uint8)
    matched = histogram_match_scratch(textured_image, reference)
    enhanced = clahe(textured_image, clip=2.0, tile=8)
    spectrum = dynamic_range_compression(
        np.array([[0.0, 1.0, 10.0, 1000.0]], dtype=np.float64)
    )

    assert np.all(matched == 160)
    assert enhanced.shape == textured_image.shape
    assert spectrum[0, 0] == 0
    assert spectrum[0, -1] == 255
    assert np.all(np.diff(spectrum[0]) >= 0)


def test_scratch_spatial_filters_match_opencv_within_one_level(
    textured_image: NDArray[np.uint8],
) -> None:
    """Compare NumPy mean and median filters with their OpenCV counterparts.

    Args:
        textured_image: Seeded synthetic grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 2 - spatial smoothing filters
    """
    mean_scratch = mean_filter_scratch(textured_image, ksize=3)
    mean_opencv = mean_filter(textured_image, ksize=3)
    median_scratch = median_filter_scratch(textured_image, ksize=3)
    median_opencv = median_filter(textured_image, ksize=3)

    assert np.max(np.abs(mean_scratch.astype(np.int16) - mean_opencv)) <= 1
    np.testing.assert_array_equal(median_scratch, median_opencv)


def test_median_filter_outperforms_mean_on_salt_and_pepper_noise() -> None:
    """Compare PSNR of mean and median restoration for impulse noise.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - spatial filtering and image quality
    """
    clean = np.full((80, 80), 128, dtype=np.uint8)
    noisy = add_salt_pepper(clean, p=0.12, seed=44)

    restored_median = median_filter(noisy, ksize=3)
    restored_mean = mean_filter(noisy, ksize=3)

    assert _psnr(clean, restored_median) > _psnr(clean, restored_mean)


def test_gaussian_and_order_statistic_filters() -> None:
    """Verify Gaussian smoothing and supported order-statistic filter modes.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - spatial filtering
    """
    image = np.array([[10, 20, 30], [40, 100, 60], [70, 80, 90]], dtype=np.uint8)
    gaussian = gaussian_filter(image, ksize=3, sigma=0)

    assert gaussian.shape == image.shape
    assert order_statistic_filter(image, 3, "min")[1, 1] == 10
    assert order_statistic_filter(image, 3, "max")[1, 1] == 100
    assert order_statistic_filter(image, 3, "midpoint")[1, 1] == 55
    assert order_statistic_filter(image, 3, "alpha_trimmed", d=2)[1, 1] == 56


def test_sharpening_relationships_and_laplacian(
    step_edge_image: NDArray[np.uint8],
) -> None:
    """Verify high boost's gain relation and run Laplacian sharpening.

    Args:
        step_edge_image: Synthetic image with a vertical step edge.

    Returns:
        None.

    Syllabus: Unit 2 - spatial sharpening
    """
    unsharp = unsharp_mask(step_edge_image, sigma=1.2, amount=1.0)
    boost = high_boost(step_edge_image, k=1.0, sigma=1.2)
    laplacian = laplacian_sharpen(step_edge_image)

    np.testing.assert_array_equal(boost, unsharp)
    assert laplacian.shape == step_edge_image.shape


def test_arithmetic_saturates_and_bitwise_operations() -> None:
    """Check byte saturation at zero and 255 for arithmetic operators.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - image arithmetic
    """
    first = np.array([[250, 10, 2, 0]], dtype=np.uint8)
    second = np.array([[20, 30, 0, 5]], dtype=np.uint8)

    np.testing.assert_array_equal(add(first, second), [[255, 40, 2, 5]])
    np.testing.assert_array_equal(subtract(first, second), [[230, 0, 2, 0]])
    np.testing.assert_array_equal(multiply(first, second), [[255, 255, 0, 0]])
    np.testing.assert_array_equal(divide(first, second), [[12, 0, 0, 0]])
    np.testing.assert_array_equal(absdiff(first, second), [[230, 20, 2, 5]])
    np.testing.assert_array_equal(weighted_blend(first, second, 0.5), [[135, 20, 1, 2]])
    np.testing.assert_array_equal(bitwise_and(first, second), np.bitwise_and(first, second))
    np.testing.assert_array_equal(bitwise_or(first, second), np.bitwise_or(first, second))
    np.testing.assert_array_equal(bitwise_xor(first, second), np.bitwise_xor(first, second))
    np.testing.assert_array_equal(bitwise_not(first), cv2.bitwise_not(first))


def test_noise_generators_are_reproducible_and_do_not_mutate_input() -> None:
    """Use local random seeds to generate repeatable noise without mutation.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - image degradation and noise
    """
    image = np.full((40, 40, 3), 128, dtype=np.uint8)
    original = image.copy()

    salt_pepper_a = add_salt_pepper(image, p=0.2, seed=12)
    salt_pepper_b = add_salt_pepper(image, p=0.2, seed=12)
    gaussian_a = add_gaussian_noise(image, sigma=15, seed=12)
    gaussian_b = add_gaussian_noise(image, sigma=15, seed=12)

    np.testing.assert_array_equal(salt_pepper_a, salt_pepper_b)
    np.testing.assert_array_equal(gaussian_a, gaussian_b)
    np.testing.assert_array_equal(image, original)


def test_background_division_corrects_page_and_validation(
    synthetic_page: NDArray[np.uint8],
) -> None:
    """Run grayscale background division and reject invalid structuring size.

    Args:
        synthetic_page: Synthetic BGR document page fixture.

    Returns:
        None.

    Syllabus: Unit 2 - illumination correction
    """
    gray = cv2.cvtColor(synthetic_page, cv2.COLOR_BGR2GRAY)
    corrected = background_division(gray, kernel=31)

    assert corrected.shape == gray.shape
    assert corrected.dtype == np.uint8
    with pytest.raises(ValueError, match="odd"):
        background_division(gray, kernel=4)


def test_invalid_filter_and_quantization_parameters_raise(
    textured_image: NDArray[np.uint8],
) -> None:
    """Reject malformed kernel sizes and gray-level counts explicitly.

    Args:
        textured_image: Seeded synthetic grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 2 - image enhancement
    """
    with pytest.raises(ValueError, match="positive odd"):
        median_filter(textured_image, ksize=2)
    with pytest.raises(ValueError, match="levels"):
        reduce_gray_levels(textured_image, levels=1)
    with pytest.raises(ValueError, match="plane"):
        bit_plane_slice(textured_image, plane=8)
    with pytest.raises(ValueError, match="alpha"):
        weighted_blend(textured_image, textured_image, alpha=1.2)

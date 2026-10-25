"""Tests for frequency-domain image processing algorithms.

Syllabus: Unit 2 - frequency-domain image processing
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import hadamard

from docuvision.frequency import (
    apply_frequency_filter,
    butterworth_highpass,
    butterworth_lowpass,
    dct2_cv,
    dft1d_scratch,
    dft2d_scratch,
    fft_spectrum,
    gaussian_highpass,
    gaussian_lowpass,
    hadamard_matrix,
    homomorphic_filter,
    ideal_highpass,
    ideal_lowpass,
    idft1d_scratch,
    iwht2d,
    notch_reject,
    reconstruct_from_spectrum,
    ringing_demo,
    walsh_matrix,
    wht2d,
    wht_basis_images,
)


def test_scratch_dfts_match_numpy() -> None:
    """Compare direct matrix DFTs and their inverse with NumPy FFT results.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - discrete Fourier transform
    """
    vector = np.array([1.0, -2.0, 3.5, 4.0, 0.25])
    image = np.arange(35, dtype=np.float64).reshape(5, 7)

    np.testing.assert_allclose(dft1d_scratch(vector), np.fft.fft(vector), atol=1e-12)
    np.testing.assert_allclose(
        idft1d_scratch(dft1d_scratch(vector)), vector, atol=1e-12
    )
    np.testing.assert_allclose(dft2d_scratch(image), np.fft.fft2(image), atol=1e-10)


def test_scratch_dft_rejects_oversized_images() -> None:
    """Keep the educational separable DFT within its documented size limit.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - discrete Fourier transform
    """
    with pytest.raises(ValueError, match="64 by 64"):
        dft2d_scratch(np.zeros((65, 2)))


def test_fft_spectrum_and_reconstruction_are_consistent(
    textured_image: NDArray[np.uint8],
) -> None:
    """Reconstruct the original image from the centered spectrum tuple.

    Args:
        textured_image: Seeded synthetic grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 2 - frequency-domain image display
    """
    spectrum, magnitude, phase = fft_spectrum(textured_image)

    assert spectrum.shape == magnitude.shape == phase.shape == textured_image.shape
    assert magnitude.dtype == np.uint8
    assert np.iscomplexobj(spectrum)
    np.testing.assert_array_equal(reconstruct_from_spectrum(spectrum), textured_image)


def test_frequency_masks_and_filtering() -> None:
    """Check complementary masks, all-pass identity, and high-frequency reduction.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    shape = (64, 64)
    lowpass = gaussian_lowpass(shape, D0=5)
    highpass = gaussian_highpass(shape, D0=5)
    np.testing.assert_allclose(lowpass + highpass, 1.0)
    np.testing.assert_allclose(
        butterworth_lowpass(shape, 5, 2) + butterworth_highpass(shape, 5, 2), 1.0
    )
    np.testing.assert_allclose(ideal_lowpass(shape, 5) + ideal_highpass(shape, 5), 1.0)

    rows, columns = np.indices(shape)
    checker = (((rows + columns) % 2) * 255).astype(np.uint8)
    all_pass = np.ones(shape, dtype=np.float64)
    np.testing.assert_array_equal(apply_frequency_filter(checker, all_pass), checker)
    filtered = apply_frequency_filter(checker, gaussian_lowpass(shape, D0=5))
    assert np.var(filtered.astype(np.float64)) < np.var(checker.astype(np.float64))


def test_notch_reject_suppresses_conjugate_peak_pair() -> None:
    """Create zero response at a periodic-noise peak and its conjugate partner.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - periodic noise filtering
    """
    mask = notch_reject((32, 32), centers=[(12, 18)], radius=2, n=2)

    assert mask[12, 18] == pytest.approx(0.0)
    assert mask[20, 14] == pytest.approx(0.0)
    assert mask[16, 16] > 0.9


def test_hadamard_matches_scipy_and_walsh_order_is_sequency_ordered() -> None:
    """Compare the recursive matrix to SciPy and inspect Walsh row sequencies.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - Walsh-Hadamard transform
    """
    np.testing.assert_array_equal(hadamard_matrix(8), hadamard(8))
    walsh = walsh_matrix(8)
    sequencies = np.count_nonzero(walsh[:, 1:] != walsh[:, :-1], axis=1)

    assert np.all(np.diff(sequencies) > 0)


def test_wht_has_orthonormal_perfect_reconstruction() -> None:
    """Round-trip a non-square-valued block with the orthonormal WHT.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - Walsh-Hadamard transform
    """
    block = np.arange(64, dtype=np.float64).reshape(8, 8) / 7.0
    coefficients = wht2d(block, ordering="walsh")
    reconstructed = iwht2d(coefficients, ordering="walsh")

    np.testing.assert_allclose(reconstructed, block, atol=1e-12)
    assert wht_basis_images(4).shape == (16, 4, 4)


def test_homomorphic_filter_reduces_shadow_gradient_background_std() -> None:
    """Reduce a smooth synthetic illumination gradient behind black page text.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - homomorphic filtering
    """
    height, width = 96, 128
    illumination = np.linspace(90.0, 220.0, width, dtype=np.float64)
    page = np.tile(illumination, (height, 1)).astype(np.uint8)
    page[35:61, 24:104:8] = 20
    background = np.ones((height, width), dtype=bool)
    background[30:66, 20:108] = False

    result = homomorphic_filter(page, gamma_l=0.5, gamma_h=1.5, c=1.0, D0=20)

    assert result.dtype == np.uint8
    assert np.std(result[background]) < np.std(page[background]) * 0.5


def test_ringing_demo_and_cv_dct_return_expected_shapes() -> None:
    """Exercise the lowpass comparison and OpenCV DCT helper.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - frequency-domain filtering and discrete cosine transform
    """
    results = ringing_demo((32, 40), D0=4)
    coefficients = dct2_cv(np.eye(8, dtype=np.float32))

    assert set(results) == {"ideal", "butterworth", "gaussian"}
    assert all(image.shape == (32, 40) for image in results.values())
    assert coefficients.shape == (8, 8)
    assert coefficients.dtype == np.float32

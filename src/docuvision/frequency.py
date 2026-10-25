"""Frequency-domain image processing algorithms.

Syllabus: Unit 2 - frequency-domain image processing
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
WHTOrdering = Literal["walsh", "hadamard"]


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate a nonempty uint8 grayscale, singleton-channel, or BGR image.

    Args:
        img: Image array to validate.

    Returns:
        The original image array, without copying or modifying it.

    Syllabus: Unit 2 - frequency-domain image processing
    """
    image = np.asarray(img)
    valid_shape = image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] in (1, 3)
    )
    if image.dtype != np.uint8 or not valid_shape or image.size == 0:
        raise ValueError("Expected a nonempty uint8 grayscale or BGR image.")
    return image


def _validate_real_array(values: NDArray[np.generic], name: str) -> RealArray:
    """Return a finite, nonempty numeric array as float64.

    Args:
        values: Numeric input array.
        name: Name used in a validation error.

    Returns:
        Float64 copy of the input values.

    Syllabus: Unit 2 - frequency-domain image processing
    """
    array = np.asarray(values)
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a nonempty numeric array.")
    if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real values.")
    return array.astype(np.float64, copy=True)


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate a two-dimensional positive image shape.

    Args:
        shape: Requested (height, width).

    Returns:
        Validated dimensions as a tuple.

    Syllabus: Unit 2 - frequency-domain image processing
    """
    if (
        len(shape) != 2
        or any(not isinstance(value, (int, np.integer)) for value in shape)
        or any(value <= 0 for value in shape)
    ):
        raise ValueError("shape must contain two positive integer dimensions.")
    return int(shape[0]), int(shape[1])


def _validate_cutoff(D0: float) -> float:
    """Validate a finite positive frequency cutoff.

    Args:
        D0: Frequency cutoff in frequency-bin units.

    Returns:
        Validated cutoff as a float.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    if not np.isfinite(D0) or D0 <= 0:
        raise ValueError("D0 must be finite and greater than zero.")
    return float(D0)


def _distance_grid(shape: tuple[int, int]) -> RealArray:
    """Create distances from the centered DC component for a frequency plane.

    Args:
        shape: Frequency-plane (height, width).

    Returns:
        Euclidean distance of each bin from the shifted-spectrum center.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    height, width = _validate_shape(shape)
    rows, columns = np.ogrid[:height, :width]
    return np.hypot(rows - height // 2, columns - width // 2)


def dft1d_scratch(x: NDArray[np.generic]) -> ComplexArray:
    """Compute a one-dimensional DFT directly from its complex DFT matrix.

    Args:
        x: Nonempty one-dimensional real or complex input vector.

    Returns:
        Complex128 DFT with the same length as x.

    Syllabus: Unit 2 - discrete Fourier transform
    """
    vector = np.asarray(x)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("x must be a nonempty one-dimensional vector.")
    if not np.issubdtype(vector.dtype, np.number) or not np.all(np.isfinite(vector)):
        raise ValueError("x must contain finite numeric values.")
    length = vector.size
    indices = np.arange(length, dtype=np.float64)
    matrix = np.exp(-2j * np.pi * np.outer(indices, indices) / length)
    return np.asarray(matrix @ vector, dtype=np.complex128)


def idft1d_scratch(X: NDArray[np.generic]) -> ComplexArray:
    """Invert a one-dimensional DFT directly with the inverse DFT matrix.

    Args:
        X: Nonempty one-dimensional real or complex frequency vector.

    Returns:
        Complex128 reconstruction with the same length as X.

    Syllabus: Unit 2 - discrete Fourier transform
    """
    spectrum = np.asarray(X)
    if spectrum.ndim != 1 or spectrum.size == 0:
        raise ValueError("X must be a nonempty one-dimensional vector.")
    if not np.issubdtype(spectrum.dtype, np.number) or not np.all(np.isfinite(spectrum)):
        raise ValueError("X must contain finite numeric values.")
    length = spectrum.size
    indices = np.arange(length, dtype=np.float64)
    matrix = np.exp(2j * np.pi * np.outer(indices, indices) / length)
    return np.asarray((matrix @ spectrum) / length, dtype=np.complex128)


def dft2d_scratch(image: NDArray[np.generic]) -> ComplexArray:
    """Compute a two-dimensional DFT separably, one row and then each column.

    Args:
        image: Nonempty two-dimensional real or complex image, at most 64 by 64.

    Returns:
        Complex128 two-dimensional DFT with the same shape.

    Syllabus: Unit 2 - discrete Fourier transform
    """
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("image must be a nonempty two-dimensional array.")
    if any(dimension > 64 for dimension in array.shape):
        raise ValueError("dft2d_scratch supports images no larger than 64 by 64.")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError("image must contain finite numeric values.")
    row_transforms = np.stack([dft1d_scratch(row) for row in array], axis=0)
    return np.stack(
        [dft1d_scratch(row_transforms[:, column]) for column in range(array.shape[1])],
        axis=1,
    )


def _map_channels(image: Image, operation: Callable[[Image], Image]) -> Image:
    """Apply a 2D numeric operation independently to image channels.

    Args:
        image: Validated grayscale or BGR image.
        operation: Callable accepting and returning a two-dimensional array.

    Returns:
        Result with the input shape and uint8 dtype.

    Syllabus: Unit 2 - frequency-domain image processing
    """
    if image.ndim == 2:
        return operation(image)
    channels = [
        operation(image[:, :, channel])
        for channel in range(image.shape[2])
    ]
    return np.stack(channels, axis=2).astype(np.uint8, copy=False)


def _normalize_uint8(values: NDArray[np.generic]) -> Image:
    """Min-max normalize finite real values to the full uint8 range.

    Args:
        values: Numeric values to normalize.

    Returns:
        uint8 array; constant input maps to all zeros.

    Syllabus: Unit 2 - frequency-domain image display
    """
    array = np.asarray(values, dtype=np.float64)
    low = float(np.min(array))
    high = float(np.max(array))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip(np.rint((array - low) * (255.0 / (high - low))), 0, 255).astype(
        np.uint8
    )


def fft_spectrum(
    img: NDArray[np.generic],
) -> tuple[ComplexArray, Image, RealArray]:
    """Calculate the centered 2D Fourier spectrum, log magnitude, and phase.

    Args:
        img: uint8 grayscale or BGR image; color channels are transformed alone.

    Returns:
        Tuple of shifted complex spectrum, normalized log-magnitude uint8 image,
        and phase in radians. Each output has the input shape.

    Syllabus: Unit 2 - frequency-domain image display
    """
    image = _validate_image(img)

    def transform(channel: Image) -> tuple[ComplexArray, RealArray]:
        spectrum = np.fft.fftshift(np.fft.fft2(channel))
        return spectrum.astype(np.complex128, copy=False), np.angle(spectrum)

    if image.ndim == 2:
        spectrum, phase = transform(image)
    else:
        transformed = [transform(image[:, :, index]) for index in range(image.shape[2])]
        spectrum = np.stack([result[0] for result in transformed], axis=2)
        phase = np.stack([result[1] for result in transformed], axis=2)
    log_magnitude = _normalize_uint8(np.log1p(np.abs(spectrum)))
    return spectrum, log_magnitude, phase


def reconstruct_from_spectrum(F: NDArray[np.generic]) -> Image:
    """Reconstruct a uint8 image from a centered complex 2D Fourier spectrum.

    Args:
        F: Finite complex grayscale spectrum or channel-last BGR spectrum,
            as returned by fft_spectrum.

    Returns:
        Reconstructed uint8 image with the same shape.

    Syllabus: Unit 2 - inverse discrete Fourier transform
    """
    spectrum = np.asarray(F)
    valid_shape = spectrum.ndim == 2 or (
        spectrum.ndim == 3 and spectrum.shape[2] in (1, 3)
    )
    if (
        not valid_shape
        or spectrum.size == 0
        or not np.issubdtype(spectrum.dtype, np.number)
        or not np.all(np.isfinite(spectrum))
    ):
        raise ValueError("F must be a finite nonempty 2D or channel-last 3D spectrum.")

    def reconstruct(channel: NDArray[np.generic]) -> Image:
        spatial = np.fft.ifft2(np.fft.ifftshift(channel)).real
        return np.clip(np.rint(spatial), 0, 255).astype(np.uint8)

    if spectrum.ndim == 2:
        return reconstruct(spectrum)
    return np.stack(
        [reconstruct(spectrum[:, :, index]) for index in range(spectrum.shape[2])],
        axis=2,
    )


def ideal_lowpass(shape: tuple[int, int], D0: float) -> RealArray:
    """Create a centered binary ideal lowpass frequency mask.

    Args:
        shape: Mask dimensions (height, width).
        D0: Radius in frequency-bin units; bins at the radius are retained.

    Returns:
        Float64 mask with values zero or one.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    cutoff = _validate_cutoff(D0)
    return (_distance_grid(shape) <= cutoff).astype(np.float64)


def butterworth_lowpass(
    shape: tuple[int, int], D0: float, n: int
) -> RealArray:
    """Create a centered Butterworth lowpass frequency mask.

    Args:
        shape: Mask dimensions (height, width).
        D0: Half-power cutoff radius in frequency-bin units.
        n: Positive integer filter order.

    Returns:
        Float64 mask ranging from zero to one.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    cutoff = _validate_cutoff(D0)
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")
    distance = _distance_grid(shape)
    return 1.0 / (1.0 + (distance / cutoff) ** (2 * int(n)))


def gaussian_lowpass(shape: tuple[int, int], D0: float) -> RealArray:
    """Create a centered Gaussian lowpass frequency mask.

    Args:
        shape: Mask dimensions (height, width).
        D0: Gaussian spread in frequency-bin units.

    Returns:
        Float64 mask ranging from zero to one.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    cutoff = _validate_cutoff(D0)
    distance = _distance_grid(shape)
    return np.exp(-(distance**2) / (2.0 * cutoff**2))


def ideal_highpass(shape: tuple[int, int], D0: float) -> RealArray:
    """Create an ideal highpass mask as the complement of ideal lowpass.

    Args:
        shape: Mask dimensions (height, width).
        D0: Radius in frequency-bin units.

    Returns:
        Float64 mask with values zero or one.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    cutoff = _validate_cutoff(D0)
    distance = _distance_grid(shape)
    return (distance > cutoff).astype(np.float64)


def butterworth_highpass(
    shape: tuple[int, int], D0: float, n: int
) -> RealArray:
    """Create a Butterworth highpass mask as the complement of lowpass.

    Args:
        shape: Mask dimensions (height, width).
        D0: Half-power cutoff radius in frequency-bin units.
        n: Positive integer filter order.

    Returns:
        Float64 mask ranging from zero to one.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    cutoff = _validate_cutoff(D0)
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")
    distance = _distance_grid(shape)
    lowpass = 1.0 / (1.0 + (distance / cutoff) ** (2 * int(n)))
    return 1.0 - lowpass


def gaussian_highpass(shape: tuple[int, int], D0: float) -> RealArray:
    """Create a Gaussian highpass mask as the complement of Gaussian lowpass.

    Args:
        shape: Mask dimensions (height, width).
        D0: Gaussian spread in frequency-bin units.

    Returns:
        Float64 mask ranging from zero to one.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    cutoff = _validate_cutoff(D0)
    distance = _distance_grid(shape)
    lowpass = np.exp(-(distance**2) / (2.0 * cutoff**2))
    return 1.0 - lowpass


def notch_reject(
    shape: tuple[int, int],
    centers: Sequence[tuple[float, float]],
    radius: float,
    n: int,
) -> RealArray:
    """Build a Butterworth notch-reject mask for periodic-noise peak pairs.

    Args:
        shape: Mask dimensions (height, width).
        centers: Shifted-spectrum (row, column) coordinates of noise peaks.
            Each center is paired with its conjugate-symmetric counterpart.
        radius: Positive notch radius in frequency-bin units.
        n: Positive Butterworth order.

    Returns:
        Float64 mask ranging from zero to one.

    Syllabus: Unit 2 - periodic noise filtering
    """
    height, width = _validate_shape(shape)
    notch_radius = _validate_cutoff(radius)
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")
    rows, columns = np.ogrid[:height, :width]
    center_row, center_column = height // 2, width // 2
    mask = np.ones((height, width), dtype=np.float64)
    for center in centers:
        if len(center) != 2 or not np.all(np.isfinite(center)):
            raise ValueError("Each notch center must contain two finite coordinates.")
        row, column = float(center[0]), float(center[1])
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError("Notch centers must lie inside the shifted spectrum.")
        paired_centers = (
            (row, column),
            ((2 * center_row - row) % height, (2 * center_column - column) % width),
        )
        for notch_row, notch_column in paired_centers:
            distance = np.hypot(rows - notch_row, columns - notch_column)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                inverse_distance = np.divide(
                    notch_radius,
                    distance,
                    out=np.full_like(distance, np.inf, dtype=np.float64),
                    where=distance > 0,
                )
                notch_pass = 1.0 / (1.0 + inverse_distance ** (2 * int(n)))
            notch_pass[distance == 0] = 0.0
            mask *= notch_pass
    return mask


def _resize_centered_mask(mask: RealArray, shape: tuple[int, int]) -> RealArray:
    """Resample a centered mask while keeping its DC bin at the new center.

    Args:
        mask: Original centered mask.
        shape: Desired padded frequency-plane dimensions.

    Returns:
        Resampled float64 mask.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    old_height, old_width = mask.shape
    new_height, new_width = shape
    old_center_y, old_center_x = old_height // 2, old_width // 2
    new_center_y, new_center_x = new_height // 2, new_width // 2
    row_coordinates = old_center_y + (
        np.arange(new_height) - new_center_y
    ) * old_height / new_height
    column_coordinates = old_center_x + (
        np.arange(new_width) - new_center_x
    ) * old_width / new_width
    row0 = np.floor(row_coordinates).astype(int)
    column0 = np.floor(column_coordinates).astype(int)
    row1 = np.minimum(row0 + 1, old_height - 1)
    column1 = np.minimum(column0 + 1, old_width - 1)
    row0 = np.clip(row0, 0, old_height - 1)
    column0 = np.clip(column0, 0, old_width - 1)
    row_fraction = np.clip(row_coordinates - row0, 0.0, 1.0)[:, None]
    column_fraction = np.clip(column_coordinates - column0, 0.0, 1.0)[None, :]
    top = (
        mask[row0[:, None], column0[None, :]] * (1.0 - column_fraction)
        + mask[row0[:, None], column1[None, :]] * column_fraction
    )
    bottom = (
        mask[row1[:, None], column0[None, :]] * (1.0 - column_fraction)
        + mask[row1[:, None], column1[None, :]] * column_fraction
    )
    return top * (1.0 - row_fraction) + bottom * row_fraction


def apply_frequency_filter(
    img: NDArray[np.generic], mask: NDArray[np.generic]
) -> Image:
    """Apply a centered frequency mask to a grayscale or BGR uint8 image.

    Args:
        img: Nonempty uint8 grayscale, singleton-channel, or BGR image.
        mask: Finite real mask in [0, 1], sized either like img's spatial
            dimensions or like its 2x padded FFT dimensions.

    Returns:
        Filtered uint8 image with the input shape.

    Syllabus: Unit 2 - frequency-domain filtering
    """
    image = _validate_image(img)
    frequency_mask = _validate_real_array(mask, "mask")
    if frequency_mask.ndim != 2 or np.any(
        (frequency_mask < 0.0) | (frequency_mask > 1.0)
    ):
        raise ValueError("mask must be a 2D array with values in [0, 1].")
    height, width = image.shape[:2]
    padded_shape = (2 * height, 2 * width)
    if frequency_mask.shape == padded_shape:
        padded_mask = frequency_mask
    elif frequency_mask.shape == (height, width):
        padded_mask = _resize_centered_mask(frequency_mask, padded_shape)
    else:
        raise ValueError("mask must match the image or its 2x padded dimensions.")

    def filter_channel(channel: Image) -> Image:
        # Padding prevents the FFT's periodic extension from wrapping opposite
        # image edges together, which otherwise creates spatial wrap-around error.
        padded = np.zeros(padded_shape, dtype=np.float64)
        padded[:height, :width] = channel
        shifted = np.fft.fftshift(np.fft.fft2(padded))
        filtered = np.fft.ifft2(
            np.fft.ifftshift(shifted * padded_mask)
        ).real
        return np.clip(np.rint(filtered[:height, :width]), 0, 255).astype(np.uint8)

    return _map_channels(image, filter_channel)


def homomorphic_filter(
    img: NDArray[np.generic],
    gamma_l: float = 0.5,
    gamma_h: float = 1.5,
    c: float = 1.0,
    D0: float = 30,
) -> Image:
    """Reduce uneven illumination using a Gaussian homomorphic high-emphasis filter.

    The illumination-reflectance model represents an image as the product of
    slowly varying illumination and reflectance containing document detail.
    Taking the logarithm turns that product into a sum; the frequency filter
    attenuates low-frequency illumination and boosts higher-frequency detail.

    Args:
        img: uint8 grayscale or BGR image.
        gamma_l: Gain for low frequencies, in [0, 1].
        gamma_h: Gain for high frequencies, at least gamma_l.
        c: Positive exponential slope controlling the transition.
        D0: Positive Gaussian transition radius in frequency-bin units.

    Returns:
        Min-max normalized uint8 image with the input shape.

    Syllabus: Unit 2 - homomorphic filtering
    """
    image = _validate_image(img)
    if (
        not np.isfinite(gamma_l)
        or not np.isfinite(gamma_h)
        or not 0 <= gamma_l <= gamma_h
    ):
        raise ValueError("gains must satisfy 0 <= gamma_l <= gamma_h.")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and greater than zero.")
    cutoff = _validate_cutoff(D0)
    height, width = image.shape[:2]
    padded_shape = (2 * height, 2 * width)
    distance = _distance_grid(padded_shape)
    high_emphasis = gamma_l + (gamma_h - gamma_l) * (
        1.0 - np.exp(-c * distance**2 / cutoff**2)
    )

    def filter_channel(channel: Image) -> Image:
        padded = np.zeros(padded_shape, dtype=np.float64)
        padded[:height, :width] = np.log1p(channel.astype(np.float64) / 255.0)
        shifted = np.fft.fftshift(np.fft.fft2(padded))
        enhanced_log = np.fft.ifft2(
            np.fft.ifftshift(shifted * high_emphasis)
        ).real[:height, :width]
        enhanced = np.expm1(enhanced_log)
        return _normalize_uint8(enhanced)

    return _map_channels(image, filter_channel)


def hadamard_matrix(n: int) -> NDArray[np.int8]:
    """Build an unnormalized Hadamard matrix recursively with Kronecker products.

    Args:
        n: Positive power-of-two matrix dimension.

    Returns:
        int8 Hadamard matrix of shape (n, n).

    Syllabus: Unit 2 - Walsh-Hadamard transform
    """
    if not isinstance(n, (int, np.integer)) or n <= 0 or int(n) & (int(n) - 1):
        raise ValueError("n must be a positive power of two.")
    if n == 1:
        return np.ones((1, 1), dtype=np.int8)
    seed = np.array([[1, 1], [1, -1]], dtype=np.int8)
    return np.kron(seed, hadamard_matrix(int(n) // 2)).astype(np.int8, copy=False)


def walsh_matrix(n: int) -> NDArray[np.int8]:
    """Reorder Hadamard rows by increasing sequency (sign-change count).

    Args:
        n: Positive power-of-two matrix dimension.

    Returns:
        int8 Walsh matrix with rows in strictly increasing sequency order.

    Syllabus: Unit 2 - Walsh-Hadamard transform
    """
    matrix = hadamard_matrix(n)
    sequencies = np.count_nonzero(matrix[:, 1:] != matrix[:, :-1], axis=1)
    return matrix[np.argsort(sequencies, kind="stable")]


def _wht_basis(n: int, ordering: WHTOrdering) -> RealArray:
    """Return an orthonormal Hadamard or Walsh transform basis.

    Args:
        n: Positive power-of-two dimension.
        ordering: Row ordering, either ``"walsh"`` or ``"hadamard"``.

    Returns:
        Float64 orthonormal transform basis.

    Syllabus: Unit 2 - Walsh-Hadamard transform
    """
    if ordering == "walsh":
        matrix = walsh_matrix(n)
    elif ordering == "hadamard":
        matrix = hadamard_matrix(n)
    else:
        raise ValueError("ordering must be 'walsh' or 'hadamard'.")
    return matrix.astype(np.float64) / np.sqrt(n)


def wht2d(
    block: NDArray[np.generic], ordering: WHTOrdering = "walsh"
) -> RealArray:
    """Apply an orthonormally scaled 2D Walsh-Hadamard or Hadamard transform.

    Args:
        block: Finite square numeric block with power-of-two side length.
        ordering: ``"walsh"`` (default) or natural ``"hadamard"`` row order.

    Returns:
        Float64 coefficient block; inverse transform is the basis transpose.

    Syllabus: Unit 2 - Walsh-Hadamard transform
    """
    values = _validate_real_array(block, "block")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("block must be a square two-dimensional array.")
    basis = _wht_basis(values.shape[0], ordering)
    return basis @ values @ basis.T


def iwht2d(
    coefficients: NDArray[np.generic], ordering: WHTOrdering = "walsh"
) -> RealArray:
    """Invert an orthonormal 2D Walsh-Hadamard or Hadamard transform.

    Args:
        coefficients: Finite square coefficient block of power-of-two size.
        ordering: ``"walsh"`` (default) or natural ``"hadamard"`` row order.

    Returns:
        Float64 reconstructed block.

    Syllabus: Unit 2 - Walsh-Hadamard transform
    """
    values = _validate_real_array(coefficients, "coefficients")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("coefficients must be square and two-dimensional.")
    basis = _wht_basis(values.shape[0], ordering)
    return basis.T @ values @ basis


def wht_basis_images(n: int) -> NDArray[np.uint8]:
    """Create a stack of display-ready 2D Walsh basis images.

    Args:
        n: Positive power-of-two basis dimension.

    Returns:
        uint8 array of shape (n*n, n, n), one black-and-white image per basis.

    Syllabus: Unit 2 - Walsh-Hadamard transform
    """
    basis = walsh_matrix(n).astype(np.int16)
    images = [
        np.where(np.outer(basis[row], basis[column]) > 0, 255, 0).astype(np.uint8)
        for row in range(n)
        for column in range(n)
    ]
    return np.stack(images)


def dct2_cv(block: NDArray[np.generic]) -> NDArray[np.float32]:
    """Compute a 2D discrete cosine transform with OpenCV.

    Args:
        block: Finite real two-dimensional grayscale block.

    Returns:
        Float32 DCT coefficients with the input shape.

    Syllabus: Unit 2 - discrete cosine transform
    """
    values = _validate_real_array(block, "block")
    if values.ndim != 2:
        raise ValueError("block must be a two-dimensional grayscale array.")
    return cv2.dct(values.astype(np.float32))


def ringing_demo(
    shape: tuple[int, int], D0: float
) -> dict[str, Image]:
    """Show ringing on an ideal step edge filtered by three lowpass masks.

    Args:
        shape: Positive (height, width) for the synthetic grayscale image.
        D0: Positive lowpass cutoff radius in frequency-bin units.

    Returns:
        Mapping of ``"ideal"``, ``"butterworth"``, and ``"gaussian"`` to
        filtered uint8 step-edge images.

    Syllabus: Unit 2 - frequency-domain filtering and ringing
    """
    height, width = _validate_shape(shape)
    cutoff = _validate_cutoff(D0)
    edge = np.zeros((height, width), dtype=np.uint8)
    edge[:, width // 2 :] = 255
    return {
        "ideal": apply_frequency_filter(edge, ideal_lowpass((height, width), cutoff)),
        "butterworth": apply_frequency_filter(
            edge, butterworth_lowpass((height, width), cutoff, n=2)
        ),
        "gaussian": apply_frequency_filter(edge, gaussian_lowpass((height, width), cutoff)),
    }

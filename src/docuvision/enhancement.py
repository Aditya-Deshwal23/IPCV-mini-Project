"""Spatial-domain image enhancement algorithms.

Syllabus: Unit 2 - image enhancement
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

Image = NDArray[np.uint8]
OrderStatisticKind = Literal["min", "max", "midpoint", "alpha_trimmed"]


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate an image array and return it without copying.

    Args:
        img: Image to validate.

    Returns:
        The original uint8 grayscale or BGR array.

    Syllabus: Unit 2 - image enhancement
    """
    image = np.asarray(img)
    valid_shape = image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] in (1, 3)
    )
    if image.dtype != np.uint8 or not valid_shape:
        raise ValueError("Expected a uint8 grayscale or three-channel BGR image.")
    if image.size == 0:
        raise ValueError("Image must not be empty.")
    return image


def _validate_odd_kernel(ksize: int) -> None:
    """Check that a spatial filter kernel has a positive odd size.

    Args:
        ksize: Requested width and height of a square kernel.

    Returns:
        None.

    Syllabus: Unit 2 - spatial filtering
    """
    if not isinstance(ksize, int) or ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")


def _map_channels(
    img: Image, operation: Callable[[Image], Image]
) -> Image:
    """Apply one point operation independently to every image channel.

    Args:
        img: Validated grayscale or BGR image.
        operation: Callable accepting and returning a two-dimensional array.

    Returns:
        A newly allocated image with the original shape.

    Syllabus: Unit 2 - point processing
    """
    if img.ndim == 2:
        return operation(img)
    if img.shape[2] == 1:
        result = operation(img[:, :, 0])
        return result[:, :, np.newaxis]
    channels = [operation(img[:, :, index]) for index in range(img.shape[2])]
    return np.stack(channels, axis=2).astype(np.uint8, copy=False)


def _round_clip(values: NDArray[np.generic]) -> Image:
    """Round numeric image values and saturate them to the byte range.

    Args:
        values: Numeric pixel values.

    Returns:
        A uint8 array with values clipped to [0, 255].

    Syllabus: Unit 2 - point processing
    """
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def digital_negative(img: NDArray[np.generic]) -> Image:
    """Invert each pixel intensity using the digital negative transform.

    Args:
        img: uint8 grayscale or BGR image.

    Returns:
        Negative image with the same shape.

    Syllabus: Unit 2 - point processing
    """
    image = _validate_image(img)
    return (255 - image).astype(np.uint8)


def contrast_stretch_minmax(img: NDArray[np.generic]) -> Image:
    """Linearly stretch each channel's observed range to the full byte range.

    Args:
        img: uint8 grayscale or BGR image.

    Returns:
        Contrast-stretched image. Constant channels map to zero.

    Syllabus: Unit 2 - point processing
    """
    image = _validate_image(img)

    def stretch(channel: Image) -> Image:
        low = int(channel.min())
        high = int(channel.max())
        if low == high:
            return np.zeros_like(channel)
        scale = 255.0 / (high - low)
        return _round_clip((channel.astype(np.float32) - low) * scale)

    return _map_channels(image, stretch)


def contrast_stretch_piecewise(
    img: NDArray[np.generic], r1: float, s1: float, r2: float, s2: float
) -> Image:
    """Apply a three-segment piecewise-linear contrast transform per channel.

    Args:
        img: uint8 grayscale or BGR image.
        r1: First input-intensity breakpoint.
        s1: Output value corresponding to r1.
        r2: Second input-intensity breakpoint.
        s2: Output value corresponding to r2.

    Returns:
        Contrast-transformed image with the same shape.

    Syllabus: Unit 2 - point processing
    """
    image = _validate_image(img)
    if not (0 <= r1 < r2 <= 255):
        raise ValueError("Breakpoints must satisfy 0 <= r1 < r2 <= 255.")
    if not (0 <= s1 <= s2 <= 255):
        raise ValueError("Output values must satisfy 0 <= s1 <= s2 <= 255.")
    input_points = np.array([0.0, r1, r2, 255.0], dtype=np.float32)
    output_points = np.array([0.0, s1, s2, 255.0], dtype=np.float32)

    def transform(channel: Image) -> Image:
        mapped = np.interp(channel, input_points, output_points)
        return _round_clip(mapped)

    return _map_channels(image, transform)


def threshold_global(img: NDArray[np.generic], t: float) -> Image:
    """Threshold intensities into a binary image using a global cutoff.

    Args:
        img: uint8 grayscale or BGR image.
        t: Cutoff intensity; values strictly greater than t become white.

    Returns:
        Single-channel uint8 binary image containing only 0 and 255.

    Syllabus: Unit 2 - point processing
    """
    image = _validate_image(img)
    if not np.isfinite(t):
        raise ValueError("Threshold must be finite.")
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.where(image > t, 255, 0).astype(np.uint8)


def log_transform(img: NDArray[np.generic], c: float | None = None) -> Image:
    """Apply logarithmic intensity expansion to a grayscale or BGR image.

    Args:
        img: uint8 grayscale or BGR image.
        c: Positive scale; if omitted, maps the largest input to 255.

    Returns:
        Log-transformed uint8 image.

    Syllabus: Unit 2 - point processing
    """
    image = _validate_image(img)
    if c is not None and (not np.isfinite(c) or c < 0):
        raise ValueError("c must be finite and non-negative.")
    scale = c
    if scale is None:
        maximum = int(image.max())
        scale = 255.0 / np.log1p(maximum) if maximum else 0.0
    return _round_clip(scale * np.log1p(image.astype(np.float32)))


def power_law(
    img: NDArray[np.generic], gamma: float, c: float = 1.0
) -> Image:
    """Apply a normalized power-law (gamma) intensity transform.

    Args:
        img: uint8 grayscale or BGR image.
        gamma: Positive exponent controlling intensity expansion or compression.
        c: Non-negative multiplier applied to normalized output intensities.

    Returns:
        Gamma-transformed uint8 image.

    Syllabus: Unit 2 - point processing
    """
    image = _validate_image(img)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive.")
    if not np.isfinite(c) or c < 0:
        raise ValueError("c must be finite and non-negative.")
    normalized = image.astype(np.float32) / 255.0
    return _round_clip(255.0 * c * np.power(normalized, gamma))


def bit_plane_slice(img: NDArray[np.generic], plane: int) -> Image:
    """Extract one binary bit plane and display set bits as white.

    Args:
        img: uint8 grayscale or BGR image.
        plane: Bit index from 0 (least significant) to 7.

    Returns:
        A single-channel uint8 image containing only 0 and 255.

    Syllabus: Unit 2 - image representation and bit-plane slicing
    """
    image = _validate_image(img)
    if not isinstance(plane, int) or plane < 0 or plane > 7:
        raise ValueError("plane must be an integer from 0 through 7.")
    return (((image >> plane) & 1) * 255).astype(np.uint8)


def bit_plane_stack(img: NDArray[np.generic]) -> list[Image]:
    """Return all eight bit planes, ordered least to most significant.

    Args:
        img: uint8 grayscale or BGR image.

    Returns:
        Eight uint8 images, with set bits represented by 255.

    Syllabus: Unit 2 - image representation and bit-plane slicing
    """
    image = _validate_image(img)
    return [bit_plane_slice(image, plane) for plane in range(8)]


def dynamic_range_compression(img: NDArray[np.generic]) -> Image:
    """Compress a nonnegative spectrum's dynamic range using a log mapping.

    Args:
        img: Nonnegative numeric array, commonly an FFT magnitude spectrum.

    Returns:
        uint8 image scaled so zero maps to zero and the maximum maps to 255.

    Syllabus: Unit 2 - frequency-domain image display
    """
    values = np.asarray(img)
    if not np.issubdtype(values.dtype, np.number) or values.size == 0:
        raise ValueError("Expected a non-empty numeric spectrum.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Spectrum values must be finite and non-negative.")
    # The logarithm makes weak FFT components visible beside dominant peaks.
    maximum = float(values.max())
    if maximum == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    compressed = np.log1p(values.astype(np.float64))
    compressed *= 255.0 / np.log1p(maximum)
    return _round_clip(compressed)


def compute_histogram_scratch(img: NDArray[np.generic]) -> NDArray[np.int64]:
    """Compute a 256-bin grayscale histogram using NumPy bincount.

    Args:
        img: uint8 two-dimensional grayscale image.

    Returns:
        int64 histogram where index i counts pixels of intensity i.

    Syllabus: Unit 2 - image histograms
    """
    image = _validate_image(img)
    if image.ndim != 2:
        raise ValueError("Histogram input must be a two-dimensional grayscale image.")
    return np.bincount(image.ravel(), minlength=256).astype(np.int64)


def histogram_stretch(img: NDArray[np.generic]) -> Image:
    """Stretch each channel's minimum and maximum to 0 and 255.

    Args:
        img: uint8 grayscale or BGR image.

    Returns:
        Contrast-stretched image.

    Syllabus: Unit 2 - image histograms
    """
    return contrast_stretch_minmax(img)


def equalize_scratch(img: NDArray[np.generic]) -> Image:
    """Equalize a grayscale histogram with a cumulative-distribution LUT.

    Args:
        img: uint8 two-dimensional grayscale image.

    Returns:
        Histogram-equalized image; constant images are returned unchanged.

    Syllabus: Unit 2 - histogram equalization
    """
    image = _validate_image(img)
    if image.ndim != 2:
        raise ValueError("Equalization input must be a two-dimensional grayscale image.")
    histogram = compute_histogram_scratch(image)
    occupied = np.flatnonzero(histogram)
    if occupied.size <= 1:
        return image.copy()
    cdf = histogram.cumsum()
    cdf_min = int(cdf[occupied[0]])
    denominator = image.size - cdf_min
    if denominator <= 0:
        return image.copy()
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[:] = _round_clip((cdf - cdf_min) * (255.0 / denominator))
    return lookup[image]


def equalize_cv(img: NDArray[np.generic]) -> Image:
    """Equalize a grayscale image using OpenCV's histogram implementation.

    Args:
        img: uint8 two-dimensional grayscale image.

    Returns:
        OpenCV histogram-equalized image.

    Syllabus: Unit 2 - histogram equalization
    """
    image = _validate_image(img)
    if image.ndim != 2:
        raise ValueError("Equalization input must be a two-dimensional grayscale image.")
    return cv2.equalizeHist(image)


def clahe(
    img: NDArray[np.generic], clip: float, tile: int
) -> Image:
    """Apply contrast-limited adaptive histogram equalization.

    Args:
        img: uint8 two-dimensional grayscale image.
        clip: Positive CLAHE contrast clipping limit.
        tile: Positive count of tiles in each image dimension.

    Returns:
        Locally equalized grayscale image.

    Syllabus: Unit 2 - adaptive histogram equalization
    """
    image = _validate_image(img)
    if image.ndim != 2:
        raise ValueError("CLAHE input must be a two-dimensional grayscale image.")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive.")
    if not isinstance(tile, int) or tile <= 0:
        raise ValueError("tile must be a positive integer.")
    # A square tile grid keeps local adaptation consistent in both dimensions.
    operator = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    return operator.apply(image)


def histogram_match_scratch(
    img: NDArray[np.generic], ref: NDArray[np.generic]
) -> Image:
    """Match a grayscale image's cumulative histogram to a reference histogram.

    Args:
        img: uint8 two-dimensional source grayscale image.
        ref: uint8 two-dimensional reference grayscale image.

    Returns:
        Source image transformed by a CDF-based histogram specification LUT.

    Syllabus: Unit 2 - histogram specification (histogram modelling)
    """
    source = _validate_image(img)
    reference = _validate_image(ref)
    if source.ndim != 2 or reference.ndim != 2:
        raise ValueError("Histogram matching requires grayscale images.")
    source_cdf = compute_histogram_scratch(source).cumsum().astype(np.float64)
    source_cdf /= source.size
    reference_cdf = compute_histogram_scratch(reference).cumsum().astype(np.float64)
    reference_cdf /= reference.size
    lookup = np.searchsorted(reference_cdf, source_cdf, side="left")
    lookup = np.clip(lookup, 0, 255).astype(np.uint8)
    return lookup[source]


def downsample(img: NDArray[np.generic], factor: int) -> Image:
    """Sample every factor-th row and column without interpolation.

    Args:
        img: uint8 grayscale or BGR image.
        factor: Positive integer sampling interval.

    Returns:
        Spatially downsampled image.

    Syllabus: Unit 2 - image sampling and quantization
    """
    image = _validate_image(img)
    if not isinstance(factor, int) or factor <= 0:
        raise ValueError("factor must be a positive integer.")
    return image[::factor, ::factor].copy()


def reduce_gray_levels(
    img: NDArray[np.generic], levels: int
) -> Image:
    """Quantize intensities uniformly to a selected number of gray levels.

    Args:
        img: uint8 grayscale or BGR image.
        levels: Desired number of representable intensities, from 2 to 256.

    Returns:
        Image with at most levels evenly-spaced intensity values.

    Syllabus: Unit 2 - image sampling and quantization
    """
    image = _validate_image(img)
    if not isinstance(levels, int) or levels < 2 or levels > 256:
        raise ValueError("levels must be an integer from 2 through 256.")
    bins = (image.astype(np.uint16) * levels) // 256
    representatives = np.rint(
        bins.astype(np.float32) * (255.0 / (levels - 1))
    )
    return np.clip(representatives, 0, 255).astype(np.uint8)


def mean_filter(img: NDArray[np.generic], ksize: int) -> Image:
    """Apply an OpenCV box mean filter with reflected borders.

    Args:
        img: uint8 grayscale or BGR image.
        ksize: Positive odd width and height of the square averaging window.

    Returns:
        Mean-filtered image.

    Syllabus: Unit 2 - spatial smoothing filters
    """
    image = _validate_image(img)
    _validate_odd_kernel(ksize)
    return cv2.blur(image, (ksize, ksize), borderType=cv2.BORDER_REFLECT_101)


def mean_filter_scratch(img: NDArray[np.generic], ksize: int) -> Image:
    """Apply a NumPy sliding-window box mean filter.

    Args:
        img: uint8 grayscale or BGR image.
        ksize: Positive odd width and height of the square averaging window.

    Returns:
        Mean-filtered image rounded to the nearest uint8 value.

    Syllabus: Unit 2 - spatial smoothing filters
    """
    image = _validate_image(img)
    _validate_odd_kernel(ksize)
    pad = ksize // 2

    def filter_channel(channel: Image) -> Image:
        padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="reflect")
        windows = sliding_window_view(padded, (ksize, ksize))
        # The mean gives each sample equal weight, as in a normalized box kernel.
        return _round_clip(windows.mean(axis=(-2, -1), dtype=np.float64))

    return _map_channels(image, filter_channel)


def gaussian_filter(
    img: NDArray[np.generic], ksize: int, sigma: float = 0
) -> Image:
    """Apply Gaussian smoothing using OpenCV.

    Args:
        img: uint8 grayscale or BGR image.
        ksize: Positive odd width and height of the Gaussian kernel.
        sigma: Positive standard deviation, or zero for OpenCV's kernel-derived value.

    Returns:
        Gaussian-smoothed image.

    Syllabus: Unit 2 - spatial smoothing filters
    """
    image = _validate_image(img)
    _validate_odd_kernel(ksize)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative.")
    return cv2.GaussianBlur(
        image,
        (ksize, ksize),
        sigmaX=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )


def median_filter(img: NDArray[np.generic], ksize: int) -> Image:
    """Apply OpenCV's nonlinear median filter.

    Args:
        img: uint8 grayscale or BGR image.
        ksize: Positive odd width and height of the median window.

    Returns:
        Median-filtered image.

    Syllabus: Unit 2 - spatial order-statistic filters
    """
    image = _validate_image(img)
    _validate_odd_kernel(ksize)
    return cv2.medianBlur(image, ksize)


def median_filter_scratch(img: NDArray[np.generic], ksize: int) -> Image:
    """Apply a median filter using NumPy sliding_window_view.

    Args:
        img: uint8 grayscale or BGR image.
        ksize: Positive odd width and height of the median window.

    Returns:
        Median-filtered image.

    Syllabus: Unit 2 - spatial order-statistic filters
    """
    image = _validate_image(img)
    _validate_odd_kernel(ksize)
    pad = ksize // 2

    def filter_channel(channel: Image) -> Image:
        # Replicated edges match OpenCV medianBlur's border handling.
        padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="edge")
        windows = sliding_window_view(padded, (ksize, ksize))
        values = np.median(windows, axis=(-2, -1))
        return _round_clip(values)

    return _map_channels(image, filter_channel)


def order_statistic_filter(
    img: NDArray[np.generic],
    ksize: int,
    kind: OrderStatisticKind,
    d: int = 2,
) -> Image:
    """Apply a min, max, midpoint, or alpha-trimmed spatial filter.

    Args:
        img: uint8 grayscale or BGR image.
        ksize: Positive odd width and height of the square neighborhood.
        kind: Selected statistic: min, max, midpoint, or alpha_trimmed.
        d: Even number of sorted samples to discard for alpha_trimmed.

    Returns:
        Filtered image with the same shape.

    Syllabus: Unit 2 - spatial order-statistic filters
    """
    image = _validate_image(img)
    _validate_odd_kernel(ksize)
    if kind not in ("min", "max", "midpoint", "alpha_trimmed"):
        raise ValueError("kind must be min, max, midpoint, or alpha_trimmed.")
    sample_count = ksize * ksize
    if kind == "alpha_trimmed":
        if not isinstance(d, int) or d < 0 or d % 2 != 0 or d >= sample_count:
            raise ValueError("d must be even, non-negative, and smaller than ksize squared.")

    pad = ksize // 2

    def filter_channel(channel: Image) -> Image:
        padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="reflect")
        windows = sliding_window_view(padded, (ksize, ksize))
        flattened = windows.reshape(*channel.shape, sample_count)
        if kind == "min":
            values = flattened.min(axis=-1)
        elif kind == "max":
            values = flattened.max(axis=-1)
        elif kind == "midpoint":
            values = (flattened.min(axis=-1) + flattened.max(axis=-1)) / 2.0
        else:
            sorted_values = np.sort(flattened, axis=-1)
            stop = sample_count - d // 2
            values = sorted_values[..., d // 2 : stop].mean(axis=-1)
        return _round_clip(values)

    return _map_channels(image, filter_channel)


def laplacian_sharpen(
    img: NDArray[np.generic], c: float = -1
) -> Image:
    """Sharpen by adding a scaled Laplacian high-frequency response.

    Args:
        img: uint8 grayscale or BGR image.
        c: Laplacian scale; -1 subtracts the Laplacian for common 4-neighbor form.

    Returns:
        Sharpened image with saturation at the byte range.

    Syllabus: Unit 2 - spatial sharpening
    """
    image = _validate_image(img)
    if not np.isfinite(c):
        raise ValueError("c must be finite.")
    laplacian = cv2.Laplacian(
        image, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT_101
    )
    sharpened = image.astype(np.float32) + float(c) * laplacian
    return _round_clip(sharpened)


def unsharp_mask(
    img: NDArray[np.generic], sigma: float, amount: float
) -> Image:
    """Sharpen by adding a scaled difference between image and Gaussian blur.

    Args:
        img: uint8 grayscale or BGR image.
        sigma: Positive Gaussian standard deviation used to form the blur.
        amount: Non-negative strength of the high-frequency detail.

    Returns:
        Unsharp-masked image with saturating uint8 arithmetic.

    Syllabus: Unit 2 - spatial sharpening
    """
    image = _validate_image(img)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    if not np.isfinite(amount) or amount < 0:
        raise ValueError("amount must be finite and non-negative.")
    blurred = cv2.GaussianBlur(
        image, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT_101
    )
    return cv2.addWeighted(image, 1.0 + float(amount), blurred, -float(amount), 0.0)


def high_boost(
    img: NDArray[np.generic], k: float, sigma: float
) -> Image:
    """Apply high-boost sharpening with gain k on the detail mask.

    Args:
        img: uint8 grayscale or BGR image.
        k: Non-negative high-frequency gain.
        sigma: Positive Gaussian standard deviation used to form the blur.

    Returns:
        High-boost sharpened image.

    Syllabus: Unit 2 - spatial sharpening
    """
    image = _validate_image(img)
    if not np.isfinite(k) or k < 0:
        raise ValueError("k must be finite and non-negative.")
    # Unsharp masking is high-boost filtering with k=1.
    return unsharp_mask(image, sigma=sigma, amount=k)


def add(img1: NDArray[np.generic], img2: NDArray[np.generic]) -> Image:
    """Add two same-shaped images using saturating uint8 arithmetic.

    Args:
        img1: First uint8 grayscale or BGR image.
        img2: Second same-shaped uint8 image.

    Returns:
        Saturated pixelwise sum.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    return cv2.add(first, second)


def subtract(img1: NDArray[np.generic], img2: NDArray[np.generic]) -> Image:
    """Subtract the second image from the first with saturation at zero.

    Args:
        img1: Minuend uint8 grayscale or BGR image.
        img2: Same-shaped subtrahend uint8 image.

    Returns:
        Saturated pixelwise difference.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    return cv2.subtract(first, second)


def multiply(img1: NDArray[np.generic], img2: NDArray[np.generic]) -> Image:
    """Multiply two same-shaped images with saturation at 255.

    Args:
        img1: First uint8 grayscale or BGR image.
        img2: Second same-shaped uint8 image.

    Returns:
        Saturated pixelwise product.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    return cv2.multiply(first, second)


def divide(img1: NDArray[np.generic], img2: NDArray[np.generic]) -> Image:
    """Divide the first image by the second using saturating byte output.

    Args:
        img1: Numerator uint8 grayscale or BGR image.
        img2: Same-shaped denominator uint8 image; zero denominators map to zero.

    Returns:
        Pixelwise quotient saturated to uint8.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    return cv2.divide(first, second)


def absdiff(img1: NDArray[np.generic], img2: NDArray[np.generic]) -> Image:
    """Compute the absolute pixelwise difference of two images.

    Args:
        img1: First uint8 grayscale or BGR image.
        img2: Second same-shaped uint8 image.

    Returns:
        Absolute difference image.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    return cv2.absdiff(first, second)


def weighted_blend(
    img1: NDArray[np.generic], img2: NDArray[np.generic], alpha: float
) -> Image:
    """Blend two images as alpha*img1 + (1-alpha)*img2.

    Args:
        img1: First uint8 grayscale or BGR image.
        img2: Second same-shaped uint8 image.
        alpha: Weight for the first image in the closed interval [0, 1].

    Returns:
        Saturated uint8 weighted blend.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    if not np.isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("alpha must be finite and in the range [0, 1].")
    return cv2.addWeighted(first, float(alpha), second, 1.0 - float(alpha), 0.0)


def bitwise_and(img1: NDArray[np.generic], img2: NDArray[np.generic]) -> Image:
    """Apply a bitwise AND to two same-shaped uint8 images.

    Args:
        img1: First uint8 grayscale or BGR image.
        img2: Second same-shaped uint8 image.

    Returns:
        Pixelwise bitwise AND.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    return cv2.bitwise_and(first, second)


def bitwise_or(img1: NDArray[np.generic], img2: NDArray[np.generic]) -> Image:
    """Apply a bitwise OR to two same-shaped uint8 images.

    Args:
        img1: First uint8 grayscale or BGR image.
        img2: Second same-shaped uint8 image.

    Returns:
        Pixelwise bitwise OR.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    return cv2.bitwise_or(first, second)


def bitwise_xor(img1: NDArray[np.generic], img2: NDArray[np.generic]) -> Image:
    """Apply a bitwise XOR to two same-shaped uint8 images.

    Args:
        img1: First uint8 grayscale or BGR image.
        img2: Second same-shaped uint8 image.

    Returns:
        Pixelwise bitwise XOR.

    Syllabus: Unit 2 - image arithmetic
    """
    first, second = _validate_image(img1), _validate_image(img2)
    if first.shape != second.shape:
        raise ValueError("Image arithmetic requires arrays with matching shapes.")
    return cv2.bitwise_xor(first, second)


def bitwise_not(img: NDArray[np.generic]) -> Image:
    """Invert every bit in a grayscale or BGR image.

    Args:
        img: uint8 grayscale or BGR image.

    Returns:
        Bitwise-complement image.

    Syllabus: Unit 2 - image arithmetic
    """
    return cv2.bitwise_not(_validate_image(img))


def background_division(
    gray: NDArray[np.generic], kernel: int
) -> Image:
    """Correct uneven illumination by dividing by a morphological closing.

    Args:
        gray: uint8 two-dimensional grayscale image.
        kernel: Positive odd square structuring-element width, chosen larger than
            text strokes so closing estimates the slowly varying background.

    Returns:
        Illumination-corrected grayscale image with mean brightness preserved.

    Syllabus: Unit 2 - image enhancement and illumination correction
    """
    image = _validate_image(gray)
    if image.ndim != 2:
        raise ValueError("background_division requires a grayscale image.")
    _validate_odd_kernel(kernel)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
    denominator = np.maximum(background.astype(np.float32), 1.0)
    reference_level = float(np.mean(background))
    corrected = image.astype(np.float32) / denominator * reference_level
    return _round_clip(corrected)


def add_salt_pepper(
    img: NDArray[np.generic], p: float, seed: int | None
) -> Image:
    """Add independent salt-and-pepper impulse noise using a local RNG.

    Args:
        img: uint8 grayscale or BGR image.
        p: Total fraction of pixels to corrupt, split evenly between salt/pepper.
        seed: Optional random seed for reproducible demonstrations.

    Returns:
        Noisy image; the input is not modified.

    Syllabus: Unit 2 - image degradation and noise
    """
    image = _validate_image(img)
    if not np.isfinite(p) or p < 0 or p > 1:
        raise ValueError("p must be finite and in the range [0, 1].")
    rng = np.random.default_rng(seed)
    noise = rng.random(image.shape[:2])
    result = image.copy()
    # Splitting the requested corruption probability evenly balances both impulses.
    result[noise < p / 2.0] = 0
    result[(noise >= p / 2.0) & (noise < p)] = 255
    return result


def add_gaussian_noise(
    img: NDArray[np.generic], sigma: float, seed: int | None
) -> Image:
    """Add zero-mean Gaussian noise with an optional reproducibility seed.

    Args:
        img: uint8 grayscale or BGR image.
        sigma: Non-negative standard deviation in intensity levels.
        seed: Optional random seed for reproducible demonstrations.

    Returns:
        Clipped uint8 noisy image; the input is not modified.

    Syllabus: Unit 2 - image degradation and noise
    """
    image = _validate_image(img)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative.")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image.shape)
    return _round_clip(image.astype(np.float32) + noise)

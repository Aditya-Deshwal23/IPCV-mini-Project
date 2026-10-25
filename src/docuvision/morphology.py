"""Binary and grayscale morphology algorithms.

Syllabus: Unit 3 - mathematical morphology
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

Image = NDArray[np.uint8]
KernelKind = Literal["rect", "ellipse", "cross"]
LineOrientation = Literal["horizontal", "vertical"]


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate a nonempty uint8 grayscale or BGR image.

    Args:
        img: Image array to validate.

    Returns:
        The input image without copying or mutating it.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = np.asarray(img)
    valid_shape = image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] in (1, 3)
    )
    if image.dtype != np.uint8 or not valid_shape or image.size == 0:
        raise ValueError("Expected a nonempty uint8 grayscale or BGR image.")
    return image


def _validate_size(size: int | tuple[int, int]) -> tuple[int, int]:
    """Convert a positive scalar or (width, height) to kernel dimensions.

    Args:
        size: Positive scalar or pair of positive integer dimensions.

    Returns:
        Structuring-element dimensions as (width, height).

    Syllabus: Unit 3 - mathematical morphology
    """
    dimensions = (size, size) if isinstance(size, (int, np.integer)) else size
    if (
        len(dimensions) != 2
        or any(not isinstance(value, (int, np.integer)) for value in dimensions)
        or any(value <= 0 for value in dimensions)
    ):
        raise ValueError("size must be a positive integer or pair of integers.")
    return int(dimensions[0]), int(dimensions[1])


def structuring_element(
    kind: KernelKind, size: int | tuple[int, int]
) -> NDArray[np.uint8]:
    """Create a rectangular, elliptical, or cross-shaped structuring element.

    Args:
        kind: One of ``"rect"``, ``"ellipse"``, or ``"cross"``.
        size: Positive side length or (width, height) kernel dimensions.

    Returns:
        OpenCV uint8 kernel with active cells set to one.

    Syllabus: Unit 3 - mathematical morphology
    """
    width, height = _validate_size(size)
    kinds = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    if kind not in kinds:
        raise ValueError("kind must be 'rect', 'ellipse', or 'cross'.")
    return cv2.getStructuringElement(kinds[kind], (width, height))


def _validate_kernel(kernel: NDArray[np.generic]) -> NDArray[np.bool_]:
    """Validate and convert a nonempty two-dimensional structuring element.

    Args:
        kernel: Numeric or boolean array where nonzero cells are active.

    Returns:
        Boolean kernel mask.

    Syllabus: Unit 3 - mathematical morphology
    """
    array = np.asarray(kernel)
    if array.ndim != 2 or array.size == 0 or not np.any(array):
        raise ValueError("kernel must be a nonempty 2D array with active cells.")
    return array != 0


def _morph_scratch(
    image: NDArray[np.generic], kernel: NDArray[np.generic], operation: str
) -> NDArray[np.generic]:
    """Apply a grayscale or binary min/max filter using sliding windows.

    Args:
        image: Nonempty 2D numeric grayscale or binary image.
        kernel: Structuring-element array; nonzero cells participate.
        operation: Either ``"erode"`` or ``"dilate"``.

    Returns:
        Array with the same shape and dtype as image.

    Syllabus: Unit 3 - mathematical morphology
    """
    values = np.asarray(image)
    element = _validate_kernel(kernel)
    if values.ndim != 2 or values.size == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("image must be a nonempty 2D numeric array.")
    if operation not in ("erode", "dilate"):
        raise ValueError("operation must be 'erode' or 'dilate'.")
    height, width = element.shape
    top, left = height // 2, width // 2
    bottom, right = height - 1 - top, width - 1 - left
    # Match OpenCV's neutral border: 255 for uint8 erosion, zero for dilation.
    if operation == "erode":
        border = (
            np.iinfo(values.dtype).max
            if np.issubdtype(values.dtype, np.integer)
            else np.inf
        )
        padded = np.pad(values, ((top, bottom), (left, right)), constant_values=border)
        windows = sliding_window_view(padded, element.shape)
        result = np.min(windows[..., element], axis=-1)
    else:
        border = (
            np.iinfo(values.dtype).min
            if np.issubdtype(values.dtype, np.integer)
            else -np.inf
        )
        padded = np.pad(values, ((top, bottom), (left, right)), constant_values=border)
        windows = sliding_window_view(padded, element.shape)
        result = np.max(windows[..., element], axis=-1)
    return result.astype(values.dtype, copy=False)


def erode_scratch(
    image: NDArray[np.generic], kernel: NDArray[np.generic]
) -> NDArray[np.generic]:
    """Erode a grayscale or binary image using NumPy sliding windows.

    Args:
        image: Nonempty two-dimensional numeric image.
        kernel: Structuring element; nonzero cells participate.

    Returns:
        Eroded image with the input shape and dtype.

    Syllabus: Unit 3 - mathematical morphology
    """
    return _morph_scratch(image, kernel, "erode")


def dilate_scratch(
    image: NDArray[np.generic], kernel: NDArray[np.generic]
) -> NDArray[np.generic]:
    """Dilate a grayscale or binary image using NumPy sliding windows.

    Args:
        image: Nonempty two-dimensional numeric image.
        kernel: Structuring element; nonzero cells participate.

    Returns:
        Dilated image with the input shape and dtype.

    Syllabus: Unit 3 - mathematical morphology
    """
    return _morph_scratch(image, kernel, "dilate")


def erode(
    img: NDArray[np.generic], kernel: NDArray[np.generic]
) -> Image:
    """Erode a grayscale or binary uint8 image with OpenCV.

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        kernel: Two-dimensional structuring element.

    Returns:
        Eroded uint8 image with the input shape.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = _validate_image(img)
    element = _validate_kernel(kernel).astype(np.uint8)
    return cv2.erode(image, element)


def dilate(
    img: NDArray[np.generic], kernel: NDArray[np.generic]
) -> Image:
    """Dilate a grayscale or binary uint8 image with OpenCV.

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        kernel: Two-dimensional structuring element.

    Returns:
        Dilated uint8 image with the input shape.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = _validate_image(img)
    element = _validate_kernel(kernel).astype(np.uint8)
    return cv2.dilate(image, element)


def opening(img: NDArray[np.generic], kernel: NDArray[np.generic]) -> Image:
    """Remove small bright foreground details using OpenCV morphological opening.

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        kernel: Two-dimensional structuring element.

    Returns:
        Opened image with the input shape.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = _validate_image(img)
    element = _validate_kernel(kernel).astype(np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, element)


def closing(img: NDArray[np.generic], kernel: NDArray[np.generic]) -> Image:
    """Fill small dark gaps using OpenCV morphological closing.

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        kernel: Two-dimensional structuring element.

    Returns:
        Closed image with the input shape.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = _validate_image(img)
    element = _validate_kernel(kernel).astype(np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)


def gradient(img: NDArray[np.generic], kernel: NDArray[np.generic]) -> Image:
    """Compute the morphological gradient (dilation minus erosion).

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        kernel: Two-dimensional structuring element.

    Returns:
        Morphological edge image with the input shape.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = _validate_image(img)
    element = _validate_kernel(kernel).astype(np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, element)


def tophat(img: NDArray[np.generic], kernel: NDArray[np.generic]) -> Image:
    """Extract small bright details as image minus its morphological opening.

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        kernel: Two-dimensional structuring element.

    Returns:
        Top-hat image with the input shape.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = _validate_image(img)
    element = _validate_kernel(kernel).astype(np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, element)


def blackhat(img: NDArray[np.generic], kernel: NDArray[np.generic]) -> Image:
    """Extract small dark details as closing minus the image.

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        kernel: Two-dimensional structuring element.

    Returns:
        Black-hat image with the input shape.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = _validate_image(img)
    element = _validate_kernel(kernel).astype(np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, element)


def _validate_binary(binary: NDArray[np.generic]) -> Image:
    """Validate a uint8 binary foreground mask.

    Args:
        binary: Two-dimensional uint8 mask.

    Returns:
        The input array without mutation.

    Syllabus: Unit 3 - mathematical morphology
    """
    image = np.asarray(binary)
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("binary must be a nonempty two-dimensional uint8 mask.")
    return image


def remove_specks(binary: NDArray[np.generic], k: int) -> Image:
    """Remove small foreground specks with a square morphological opening.

    Args:
        binary: uint8 binary image with foreground nonzero.
        k: Positive odd structuring-element side length.

    Returns:
        Speck-reduced uint8 binary mask.

    Syllabus: Unit 3 - morphological noise removal
    """
    image = _validate_binary(binary)
    if not isinstance(k, (int, np.integer)) or k <= 0 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer.")
    return opening(image, structuring_element("rect", int(k)))


def join_broken_strokes(binary: NDArray[np.generic], k: int) -> Image:
    """Join small gaps in foreground strokes with square morphological closing.

    Args:
        binary: uint8 binary image with foreground nonzero.
        k: Positive odd structuring-element side length.

    Returns:
        Stroke-connected uint8 binary mask.

    Syllabus: Unit 3 - morphological noise removal
    """
    image = _validate_binary(binary)
    if not isinstance(k, (int, np.integer)) or k <= 0 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer.")
    return closing(image, structuring_element("rect", int(k)))


def extract_lines(
    binary: NDArray[np.generic], orientation: LineOrientation, length: int
) -> Image:
    """Extract long horizontal or vertical lines with a directional opening.

    Args:
        binary: uint8 binary image with foreground nonzero.
        orientation: Either ``"horizontal"`` or ``"vertical"``.
        length: Positive line-kernel length.

    Returns:
        uint8 mask containing the extracted lines.

    Syllabus: Unit 3 - document morphology
    """
    image = _validate_binary(binary)
    if not isinstance(length, (int, np.integer)) or length <= 0:
        raise ValueError("length must be a positive integer.")
    if orientation == "horizontal":
        kernel_size = (int(length), 1)
    elif orientation == "vertical":
        kernel_size = (1, int(length))
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'.")
    return opening(image, structuring_element("rect", kernel_size))


def text_line_mask(
    binary: NDArray[np.generic], kw: int, kh: int
) -> Image:
    """Merge nearby glyphs into horizontal text-line components by dilation.

    Args:
        binary: uint8 binary image with foreground nonzero.
        kw: Positive horizontal dilation-kernel width.
        kh: Positive vertical dilation-kernel height.

    Returns:
        Dilated uint8 text-line mask.

    Syllabus: Unit 3 - document morphology
    """
    image = _validate_binary(binary)
    width, height = _validate_size((kw, kh))
    return dilate(image, structuring_element("rect", (width, height)))

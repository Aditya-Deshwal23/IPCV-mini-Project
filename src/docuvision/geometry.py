"""Geometric transforms and document rectification.

Syllabus: Unit 5 - geometric image transformations
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def affine_transform(
    img: NDArray[np.generic],
    translation: tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
    scale: float = 1.0,
    shear: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[Image, NDArray[np.float64]]:
    """Apply a centered rotation/scale, horizontal shear, and translation.

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        translation: Horizontal and vertical offsets in output pixels.
        rotation: Counter-clockwise rotation angle in degrees.
        scale: Positive isotropic scale factor.
        shear: Horizontal shear coefficient.
        interpolation: OpenCV interpolation constant such as INTER_NEAREST,
            INTER_LINEAR, or INTER_CUBIC.

    Returns:
        Transformed image and the corresponding 3x3 homogeneous matrix.

    Syllabus: Unit 5 - geometric image transformations
    """
    image = np.asarray(img)
    valid_shape = image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 3))
    if image.dtype != np.uint8 or image.size == 0 or not valid_shape:
        raise ValueError("img must be a nonempty uint8 grayscale or BGR image.")
    if (
        len(translation) != 2
        or not np.all(np.isfinite(translation))
        or not np.isfinite(rotation)
        or not np.isfinite(scale)
        or scale <= 0
        or not np.isfinite(shear)
    ):
        raise ValueError("transform parameters must be finite and scale positive.")
    allowed_interpolations = {
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    }
    if interpolation not in allowed_interpolations:
        raise ValueError("interpolation must be an OpenCV geometric interpolation mode.")
    height, width = image.shape[:2]
    affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotation, scale)
    affine[0, 1] += shear
    affine[0, 2] += translation[0]
    affine[1, 2] += translation[1]
    matrix = np.vstack((affine, [0.0, 0.0, 1.0])).astype(np.float64)
    transformed = cv2.warpAffine(
        image,
        affine,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return transformed, matrix


def detect_document_corners(img: NDArray[np.generic]) -> NDArray[np.float32]:
    """Detect the largest four-corner document contour in a grayscale or BGR image.

    Args:
        img: Nonempty uint8 grayscale or BGR image.

    Returns:
        Float32 corner array in clockwise (top-left, top-right, bottom-right,
        bottom-left) order, with shape (4, 2).

    Raises:
        ValueError: If no quadrilateral document contour can be identified.

    Syllabus: Unit 5 - geometric image transformations
    """
    image = np.asarray(img)
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("img must be a nonempty uint8 image.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        raise ValueError("img must be grayscale or three-channel BGR.")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(gray.shape[0] * gray.shape[1])
    candidates: list[tuple[float, NDArray[np.float32]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.05:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            candidates.append((area, polygon.reshape(4, 2).astype(np.float32)))
    if not candidates:
        raise ValueError("No document quadrilateral was detected.")
    corners = max(candidates, key=lambda candidate: candidate[0])[1]
    sums = corners.sum(axis=1)
    differences = np.diff(corners, axis=1).reshape(-1)
    ordered = np.array(
        [
            corners[np.argmin(sums)],
            corners[np.argmin(differences)],
            corners[np.argmax(sums)],
            corners[np.argmax(differences)],
        ],
        dtype=np.float32,
    )
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("Detected document corners could not be ordered.")
    return ordered

"""Image, OCR, geometry, and focus metrics used in evaluation.

Syllabus: Unit 1 - image processing evaluation
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity

from docuvision.ocr import cer as _cer
from docuvision.ocr import wer as _wer


def _validate_pair(
    reference: NDArray[np.generic], image: NDArray[np.generic]
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    """Validate same-shaped finite numeric image arrays."""
    first, second = np.asarray(reference), np.asarray(image)
    if (
        first.shape != second.shape
        or first.size == 0
        or not np.issubdtype(first.dtype, np.number)
        or not np.issubdtype(second.dtype, np.number)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("images must be nonempty, finite, numeric, and same-shaped.")
    return first, second


def psnr(reference: NDArray[np.generic], image: NDArray[np.generic]) -> float:
    """Compute peak signal-to-noise ratio with 8-bit peak value 255.

    Args:
        reference: Reference image.
        image: Image being evaluated, same shape as reference.

    Returns:
        PSNR in decibels; infinity for exact equality.

    Syllabus: Unit 1 - image quality metrics
    """
    first, second = _validate_pair(reference, image)
    mse = float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0**2 / mse))


def ssim(reference: NDArray[np.generic], image: NDArray[np.generic]) -> float:
    """Compute structural similarity for grayscale or multichannel images.

    Args:
        reference: Reference image.
        image: Image being evaluated, same shape as reference.

    Returns:
        SSIM score using data range 255.

    Syllabus: Unit 1 - image quality metrics
    """
    first, second = _validate_pair(reference, image)
    if first.ndim not in (2, 3):
        raise ValueError("SSIM accepts 2D grayscale or 3D channel-last images.")
    minimum_side = min(first.shape[:2])
    window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
    if window < 3:
        raise ValueError("SSIM requires image dimensions of at least three pixels.")
    return float(
        structural_similarity(
            first,
            second,
            data_range=255,
            channel_axis=-1 if first.ndim == 3 else None,
            win_size=window,
        )
    )


def cer(reference: str, hypothesis: str) -> float:
    """Compute normalized-text character error rate.

    Args:
        reference: Ground-truth text.
        hypothesis: Recognized text.

    Returns:
        Character Levenshtein distance divided by normalized reference length.

    Syllabus: Unit 1 - OCR evaluation
    """
    return _cer(reference, hypothesis)


def wer(reference: str, hypothesis: str) -> float:
    """Compute normalized-text word error rate.

    Args:
        reference: Ground-truth text.
        hypothesis: Recognized text.

    Returns:
        Word Levenshtein distance divided by normalized reference word count.

    Syllabus: Unit 1 - OCR evaluation
    """
    return _wer(reference, hypothesis)


def corner_error(
    predicted: NDArray[np.generic], true: NDArray[np.generic]
) -> float:
    """Compute mean Euclidean error between corresponding document corners.

    Args:
        predicted: Four predicted (x, y) coordinates.
        true: Four ground-truth (x, y) coordinates in the same order.

    Returns:
        Mean corner displacement in pixels.

    Syllabus: Unit 1 - geometric evaluation metrics
    """
    prediction = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(true, dtype=np.float64)
    if (
        prediction.shape != (4, 2)
        or target.shape != (4, 2)
        or not np.all(np.isfinite(prediction))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("predicted and true corners must be finite arrays of shape (4, 2).")
    return float(np.linalg.norm(prediction - target, axis=1).mean())


def sharpness(img: NDArray[np.generic]) -> float:
    """Measure focus as variance of the grayscale 3x3 Laplacian.

    Args:
        img: Nonempty uint8 grayscale or BGR image.

    Returns:
        Laplacian response variance.

    Syllabus: Unit 1 - image quality metrics
    """
    image = np.asarray(img)
    if image.dtype != np.uint8 or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("img must be a nonempty uint8 grayscale or color image.")
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            raise ValueError("color images must have one or three channels.")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def rms_contrast(img: NDArray[np.generic]) -> float:
    """Measure root-mean-square intensity contrast as standard deviation.

    Args:
        img: Nonempty numeric grayscale image.

    Returns:
        Pixel-intensity standard deviation.

    Syllabus: Unit 1 - image quality metrics
    """
    image = np.asarray(img)
    if image.size == 0 or image.ndim != 2 or not np.issubdtype(image.dtype, np.number):
        raise ValueError("img must be a nonempty numeric grayscale image.")
    if not np.all(np.isfinite(image)):
        raise ValueError("img must contain only finite values.")
    return float(image.astype(np.float64).std())

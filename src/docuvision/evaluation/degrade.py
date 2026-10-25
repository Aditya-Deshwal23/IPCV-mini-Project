"""Synthetic degradation operators for document evaluation data.

Syllabus: Unit 2 - image degradation and restoration
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
ShadowKind = Literal["linear", "radial"]


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate an image as nonempty uint8 grayscale or BGR.

    Args:
        img: Image array to validate.

    Returns:
        The input image without mutation.

    Syllabus: Unit 2 - image degradation and restoration
    """
    image = np.asarray(img)
    if (
        image.dtype != np.uint8
        or image.size == 0
        or image.ndim not in (2, 3)
        or (image.ndim == 3 and image.shape[2] not in (1, 3))
    ):
        raise ValueError("Expected a nonempty uint8 grayscale or BGR image.")
    return image


def add_shadow(
    img: NDArray[np.generic],
    strength: float,
    kind: ShadowKind,
    seed: int,
) -> Image:
    """Apply a deterministic smooth linear or radial illumination shadow.

    Args:
        img: Nonempty uint8 grayscale or BGR source.
        strength: Shadow attenuation in [0, 1]; zero leaves input unchanged.
        kind: ``"linear"`` varying across a random direction or ``"radial"``
            centered at a seeded location.
        seed: Random seed for the shadow direction/center.

    Returns:
        Shadowed uint8 image with the original shape.

    Syllabus: Unit 2 - illumination degradation
    """
    image = _validate_image(img)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and in [0, 1].")
    if kind not in ("linear", "radial"):
        raise ValueError("kind must be 'linear' or 'radial'.")
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    if kind == "linear":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = (x / max(width - 1, 1) - 0.5) * np.cos(angle) + (
            y / max(height - 1, 1) - 0.5
        ) * np.sin(angle)
        normalized = (direction - direction.min()) / max(
            float(direction.max() - direction.min()), 1e-12
        )
        field = 1.0 - float(strength) * normalized
    else:
        center_x = float(rng.uniform(0.25, 0.75) * width)
        center_y = float(rng.uniform(0.25, 0.75) * height)
        distance = np.hypot((x - center_x) / max(width, 1), (y - center_y) / max(height, 1))
        normalized = distance / max(float(distance.max()), 1e-12)
        field = 1.0 - float(strength) * (1.0 - normalized) ** 2
    if image.ndim == 3:
        field = field[:, :, None]
    return np.clip(np.rint(image.astype(np.float32) * field), 0, 255).astype(np.uint8)


def add_noise(
    img: NDArray[np.generic], sigma: float, sp_prob: float, seed: int
) -> Image:
    """Add seeded Gaussian noise and salt-and-pepper impulses.

    Args:
        img: Nonempty uint8 grayscale or BGR source.
        sigma: Gaussian noise standard deviation in intensity levels.
        sp_prob: Fraction of pixels to replace with salt/pepper impulses.
        seed: Random seed for both noise sources.

    Returns:
        Noisy uint8 image with the original shape.

    Syllabus: Unit 2 - image noise degradation
    """
    image = _validate_image(img)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")
    if not np.isfinite(sp_prob) or not 0 <= sp_prob <= 1:
        raise ValueError("sp_prob must be finite and in [0, 1].")
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    result = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    if sp_prob:
        random_values = rng.random(image.shape[:2])
        pepper = random_values < sp_prob / 2.0
        salt = (random_values >= sp_prob / 2.0) & (random_values < sp_prob)
        result[pepper] = 0
        result[salt] = 255
    return result


def add_blur(
    img: NDArray[np.generic],
    seed: int,
    sigma: float | None = None,
    motion_length: int | None = None,
) -> Image:
    """Apply seeded Gaussian blur or linear motion blur.

    Args:
        img: Nonempty uint8 grayscale or BGR source.
        seed: Random seed for motion direction when using a motion kernel.
        sigma: Positive Gaussian standard deviation; takes precedence when set.
        motion_length: Positive length for a motion-blur kernel.

    Returns:
        Blurred uint8 image with the original shape.

    Syllabus: Unit 2 - image blur degradation
    """
    image = _validate_image(img)
    if sigma is not None:
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive.")
        return cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    if motion_length is None or not isinstance(motion_length, int) or motion_length <= 0:
        raise ValueError("motion_length must be a positive integer when sigma is omitted.")
    rng = np.random.default_rng(seed)
    kernel = np.zeros((motion_length, motion_length), dtype=np.float32)
    center = (motion_length - 1) / 2.0
    angle = float(rng.uniform(0.0, np.pi))
    half = center
    dx, dy = half * np.cos(angle), half * np.sin(angle)
    start = (int(round(center - dx)), int(round(center - dy)))
    end = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, start, end, 1.0, thickness=1)
    if float(kernel.sum()) == 0:
        kernel[motion_length // 2, :] = 1.0
    kernel /= float(kernel.sum())
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def reduce_contrast(
    img: NDArray[np.generic], alpha: float, seed: int = 0
) -> Image:
    """Reduce image contrast around its mid-gray center.

    Args:
        img: Nonempty uint8 grayscale or BGR source.
        alpha: Contrast multiplier in [0, 1]; one preserves contrast.
        seed: Reserved common degradation seed; the point transform is deterministic.

    Returns:
        Contrast-reduced uint8 image.

    Syllabus: Unit 2 - contrast degradation
    """
    image = _validate_image(img)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1].")
    np.random.default_rng(seed)
    return np.clip(
        np.rint((image.astype(np.float32) - 127.5) * float(alpha) + 127.5),
        0,
        255,
    ).astype(np.uint8)


def jpeg_artifacts(
    img: NDArray[np.generic], q: int, seed: int = 0
) -> Image:
    """Encode and decode an image as JPEG at a specified quality.

    Args:
        img: Nonempty uint8 grayscale or BGR source.
        q: JPEG quality from 0 (lowest) through 100 (highest).
        seed: Reserved common degradation seed; JPEG encoding is deterministic.

    Returns:
        JPEG-degraded uint8 image with the original shape.

    Syllabus: Unit 2 - compression artifacts
    """
    image = _validate_image(img)
    if not isinstance(q, int) or not 0 <= q <= 100:
        raise ValueError("q must be an integer from 0 through 100.")
    np.random.default_rng(seed)
    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(q)]
    )
    if not success:
        raise RuntimeError("OpenCV could not encode JPEG degradation.")
    mode = cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR
    decoded = cv2.imdecode(encoded, mode)
    if decoded is None:
        raise RuntimeError("OpenCV could not decode JPEG degradation.")
    if image.ndim == 3 and image.shape[2] == 1:
        decoded = decoded[:, :, None]
    return decoded


def random_perspective(
    img: NDArray[np.generic],
    max_shift: float,
    bg: NDArray[np.generic],
    seed: int,
) -> tuple[Image, NDArray[np.float32]]:
    """Place a page under a seeded random perspective on a supplied background.

    Args:
        img: Nonempty uint8 grayscale or BGR page image.
        max_shift: Maximum corner offset as a fraction of the smaller page side.
        bg: uint8 grayscale or BGR background canvas.
        seed: Random seed for destination-corner perturbations.

    Returns:
        (Perspective-degraded canvas, true destination corners) where corners
        are ordered TL, TR, BR, BL in destination pixel coordinates.

    Syllabus: Unit 2 - projective degradation
    """
    image, background = _validate_image(img), _validate_image(bg)
    if not np.isfinite(max_shift) or not 0 <= max_shift < 0.5:
        raise ValueError("max_shift must be finite and in [0, 0.5).")
    if image.ndim != background.ndim or (
        image.ndim == 3 and image.shape[2] != background.shape[2]
    ):
        raise ValueError("img and bg must have matching channel formats.")
    height, width = image.shape[:2]
    bg_height, bg_width = background.shape[:2]
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    base_x = (bg_width - width) / 2.0
    base_y = (bg_height - height) / 2.0
    destination = source + np.array([base_x, base_y], dtype=np.float32)
    rng = np.random.default_rng(seed)
    max_pixels = float(max_shift) * min(height, width)
    perturbation = rng.uniform(-max_pixels, max_pixels, size=(4, 2)).astype(np.float32)
    destination += perturbation
    destination[:, 0] = np.clip(destination[:, 0], 0, bg_width - 1)
    destination[:, 1] = np.clip(destination[:, 1], 0, bg_height - 1)
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image,
        transform,
        (bg_width, bg_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        transform,
        (bg_width, bg_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if background.ndim == 3:
        mask = mask[:, :, None]
    output = np.where(mask > 0, warped, background).astype(np.uint8)
    return output, destination


def random_rotation(
    img: NDArray[np.generic], max_angle: float, seed: int
) -> Image:
    """Apply a seeded random in-plane rotation without changing image dimensions.

    Args:
        img: Nonempty uint8 grayscale or BGR source.
        max_angle: Maximum absolute rotation angle in degrees.
        seed: Random seed for the sampled angle.

    Returns:
        Rotated uint8 image with the original shape.

    Syllabus: Unit 2 - geometric degradation
    """
    image = _validate_image(img)
    if not np.isfinite(max_angle) or max_angle < 0 or max_angle > 45:
        raise ValueError("max_angle must be finite and in [0, 45].")
    angle = float(np.random.default_rng(seed).uniform(-max_angle, max_angle))
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

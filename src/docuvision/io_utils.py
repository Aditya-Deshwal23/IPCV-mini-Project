"""Image and video input/output helpers for DocuVision.

Syllabus: Unit 1 - image types, formats, and representation
"""

from collections.abc import Generator
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

Image: TypeAlias = NDArray[np.uint8]
ImagePath: TypeAlias = str | Path
ImageSource: TypeAlias = ImagePath | bytes | bytearray | memoryview


def load_image(path_or_bytes: ImageSource) -> Image:
    """Decode an image as a three-channel BGR uint8 array.

    Args:
        path_or_bytes: Filesystem path or encoded image data.

    Returns:
        Decoded image in BGR channel order.

    Syllabus: Unit 1 - image types, formats, and representation
    """
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(path_or_bytes, dtype=np.uint8)
    else:
        encoded = np.frombuffer(Path(path_or_bytes).read_bytes(), dtype=np.uint8)

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The provided data could not be decoded as an image.")
    return ensure_uint8(image)


def save_image(path: ImagePath, img: NDArray[np.generic]) -> None:
    """Encode and write an image to a path.

    Args:
        path: Destination path; its extension selects the image format.
        img: Image array to save.

    Returns:
        None.

    Syllabus: Unit 1 - image types, formats, and representation
    """
    destination = Path(path)
    suffix = destination.suffix.lower()
    if not suffix:
        raise ValueError("The destination path must include an image extension.")

    image = ensure_uint8(img)
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"OpenCV could not encode the image as {suffix}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded.tobytes())


def to_gray(img: NDArray[np.generic]) -> Image:
    """Convert a grayscale or BGR image to grayscale uint8.

    Args:
        img: Two-dimensional grayscale or three-channel BGR image.

    Returns:
        Grayscale image as a new uint8 array.

    Syllabus: Unit 1 - image types, formats, and representation
    """
    image = ensure_uint8(img)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].copy()
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a grayscale image or a three-channel BGR image.")


def ensure_uint8(img: NDArray[np.generic]) -> Image:
    """Return an independent uint8 image, clipping other numeric types.

    Args:
        img: Numeric image array.

    Returns:
        A copied uint8 array with values clipped to the uint8 range.

    Syllabus: Unit 1 - image types, formats, and representation
    """
    image = np.asarray(img)
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("Image data must have a numeric dtype.")
    if not np.isfinite(image).all():
        raise ValueError("Image data must contain only finite values.")
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def resize_max_side(img: NDArray[np.generic], max_side: int) -> Image:
    """Resize an image so its longest side does not exceed a limit.

    Args:
        img: Two-dimensional or channel-last image.
        max_side: Positive maximum width or height.

    Returns:
        A resized uint8 image, or a copy if already within the limit.

    Syllabus: Unit 1 - image sampling and spatial resolution
    """
    if max_side <= 0:
        raise ValueError("max_side must be a positive integer.")
    image = ensure_uint8(img)
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Expected a non-empty two-dimensional or channel-last image.")
    longest_side = max(image.shape[:2])
    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    new_width = max(1, round(image.shape[1] * scale))
    new_height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(img: NDArray[np.generic]) -> Image:
    """Convert a three-channel BGR image to RGB for display.

    Args:
        img: Three-channel BGR image.

    Returns:
        RGB image as a new uint8 array.

    Syllabus: Unit 1 - image types, formats, and representation
    """
    image = ensure_uint8(img)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_info(img: NDArray[np.generic]) -> dict[str, str | tuple[int, ...] | int]:
    """Describe image representation and storage characteristics.

    Args:
        img: Image array to inspect.

    Returns:
        Dictionary with type, shape, dtype, bit depth, unique levels, and bytes.

    Syllabus: Unit 1 - image types and formats
    """
    image = np.asarray(img)
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image_type = "binary" if np.unique(image).size <= 2 else "gray"
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        image_type = "color"
    else:
        raise ValueError("Expected a grayscale or color image array.")

    return {
        "type": image_type,
        "shape": tuple(image.shape),
        "dtype": str(image.dtype),
        "bit_depth": image.dtype.itemsize * 8,
        "n_unique_levels": int(np.unique(image).size),
        "size_bytes": int(image.nbytes),
    }


def read_video_frames(path: ImagePath, stride: int) -> Generator[Image, None, None]:
    """Yield decoded BGR frames at the requested frame interval.

    Args:
        path: Path to the video file.
        stride: Yield every stride-th frame, starting with frame zero.

    Returns:
        Generator yielding BGR uint8 video frames.

    Syllabus: Unit 1 - image sequences and video representation
    """
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {path}")

    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % stride == 0:
                yield ensure_uint8(frame)
            frame_index += 1
    finally:
        capture.release()

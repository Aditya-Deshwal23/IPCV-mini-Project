"""Tests for bundled app assets that should work in a fresh deployment."""

from pathlib import Path

import cv2


def test_bundled_sample_image_is_readable() -> None:
    """Ensure the Sample input has an image available in a clean checkout."""
    project_root = Path(__file__).resolve().parents[1]
    sample = project_root / "data" / "samples" / "document_sample.png"

    image = cv2.imread(str(sample), cv2.IMREAD_COLOR)

    assert image is not None
    assert image.ndim == 3
    assert image.dtype.name == "uint8"

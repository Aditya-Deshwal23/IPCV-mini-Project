"""Tests for reusable geometric transforms.

Syllabus: Unit 5 - geometric image transformations
"""

import numpy as np
import pytest

from docuvision.geometry import affine_transform


def test_affine_transform_identity_preserves_pixels_and_reports_matrix() -> None:
    """Identity parameters preserve image content and return homogeneous identity.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 5 - geometric image transformations
    """
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    original = image.copy()
    transformed, matrix = affine_transform(image)
    np.testing.assert_array_equal(transformed, image)
    np.testing.assert_array_equal(matrix, np.eye(3))
    np.testing.assert_array_equal(image, original)


def test_affine_transform_validates_scale() -> None:
    """Reject zero scale instead of returning a success-shaped result.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 5 - geometric image transformations
    """
    with pytest.raises(ValueError, match="scale positive"):
        affine_transform(np.zeros((4, 4), dtype=np.uint8), scale=0)

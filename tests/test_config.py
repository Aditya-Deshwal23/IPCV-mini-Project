"""Tests for pipeline configuration defaults and immutable updates.

Syllabus: Unit 1 - image processing workflow configuration
"""

from dataclasses import FrozenInstanceError

import pytest

from docuvision.config import PipelineConfig


def test_pipeline_config_has_requested_defaults() -> None:
    """Verify all baseline values match the project specification.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image processing workflow configuration
    """
    config = PipelineConfig()

    assert config.to_dict() == {
        "work_size": 1600,
        "illum": "homomorphic",
        "denoise": "median",
        "denoise_ksize": 3,
        "contrast": "clahe",
        "clahe_clip": 2.0,
        "clahe_tile": 8,
        "sharpen": "unsharp",
        "sharpen_amount": 1.0,
        "binarize": "sauvola",
        "sauvola_window": 25,
        "sauvola_k": 0.2,
        "morph_cleanup": True,
        "deskew": True,
        "ocr_source": "enhanced",
        "disabled_steps": (),
    }


def test_replace_returns_a_new_config_and_preserves_original() -> None:
    """Replace selected settings without mutating the frozen configuration.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image processing workflow configuration
    """
    original = PipelineConfig()

    updated = original.replace(work_size=800, deskew=False)

    assert updated.work_size == 800
    assert updated.deskew is False
    assert original.work_size == 1600
    assert original.deskew is True
    with pytest.raises(FrozenInstanceError):
        original.work_size = 800  # type: ignore[misc]


def test_pipeline_presets_are_available() -> None:
    """Construct quality and single-threshold baseline configurations.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image processing workflow configuration
    """
    fast = PipelineConfig.preset("fast")
    balanced = PipelineConfig.preset("balanced")
    quality = PipelineConfig.preset("max_quality")
    otsu = PipelineConfig.preset("baseline_otsu")
    adaptive = PipelineConfig.preset("baseline_adaptive")

    assert fast.work_size < balanced.work_size < quality.work_size
    assert otsu.binarize == "otsu"
    assert adaptive.binarize == "adaptive"
    assert "denoise" in otsu.disabled_steps
    assert "contrast" in adaptive.disabled_steps
    with pytest.raises(ValueError, match="name"):
        PipelineConfig.preset("unknown")

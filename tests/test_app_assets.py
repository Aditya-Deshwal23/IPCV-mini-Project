"""Tests for bundled app assets that should work in a fresh deployment."""

from pathlib import Path

import cv2
from streamlit.testing.v1 import AppTest

from docuvision.config import PipelineConfig
from docuvision.geometry import detect_document_corners
from docuvision.pipeline import run_pipeline


def test_sample_picker_loads_the_clean_demo_page() -> None:
    """Check that the app exposes and loads the clean sample from its sidebar."""
    project_root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(project_root / "app.py", default_timeout=30).run()

    app.radio(key="source_mode").set_value("Sample").run()

    image = app.session_state["source_image"]
    assert not app.exception
    assert app.session_state["source_name"] == "01_clean_print_page.jpg"
    assert image.ndim == 3
    assert any(
        "Best for your first scan" in caption.value for caption in app.caption
    )


def test_demo_sample_images_are_readable_and_detectable() -> None:
    """Validate both demo images, their truth text, and page-corner processing."""
    project_root = Path(__file__).resolve().parents[1]
    sample_directory = project_root / "data" / "samples"
    samples = {
        "01_clean_print_page.jpg",
        "02_shadowed_tilted_page.jpg",
    }
    sample_paths = {path.name: path for path in sample_directory.glob("*.jpg")}

    assert set(sample_paths) == samples

    for name, sample_path in sample_paths.items():
        image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
        ground_truth_path = sample_path.with_suffix(".txt")

        assert image is not None
        assert image.ndim == 3
        assert image.dtype.name == "uint8"
        assert ground_truth_path.is_file()
        assert ground_truth_path.read_text(encoding="utf-8").strip()
        assert detect_document_corners(image).shape == (4, 2)

        result = run_pipeline(image, PipelineConfig.preset("fast"))

        assert result.metrics["corners_detected"] is True
        assert result.final_gray.ndim == 2
        assert set(result.final_binary.ravel()) <= {0, 255}

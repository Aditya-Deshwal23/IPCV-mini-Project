"""Focused tests for Scan-tab source selection and OCR retries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from docuvision.config import PipelineConfig
from docuvision.pipeline import PipelineResult
from docuvision.ui import tab_scan


def _pipeline_result() -> PipelineResult:
    """Create a small completed result for the UI source-selection helpers."""
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    binary = np.where(gray > 7, 255, 0).astype(np.uint8)
    return PipelineResult(
        stages=[],
        final_gray=gray,
        final_binary=binary,
        text="recognized",
        boxes=[],
        corners=None,
        metrics={},
    )


def test_select_ocr_image_returns_requested_representation() -> None:
    """Choose enhanced grayscale, binary, or original grayscale as requested."""
    result = _pipeline_result()
    source = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)

    enhanced = tab_scan._select_ocr_image("Enhanced grayscale", source, result)
    binary = tab_scan._select_ocr_image("Black and white", source, result)
    original = tab_scan._select_ocr_image("Original grayscale", source, result)

    assert np.array_equal(enhanced, result.final_gray)
    assert np.array_equal(binary, result.final_binary)
    assert original.shape == (4, 4)
    assert original.dtype == np.uint8
    with pytest.raises(ValueError, match="Unsupported OCR image choice"):
        tab_scan._select_ocr_image("unknown", source, result)


def test_source_match_prevents_showing_stale_pipeline_result() -> None:
    """Treat replaced or reshaped images as new inputs."""
    first = np.zeros((3, 4), dtype=np.uint8)
    same_pixels = first.copy()
    changed_pixels = first.copy()
    changed_pixels[0, 0] = 1

    assert tab_scan._source_matches(first, same_pixels)
    assert not tab_scan._source_matches(first, changed_pixels)
    assert not tab_scan._source_matches(first, np.zeros((4, 3), dtype=np.uint8))
    assert not tab_scan._source_matches(first, None)


def test_crop_ocr_image_returns_region_copy_and_rejects_empty_ranges() -> None:
    """Crop a nonempty OCR region without changing the source image."""
    image = np.arange(60, dtype=np.uint8).reshape(6, 10)

    crop = tab_scan._crop_ocr_image(image, (2, 7), (1, 5))

    assert crop.shape == (4, 5)
    assert np.array_equal(crop, image[1:5, 2:7])
    crop[:] = 0
    assert not np.array_equal(crop, image[1:5, 2:7])
    with pytest.raises(ValueError, match="nonempty region"):
        tab_scan._crop_ocr_image(image, (4, 4), (1, 5))


def test_cached_ocr_passes_selected_segmentation_mode(monkeypatch) -> None:
    """Pass the chosen Tesseract PSM to the OCR adapter."""
    calls: list[int] = []

    def fake_run_ocr(image: np.ndarray, psm: int) -> tuple[str, pd.DataFrame]:
        calls.append(psm)
        assert image.shape == (2, 3)
        return f"mode {psm}", pd.DataFrame(columns=["text"])

    monkeypatch.setattr(tab_scan, "run_ocr", fake_run_ocr)
    tab_scan._cached_ocr.clear()
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)

    text, boxes = tab_scan._cached_ocr(image.tobytes(), image.shape, 11, True)

    assert text == "mode 11"
    assert boxes.empty
    assert calls == [11]


def test_cached_ocr_invalidates_when_tesseract_availability_changes(
    monkeypatch,
) -> None:
    """Do not reuse an unavailable OCR result after Tesseract is installed."""
    calls: list[int] = []

    def fake_run_ocr(image: np.ndarray, psm: int) -> tuple[str, pd.DataFrame]:
        calls.append(psm)
        return "recognized", pd.DataFrame(columns=["text"])

    monkeypatch.setattr(tab_scan, "run_ocr", fake_run_ocr)
    tab_scan._cached_ocr.clear()
    image = np.zeros((2, 3), dtype=np.uint8)

    tab_scan._cached_ocr(image.tobytes(), image.shape, 6, False)
    tab_scan._cached_ocr(image.tobytes(), image.shape, 6, True)

    assert calls == [6, 6]
    tab_scan._cached_ocr.clear()


def test_cached_pipeline_invalidates_when_tesseract_availability_changes(
    monkeypatch,
) -> None:
    """Recompute a cached scan when the OCR dependency becomes available."""
    calls: list[PipelineConfig] = []

    def fake_run_pipeline(image: np.ndarray, config: PipelineConfig) -> PipelineResult:
        calls.append(config)
        return _pipeline_result()

    monkeypatch.setattr(tab_scan, "run_pipeline", fake_run_pipeline)
    tab_scan._cached_pipeline.clear()
    image = np.zeros((2, 3), dtype=np.uint8)
    config = PipelineConfig()
    config_json = json.dumps(config.to_dict(), sort_keys=True)

    tab_scan._cached_pipeline(image.tobytes(), image.shape, config_json, False)
    tab_scan._cached_pipeline(image.tobytes(), image.shape, config_json, True)

    assert len(calls) == 2
    tab_scan._cached_pipeline.clear()


def test_scan_tab_defaults_to_final_stage_and_exposes_ocr_choices(monkeypatch) -> None:
    """Render a synthetic scan and verify its final-stage and OCR retry controls."""
    ocr_calls: list[tuple[tuple[int, ...], int]] = []

    def fake_run_ocr(
        image: np.ndarray, psm: int = 6, lang: str = "eng"
    ) -> tuple[str, pd.DataFrame]:
        ocr_calls.append((image.shape, psm))
        return f"recognized with mode {psm}", pd.DataFrame(columns=["text"])

    monkeypatch.setattr(tab_scan, "run_ocr", fake_run_ocr)
    tab_scan._cached_ocr.clear()
    image = np.full((48, 64, 3), 245, dtype=np.uint8)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    app.session_state["source_image"] = image
    app.session_state["source_name"] = "synthetic.png"
    app.session_state["pipeline_result"] = _pipeline_result()
    app.session_state["last_scan_source"] = image.copy()
    app.session_state["last_scan_ocr_available"] = not tab_scan.tesseract_available()
    app.run()
    assert "pipeline_result" not in app.session_state
    app.button(key="scan_run").click().run()

    assert not app.exception
    stage_selector = next(
        widget for widget in app.select_slider if widget.label == "Pipeline stage"
    )
    assert stage_selector.value == len(stage_selector.options) - 1
    assert stage_selector.options[-1].endswith("Ocr")
    assert {widget.label for widget in app.selectbox} >= {
        "Image sent to OCR",
        "Text layout",
    }
    app.selectbox(key="scan_ocr_image_option").select("Black and white").run()
    app.selectbox(key="scan_ocr_psm").select(11).run()
    app.checkbox(key="scan_ocr_crop_enabled").check().run()
    next(
        widget for widget in app.slider if widget.label == "Horizontal crop: left to right"
    ).set_value((10, 40)).run()
    next(
        widget for widget in app.slider if widget.label == "Vertical crop: top to bottom"
    ).set_value((5, 30)).run()
    app.button(key="scan_ocr_retry").click().run()

    recognized = next(
        widget.value for widget in app.text_area if widget.label == "Recognized text"
    )
    assert not app.exception
    assert recognized == "recognized with mode 11"
    assert ocr_calls == [((25, 30), 11)]

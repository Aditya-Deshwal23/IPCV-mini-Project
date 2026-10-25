"""Tests for the ordered document-processing pipeline.

Syllabus: Unit 1 - digital image processing pipeline
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from docuvision.config import PipelineConfig
from docuvision.pipeline import PIPELINE_STEPS, run_pipeline, scan_to_pdf


def _degraded_page() -> NDArray[np.uint8]:
    """Create a small shaded, noisy document page without external assets.

    Args:
        None.

    Returns:
        BGR uint8 page with a smooth illumination gradient and printed lines.

    Syllabus: Unit 1 - image-processing test data
    """
    height, width = 180, 240
    gradient = np.linspace(175, 240, width, dtype=np.uint8)
    page = np.tile(gradient, (height, 1))
    cv2.putText(
        page,
        "DOCUVISION",
        (24, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        25,
        2,
        cv2.LINE_AA,
    )
    for row in (82, 105, 128, 151):
        cv2.line(page, (24, row), (205, row), 35, 2, cv2.LINE_AA)
    random = np.random.default_rng(41)
    noise = random.normal(0, 3, page.shape)
    degraded = np.clip(page.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(degraded, cv2.COLOR_GRAY2BGR)


def test_pipeline_runs_without_tesseract_and_records_all_stages(
    monkeypatch,
) -> None:
    """Run a degraded synthetic page and validate output and step ordering.

    Args:
        monkeypatch: Pytest patch helper for simulating missing Tesseract.

    Returns:
        None.

    Syllabus: Unit 1 - digital image processing pipeline
    """
    from docuvision import ocr

    monkeypatch.setattr(ocr, "tesseract_available", lambda: False)
    config = PipelineConfig(
        work_size=200,
        illum="none",
        denoise="median",
        contrast="clahe",
        sharpen="none",
        binarize="sauvola",
        sauvola_window=15,
        morph_cleanup=True,
        deskew=True,
    )

    result = run_pipeline(_degraded_page(), config)

    assert [stage.name for stage in result.stages] == list(PIPELINE_STEPS)
    assert result.final_gray.ndim == 2
    assert result.final_binary.ndim == 2
    assert set(np.unique(result.final_binary)) <= {0, 255}
    assert result.metrics["ocr_status"] == "unavailable"
    assert result.text.startswith("Tesseract unavailable:")


def test_disabled_step_is_recorded_as_skipped(monkeypatch) -> None:
    """Mark a configured-disabled denoise stage while keeping all stages visible.

    Args:
        monkeypatch: Pytest patch helper for simulating missing Tesseract.

    Returns:
        None.

    Syllabus: Unit 1 - configurable image-processing stages
    """
    from docuvision import ocr

    monkeypatch.setattr(ocr, "tesseract_available", lambda: False)
    config = PipelineConfig(
        work_size=180,
        illum="none",
        contrast="none",
        sharpen="none",
        binarize="otsu",
        morph_cleanup=False,
        deskew=False,
        disabled_steps=("denoise",),
    )

    result = run_pipeline(_degraded_page(), config)
    denoise = next(stage for stage in result.stages if stage.name == "denoise")

    assert denoise.skipped
    assert denoise.seconds == 0
    assert len(result.stages) == len(PIPELINE_STEPS)
    assert set(np.unique(result.final_binary)) <= {0, 255}


def test_scan_to_pdf_writes_readable_pdf(tmp_path, monkeypatch) -> None:
    """Export a result as a PDF without depending on OCR availability.

    Args:
        tmp_path: Pytest-managed destination directory.
        monkeypatch: Pytest patch helper for simulating missing Tesseract.

    Returns:
        None.

    Syllabus: Unit 1 - document input and output
    """
    from docuvision import ocr

    monkeypatch.setattr(ocr, "tesseract_available", lambda: False)
    result = run_pipeline(
        _degraded_page(),
        PipelineConfig(
            work_size=180,
            illum="none",
            denoise="none",
            contrast="none",
            sharpen="none",
            binarize="otsu",
            morph_cleanup=False,
            deskew=False,
            disabled_steps=("layout", "ocr"),
        ),
    )
    destination = tmp_path / "scan.pdf"

    scan_to_pdf(result, destination)

    assert destination.read_bytes().startswith(b"%PDF")

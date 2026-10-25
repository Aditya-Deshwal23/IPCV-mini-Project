"""Composition of DocuVision processing stages and intermediate outputs.

Syllabus: Unit 1 - digital image processing pipeline
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from docuvision.config import PipelineConfig
from docuvision.enhancement import (
    background_division,
    clahe,
    contrast_stretch_minmax,
    equalize_cv,
    gaussian_filter,
    high_boost,
    median_filter,
    unsharp_mask,
)
from docuvision.features import align_to_template
from docuvision.frequency import homomorphic_filter
from docuvision.geometry import detect_document_corners
from docuvision.morphology import closing, remove_specks, structuring_element
from docuvision.ocr import run_ocr
from docuvision.segmentation import (
    adaptive_threshold,
    draw_boxes,
    global_threshold,
    otsu_cv,
    sauvola,
    segment_text_blocks,
)

Image = NDArray[np.uint8]
PipelineOperation = Callable[[Image], Image]
PIPELINE_STEPS = (
    "resize_working",
    "document_rectification",
    "gray_conversion",
    "illumination_correction",
    "denoise",
    "contrast",
    "sharpen",
    "binarize",
    "morphology_cleanup",
    "deskew",
    "layout",
    "ocr",
)


@dataclass
class Stage:
    """A single timed pipeline operation and its image output.

    Args:
        name: Stable pipeline step identifier.
        image: Image produced at this stage or unchanged input when skipped.
        unit: Course unit associated with the operation.
        technique: Human-readable method name.
        params: Parameters used for this stage.
        seconds: Wall-clock duration; zero for skipped stages.
        skipped: Whether the stage was explicitly bypassed.

    Returns:
        Mutable stage record for UI presentation and analysis.

    Syllabus: Unit 1 - digital image processing pipeline
    """

    name: str
    image: Image
    unit: str
    technique: str
    params: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    skipped: bool = False


@dataclass
class PipelineResult:
    """All intermediate and final outputs of one document-processing run.

    Args:
        stages: Ordered per-step stage records.
        final_gray: Final enhanced grayscale document.
        final_binary: Final binary document image.
        text: OCR text or an explicit OCR-unavailable status.
        boxes: Text-layout boxes in (x, y, width, height) order.
        corners: Detected/rectified page corners, or None if unavailable.
        metrics: Timing, OCR status, and geometric processing details.

    Returns:
        Container used by the UI, evaluation, and PDF export.

    Syllabus: Unit 1 - digital image processing pipeline
    """

    stages: list[Stage]
    final_gray: Image
    final_binary: Image
    text: str
    boxes: list[tuple[int, int, int, int]]
    corners: NDArray[np.float32] | None
    metrics: dict[str, Any]


def _validate_input(img_bgr: NDArray[np.generic]) -> Image:
    """Validate a nonempty uint8 grayscale or BGR input.

    Args:
        img_bgr: Input document image.

    Returns:
        Original image array without mutation.

    Syllabus: Unit 1 - digital image processing pipeline
    """
    image = np.asarray(img_bgr)
    valid = image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] in (1, 3)
    )
    if image.dtype != np.uint8 or not valid or image.size == 0:
        raise ValueError("Expected a nonempty uint8 grayscale or BGR image.")
    return image


def _four_point_transform(
    image: Image, corners: NDArray[np.float32]
) -> Image:
    """Perspective rectify a page using clockwise ordered document corners.

    Args:
        image: BGR uint8 input.
        corners: (4, 2) float32 points ordered TL, TR, BR, BL.

    Returns:
        Warped BGR image sized to the page's estimated edge lengths.

    Syllabus: Unit 5 - perspective image transformation
    """
    top_left, top_right, bottom_right, bottom_left = corners
    width = max(
        int(round(np.linalg.norm(bottom_right - bottom_left))),
        int(round(np.linalg.norm(top_right - top_left))),
        1,
    )
    height = max(
        int(round(np.linalg.norm(top_right - bottom_right))),
        int(round(np.linalg.norm(bottom_left - top_left))),
        1,
    )
    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _run_step(
    stages: list[Stage],
    name: str,
    unit: str,
    technique: str,
    params: dict[str, Any],
    image: Image,
    operation: PipelineOperation,
    enabled: bool,
) -> Image:
    """Time an operation, retain its image, or record an explicit skipped step.

    Args:
        stages: Mutable ordered stage accumulator.
        name: Stable step identifier.
        unit: Course unit label.
        technique: Method label displayed to users.
        params: Operation parameters.
        image: Current image input.
        operation: Callable producing this stage's image.
        enabled: Whether to execute the operation.

    Returns:
        Operation output or an independent copy of input when disabled.

    Syllabus: Unit 1 - digital image processing pipeline
    """
    if not enabled:
        output = image.copy()
        stages.append(
            Stage(
                name=name,
                image=output,
                unit=unit,
                technique=technique,
                params=params.copy(),
                seconds=0.0,
                skipped=True,
            )
        )
        return output
    started = time.perf_counter()
    output = operation(image)
    elapsed = time.perf_counter() - started
    if not isinstance(output, np.ndarray) or output.dtype != np.uint8:
        raise TypeError(f"pipeline step {name!r} must return a uint8 image.")
    stages.append(
        Stage(
            name=name,
            image=output,
            unit=unit,
            technique=technique,
            params=params.copy(),
            seconds=elapsed,
            skipped=False,
        )
    )
    return output


def _odd_kernel(value: int, minimum: int = 3) -> int:
    """Return an odd kernel size near a requested scale, bounded below."""
    size = max(minimum, int(value))
    return size if size % 2 else size + 1


def _deskew_pair(
    gray: Image, binary: Image
) -> tuple[Image, Image, float]:
    """Rotate gray and binary images to align their foreground text baseline.

    Args:
        gray: Enhanced grayscale image with dark foreground.
        binary: Binary image with black foreground and white background.

    Returns:
        (deskewed gray, deskewed binary, clockwise angle in degrees).

    Syllabus: Unit 4 - fine geometric deskewing
    """
    foreground = (binary == 0).astype(np.uint8)
    points = cv2.findNonZero(foreground)
    if points is None or len(points) < 5:
        return gray.copy(), binary.copy(), 0.0
    angle = float(cv2.minAreaRect(points)[-1])
    if angle < -45.0:
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    if abs(angle) < 0.15 or abs(angle) > 15.0:
        return gray.copy(), binary.copy(), 0.0
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    corrected_gray = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    corrected_binary = cv2.warpAffine(
        binary,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return corrected_gray, corrected_binary, angle


def run_pipeline(
    img_bgr: NDArray[np.generic],
    cfg: PipelineConfig,
    template: NDArray[np.generic] | None = None,
) -> PipelineResult:
    """Run an ordered document pipeline and retain each stage image and timing.

    Individual stages can be bypassed by listing their identifiers in
    ``cfg.disabled_steps``. All steps remain visible in ``stages``; bypassed
    steps are marked ``skipped=True`` and preserve their incoming image.

    Args:
        img_bgr: Nonempty uint8 grayscale or BGR document image.
        cfg: Immutable pipeline settings and optional disabled step identifiers.
        template: Optional document template used for SIFT alignment when page
            corner detection fails.

    Returns:
        PipelineResult with intermediate stages, enhanced grayscale, binary
        result, OCR text, layout boxes, corners, and metrics.

    Raises:
        ValueError: If input, configuration, or a selected technique is invalid.

    Syllabus: Unit 1 - digital image processing pipeline
    """
    image = _validate_input(img_bgr)
    if cfg.work_size <= 0:
        raise ValueError("cfg.work_size must be positive.")
    unknown_steps = set(cfg.disabled_steps) - set(PIPELINE_STEPS)
    if unknown_steps:
        raise ValueError(f"unknown disabled pipeline steps: {sorted(unknown_steps)}")
    stages: list[Stage] = []
    metrics: dict[str, Any] = {}
    disabled = set(cfg.disabled_steps)

    def is_enabled(step: str) -> bool:
        """Check named step toggles in both global and category settings."""
        category_enabled = {
            "morphology_cleanup": bool(cfg.morph_cleanup),
            "deskew": bool(cfg.deskew),
        }
        return step not in disabled and category_enabled.get(step, True)

    def record_special(
        step: str,
        unit: str,
        technique: str,
        params: dict[str, Any],
        output: Image,
        elapsed: float,
        skipped: bool,
    ) -> Image:
        """Append metadata for a step with auxiliary non-image results."""
        stages.append(
            Stage(
                step,
                output,
                unit,
                technique,
                params.copy(),
                elapsed,
                skipped,
            )
        )
        return output

    source = image.copy()

    def resize_operation(current: Image) -> Image:
        """Resize only when the configured longest-side bound is exceeded."""
        longest = max(current.shape[:2])
        if longest <= cfg.work_size:
            return current.copy()
        factor = cfg.work_size / longest
        size = (
            max(1, int(round(current.shape[1] * factor))),
            max(1, int(round(current.shape[0] * factor))),
        )
        return cv2.resize(current, size, interpolation=cv2.INTER_AREA)

    working = _run_step(
        stages,
        "resize_working",
        "Unit 1",
        "aspect-preserving resize",
        {"work_size": cfg.work_size},
        source,
        resize_operation,
        is_enabled("resize_working"),
    )

    corners: NDArray[np.float32] | None = None
    rectification_method = "full image"
    if is_enabled("document_rectification"):
        started = time.perf_counter()
        try:
            corners = detect_document_corners(working)
            color_working = (
                cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
                if working.ndim == 2
                else working[:, :, 0]
                if working.shape[2] == 1
                else working
            )
            rectified = _four_point_transform(color_working, corners)
            rectification_method = "document corners + four-point transform"
        except ValueError:
            corners = None
            if template is not None:
                try:
                    rectified, _ = align_to_template(working, template)
                    corners = np.array(
                        [
                            [0.0, 0.0],
                            [float(rectified.shape[1] - 1), 0.0],
                            [
                                float(rectified.shape[1] - 1),
                                float(rectified.shape[0] - 1),
                            ],
                            [0.0, float(rectified.shape[0] - 1)],
                        ],
                        dtype=np.float32,
                    )
                    rectification_method = "SIFT template alignment"
                except ValueError:
                    rectified = working.copy()
                    rectification_method = "full image (alignment unavailable)"
            else:
                rectified = working.copy()
                rectification_method = "full image (corners unavailable)"
        elapsed = time.perf_counter() - started
        record_special(
            "document_rectification",
            "Units 4/5",
            rectification_method,
            {"template": template is not None},
            rectified,
            elapsed,
            False,
        )
    else:
        rectified = working.copy()
        record_special(
            "document_rectification",
            "Units 4/5",
            "disabled",
            {"template": template is not None},
            rectified,
            0.0,
            True,
        )

    gray = _run_step(
        stages,
        "gray_conversion",
        "Unit 1",
        "BGR to grayscale",
        {},
        rectified,
        lambda current: (
            current.copy()
            if current.ndim == 2
            else current[:, :, 0].copy()
            if current.shape[2] == 1
            else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        ),
        is_enabled("gray_conversion"),
    )
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if gray.shape[2] == 3 else gray[:, :, 0]

    def illumination(current: Image) -> Image:
        """Select configured illumination correction."""
        if cfg.illum == "homomorphic":
            return homomorphic_filter(current)
        if cfg.illum == "background_division":
            return background_division(current, _odd_kernel(max(current.shape) // 12))
        if cfg.illum == "none":
            return current.copy()
        raise ValueError("cfg.illum must be homomorphic, background_division, or none.")

    enhanced = _run_step(
        stages,
        "illumination_correction",
        "Units 3/1",
        cfg.illum,
        {"method": cfg.illum},
        gray,
        illumination,
        is_enabled("illumination_correction"),
    )

    def denoise(current: Image) -> Image:
        """Apply configured denoising or preserve the image."""
        if cfg.denoise == "median":
            return median_filter(current, cfg.denoise_ksize)
        if cfg.denoise == "gaussian":
            return gaussian_filter(current, cfg.denoise_ksize)
        if cfg.denoise == "none":
            return current.copy()
        raise ValueError("cfg.denoise must be median, gaussian, or none.")

    enhanced = _run_step(
        stages,
        "denoise",
        "Unit 1",
        cfg.denoise,
        {"method": cfg.denoise, "ksize": cfg.denoise_ksize},
        enhanced,
        denoise,
        is_enabled("denoise"),
    )

    def contrast(current: Image) -> Image:
        """Apply selected local/global contrast enhancement."""
        if cfg.contrast == "clahe":
            return clahe(current, cfg.clahe_clip, cfg.clahe_tile)
        if cfg.contrast == "equalize":
            return equalize_cv(current)
        if cfg.contrast == "stretch":
            return contrast_stretch_minmax(current)
        if cfg.contrast == "none":
            return current.copy()
        raise ValueError("cfg.contrast must be clahe, equalize, stretch, or none.")

    enhanced = _run_step(
        stages,
        "contrast",
        "Unit 1",
        cfg.contrast,
        {"method": cfg.contrast, "clip": cfg.clahe_clip, "tile": cfg.clahe_tile},
        enhanced,
        contrast,
        is_enabled("contrast"),
    )

    def sharpen(current: Image) -> Image:
        """Apply selected unsharp or high-boost filter."""
        if cfg.sharpen == "unsharp":
            return unsharp_mask(current, sigma=1.0, amount=cfg.sharpen_amount)
        if cfg.sharpen == "high_boost":
            return high_boost(current, k=cfg.sharpen_amount, sigma=1.0)
        if cfg.sharpen == "none":
            return current.copy()
        raise ValueError("cfg.sharpen must be unsharp, high_boost, or none.")

    enhanced = _run_step(
        stages,
        "sharpen",
        "Unit 1",
        cfg.sharpen,
        {"method": cfg.sharpen, "amount": cfg.sharpen_amount},
        enhanced,
        sharpen,
        is_enabled("sharpen"),
    )

    def binarize(current: Image) -> Image:
        """Apply the configured threshold method."""
        if cfg.binarize == "sauvola":
            window = _odd_kernel(cfg.sauvola_window, minimum=3)
            return sauvola(current, window=window, k=cfg.sauvola_k)
        if cfg.binarize == "otsu":
            threshold = otsu_cv(current)
            return global_threshold(current, threshold)
        if cfg.binarize == "adaptive":
            block = _odd_kernel(max(3, cfg.sauvola_window))
            return adaptive_threshold(current, "gaussian", block, C=2.0)
        if cfg.binarize == "none":
            return current.copy()
        raise ValueError("cfg.binarize must be sauvola, otsu, adaptive, or none.")

    binary_stage_image = _run_step(
        stages,
        "binarize",
        "Unit 5",
        cfg.binarize,
        {
            "method": cfg.binarize,
            "sauvola_window": cfg.sauvola_window,
            "sauvola_k": cfg.sauvola_k,
        },
        enhanced,
        binarize,
        is_enabled("binarize"),
    )
    if cfg.binarize == "none" or not is_enabled("binarize"):
        # Keep the public final_binary contract genuinely binary when thresholding is bypassed.
        binary = global_threshold(enhanced, otsu_cv(enhanced))
    else:
        binary = np.where(binary_stage_image > 0, 255, 0).astype(np.uint8)

    def cleanup(current: Image) -> Image:
        """Clean black foreground text via inverted binary morphology."""
        if not cfg.morph_cleanup:
            return current.copy()
        text_foreground = cv2.bitwise_not(current)
        cleaned = remove_specks(text_foreground, k=3)
        cleaned = closing(cleaned, structuring_element("rect", 3))
        return cv2.bitwise_not(cleaned)

    binary = _run_step(
        stages,
        "morphology_cleanup",
        "Unit 3",
        "remove specks + closing",
        {"enabled": cfg.morph_cleanup, "kernel": 3},
        binary,
        cleanup,
        is_enabled("morphology_cleanup"),
    )
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)
    deskew_result: list[tuple[Image, Image, float]] = []

    def deskew_operation(current: Image) -> Image:
        """Deskew paired grayscale and binary intermediates."""
        pair = _deskew_pair(enhanced, current)
        deskew_result.append(pair)
        return pair[1]

    binary = _run_step(
        stages,
        "deskew",
        "Unit 4",
        "fine min-area-rectangle deskew",
        {"enabled": cfg.deskew},
        binary,
        deskew_operation,
        is_enabled("deskew"),
    )
    if deskew_result:
        enhanced, binary, deskew_angle = deskew_result[-1]
    else:
        deskew_angle = 0.0
    metrics["deskew_angle"] = float(deskew_angle)

    boxes: list[tuple[int, int, int, int]] = []
    rectified_color = (
        cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
        if rectified.ndim == 2
        else rectified[:, :, 0]
        if rectified.shape[2] == 1
        else rectified.copy()
    )

    def layout(current: Image) -> Image:
        """Find line-level text boxes and draw them on a page copy."""
        nonlocal boxes
        text_mask = cv2.bitwise_not(current)
        boxes = segment_text_blocks(text_mask)
        return draw_boxes(rectified_color, boxes)

    layout_image = _run_step(
        stages,
        "layout",
        "Units 2/3",
        "text-line connected components",
        {},
        binary,
        layout,
        is_enabled("layout"),
    )
    if not is_enabled("layout"):
        boxes = []

    if cfg.ocr_source == "enhanced":
        ocr_image = enhanced
    elif cfg.ocr_source == "binary":
        ocr_image = binary
    elif cfg.ocr_source == "gray":
        ocr_image = gray
    elif cfg.ocr_source == "original":
        ocr_image = (
            cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
            if working.ndim == 3 and working.shape[2] == 3
            else working[:, :, 0]
            if working.ndim == 3
            else working
        )
    else:
        raise ValueError("cfg.ocr_source must be enhanced, binary, gray, or original.")
    ocr_text = ""
    ocr_words = 0
    ocr_status = "skipped"

    def ocr_operation(current: Image) -> Image:
        """Run OCR while retaining its explicit status and word-box count."""
        nonlocal ocr_text, ocr_words, ocr_status
        ocr_text, word_data = run_ocr(current)
        ocr_words = len(word_data)
        ocr_status = (
            "unavailable"
            if ocr_text.startswith("Tesseract unavailable:")
            else "failed"
            if ocr_text.startswith("OCR failed:")
            else "available"
        )
        return current.copy()

    _run_step(
        stages,
        "ocr",
        "Unit 9",
        "Tesseract OCR",
        {"source": cfg.ocr_source, "psm": 6, "lang": "eng"},
        ocr_image,
        ocr_operation,
        is_enabled("ocr"),
    )
    metrics.update(
        {
            "rectification": rectification_method,
            "corners_detected": corners is not None,
            "ocr_status": ocr_status,
            "ocr_word_count": ocr_words,
            "total_seconds": float(sum(stage.seconds for stage in stages)),
            "final_sharpness": float(cv2.Laplacian(enhanced, cv2.CV_64F).var()),
        }
    )
    del layout_image
    return PipelineResult(
        stages=stages,
        final_gray=enhanced.copy(),
        final_binary=binary.copy(),
        text=ocr_text,
        boxes=boxes,
        corners=None if corners is None else corners.copy(),
        metrics=metrics,
    )


def scan_to_pdf(result: PipelineResult, path: str | Path) -> None:
    """Export the final grayscale document image as a single-page PDF.

    Args:
        result: Pipeline result containing the final enhanced grayscale image.
        path: Destination PDF path.

    Returns:
        None.

    Raises:
        ValueError: If final_gray is not a nonempty uint8 grayscale image.

    Syllabus: Unit 1 - document input and output
    """
    image = np.asarray(result.final_gray)
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("result.final_gray must be a nonempty uint8 grayscale image.")
    buffer = io.BytesIO()
    PILImage.fromarray(image, mode="L").save(buffer, format="PDF", resolution=300.0)
    Path(path).write_bytes(buffer.getvalue())

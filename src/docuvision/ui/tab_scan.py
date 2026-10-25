"""Scan tab for end-to-end document processing and output.

Syllabus: Unit 1 - digital image processing pipeline
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray

from docuvision.config import PipelineConfig
from docuvision.evaluation.metrics import cer
from docuvision.ocr import run_ocr, tesseract_available
from docuvision.pipeline import PipelineResult, run_pipeline, scan_to_pdf
from docuvision.ui.common import gray, require_image, rgb, show_pair, syllabus_info


@st.cache_data(show_spinner=False)
def _cached_pipeline(
    image_bytes: bytes,
    shape: tuple[int, ...],
    config_json: str,
    ocr_available: bool,
) -> PipelineResult:
    """Cache expensive pipeline work by source image and serialized config.

    Args:
        image_bytes: Contiguous source pixel bytes.
        shape: Original array dimensions used to restore the image.
        config_json: Stable JSON serialization of the active pipeline preset.
        ocr_available: Included in the cache key so an old no-OCR result is not
            reused after Tesseract becomes available.

    Returns:
        Complete pipeline result containing all intermediate stages.

    Syllabus: Unit 1 - image processing pipeline
    """
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape).copy()
    config = PipelineConfig(**json.loads(config_json))
    del ocr_available  # Streamlit uses this argument in the cache key.
    return run_pipeline(image, config)


@st.cache_data(show_spinner=False)
def _cached_ocr(
    image_bytes: bytes, shape: tuple[int, ...], psm: int, ocr_available: bool
) -> tuple[str, pd.DataFrame]:
    """Cache OCR for a selected image and Tesseract page-segmentation mode.

    Args:
        image_bytes: Contiguous uint8 image bytes.
        shape: Image dimensions used to restore the array.
        psm: Tesseract page-segmentation mode.
        ocr_available: Included in the cache key to invalidate an unavailable
            result after Tesseract is installed.

    Returns:
        Recognized text and the word-box DataFrame from Tesseract.

    Syllabus: Unit 9 - character recognition
    """
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape).copy()
    del ocr_available  # Streamlit uses this argument in the cache key.
    return run_ocr(image, psm=psm)


def _select_ocr_image(
    option: str, source: NDArray[np.generic], result: PipelineResult
) -> NDArray[np.uint8]:
    """Select the source representation used for an optional OCR retry.

    Args:
        option: User-facing OCR image choice.
        source: Original uploaded/captured image.
        result: Completed scan pipeline output.

    Returns:
        Grayscale, binary, or original grayscale image for OCR.

    Raises:
        ValueError: If option is not a supported choice.

    Syllabus: Unit 9 - character recognition
    """
    if option == "Enhanced grayscale":
        return result.final_gray
    if option == "Black and white":
        return result.final_binary
    if option == "Original grayscale":
        return gray(source)
    raise ValueError(f"Unsupported OCR image choice: {option}")


def _crop_ocr_image(
    image: NDArray[np.uint8],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
) -> NDArray[np.uint8]:
    """Return a copied, validated rectangular crop for a focused OCR attempt.

    Args:
        image: Nonempty grayscale or color image.
        x_range: Horizontal start and exclusive end coordinates.
        y_range: Vertical start and exclusive end coordinates.

    Returns:
        Cropped image copy.

    Raises:
        ValueError: If coordinates are invalid or produce an empty crop.

    Syllabus: Unit 9 - character recognition
    """
    height, width = image.shape[:2]
    x_start, x_end = x_range
    y_start, y_end = y_range
    if not (
        0 <= x_start < x_end <= width and 0 <= y_start < y_end <= height
    ):
        raise ValueError("OCR crop coordinates must form a nonempty region inside the image.")
    return image[y_start:y_end, x_start:x_end].copy()


def _source_matches(
    current: NDArray[np.generic], previous: object
) -> bool:
    """Return whether a stored scan result belongs to the current source image.

    Args:
        current: Current source image supplied to the Scan tab.
        previous: Image saved when the pipeline last ran, or another value.

    Returns:
        True only when both values are arrays with equal shape and pixels.

    Syllabus: Unit 1 - document image input
    """
    return (
        isinstance(previous, np.ndarray)
        and previous.shape == current.shape
        and np.array_equal(previous, current)
    )


def render(
    image: NDArray[np.generic] | None, config: PipelineConfig
) -> None:
    """Render the pipeline stage viewer, OCR text, and downloads.

    Args:
        image: Shared uploaded/captured image or None.
        config: Current sidebar pipeline preset.

    Returns:
        None.

    Syllabus: Unit 1 - digital image processing pipeline
    """
    syllabus_info("sampling, enhancement, thresholding, morphology, OCR", "Lab 1")
    source = require_image(image)
    if source is None:
        return
    source_id = hashlib.blake2b(source.tobytes(), digest_size=6).hexdigest()
    ocr_available = tesseract_available()
    previous_source = st.session_state.get("last_scan_source")
    if (
        "pipeline_result" in st.session_state
        and (
            not _source_matches(source, previous_source)
            or st.session_state.get("last_scan_ocr_available") is not ocr_available
        )
    ):
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("last_scan_source", None)
        st.session_state.pop("last_scan_ocr_available", None)
        st.session_state.pop("scan_ocr_override", None)
        st.session_state.pop("scan_ocr_text", None)
        st.session_state.pop("scan_ocr_display_signature", None)
    if st.button("Run document pipeline", type="primary", key="scan_run"):
        config_json = json.dumps(config.to_dict(), sort_keys=True)
        with st.spinner("Processing document stages..."):
            result = _cached_pipeline(
                source.tobytes(), source.shape, config_json, ocr_available
            )
        st.session_state["pipeline_result"] = result
        st.session_state["last_scan_source"] = source.copy()
        st.session_state["last_scan_ocr_available"] = ocr_available
        st.session_state.pop("scan_ocr_text", None)
        st.session_state.pop("scan_ground_truth", None)
        st.session_state.pop("scan_ocr_override", None)
    result = st.session_state.get("pipeline_result")
    if not isinstance(result, PipelineResult):
        show_pair(source, gray(source), "Source", "Grayscale input")
        st.caption("Run the pipeline to inspect stages and export a scan.")
        return
    before_column, after_column = st.columns(2)
    before_column.image(rgb(source), caption="Before", use_container_width=True)
    after_column.image(
        rgb(result.final_gray), caption="Enhanced grayscale", use_container_width=True
    )
    st.subheader("Pipeline stages")
    stage_options = range(len(result.stages))
    stage_index = st.select_slider(
        "Pipeline stage",
        options=stage_options,
        value=max(0, len(result.stages) - 1),
        format_func=lambda index: (
            f"{index + 1}. {result.stages[index].name.replace('_', ' ').title()}"
        ),
        key=f"scan_stage_slider_{source_id}",
    )
    stage = result.stages[stage_index]
    badge = f"**{stage.unit}** · {stage.technique}"
    if stage.skipped:
        badge += " · skipped"
    st.markdown(badge)
    st.caption(f"{stage.name} · {stage.seconds * 1000:.2f} ms · {stage.params}")
    st.image(rgb(stage.image), caption=stage.name, use_container_width=True)
    st.subheader("OCR text")
    st.caption(
        "OCR works best with clear, straight, ordinary printed text. "
        "Decorative lettering, artwork, texture, and shadows can be misread."
    )
    ocr_options = (
        "Enhanced grayscale",
        "Black and white",
        "Original grayscale",
    )
    ocr_columns = st.columns(2)
    ocr_image_option = ocr_columns[0].selectbox(
        "Image sent to OCR",
        ocr_options,
        key="scan_ocr_image_option",
        help="Try black and white for uneven backgrounds, or original grayscale if enhancement distorted the letters.",
    )
    psm_options = (6, 3, 11, 7, 8, 13)
    psm_labels = {
        6: "Block of text (default)",
        3: "Automatic page layout",
        11: "Sparse text / poster",
        7: "Single text line",
        8: "Single word",
        13: "Raw text line",
    }
    psm = ocr_columns[1].selectbox(
        "Text layout",
        psm_options,
        index=0,
        format_func=lambda mode: psm_labels[mode],
        key="scan_ocr_psm",
        help="For cover titles, posters, or widely spaced text, try Sparse text / poster.",
    )
    ocr_image = _select_ocr_image(ocr_image_option, source, result)
    crop_enabled = st.checkbox(
        "Crop to a smaller region before OCR",
        key="scan_ocr_crop_enabled",
        help="Useful when a cover or poster has a title mixed with illustrations. Crop coordinates refer to the selected OCR image.",
    )
    crop_ranges: tuple[tuple[int, int], tuple[int, int]] | None = None
    if crop_enabled:
        image_height, image_width = ocr_image.shape[:2]
        crop_columns = st.columns(2)
        x_range = crop_columns[0].slider(
            "Horizontal crop: left to right",
            min_value=0,
            max_value=image_width,
            value=(0, image_width),
            key=f"scan_ocr_x_range_{source_id}_{ocr_image_option}",
        )
        y_range = crop_columns[1].slider(
            "Vertical crop: top to bottom",
            min_value=0,
            max_value=image_height,
            value=(0, image_height),
            key=f"scan_ocr_y_range_{source_id}_{ocr_image_option}",
        )
        crop_ranges = (x_range, y_range)
        if x_range[0] < x_range[1] and y_range[0] < y_range[1]:
            ocr_image = _crop_ocr_image(ocr_image, x_range, y_range)
            st.image(
                rgb(ocr_image),
                caption=f"OCR crop preview · {ocr_image.shape[1]} × {ocr_image.shape[0]} pixels",
                use_container_width=True,
            )
        else:
            st.warning("Choose a crop with nonzero width and height before retrying OCR.")
    valid_crop = (
        not crop_enabled
        or (
            crop_ranges is not None
            and crop_ranges[0][0] < crop_ranges[0][1]
            and crop_ranges[1][0] < crop_ranges[1][1]
        )
    )
    if st.button(
        "Re-run OCR with these settings",
        key="scan_ocr_retry",
        disabled=not valid_crop,
    ):
        with st.spinner("Recognizing text with the selected image and layout..."):
            text, word_boxes = _cached_ocr(
                ocr_image.tobytes(), ocr_image.shape, psm, ocr_available
            )
        st.session_state["scan_ocr_override"] = {
            "text": text,
            "boxes": word_boxes,
            "source": ocr_image_option,
            "psm": psm,
            "crop": crop_ranges if crop_enabled else None,
            "ocr_available": ocr_available,
        }
    override = st.session_state.get("scan_ocr_override")
    if (
        isinstance(override, dict)
        and override.get("source") == ocr_image_option
        and override.get("psm") == psm
        and override.get("crop") == (crop_ranges if crop_enabled else None)
        and override.get("ocr_available") is ocr_available
    ):
        recognized_text = str(override.get("text", ""))
        word_boxes = override.get("boxes")
        st.caption(f"OCR retry used {ocr_image_option.lower()} · {psm_labels[psm]}.")
        if word_boxes is not None and not word_boxes.empty:
            with st.expander("OCR word boxes and confidence"):
                st.dataframe(word_boxes, hide_index=True, use_container_width=True)
    else:
        recognized_text = result.text
    display_signature = (
        source_id,
        ocr_image_option,
        psm,
        crop_ranges if crop_enabled else None,
        recognized_text,
    )
    if st.session_state.get("scan_ocr_display_signature") != display_signature:
        st.session_state["scan_ocr_text"] = recognized_text
        st.session_state["scan_ocr_display_signature"] = display_signature
    st.text_area(
        "Recognized text",
        value=recognized_text,
        height=180,
        key="scan_ocr_text",
    )
    reference = st.text_area(
        "Optional ground-truth text for CER",
        value="",
        height=90,
        key="scan_ground_truth",
    )
    if reference:
        st.metric("Character error rate", f"{cer(reference, recognized_text):.4f}")
    png_ok, png_data = cv2.imencode(".png", result.final_gray)
    if not png_ok:
        st.error("Could not encode the processed image for download.")
        return
    try:
        with tempfile.TemporaryDirectory(prefix="docuvision-pdf-") as directory:
            pdf_path = Path(directory) / "scan.pdf"
            scan_to_pdf(result, pdf_path)
            pdf_data = pdf_path.read_bytes()
    except (OSError, ValueError) as error:
        st.error(f"Could not prepare PDF download: {error}")
        pdf_data = b""
    downloads = st.columns(3)
    downloads[0].download_button(
        "Download PNG",
        png_data.tobytes(),
        file_name="docuvision_scan.png",
        mime="image/png",
    )
    downloads[1].download_button(
        "Download TXT",
        recognized_text,
        file_name="docuvision_ocr.txt",
        mime="text/plain",
    )
    downloads[2].download_button(
        "Download PDF",
        pdf_data,
        file_name="docuvision_scan.pdf",
        mime="application/pdf",
        disabled=not bool(pdf_data),
    )
    if result.metrics.get("ocr_status") == "unavailable":
        st.info(result.text)

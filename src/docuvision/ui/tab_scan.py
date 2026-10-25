"""Scan tab for end-to-end document processing and output.

Syllabus: Unit 1 - digital image processing pipeline
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from numpy.typing import NDArray

from docuvision.config import PipelineConfig
from docuvision.evaluation.metrics import cer
from docuvision.pipeline import PipelineResult, run_pipeline, scan_to_pdf
from docuvision.ui.common import gray, require_image, rgb, show_pair, syllabus_info


@st.cache_data(show_spinner=False)
def _cached_pipeline(
    image_bytes: bytes, shape: tuple[int, ...], config_json: str
) -> PipelineResult:
    """Cache expensive pipeline work by source image and serialized config.

    Args:
        image_bytes: Contiguous source pixel bytes.
        shape: Original array dimensions used to restore the image.
        config_json: Stable JSON serialization of the active pipeline preset.

    Returns:
        Complete pipeline result containing all intermediate stages.

    Syllabus: Unit 1 - image processing pipeline
    """
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape).copy()
    config = PipelineConfig(**json.loads(config_json))
    return run_pipeline(image, config)


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
    if st.button("Run document pipeline", type="primary", key="scan_run"):
        config_json = json.dumps(config.to_dict(), sort_keys=True)
        with st.spinner("Processing document stages..."):
            result = _cached_pipeline(source.tobytes(), source.shape, config_json)
        st.session_state["pipeline_result"] = result
        st.session_state["last_scan_source"] = source.copy()
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
    stage_index = st.slider(
        "Stage",
        min_value=0,
        max_value=max(0, len(result.stages) - 1),
        value=0,
        format="%d",
        key="scan_stage_slider",
    )
    stage = result.stages[stage_index]
    badge = f"**{stage.unit}** · {stage.technique}"
    if stage.skipped:
        badge += " · skipped"
    st.markdown(badge)
    st.caption(f"{stage.name} · {stage.seconds * 1000:.2f} ms · {stage.params}")
    st.image(rgb(stage.image), caption=stage.name, use_container_width=True)
    st.subheader("OCR text")
    st.text_area("Recognized text", value=result.text, height=180, key="scan_ocr_text")
    reference = st.text_area(
        "Optional ground-truth text for CER",
        value="",
        height=90,
        key="scan_ground_truth",
    )
    if reference:
        st.metric("Character error rate", f"{cer(reference, result.text):.4f}")
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
        result.text,
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

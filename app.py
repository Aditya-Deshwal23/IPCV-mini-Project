"""Streamlit application entry point for the DocuVision learning laboratory.

Syllabus: Unit 1 - image processing application workflow
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from docuvision.config import PipelineConfig
from docuvision.ocr import tesseract_available
from docuvision.ui import (
    tab_compression,
    tab_enhancement,
    tab_evaluation,
    tab_features,
    tab_frequency,
    tab_motion,
    tab_scan,
    tab_segmentation,
)

st.set_page_config(
    page_title="DocuVision | Document Processing Lab",
    page_icon="📄",
    layout="wide",
)


def _decode_upload(data: bytes) -> np.ndarray | None:
    """Decode an uploaded image as BGR using OpenCV.

    Args:
        data: Encoded image bytes.

    Returns:
        BGR uint8 image, or None if OpenCV cannot decode it.

    Syllabus: Unit 1 - image acquisition
    """
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image


def _sidebar_image() -> None:
    """Render source selection and update shared session image state.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image acquisition
    """
    st.sidebar.header("Document source")
    source_mode = st.sidebar.radio(
        "Choose input",
        ("Upload image", "Camera", "Sample"),
        horizontal=False,
        key="source_mode",
    )
    if source_mode == "Upload image":
        uploaded = st.sidebar.file_uploader(
            "Image file",
            type=("png", "jpg", "jpeg", "bmp", "tif", "tiff"),
            key="image_upload",
        )
        if uploaded is not None:
            decoded = _decode_upload(uploaded.getvalue())
            if decoded is None:
                st.sidebar.error("That file could not be decoded as an image.")
            else:
                st.session_state["source_image"] = decoded
                st.session_state["source_name"] = uploaded.name
    elif source_mode == "Camera":
        camera = st.sidebar.camera_input("Capture a document")
        if camera is not None:
            decoded = _decode_upload(camera.getvalue())
            if decoded is None:
                st.sidebar.error("The camera image could not be decoded.")
            else:
                st.session_state["source_image"] = decoded
                st.session_state["source_name"] = "camera_capture.png"
    else:
        sample_directory = Path("data/samples")
        samples = sorted(
            path
            for path in sample_directory.glob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ) if sample_directory.exists() else []
        if not samples:
            st.sidebar.info("No sample images yet. Add an image under data/samples/.")
        else:
            selected = st.sidebar.selectbox("Sample image", samples, format_func=lambda p: p.name)
            file_bytes = selected.read_bytes()
            decoded = _decode_upload(file_bytes)
            if decoded is not None:
                st.session_state["source_image"] = decoded
                st.session_state["source_name"] = selected.name
                if selected.stem == "01_clean_print_page":
                    st.sidebar.caption(
                        "Demo 1: clean, ordinary print. Best for your first scan "
                        "and OCR demonstration."
                    )
                elif selected.stem == "02_shadowed_tilted_page":
                    st.sidebar.caption(
                        "Demo 2: simulated shadow and perspective. Best for showing "
                        "page detection and rectification; OCR may be less accurate."
                    )
                reference_path = selected.with_suffix(".txt")
                if reference_path.is_file():
                    st.sidebar.caption(
                        f"Optional ground-truth text for CER: {reference_path.name}"
                    )
    if "source_image" in st.session_state:
        st.sidebar.caption(
            f"Current: {st.session_state.get('source_name', 'image')} "
            f"· {st.session_state['source_image'].shape[1]}×"
            f"{st.session_state['source_image'].shape[0]}"
        )
        if st.sidebar.button("Clear current image"):
            st.session_state.pop("source_image", None)
            st.session_state.pop("source_name", None)


def _sidebar_settings() -> None:
    """Select a pipeline preset and show the syllabus mapping."""
    preset = st.sidebar.selectbox(
        "Pipeline preset",
        ("balanced", "fast", "max_quality", "baseline_otsu", "baseline_adaptive"),
        key="pipeline_preset",
    )
    st.session_state["pipeline_config"] = PipelineConfig.preset(preset)
    with st.sidebar.expander("Syllabus map"):
        st.markdown(
            """
            - **Unit 1:** image fundamentals, enhancement, sampling
            - **Unit 2:** segmentation and edge detection
            - **Unit 3:** morphology and frequency processing
            - **Unit 4:** geometry, feature detection and registration
            - **Unit 5:** motion analysis and action detection
            - **Unit 6:** compression and coding
            - **Unit 7:** evaluation and reproducibility
            - **Unit 9:** OCR
            """
        )


def main() -> None:
    """Render the shared sidebar, availability banner, and eight lab tabs."""
    st.title("DocuVision")
    st.caption("Scan, clean, and recognize text in document images.")
    _sidebar_settings()
    _sidebar_image()
    if not tesseract_available():
        st.warning(
            "Tesseract OCR is unavailable. Scan and OCR features will show a clear "
            "status; install Tesseract or set TESSERACT_CMD to enable recognition."
        )
    image = st.session_state.get("source_image")
    config = st.session_state.get("pipeline_config", PipelineConfig.preset("balanced"))
    tabs = st.tabs(
        [
            "Scan",
            "Unit 1 · Enhancement",
            "Unit 2 · Segmentation",
            "Unit 3 · Frequency & Morphology",
            "Unit 4 · Geometry & Features",
            "Unit 5 · Motion",
            "Unit 6 · Compression",
            "Evaluation",
        ]
    )
    with tabs[0]:
        tab_scan.render(image, config)
    with tabs[1]:
        tab_enhancement.render(image)
    with tabs[2]:
        tab_segmentation.render(image)
    with tabs[3]:
        tab_frequency.render(image)
    with tabs[4]:
        tab_features.render(image)
    with tabs[5]:
        tab_motion.render(image)
    with tabs[6]:
        tab_compression.render(image)
    with tabs[7]:
        tab_evaluation.render()


if __name__ == "__main__":
    main()

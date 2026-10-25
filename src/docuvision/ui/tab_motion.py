"""Interactive Unit 5 motion and document-capture laboratory.

Syllabus: Unit 5 - motion analysis and action detection
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray

from docuvision.features import shi_tomasi
from docuvision.motion import (
    draw_flow_arrows,
    flow_to_color,
    horn_schunck_scratch,
    lucas_kanade_cv,
    make_demo_video,
    process_video,
)
from docuvision.segmentation import (
    adaptive_threshold,
    otsu_cv,
    sauvola,
    watershed_split,
)
from docuvision.ui.common import gray, require_image, syllabus_info


def _video_result(
    video_bytes: bytes, stride: int, progress: Any
) -> dict[str, Any]:
    """Analyze encoded video bytes and report frame progress to a Streamlit bar.

    Args:
        video_bytes: Encoded local video contents.
        stride: Frame sampling interval passed to the motion library.
        progress: Streamlit progress element updated for each yielded frame.

    Returns:
        Analysis result containing events, captures, and a timeline DataFrame.

    Syllabus: Unit 5 - motion analysis
    """
    with tempfile.TemporaryDirectory(prefix="docuvision-motion-") as directory:
        video_path = Path(directory) / "input_video.avi"
        video_path.write_bytes(video_bytes)
        generator = process_video(video_path, stride=stride, max_frames=500)
        while True:
            try:
                update = next(generator)
                progress.progress(
                    min(1.0, max(0.0, float(update["progress"]))),
                    text=f"Analyzing frame {int(update['frame'])}...",
                )
            except StopIteration as completed:
                return completed.value


@st.cache_data(show_spinner=False)
def _cached_demo_video() -> bytes:
    """Generate a deterministic demo clip once and cache its encoded bytes.

    Args:
        None.

    Returns:
        Encoded AVI video contents for the built-in page-motion demonstration.

    Syllabus: Unit 5 - synthetic motion demonstration
    """
    with tempfile.TemporaryDirectory(prefix="docuvision-demo-") as directory:
        video_path = Path(directory) / "demo.avi"
        make_demo_video(video_path)
        return video_path.read_bytes()


def _resize_pair(
    first: NDArray[np.uint8], second: NDArray[np.uint8]
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Resize two frames to equal bounded dimensions for flow visualization.

    Args:
        first: First grayscale uint8 frame.
        second: Second grayscale uint8 frame.

    Returns:
        Pair of area-resized frames with equal dimensions and longest side <=192.

    Syllabus: Unit 5 - optical flow visualization
    """
    height = min(first.shape[0], second.shape[0])
    width = min(first.shape[1], second.shape[1])
    scale = min(1.0, 192.0 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return (
        cv2.resize(first, size, interpolation=cv2.INTER_AREA),
        cv2.resize(second, size, interpolation=cv2.INTER_AREA),
    )


def render(image: NDArray[np.generic] | None) -> None:
    """Render video capture analysis, flow, thresholding, and watershed demos.

    Args:
        image: Shared grayscale or BGR source image or None.

    Returns:
        None.

    Syllabus: Unit 5 - motion analysis and action detection
    """
    syllabus_info("Lucas-Kanade, Horn-Schunck, stability states, thresholding, watershed", "Lab 13")
    st.subheader("Document capture action detection")
    video_upload = st.file_uploader("Upload a video", type=("avi", "mp4", "mov", "mkv"), key="motion_video")
    use_demo = st.button("Generate demo video", key="motion_demo")
    if use_demo:
        try:
            st.session_state["motion_video_bytes"] = _cached_demo_video()
            st.success("Synthetic drifting and page-change video is ready.")
        except (OSError, RuntimeError, cv2.error) as error:
            st.error(f"Could not generate the demo video: {error}")
    if video_upload is not None:
        video_bytes = video_upload.getvalue()
        st.session_state["motion_video_bytes"] = video_bytes
    else:
        video_bytes = st.session_state.get("motion_video_bytes")
    stride = st.slider("Analyze every Nth frame", 1, 5, 2, key="motion_stride")
    run = st.button("Analyze video", type="primary", disabled=not video_bytes, key="motion_run")
    if run and video_bytes:
        progress = st.progress(0.0, text="Analyzing frames...")
        try:
            result = _video_result(video_bytes, stride, progress)
            progress.progress(1.0, text="Motion analysis complete")
            st.session_state["motion_result"] = result
        except (OSError, ValueError, cv2.error) as error:
            progress.empty()
            st.error(f"Video analysis failed: {error}")
    result = st.session_state.get("motion_result")
    if isinstance(result, dict):
        timeline = result.get("timeline", pd.DataFrame())
        captures = result.get("captured_frames", [])
        events = result.get("events", [])
        if isinstance(timeline, pd.DataFrame) and not timeline.empty:
            chart_timeline = timeline.copy()
            capture_frames = {
                int(event["frame"])
                for event in events
                if isinstance(event, dict) and event.get("type") == "CAPTURE"
            }
            marker_height = max(1.0, float(chart_timeline["motion"].max()) * 1.1)
            chart_timeline["CAPTURE"] = np.where(
                chart_timeline["frame"].isin(capture_frames), marker_height, np.nan
            )
            st.line_chart(
                chart_timeline.set_index("frame")[["motion", "sharpness", "CAPTURE"]]
            )
            st.dataframe(timeline, hide_index=True, use_container_width=True)
            st.caption(f"Events: {events}")
        if captures:
            st.subheader("Captured frames")
            st.image(captures, caption=[f"Capture {index + 1}" for index in range(len(captures))], use_container_width=True)
            selection = st.selectbox("Capture to send to Scan", range(len(captures)), format_func=lambda index: f"Capture {index + 1}")
            if st.button("Send captured frame to Scan tab", key="motion_send_scan"):
                captured = np.asarray(captures[selection], dtype=np.uint8)
                st.session_state["source_image"] = captured.copy()
                st.session_state["source_name"] = f"motion_capture_{selection + 1}.png"
                st.success("Capture set as the shared source image. Open the Scan tab.")
        else:
            st.info("No stable document capture was detected in this video.")
        if len(captures) >= 2:
            frame_a, frame_b = _resize_pair(gray(captures[0]), gray(captures[1]))
            points = shi_tomasi(frame_a, max_corners=150, min_distance=5)
            if len(points):
                tracked, status = lucas_kanade_cv(frame_a, frame_b, points)
                arrow_image = cv2.cvtColor(frame_a, cv2.COLOR_GRAY2BGR)
                for point, target, valid in zip(points, tracked, status):
                    if valid:
                        cv2.arrowedLine(arrow_image, tuple(np.rint(point).astype(int)), tuple(np.rint(target).astype(int)), (0, 255, 0), 1)
                st.image(arrow_image, caption="Lucas-Kanade feature flow", use_container_width=True)
            u, v = horn_schunck_scratch(frame_a, frame_b, iters=60)
            flow_columns = st.columns(2)
            flow_columns[0].image(flow_to_color(u, v), caption="Horn-Schunck flow color", use_container_width=True)
            flow_columns[1].image(draw_flow_arrows(frame_a, u, v, step=12), caption="Horn-Schunck flow arrows", use_container_width=True)

    st.subheader("Threshold and touching-character demos")
    source = require_image(image)
    if source is None:
        return
    gray_image = gray(source)
    threshold = otsu_cv(gray_image)
    binary_otsu = np.where(gray_image > threshold, 255, 0).astype(np.uint8)
    block = max(3, min(31, min(gray_image.shape) | 1))
    block += 1 if block % 2 == 0 else 0
    block = min(block, min(gray_image.shape) if min(gray_image.shape) % 2 else max(1, min(gray_image.shape) - 1))
    sauvola_window = max(3, block)
    if sauvola_window > min(gray_image.shape):
        st.info("Adaptive threshold comparisons require an image at least 3×3.")
        return
    adaptive = adaptive_threshold(gray_image, "gaussian", sauvola_window, C=2.0)
    sauvola_result = sauvola(gray_image, sauvola_window, 0.2)
    columns = st.columns(3)
    for column, title, output in zip(columns, ("Otsu", "Adaptive Gaussian", "Sauvola"), (binary_otsu, adaptive, sauvola_result)):
        column.image(output, caption=title, use_container_width=True)
    if st.checkbox("Show watershed split of threshold foreground", key="motion_watershed"):
        if min(gray_image.shape) < 3:
            st.info("Watershed needs at least a 3×3 image.")
        else:
            foreground = np.where(binary_otsu == 0, 255, 0).astype(np.uint8)
            labels = watershed_split(foreground)
            label_display = cv2.applyColorMap(
                cv2.normalize(
                    labels,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                ),
                cv2.COLORMAP_TURBO,
            )
            st.image(label_display, caption="Watershed labels", use_container_width=True)

"""Tests for the hardware-independent portions of the live demonstration.

Syllabus: Unit 5 - motion analysis and action detection
"""

from pathlib import Path

import cv2
import numpy as np

from docuvision.config import PipelineConfig
from docuvision.motion import StabilityState
from live_demo import _draw_overlay, _parse_args, _save_scan


def test_live_demo_parses_video_file_argument() -> None:
    """Accept a local replay path without initializing camera hardware.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 5 - motion analysis and action detection
    """
    arguments = _parse_args(["--video", "demo.avi"])
    assert str(arguments.video) == "demo.avi"


def test_overlay_and_capture_pipeline_work_on_synthetic_page(tmp_path: Path) -> None:
    """Draw a no-track status and save both rectified and processed scans.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.

    Syllabus: Unit 5 - motion analysis and action detection
    """
    frame = np.full((120, 160, 3), 60, dtype=np.uint8)
    cv2.rectangle(frame, (12, 10), (148, 110), (245, 245, 245), -1)
    cv2.putText(frame, "PAGE", (36, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    overlay = _draw_overlay(frame, None, StabilityState.IDLE, 0)
    assert overlay.shape == frame.shape
    warped_path = _save_scan(frame, tmp_path, PipelineConfig.preset("fast"))
    assert warped_path.is_file()
    assert (tmp_path / warped_path.name.replace("warped_", "scan_")).is_file()

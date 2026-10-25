"""Tests for optical flow, document tracking, and temporal action detection.

Syllabus: Unit 7 - motion analysis and tracking
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from docuvision.motion import (
    DocumentTracker,
    StabilityDetector,
    StabilityState,
    draw_flow_arrows,
    flow_to_color,
    frame_difference,
    horn_schunck_scratch,
    lucas_kanade_cv,
    lucas_kanade_scratch,
    make_demo_video,
    process_video,
    running_average_background,
)


def _translated(
    image: NDArray[np.uint8], dx: float, dy: float
) -> NDArray[np.uint8]:
    """Translate an image with reflected boundaries for optical-flow tests.

    Args:
        image: Grayscale uint8 source.
        dx: Horizontal pixel displacement.
        dy: Vertical pixel displacement.

    Returns:
        Translated image with the same shape.

    Syllabus: Unit 7 - optical flow testing
    """
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def test_frame_difference_and_running_average() -> None:
    """Measure a known changed region and average synthetic frames.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 7 - frame differencing and background modeling
    """
    first = np.zeros((10, 10), dtype=np.uint8)
    second = first.copy()
    second[2:7, 2:7] = 100

    mask, score = frame_difference(first, second, thr=20)
    background = running_average_background([first, second], alpha=0.5)

    assert score == pytest.approx(0.25)
    assert np.all(mask[2:7, 2:7] == 255)
    assert np.all(mask[:2] == 0)
    assert np.all(background[2:7, 2:7] == 50)


def test_lucas_kanade_scratch_recovers_two_pixel_translation(
    textured_image: NDArray[np.uint8],
) -> None:
    """Recover +2-pixel horizontal motion with the per-window least-squares LK.

    Args:
        textured_image: Seeded grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 7 - Lucas-Kanade optical flow
    """
    current = _translated(textured_image, 2, 0)
    rows, columns = np.mgrid[18:78:12, 18:78:12]
    points = np.column_stack((columns.ravel(), rows.ravel())).astype(np.float32)

    flow, valid = lucas_kanade_scratch(textured_image, current, points, win=15)

    assert np.count_nonzero(valid) >= 20
    assert float(np.median(np.abs(flow[valid, 0] - 2.0))) < 0.3
    assert float(np.median(np.abs(flow[valid, 1]))) < 0.3


def test_lucas_kanade_cv_recovers_translation_with_fb_check(
    textured_image: NDArray[np.uint8],
) -> None:
    """Verify pyramidal OpenCV LK accepts accurate forward-backward tracks.

    Args:
        textured_image: Seeded grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 7 - Lucas-Kanade optical flow
    """
    current = _translated(textured_image, 2, 0)
    rows, columns = np.mgrid[18:78:12, 18:78:12]
    points = np.column_stack((columns.ravel(), rows.ravel())).astype(np.float32)

    new_points, valid = lucas_kanade_cv(
        textured_image, current, points, win=21, levels=3
    )
    displacement = new_points[valid] - points[valid]

    assert np.count_nonzero(valid) >= 20
    assert float(np.median(np.abs(displacement[:, 0] - 2.0))) < 0.3
    assert float(np.median(np.abs(displacement[:, 1]))) < 0.3


def test_horn_schunck_flow_direction_and_visualizations(
    textured_image: NDArray[np.uint8],
) -> None:
    """Check that dense Horn-Schunck flow points toward a known image shift.

    Args:
        textured_image: Seeded grayscale texture fixture.

    Returns:
        None.

    Syllabus: Unit 7 - Horn-Schunck optical flow
    """
    current = _translated(textured_image, 2, 0)

    u, v = horn_schunck_scratch(textured_image, current, alpha=0.2, iters=150)
    color = flow_to_color(u, v)
    arrows = draw_flow_arrows(textured_image, u, v, step=16)

    assert np.median(u[20:-20, 20:-20]) > 0
    assert np.median(np.abs(v[20:-20, 20:-20])) < 0.5
    assert color.shape == (*textured_image.shape, 3)
    assert arrows.shape == (*textured_image.shape, 3)


def test_document_tracker_tracks_page_motion() -> None:
    """Track synthetic document corners through a small translation.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 7 - document tracking
    """
    image = np.full((160, 200, 3), 70, dtype=np.uint8)
    cv2.rectangle(image, (30, 20), (170, 140), (245, 245, 245), -1)
    cv2.rectangle(image, (30, 20), (170, 140), (0, 0, 0), 2)
    cv2.putText(image, "PAGE", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for row in range(85, 125, 12):
        cv2.line(image, (48, row), (150 - row % 13, row), (0, 0, 0), 1)

    tracker = DocumentTracker(image)
    moved = _translated(image, 3, 0)
    state = tracker.update(moved)

    assert state.corners is not None and state.corners.shape == (4, 2)
    assert state.n_points >= 8
    assert state.mean_flow == pytest.approx(3.0, abs=0.5)
    assert state.sharpness > 0


def test_stability_detector_captures_sharpest_then_detects_page_change() -> None:
    """Exercise movement, stable capture, and the page-changed action state.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 7 - action detection
    """
    base = np.full((40, 40), 200, dtype=np.uint8)
    sharp = base.copy()
    sharp[10:30:2, 10:30:2] = 0
    detector = StabilityDetector(
        stable_frames=3, motion_threshold=0.5, page_change_threshold=0.1
    )

    assert detector.update(base, 0)[0] == StabilityState.IDLE
    assert detector.update(base, 2)[0] == StabilityState.MOVING
    detector.update(base, 0.1)
    detector.update(sharp, 0.1)
    state, captured = detector.update(sharp, 0.1)
    assert state == StabilityState.CAPTURED
    assert captured is not None
    assert np.array_equal(captured, sharp)

    replacement = np.zeros_like(sharp)
    state, result = detector.update(replacement, 0)
    assert state == StabilityState.PAGE_CHANGED
    assert result is None


def _run_generator(
    generator: Generator[dict[str, int | float], None, dict[str, Any]],
) -> tuple[list[dict[str, int | float]], dict[str, Any]]:
    """Exhaust a process_video generator and collect progress and return value.

    Args:
        generator: Initialized video-processing generator.

    Returns:
        (yielded progress values, final generator return mapping).

    Syllabus: Unit 7 - motion analysis and action detection
    """
    progress: list[dict[str, int | float]] = []
    while True:
        try:
            progress.append(next(generator))
        except StopIteration as stopped:
            return progress, stopped.value


def test_demo_video_processes_to_exactly_two_captures(tmp_path: Path) -> None:
    """Create and analyze the synthetic drifting/two-page video end to end.

    Args:
        tmp_path: Pytest-managed directory for generated video output.

    Returns:
        None.

    Syllabus: Unit 7 - motion analysis and action detection
    """
    video_path = tmp_path / "motion_demo.avi"
    make_demo_video(video_path)

    progress, result = _run_generator(process_video(video_path, stride=2))

    captures = [event for event in result["events"] if event["type"] == "CAPTURE"]
    assert len(progress) > 0
    assert len(captures) == 2
    assert len(result["captured_frames"]) == 2
    assert list(result["timeline"].columns) == [
        "frame",
        "motion",
        "sharpness",
        "state",
    ]
    assert StabilityState.PAGE_CHANGED.value in result["timeline"]["state"].values

"""OpenCV webcam/file demo for document tracking and stable scan capture.

Syllabus: Unit 5 - motion analysis and action detection
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from docuvision.config import PipelineConfig
from docuvision.motion import (
    DocumentTracker,
    StabilityDetector,
    StabilityState,
    TrackState,
)
from docuvision.pipeline import PipelineResult, run_pipeline

Image = NDArray[np.uint8]


def _draw_overlay(
    frame: Image, track: TrackState | None, state: StabilityState, stable_count: int
) -> Image:
    """Draw the tracked page polygon and live state metrics on a frame.

    Args:
        frame: Current BGR uint8 camera or file frame.
        track: Latest tracker output, or None before a page is found.
        state: Current stability state from the temporal detector.
        stable_count: Number of consecutive stable frames observed so far.

    Returns:
        Independent BGR visualization frame.

    Syllabus: Unit 5 - document tracking and action detection
    """
    display = frame.copy()
    if track is not None and track.corners is not None:
        corners = np.rint(track.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(display, [corners], True, (0, 220, 0), 2, cv2.LINE_AA)
        sharpness = track.sharpness
        points = track.n_points
        motion = track.mean_flow
    else:
        sharpness = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        points = 0
        motion = 0.0
        cv2.putText(
            display,
            "Searching for a document quadrilateral...",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    label = f"{state.value} | tracked points: {points} | motion: {motion:.2f}px | sharpness: {sharpness:.1f}"
    if state == StabilityState.STABLE:
        label += f" | stable frames: {stable_count}"
    cv2.putText(
        display,
        label,
        (16, max(28, display.shape[0] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 30, 230) if state in (StabilityState.MOVING, StabilityState.PAGE_CHANGED) else (0, 180, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "q: quit  |  s: capture",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def _save_scan(frame: Image, output_dir: Path, config: PipelineConfig) -> Path:
    """Run the document pipeline, saving a rectified scan and enhanced output.

    Args:
        frame: Captured BGR uint8 source frame.
        output_dir: Destination directory for live scan images.
        config: Pipeline configuration used for processing.

    Returns:
        Path to the saved rectified scan.

    Raises:
        OSError: If processing cannot be encoded or the output cannot be written.

    Syllabus: Unit 1 - document processing pipeline
    """
    result: PipelineResult = run_pipeline(frame, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    rectification = next(
        (stage.image for stage in result.stages if stage.name == "document_rectification"),
        frame,
    )
    warped_path = output_dir / f"warped_{timestamp}.png"
    enhanced_path = output_dir / f"scan_{timestamp}.png"
    if not cv2.imwrite(str(warped_path), rectification):
        raise OSError(f"Could not save rectified scan: {warped_path}")
    if not cv2.imwrite(str(enhanced_path), result.final_gray):
        raise OSError(f"Could not save enhanced scan: {enhanced_path}")
    print(
        f"Saved warped scan: {warped_path}\n"
        f"Saved enhanced scan: {enhanced_path}\n"
        f"Pipeline status: {result.metrics.get('rectification', 'processed')}; "
        f"OCR: {result.metrics.get('ocr_status', 'not reported')}"
    )
    if result.text:
        print(f"OCR: {result.text[:300].replace(chr(10), ' ')}")
    return warped_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse camera or video-file input for the live OpenCV demonstration.

    Args:
        argv: Optional command-line arguments; None parses process arguments.

    Returns:
        Parsed namespace with optional video path.

    Syllabus: Unit 5 - motion analysis and action detection
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        help="Replay a video file instead of opening camera index 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the live webcam/file tracking window until q or end of video.

    Args:
        argv: Optional command-line arguments for testing or embedding.

    Returns:
        Zero on normal exit, one if the selected video or camera cannot open.

    Syllabus: Unit 5 - motion analysis and action detection
    """
    args = _parse_args(argv)
    source: int | str = str(args.video) if args.video is not None else 0
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        if args.video is None:
            print(
                "No camera found at index 0. Connect or enable a webcam, or replay "
                "a file with: python live_demo.py --video path/to/video.mp4"
            )
        else:
            print(f"Could not open video file: {args.video}")
        return 1

    tracker: DocumentTracker | None = None
    stability = StabilityDetector(stable_frames=8, motion_threshold=0.5)
    track_state: TrackState | None = None
    stable_count = 0
    config = PipelineConfig.preset("balanced")
    output_dir = Path("results/live")
    state = StabilityState.IDLE
    try:
        print("Live demo running. Press q to quit or s to capture the current frame.")
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = np.asarray(frame, dtype=np.uint8)
            if tracker is None:
                try:
                    tracker = DocumentTracker(frame)
                    track_state = TrackState(
                        corners=tracker.corners,
                        n_points=tracker.n_points,
                        mean_flow=0.0,
                        sharpness=float(
                            cv2.Laplacian(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                            ).var()
                        ),
                    )
                except ValueError:
                    track_state = None
                state, _ = stability.update(frame, 0.0)
            else:
                track_state = tracker.update(frame)
                state, captured_frame = stability.update(frame, track_state.mean_flow)
                if captured_frame is not None:
                    print("Stable document detected; capturing the sharpest frame...")
                    _save_scan(captured_frame, output_dir, config)
            stable_count = stable_count + 1 if state == StabilityState.STABLE else 0
            display = _draw_overlay(
                frame, track_state, state, stable_count
            )
            cv2.imshow("DocuVision live document capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                print("Manual capture requested.")
                _save_scan(frame.copy(), output_dir, config)
    except KeyboardInterrupt:
        print("Live demo interrupted.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

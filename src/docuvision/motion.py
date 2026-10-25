"""Optical flow, document tracking, and temporal stability detection.

Syllabus: Unit 7 - motion analysis and tracking
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from docuvision.geometry import detect_document_corners

Image = NDArray[np.uint8]
FloatImage = NDArray[np.float32]
FlowField = NDArray[np.float32]
PointArray = NDArray[np.float32]


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate a nonempty uint8 grayscale, singleton-channel, or BGR image.

    Args:
        img: Image array to validate.

    Returns:
        The original image array without copying or mutation.

    Syllabus: Unit 7 - motion analysis and tracking
    """
    image = np.asarray(img)
    valid_shape = image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] in (1, 3)
    )
    if image.dtype != np.uint8 or not valid_shape or image.size == 0:
        raise ValueError("Expected a nonempty uint8 grayscale or BGR image.")
    return image


def _gray(img: NDArray[np.generic]) -> Image:
    """Convert a validated image to uint8 grayscale.

    Args:
        img: Nonempty uint8 grayscale or BGR image.

    Returns:
        Grayscale uint8 image.

    Syllabus: Unit 7 - motion analysis and tracking
    """
    image = _validate_image(img)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def frame_difference(
    prev: NDArray[np.generic], curr: NDArray[np.generic], thr: int
) -> tuple[Image, float]:
    """Threshold absolute frame difference and measure the changed-pixel fraction.

    Args:
        prev: Previous uint8 grayscale or BGR frame.
        curr: Current uint8 grayscale or BGR frame of the same shape.
        thr: Nonnegative uint8 intensity-difference threshold.

    Returns:
        (uint8 binary change mask, motion score in [0, 1]).

    Syllabus: Unit 7 - frame differencing
    """
    previous = _gray(prev)
    current = _gray(curr)
    if previous.shape != current.shape:
        raise ValueError("prev and curr must have the same spatial dimensions.")
    if not isinstance(thr, (int, np.integer)) or not 0 <= thr <= 255:
        raise ValueError("thr must be an integer in [0, 255].")
    difference = cv2.absdiff(previous, current)
    mask = np.where(difference > thr, 255, 0).astype(np.uint8)
    return mask, float(np.count_nonzero(mask) / mask.size)


def running_average_background(
    frames: Sequence[NDArray[np.generic]], alpha: float
) -> Image:
    """Estimate a background by recursively averaging a sequence of frames.

    Args:
        frames: Nonempty sequence of same-shape uint8 grayscale or BGR frames.
        alpha: New-frame contribution in (0, 1]; one uses only the final frame.

    Returns:
        Rounded uint8 running-average background in the input color format.

    Syllabus: Unit 7 - background modeling
    """
    if len(frames) == 0:
        raise ValueError("frames must contain at least one image.")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1].")
    first = _validate_image(frames[0])
    average = first.astype(np.float32)
    for frame in frames[1:]:
        image = _validate_image(frame)
        if image.shape != first.shape:
            raise ValueError("all frames must have the same shape.")
        average = (1.0 - alpha) * average + float(alpha) * image
    return np.clip(np.rint(average), 0, 255).astype(np.uint8)


def _validate_points(points: NDArray[np.generic]) -> NDArray[np.float32]:
    """Validate floating-point (x, y) point coordinates.

    Args:
        points: Array with shape (N, 2).

    Returns:
        Float32 point array.

    Syllabus: Unit 7 - optical flow
    """
    coordinates = np.asarray(points, dtype=np.float32)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("points must be a finite array with shape (N, 2).")
    return coordinates


def lucas_kanade_scratch(
    prev: NDArray[np.generic],
    curr: NDArray[np.generic],
    points: NDArray[np.generic],
    win: int = 15,
) -> tuple[FlowField, NDArray[np.bool_]]:
    """Estimate sparse optical flow using per-point least-squares Lucas-Kanade.

    The aperture problem makes motion along an edge ambiguous: local gradients
    constrain flow only perpendicular to that edge. Corners have gradients in
    multiple directions, so their structure matrix is well-conditioned and
    they track reliably.

    Args:
        prev: Previous uint8 grayscale or BGR frame.
        curr: Current uint8 frame with the same dimensions.
        points: (N, 2) floating-point coordinates in (x, y) order.
        win: Positive odd side length of each local gradient window.

    Returns:
        (N, 2) float32 flow vectors (dx, dy) and N boolean validity flags.
        A point is valid when the minimum eigenvalue of A.T @ A exceeds the
        texture threshold and the least-squares solve returns finite flow.

    Syllabus: Unit 7 - Lucas-Kanade optical flow
    """
    previous = _gray(prev)
    current = _gray(curr)
    coordinates = _validate_points(points)
    if previous.shape != current.shape:
        raise ValueError("prev and curr must have the same spatial dimensions.")
    if not isinstance(win, (int, np.integer)) or win <= 1 or win % 2 == 0:
        raise ValueError("win must be an odd integer greater than one.")
    previous_float = previous.astype(np.float32) / 255.0
    current_float = current.astype(np.float32) / 255.0
    ix_previous = cv2.Sobel(
        previous_float, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0
    )
    iy_previous = cv2.Sobel(
        previous_float, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0
    )
    ix_current = cv2.Sobel(
        current_float, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0
    )
    iy_current = cv2.Sobel(
        current_float, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0
    )
    radius = int(win) // 2
    flow = np.zeros((len(coordinates), 2), dtype=np.float32)
    valid = np.zeros(len(coordinates), dtype=bool)
    height, width = previous.shape
    # Reject nearly flat windows: this eigenvalue floor keeps the solve stable
    # and marks the aperture problem instead of reporting arbitrary motion.
    eigenvalue_threshold = 1e-5
    for index, (x, y) in enumerate(coordinates):
        column, row = int(round(float(x))), int(round(float(y)))
        if (
            row - radius < 0
            or row + radius >= height
            or column - radius < 0
            or column + radius >= width
        ):
            continue
        row_slice = slice(row - radius, row + radius + 1)
        column_slice = slice(column - radius, column + radius + 1)
        previous_patch = previous_float[row_slice, column_slice]
        previous_ix = ix_previous[row_slice, column_slice]
        previous_iy = iy_previous[row_slice, column_slice]
        displacement = np.zeros(2, dtype=np.float32)
        point_is_valid = True
        # Re-linearize the current patch after each update to accommodate larger
        # displacements than a single first-order Taylor approximation.
        for _ in range(3):
            center = (float(x + displacement[0]), float(y + displacement[1]))
            current_patch = cv2.getRectSubPix(
                current_float, (int(win), int(win)), center
            )
            current_ix = cv2.getRectSubPix(
                ix_current, (int(win), int(win)), center
            )
            current_iy = cv2.getRectSubPix(
                iy_current, (int(win), int(win)), center
            )
            grad_x = (0.5 * (previous_ix + current_ix)).reshape(-1)
            grad_y = (0.5 * (previous_iy + current_iy)).reshape(-1)
            temporal = (current_patch - previous_patch).reshape(-1)
            matrix = np.column_stack((grad_x, grad_y))
            normal = matrix.T @ matrix
            if float(np.linalg.eigvalsh(normal)[0]) <= eigenvalue_threshold:
                point_is_valid = False
                break
            correction, _, _, _ = np.linalg.lstsq(matrix, -temporal, rcond=None)
            if not np.all(np.isfinite(correction)):
                point_is_valid = False
                break
            displacement += correction.astype(np.float32)
            if float(np.linalg.norm(correction)) < 0.01:
                break
        if point_is_valid:
            flow[index] = displacement
            valid[index] = True
    return flow, valid


def lucas_kanade_cv(
    prev: NDArray[np.generic],
    curr: NDArray[np.generic],
    points: NDArray[np.generic],
    win: int = 21,
    levels: int = 3,
) -> tuple[PointArray, NDArray[np.bool_]]:
    """Track points with pyramidal OpenCV Lucas-Kanade and forward-backward check.

    Image pyramids handle large motion by estimating coarse-scale displacement
    first, then refining it at finer scales. A forward-backward error threshold
    below one pixel rejects inconsistent tracks.

    Args:
        prev: Previous uint8 grayscale or BGR frame.
        curr: Current uint8 frame with the same dimensions.
        points: (N, 2) input coordinates in (x, y) order.
        win: Positive odd LK window side length.
        levels: Nonnegative pyramid maximum level.

    Returns:
        (N, 2) float32 new positions and N boolean valid statuses.

    Syllabus: Unit 7 - Lucas-Kanade optical flow
    """
    previous = _gray(prev)
    current = _gray(curr)
    coordinates = _validate_points(points)
    if previous.shape != current.shape:
        raise ValueError("prev and curr must have the same spatial dimensions.")
    if not isinstance(win, (int, np.integer)) or win <= 1 or win % 2 == 0:
        raise ValueError("win must be an odd integer greater than one.")
    if not isinstance(levels, (int, np.integer)) or levels < 0:
        raise ValueError("levels must be a nonnegative integer.")
    if len(coordinates) == 0:
        return coordinates.copy(), np.empty((0,), dtype=bool)
    initial = coordinates.reshape(-1, 1, 2)
    tracked, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        initial,
        None,
        winSize=(int(win), int(win)),
        maxLevel=int(levels),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status_forward is None:
        return coordinates.copy(), np.zeros(len(coordinates), dtype=bool)
    returned, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        tracked,
        None,
        winSize=(int(win), int(win)),
        maxLevel=int(levels),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    new_points = tracked.reshape(-1, 2).astype(np.float32, copy=False)
    if returned is None or status_backward is None:
        return new_points, np.zeros(len(coordinates), dtype=bool)
    forward_backward_error = np.linalg.norm(
        returned.reshape(-1, 2) - coordinates, axis=1
    )
    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & (forward_backward_error < 1.0)
    )
    return new_points, valid


def horn_schunck_scratch(
    prev: NDArray[np.generic],
    curr: NDArray[np.generic],
    alpha: float = 1.0,
    iters: int = 100,
) -> tuple[FlowField, FlowField]:
    """Estimate dense flow with the Horn-Schunck global smoothness constraint.

    Frames are reduced to at most 256 pixels per side for this educational
    iterative implementation, then the flow is resized to the original size.

    Args:
        prev: Previous uint8 grayscale or BGR frame.
        curr: Current uint8 frame with the same dimensions.
        alpha: Positive smoothness regularization strength.
        iters: Positive number of Jacobi flow-refinement iterations.

    Returns:
        Dense float32 horizontal and vertical flow fields (u, v), in original
        image-pixel displacement units.

    Syllabus: Unit 7 - Horn-Schunck optical flow
    """
    previous = _gray(prev)
    current = _gray(curr)
    if previous.shape != current.shape:
        raise ValueError("prev and curr must have the same spatial dimensions.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    if not isinstance(iters, (int, np.integer)) or iters <= 0:
        raise ValueError("iters must be a positive integer.")
    original_height, original_width = previous.shape
    scale = min(1.0, 256.0 / max(original_height, original_width))
    small_width = max(1, int(round(original_width * scale)))
    small_height = max(1, int(round(original_height * scale)))
    previous_float = cv2.resize(
        previous.astype(np.float32) / 255.0,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    current_float = cv2.resize(
        current.astype(np.float32) / 255.0,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    average = 0.5 * (previous_float + current_float)
    ix = cv2.Sobel(average, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    iy = cv2.Sobel(average, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    it = current_float - previous_float
    u = np.zeros_like(previous_float)
    v = np.zeros_like(previous_float)
    # This 3x3 neighborhood average approximates the smoothness term.
    averaging_kernel = np.array(
        [[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype=np.float32
    ) / 12.0
    denominator = float(alpha) ** 2 + ix * ix + iy * iy
    for _ in range(int(iters)):
        mean_u = cv2.filter2D(u, cv2.CV_32F, averaging_kernel)
        mean_v = cv2.filter2D(v, cv2.CV_32F, averaging_kernel)
        correction = (ix * mean_u + iy * mean_v + it) / denominator
        u = mean_u - ix * correction
        v = mean_v - iy * correction
    if (small_height, small_width) != (original_height, original_width):
        u = cv2.resize(u, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        u *= original_width / small_width
        v *= original_height / small_height
    return u.astype(np.float32, copy=False), v.astype(np.float32, copy=False)


def flow_to_color(u: NDArray[np.generic], v: NDArray[np.generic]) -> Image:
    """Visualize dense optical flow as an HSV-to-BGR color image.

    Args:
        u: Finite horizontal flow component.
        v: Finite vertical flow component of the same shape.

    Returns:
        BGR uint8 flow visualization; hue encodes direction and value encodes
        magnitude relative to the strongest vector.

    Syllabus: Unit 7 - optical flow visualization
    """
    horizontal = np.asarray(u, dtype=np.float32)
    vertical = np.asarray(v, dtype=np.float32)
    if (
        horizontal.ndim != 2
        or horizontal.shape != vertical.shape
        or horizontal.size == 0
        or not np.all(np.isfinite(horizontal))
        or not np.all(np.isfinite(vertical))
    ):
        raise ValueError("u and v must be same-shape finite 2D arrays.")
    magnitude, angle = cv2.cartToPolar(horizontal, vertical, angleInDegrees=True)
    hsv = np.zeros((*horizontal.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.mod(angle * 0.5, 180).astype(np.uint8)
    hsv[:, :, 1] = 255
    max_magnitude = float(magnitude.max())
    hsv[:, :, 2] = (
        np.clip(magnitude * (255.0 / max_magnitude), 0, 255).astype(np.uint8)
        if max_magnitude > 0
        else 0
    )
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(
    img: NDArray[np.generic],
    u: NDArray[np.generic],
    v: NDArray[np.generic],
    step: int = 16,
) -> Image:
    """Draw sampled dense-flow vectors on a copy of an input frame.

    Args:
        img: uint8 grayscale or BGR image matching the flow dimensions.
        u: Horizontal flow component.
        v: Vertical flow component of the same shape.
        step: Positive grid spacing in pixels for arrow placement.

    Returns:
        BGR uint8 frame with green flow arrows.

    Syllabus: Unit 7 - optical flow visualization
    """
    image = _validate_image(img)
    horizontal = np.asarray(u, dtype=np.float32)
    vertical = np.asarray(v, dtype=np.float32)
    if (
        horizontal.shape != vertical.shape
        or horizontal.shape != image.shape[:2]
        or not np.all(np.isfinite(horizontal))
        or not np.all(np.isfinite(vertical))
    ):
        raise ValueError("flow arrays must be finite and match the image size.")
    if not isinstance(step, (int, np.integer)) or step <= 0:
        raise ValueError("step must be a positive integer.")
    canvas = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 2
        else image[:, :, 0].copy()
        if image.shape[2] == 1
        else image.copy()
    )
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    height, width = horizontal.shape
    for y in range(int(step) // 2, height, int(step)):
        for x in range(int(step) // 2, width, int(step)):
            end_x = int(round(x + float(horizontal[y, x])))
            end_y = int(round(y + float(vertical[y, x])))
            cv2.arrowedLine(canvas, (x, y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)
    return canvas


def _sharpness(frame: Image) -> float:
    """Compute variance-of-Laplacian sharpness for a frame.

    Args:
        frame: uint8 grayscale or BGR image.

    Returns:
        Nonnegative Laplacian variance.

    Syllabus: Unit 7 - frame quality selection
    """
    return float(cv2.Laplacian(_gray(frame), cv2.CV_64F).var())


@dataclass(frozen=True)
class TrackState:
    """Snapshot of tracked document corners and motion quality.

    Args:
        corners: Ordered float32 page corners or None when unavailable.
        n_points: Number of valid tracked feature points.
        mean_flow: Mean tracked-point displacement magnitude in pixels.
        sharpness: Variance of the grayscale Laplacian in the current frame.

    Returns:
        Immutable tracking result.

    Syllabus: Unit 7 - document tracking
    """

    corners: NDArray[np.float32] | None
    n_points: int
    mean_flow: float
    sharpness: float


class DocumentTracker:
    """Track a detected page with Shi-Tomasi points and pyramidal LK flow.

    Args:
        first_frame: Initial uint8 grayscale or BGR frame containing a page.
        redetect_interval: Positive frame interval for re-detecting page corners.

    Returns:
        Stateful tracker whose ``update`` method returns a TrackState.

    Syllabus: Unit 7 - document tracking
    """

    def __init__(
        self, first_frame: NDArray[np.generic], redetect_interval: int = 15
    ) -> None:
        self._frame = _validate_image(first_frame).copy()
        if (
            not isinstance(redetect_interval, (int, np.integer))
            or redetect_interval <= 0
        ):
            raise ValueError("redetect_interval must be a positive integer.")
        self._redetect_interval = int(redetect_interval)
        self._frame_index = 0
        self._corners = detect_document_corners(self._frame)
        self._points = self._points_inside(self._frame, self._corners)

    @staticmethod
    def _points_inside(frame: Image, corners: NDArray[np.float32]) -> PointArray:
        """Detect Shi-Tomasi points constrained to the document polygon.

        Args:
            frame: Current uint8 grayscale or BGR image.
            corners: Four ordered page corners in (x, y) coordinates.

        Returns:
            Float32 point array in (x, y) order, shape (N, 2).

        Syllabus: Unit 7 - document tracking
        """
        gray = _gray(frame)
        polygon_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(polygon_mask, np.rint(corners).astype(np.int32), 255)
        found = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=7,
            mask=polygon_mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if found is None:
            return np.empty((0, 2), dtype=np.float32)
        return found.reshape(-1, 2).astype(np.float32, copy=False)

    @property
    def corners(self) -> NDArray[np.float32]:
        """Return a copy of the current ordered document corners.

        Args:
            None.

        Returns:
            Float32 corner array with shape (4, 2).

        Syllabus: Unit 7 - document tracking
        """
        return self._corners.copy()

    @property
    def n_points(self) -> int:
        """Return the number of Shi-Tomasi points currently being tracked.

        Args:
            None.

        Returns:
            Number of current points.

        Syllabus: Unit 7 - document tracking
        """
        return len(self._points)

    def update(self, frame: NDArray[np.generic]) -> TrackState:
        """Track document features into a new frame and update page corners.

        Args:
            frame: Next uint8 grayscale or BGR frame, same size as first_frame.

        Returns:
            TrackState with updated corners, valid point count, mean point flow,
            and current frame sharpness.

        Syllabus: Unit 7 - document tracking
        """
        current = _validate_image(frame)
        if current.shape != self._frame.shape:
            raise ValueError("tracked frames must have identical shapes.")
        previous_points = self._points
        flow_magnitudes = np.empty((0,), dtype=np.float32)
        if len(previous_points):
            tracked_points, valid = lucas_kanade_cv(
                self._frame, current, previous_points, win=21, levels=3
            )
            valid &= (
                (tracked_points[:, 0] >= 0)
                & (tracked_points[:, 0] < current.shape[1])
                & (tracked_points[:, 1] >= 0)
                & (tracked_points[:, 1] < current.shape[0])
            )
            accepted_old = previous_points[valid]
            accepted_new = tracked_points[valid]
            flow_magnitudes = np.linalg.norm(accepted_new - accepted_old, axis=1)
            if len(accepted_new) >= 4:
                homography, inliers = cv2.findHomography(
                    accepted_old,
                    accepted_new,
                    cv2.RANSAC,
                    3.0,
                )
                if homography is not None and inliers is not None and int(inliers.sum()) >= 4:
                    self._corners = cv2.perspectiveTransform(
                        self._corners.reshape(1, -1, 2), homography
                    ).reshape(-1, 2).astype(np.float32, copy=False)
                    accepted_new = accepted_new[inliers.reshape(-1).astype(bool)]
            self._points = accepted_new.astype(np.float32, copy=False)
        self._frame_index += 1
        if (
            len(self._points) < 8
            or self._frame_index % self._redetect_interval == 0
        ):
            try:
                detected = detect_document_corners(current)
            except ValueError:
                detected = None
            if detected is not None:
                self._corners = detected
                self._points = self._points_inside(current, self._corners)
        self._frame = current.copy()
        mean_flow = float(flow_magnitudes.mean()) if flow_magnitudes.size else 0.0
        return TrackState(
            corners=self._corners.copy(),
            n_points=len(self._points),
            mean_flow=mean_flow,
            sharpness=_sharpness(current),
        )


class StabilityState(str, Enum):
    """Finite-state labels emitted by the temporal stability detector.

    Args:
        None.

    Returns:
        String-valued states IDLE, MOVING, STABLE, CAPTURED, and PAGE_CHANGED.

    Syllabus: Unit 7 - action detection
    """

    IDLE = "IDLE"
    MOVING = "MOVING"
    STABLE = "STABLE"
    CAPTURED = "CAPTURED"
    PAGE_CHANGED = "PAGE_CHANGED"


class StabilityDetector:
    """Detect motion pauses, choose sharp frames, and report page-change actions.

    Args:
        stable_frames: Consecutive low-motion frames required before capture.
        motion_threshold: Corner displacement below this value is stable.
        page_change_threshold: Changed-pixel fraction indicating a new page.

    Returns:
        Stateful detector whose ``update`` method returns a state and optional
        captured frame.

    Syllabus: Unit 7 - action detection
    """

    def __init__(
        self,
        stable_frames: int = 5,
        motion_threshold: float = 0.5,
        page_change_threshold: float = 0.12,
    ) -> None:
        if not isinstance(stable_frames, (int, np.integer)) or stable_frames <= 0:
            raise ValueError("stable_frames must be a positive integer.")
        if not np.isfinite(motion_threshold) or motion_threshold < 0:
            raise ValueError("motion_threshold must be finite and nonnegative.")
        if not np.isfinite(page_change_threshold) or not 0 < page_change_threshold <= 1:
            raise ValueError("page_change_threshold must be in (0, 1].")
        self._stable_frames = int(stable_frames)
        self._motion_threshold = float(motion_threshold)
        self._page_change_threshold = float(page_change_threshold)
        self._state = StabilityState.IDLE
        self._previous: Image | None = None
        self._stable_count = 0
        self._sharpest_frame: Image | None = None
        self._sharpest_value = -1.0

    @property
    def state(self) -> StabilityState:
        """Return the current detector state.

        Args:
            None.

        Returns:
            Current finite-state value.

        Syllabus: Unit 7 - action detection
        """
        return self._state

    def update(
        self, frame: NDArray[np.generic], mean_corner_motion: float
    ) -> tuple[StabilityState, Image | None]:
        """Advance the state machine and optionally return a captured sharp frame.

        Args:
            frame: Current uint8 grayscale or BGR frame.
            mean_corner_motion: Mean tracked-corner displacement from the prior frame.

        Returns:
            (current state, captured-frame copy or None). A frame is returned
            exactly once when N stable frames complete; after capture, a large
            frame difference transitions to PAGE_CHANGED.

        Syllabus: Unit 7 - action detection
        """
        image = _validate_image(frame)
        if not np.isfinite(mean_corner_motion) or mean_corner_motion < 0:
            raise ValueError("mean_corner_motion must be finite and nonnegative.")
        if self._previous is None:
            self._previous = image.copy()
            return self._state, None
        if image.shape != self._previous.shape:
            raise ValueError("stability frames must have identical shapes.")
        _, difference_score = frame_difference(self._previous, image, thr=25)
        self._previous = image.copy()
        if (
            self._state == StabilityState.CAPTURED
            and difference_score >= self._page_change_threshold
        ):
            self._state = StabilityState.PAGE_CHANGED
            self._stable_count = 0
            self._sharpest_frame = image.copy()
            self._sharpest_value = _sharpness(image)
            return self._state, None
        if self._state == StabilityState.PAGE_CHANGED:
            self._state = (
                StabilityState.MOVING
                if mean_corner_motion >= self._motion_threshold
                else StabilityState.IDLE
            )
            self._stable_count = 0
        if self._state in (StabilityState.IDLE, StabilityState.CAPTURED):
            if self._state == StabilityState.CAPTURED:
                return self._state, None
            if mean_corner_motion >= self._motion_threshold:
                self._state = StabilityState.MOVING
                self._stable_count = 0
            return self._state, None
        if self._state in (StabilityState.MOVING, StabilityState.STABLE):
            if mean_corner_motion >= self._motion_threshold:
                self._state = StabilityState.MOVING
                self._stable_count = 0
                self._sharpest_frame = image.copy()
                self._sharpest_value = _sharpness(image)
                return self._state, None
            self._stable_count += 1
            sharpness = _sharpness(image)
            if sharpness > self._sharpest_value:
                self._sharpest_frame = image.copy()
                self._sharpest_value = sharpness
            if self._stable_count >= self._stable_frames:
                self._state = StabilityState.CAPTURED
                if self._sharpest_frame is None:
                    raise RuntimeError("stable frame selection has no candidate.")
                return self._state, self._sharpest_frame.copy()
            self._state = StabilityState.STABLE
        return self._state, None


def process_video(
    path: str | Path,
    stride: int = 2,
    max_frames: int | None = None,
) -> Generator[dict[str, int | float], None, dict[str, Any]]:
    """Analyze a video and return events, captured frames, and a timeline.

    The generator yields progress dictionaries while reading. Its final return
    value (available from the generator's StopIteration.value) contains
    ``events``, ``captured_frames``, and a pandas timeline with columns
    ``frame``, ``motion``, ``sharpness``, and ``state``.

    Args:
        path: Local video path readable by OpenCV.
        stride: Positive frame sampling interval.
        max_frames: Optional positive cap on decoded frames.

    Returns:
        Generator yielding progress and returning the completed analysis bundle.

    Syllabus: Unit 7 - motion analysis and action detection
    """
    if not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError("stride must be a positive integer.")
    if max_frames is not None and (
        not isinstance(max_frames, (int, np.integer)) or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer or None.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {path}")
    tracker: DocumentTracker | None = None
    detector = StabilityDetector(stable_frames=5)
    events: list[dict[str, int | float | str]] = []
    captured_frames: list[Image] = []
    timeline_rows: list[dict[str, int | float | str]] = []
    previous_sample: Image | None = None
    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % int(stride) == 0:
                image = _validate_image(frame)
                if tracker is None:
                    tracker = DocumentTracker(image)
                    state = TrackState(
                        corners=tracker.corners,
                        n_points=tracker.n_points,
                        mean_flow=0.0,
                        sharpness=_sharpness(image),
                    )
                    stability_state, captured_frame = detector.update(image, 0.0)
                else:
                    state = tracker.update(image)
                    stability_state, captured_frame = detector.update(
                        image, state.mean_flow
                    )
                if previous_sample is None:
                    frame_motion = 0.0
                else:
                    _, frame_motion = frame_difference(previous_sample, image, thr=25)
                previous_sample = image.copy()
                if captured_frame is not None:
                    captured_frames.append(captured_frame)
                    events.append(
                        {
                            "type": "CAPTURE",
                            "frame": frame_index,
                            "sharpness": _sharpness(captured_frame),
                            "state": stability_state.value,
                        }
                    )
                elif stability_state == StabilityState.PAGE_CHANGED:
                    events.append(
                        {
                            "type": "PAGE_CHANGED",
                            "frame": frame_index,
                            "motion": frame_motion,
                            "state": stability_state.value,
                        }
                    )
                timeline_rows.append(
                    {
                        "frame": frame_index,
                        "motion": float(state.mean_flow),
                        "sharpness": float(state.sharpness),
                        "state": stability_state.value,
                    }
                )
                yield {
                    "frame": frame_index,
                    "progress": float(
                        (frame_index + 1)
                        / max(1.0, capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    ),
                }
            frame_index += 1
    finally:
        capture.release()
    timeline = pd.DataFrame(
        timeline_rows, columns=["frame", "motion", "sharpness", "state"]
    )
    return {
        "events": events,
        "captured_frames": captured_frames,
        "timeline": timeline,
    }


def _demo_page(index: int, size: tuple[int, int]) -> Image:
    """Create one deterministic synthetic page for the built-in motion demo.

    Args:
        index: Page identifier, one or two.
        size: Requested (height, width) in pixels.

    Returns:
        BGR uint8 page with a border and deterministic printed marks.

    Syllabus: Unit 7 - synthetic motion demonstration
    """
    height, width = size
    # A tinted second sheet makes page replacement detectable by frame differencing.
    paper = 248 if index == 1 else 218
    page = np.full((height, width, 3), paper, dtype=np.uint8)
    cv2.rectangle(page, (5, 5), (width - 6, height - 6), (15, 15, 15), 2)
    cv2.putText(
        page,
        f"DEMO PAGE {index}",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    for line in range(5):
        y = 62 + line * 19
        cv2.line(page, (18, y), (width - 20 - (line % 2) * 28, y), (40, 40, 40), 2)
        cv2.circle(page, (28 + (line % 3) * 25, y - 4), 3, (0, 0, 0), -1)
    return page


def make_demo_video(path: str | Path) -> None:
    """Write a deterministic drifting, pausing, and page-changing demo video.

    Args:
        path: Destination video file path; parent directory must exist.

    Returns:
        None.

    Raises:
        OSError: If OpenCV cannot create the destination video.

    Syllabus: Unit 7 - motion analysis and action detection
    """
    destination = Path(path)
    width, height = 320, 240
    fps = 12.0
    # MJPG is broadly available and keeps this tiny synthetic test video deterministic.
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise OSError(f"Could not create demo video: {destination}")
    background = np.full((height, width, 3), 72, dtype=np.uint8)
    pages = [_demo_page(1, (174, 132)), _demo_page(2, (174, 132))]
    # Each page drifts, pauses long enough for capture, then the second replaces it.
    sequence: list[tuple[int, int, float]] = []
    for index in range(5):
        sequence.append((0, 0, 0.0))
    for index in range(12):
        sequence.append((index - 6, (index - 6) // 2, (index - 6) * 0.8))
    for _ in range(20):
        sequence.append((5, 3, 4.8))
    for _ in range(2):
        sequence.append((5, 3, 4.8))
    for index in range(10):
        sequence.append((5 - index // 2, 3 - index // 3, 4.8 - index * 0.6))
    for _ in range(24):
        sequence.append((0, 0, -0.6))
    try:
        for frame_index, (offset_x, offset_y, angle) in enumerate(sequence):
            page_index = 0 if frame_index < 39 else 1
            page = pages[page_index]
            matrix = cv2.getRotationMatrix2D(
                (page.shape[1] / 2, page.shape[0] / 2), angle, 1.0
            )
            matrix[0, 2] += width // 2 - page.shape[1] // 2 + offset_x
            matrix[1, 2] += height // 2 - page.shape[0] // 2 + offset_y
            warped = cv2.warpAffine(
                page,
                matrix,
                (width, height),
                borderValue=(72, 72, 72),
            )
            frame = background.copy()
            page_mask = cv2.warpAffine(
                np.full(page.shape[:2], 255, dtype=np.uint8),
                matrix,
                (width, height),
                borderValue=0,
            )
            frame[page_mask > 0] = warped[page_mask > 0]
            writer.write(frame)
    finally:
        writer.release()

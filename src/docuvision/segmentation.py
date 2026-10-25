"""Classical document segmentation and edge-detection algorithms.

Syllabus: Unit 2 - segmentation and edge detection
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.feature import peak_local_max
from skimage.filters import threshold_sauvola
from skimage.segmentation import felzenszwalb

Image = NDArray[np.uint8]
LabelImage = NDArray[np.int32]
Connectivity = Literal[4, 8]
GradientOperator = Literal["sobel", "prewitt", "roberts"]
DistanceMetric = Literal["euclidean", "cityblock", "chessboard"]
ThresholdMethod = Literal["mean", "gaussian"]


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate a nonempty uint8 grayscale, singleton-channel, or BGR image.

    Args:
        img: Image array to validate.

    Returns:
        Original array without copying or mutation.

    Syllabus: Unit 2 - segmentation and edge detection
    """
    image = np.asarray(img)
    valid_shape = image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] in (1, 3)
    )
    if image.dtype != np.uint8 or not valid_shape or image.size == 0:
        raise ValueError("Expected a nonempty uint8 grayscale or BGR image.")
    return image


def _gray_float(img: NDArray[np.generic]) -> NDArray[np.float32]:
    """Convert a grayscale or BGR uint8 image to float32 grayscale.

    Args:
        img: Nonempty uint8 grayscale or BGR image.

    Returns:
        Two-dimensional float32 grayscale image.

    Syllabus: Unit 2 - image representation
    """
    image = _validate_image(img)
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def _validate_connectivity(connectivity: int) -> Connectivity:
    """Validate a 4- or 8-neighbor pixel connectivity.

    Args:
        connectivity: Requested neighborhood size.

    Returns:
        Validated connectivity literal.

    Syllabus: Unit 2 - connected components
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")
    return connectivity


def label_components(
    binary: NDArray[np.generic], connectivity: Connectivity = 8
) -> tuple[LabelImage, NDArray[np.int32]]:
    """Label connected foreground components and return OpenCV component stats.

    Args:
        binary: Two-dimensional uint8 or bool mask; nonzero is foreground.
        connectivity: Pixel connectivity, either 4 or 8.

    Returns:
        Pair (labels, stats); labels are int32, while stats rows contain
        x, y, width, height, and area for background followed by each component.

    Syllabus: Unit 2 - connected components
    """
    mask = np.asarray(binary)
    if mask.ndim != 2 or mask.size == 0 or mask.dtype not in (np.dtype("uint8"), np.dtype("bool")):
        raise ValueError("binary must be a nonempty 2D uint8 or bool mask.")
    neighborhood = _validate_connectivity(connectivity)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask != 0, dtype=np.uint8),
        connectivity=neighborhood,
        ltype=cv2.CV_32S,
    )
    if count < 1:
        raise RuntimeError("Connected-component labeling failed.")
    return labels.astype(np.int32, copy=False), stats.astype(np.int32, copy=False)


def connectivity_comparison(
    binary: NDArray[np.generic],
) -> tuple[LabelImage, LabelImage]:
    """Return both 4-connected and 8-connected label maps for a binary image.

    Args:
        binary: Two-dimensional uint8 or bool foreground mask.

    Returns:
        (labels4, labels8), useful for visualizing diagonal connectivity.

    Syllabus: Unit 2 - connected components
    """
    labels4, _ = label_components(binary, connectivity=4)
    labels8, _ = label_components(binary, connectivity=8)
    return labels4, labels8


def connectivity_difference(binary: NDArray[np.generic]) -> tuple[int, int]:
    """Report foreground component counts for 4- and 8-connectivity.

    Args:
        binary: Two-dimensional uint8 or bool foreground mask.

    Returns:
        Tuple (four-connected count, eight-connected count), excluding background.

    Syllabus: Unit 2 - connected components
    """
    _, stats4 = label_components(binary, connectivity=4)
    _, stats8 = label_components(binary, connectivity=8)
    return int(stats4.shape[0] - 1), int(stats8.shape[0] - 1)


def detect_points(img: NDArray[np.generic], thr: float) -> Image:
    """Detect isolated bright points with the textbook 3x3 point mask.

    Args:
        img: uint8 grayscale or BGR image.
        thr: Nonnegative response threshold; strict exceedance is detected.

    Returns:
        uint8 binary point mask.

    Syllabus: Unit 2 - point and line detection
    """
    gray = _gray_float(img)
    if not np.isfinite(thr) or thr < 0:
        raise ValueError("thr must be finite and nonnegative.")
    # Textbook point mask: [-1,-1,-1; -1,8,-1; -1,-1,-1].
    point_mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    response = cv2.filter2D(gray, cv2.CV_32F, point_mask)
    return np.where(np.abs(response) > thr, 255, 0).astype(np.uint8)


def detect_lines(
    img: NDArray[np.generic],
    direction: Literal[
        "horizontal", "vertical", "diagonal_45", "diagonal_135", "45", "135"
    ],
) -> Image:
    """Detect bright lines with one of four textbook 3x3 directional masks.

    Args:
        img: uint8 grayscale or BGR image.
        direction: Horizontal, vertical, 45-degree, or 135-degree line mask.

    Returns:
        uint8 binary response for positive mask responses.

    Syllabus: Unit 2 - point and line detection
    """
    gray = _gray_float(img)
    # The 3x3 masks have two center-line values of +2 and -1 elsewhere.
    masks: dict[str, NDArray[np.float32]] = {
        "horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], np.float32),
        "vertical": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], np.float32),
        "diagonal_45": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], np.float32),
        "diagonal_135": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], np.float32),
    }
    masks["45"] = masks["diagonal_45"]
    masks["135"] = masks["diagonal_135"]
    if direction not in masks:
        raise ValueError("direction must be a supported 3x3 line orientation.")
    response = cv2.filter2D(gray, cv2.CV_32F, masks[direction])
    return np.where(response > 0, 255, 0).astype(np.uint8)


def gradient_edges(
    img: NDArray[np.generic], op: GradientOperator = "sobel"
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Compute directional gradients, magnitude, and angle in radians.

    Args:
        img: uint8 grayscale or BGR image.
        op: ``"sobel"``, ``"prewitt"``, or ``"roberts"`` operator.

    Returns:
        (Gx, Gy, magnitude, direction), each a float32 array with grayscale shape.

    Syllabus: Unit 2 - gradient edge detection
    """
    gray = _gray_float(img)
    if op == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    elif op == "prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
        gy = cv2.filter2D(gray, cv2.CV_32F, kernel_x.T)
    elif op == "roberts":
        kernel_x = np.array([[1, 0], [0, -1]], np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], np.float32)
        gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
        gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    else:
        raise ValueError("op must be 'sobel', 'prewitt', or 'roberts'.")
    magnitude = np.hypot(gx, gy).astype(np.float32, copy=False)
    direction = np.arctan2(gy, gx).astype(np.float32, copy=False)
    return gx, gy, magnitude, direction


def laplacian_edges(img: NDArray[np.generic]) -> Image:
    """Return the absolute 3x3 Laplacian response as an edge-strength image.

    Args:
        img: uint8 grayscale or BGR image.

    Returns:
        uint8 absolute Laplacian response, clipped to [0, 255].

    Syllabus: Unit 2 - Laplacian edge detection
    """
    gray = _gray_float(img)
    response = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return np.clip(np.rint(np.abs(response)), 0, 255).astype(np.uint8)


def log_edges(img: NDArray[np.generic], sigma: float) -> Image:
    """Detect zero crossings in a Gaussian-smoothed Laplacian response.

    Args:
        img: uint8 grayscale or BGR image.
        sigma: Positive Gaussian standard deviation in pixels.

    Returns:
        uint8 binary edge map.

    Syllabus: Unit 2 - Laplacian of Gaussian edge detection
    """
    gray = _gray_float(img)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and greater than zero.")
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    response = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    windows = np.pad(response, 1, mode="edge")
    neighborhoods = np.stack(
        [
            windows[row : row + response.shape[0], col : col + response.shape[1]]
            for row in range(3)
            for col in range(3)
        ],
        axis=0,
    )
    crosses = (neighborhoods.min(axis=0) < 0) & (neighborhoods.max(axis=0) > 0)
    contrast = neighborhoods.max(axis=0) - neighborhoods.min(axis=0)
    threshold = max(float(contrast.max()) * 0.05, np.finfo(np.float32).eps)
    return np.where(crosses & (contrast >= threshold), 255, 0).astype(np.uint8)


def region_growing(
    img: NDArray[np.generic],
    seeds: Sequence[tuple[int, int]],
    tol: float,
    connectivity: Connectivity = 8,
) -> Image:
    """Grow regions from seed points using a breadth-first intensity tolerance.

    Pixels join a seed's region when their grayscale intensity is within
    ``tol`` of that seed's intensity; accepted pixels expand the deque frontier.

    Args:
        img: uint8 grayscale or BGR image.
        seeds: (row, column) seed coordinates.
        tol: Nonnegative maximum absolute intensity difference from a seed.
        connectivity: Neighborhood connectivity, either 4 or 8.

    Returns:
        uint8 mask with reached pixels set to 255.

    Syllabus: Unit 2 - region-based segmentation
    """
    gray = _gray_float(img)
    neighborhood = _validate_connectivity(connectivity)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    if len(seeds) == 0:
        raise ValueError("at least one seed is required.")
    height, width = gray.shape
    coordinates: list[tuple[int, int]] = []
    for seed in seeds:
        if len(seed) != 2:
            raise ValueError("each seed must be a (row, column) pair.")
        row, column = int(seed[0]), int(seed[1])
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError("seed coordinates must be inside the image.")
        coordinates.append((row, column))
    offsets = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if neighborhood == 4
        else [
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        ]
    )
    visited = np.zeros(gray.shape, dtype=bool)
    queue: deque[tuple[int, int, float]] = deque()
    for row, column in coordinates:
        reference = float(gray[row, column])
        if not visited[row, column]:
            visited[row, column] = True
            queue.append((row, column, reference))
    while queue:
        row, column, reference = queue.popleft()
        for row_offset, column_offset in offsets:
            next_row, next_column = row + row_offset, column + column_offset
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and not visited[next_row, next_column]
                and abs(float(gray[next_row, next_column]) - reference) <= tol
            ):
                visited[next_row, next_column] = True
                queue.append((next_row, next_column, reference))
    return visited.astype(np.uint8) * 255


def split_and_merge(
    img: NDArray[np.generic], min_size: int, std_thr: float
) -> LabelImage:
    """Split into quadtree regions and merge adjacent regions with similar means.

    Images are limited to 256 pixels per side to keep this educational
    quadtree/union-find implementation responsive and its intermediate regions
    small.

    Args:
        img: uint8 grayscale or BGR image no larger than 256 by 256.
        min_size: Positive minimum quadtree tile side length.
        std_thr: Nonnegative standard-deviation split and mean-merge threshold.

    Returns:
        Contiguous int32 label map, with background not treated specially.

    Syllabus: Unit 2 - split-and-merge segmentation
    """
    gray = _gray_float(img)
    height, width = gray.shape
    if height > 256 or width > 256:
        raise ValueError("split_and_merge supports images no larger than 256 pixels per side.")
    if not isinstance(min_size, (int, np.integer)) or min_size <= 0:
        raise ValueError("min_size must be a positive integer.")
    if not np.isfinite(std_thr) or std_thr < 0:
        raise ValueError("std_thr must be finite and nonnegative.")
    leaves: list[tuple[int, int, int, int]] = []

    def split(y: int, x: int, h: int, w: int) -> None:
        """Append a homogeneous leaf or recursively divide a variable region."""
        tile = gray[y : y + h, x : x + w]
        if (float(tile.std()) <= std_thr) or (h <= min_size and w <= min_size):
            leaves.append((y, x, h, w))
            return
        if h <= min_size and w <= min_size:
            leaves.append((y, x, h, w))
            return
        half_h, half_w = max(1, h // 2), max(1, w // 2)
        row_parts = [(y, half_h), (y + half_h, h - half_h)] if h > min_size else [(y, h)]
        column_parts = [(x, half_w), (x + half_w, w - half_w)] if w > min_size else [(x, w)]
        for child_y, child_h in row_parts:
            for child_x, child_w in column_parts:
                if child_h > 0 and child_w > 0:
                    split(child_y, child_x, child_h, child_w)

    split(0, 0, height, width)
    count = len(leaves)
    parents = list(range(count))

    def find(index: int) -> int:
        """Find and compress a union-find root."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        """Join region sets, preferring the smaller root index."""
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parents[max(root_first, root_second)] = min(root_first, root_second)

    leaf_map = np.full((height, width), -1, dtype=np.int32)
    means = np.empty(count, dtype=np.float64)
    for index, (y, x, h, w) in enumerate(leaves):
        leaf_map[y : y + h, x : x + w] = index
        means[index] = float(gray[y : y + h, x : x + w].mean())
    adjacent: set[tuple[int, int]] = set()
    for first, second in (
        (leaf_map[:, :-1], leaf_map[:, 1:]),
        (leaf_map[:-1, :], leaf_map[1:, :]),
    ):
        boundaries = (first != second) & (first >= 0) & (second >= 0)
        pairs = np.stack((first[boundaries], second[boundaries]), axis=1)
        adjacent.update((min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs)
    for first, second in adjacent:
        if abs(means[first] - means[second]) <= std_thr:
            union(first, second)
    root_labels: dict[int, int] = {}
    for index in range(count):
        root = find(index)
        if root not in root_labels:
            root_labels[root] = len(root_labels) + 1
    labels = np.empty((height, width), dtype=np.int32)
    for index, (y, x, h, w) in enumerate(leaves):
        labels[y : y + h, x : x + w] = root_labels[find(index)]
    return labels


def canny(img: NDArray[np.generic], sigma: float = 0.33) -> Image:
    """Apply Canny edge detection with lower and upper thresholds from the median.

    Args:
        img: uint8 grayscale or BGR image.
        sigma: Nonnegative fraction controlling thresholds around the median.

    Returns:
        uint8 binary Canny edge map.

    Syllabus: Unit 5 - Canny edge detection
    """
    gray = _validate_image(img)
    if gray.ndim == 3:
        gray = (
            cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            if gray.shape[2] == 3
            else gray[:, :, 0]
        )
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, max(lower, upper))


def dog_edges(
    img: NDArray[np.generic], s1: float, s2: float
) -> Image:
    """Detect edges using the difference of two Gaussian-smoothed images.

    Args:
        img: uint8 grayscale or BGR image.
        s1: Positive standard deviation for the narrower Gaussian.
        s2: Positive standard deviation for the wider Gaussian, greater than s1.

    Returns:
        uint8 binary edge map thresholded relative to DoG response strength.

    Syllabus: Unit 5 - difference-of-Gaussians edges
    """
    gray = _gray_float(img)
    if (
        not np.isfinite(s1)
        or not np.isfinite(s2)
        or s1 <= 0
        or s2 <= s1
    ):
        raise ValueError("s1 and s2 must be finite, positive, and satisfy s2 > s1.")
    first = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(s1), sigmaY=float(s1))
    second = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(s2), sigmaY=float(s2))
    response = np.abs(first - second)
    # A tenth of the strongest response suppresses weak blur differences as noise.
    threshold = float(response.max()) * 0.1
    return np.where(response > threshold, 255, 0).astype(np.uint8)


def otsu_scratch(img: NDArray[np.generic]) -> int:
    """Compute the Otsu intensity threshold by maximizing inter-class variance.

    Args:
        img: uint8 grayscale or BGR image.

    Returns:
        Integer threshold in [0, 255].

    Syllabus: Unit 5 - Otsu thresholding
    """
    gray = _validate_image(img)
    if gray.ndim == 3:
        gray = (
            cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            if gray.shape[2] == 3
            else gray[:, :, 0]
        )
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    intensities = np.arange(256, dtype=np.float64)
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * intensities)
    total_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    between_class = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    between_class[valid] = (
        (total_mean * cumulative_probability[valid] - cumulative_mean[valid]) ** 2
        / denominator[valid]
    )
    return int(np.argmax(between_class))


def otsu_cv(img: NDArray[np.generic]) -> int:
    """Compute an Otsu threshold with OpenCV's optimized implementation.

    Args:
        img: uint8 grayscale or BGR image.

    Returns:
        Integer threshold in [0, 255].

    Syllabus: Unit 5 - Otsu thresholding
    """
    gray = _validate_image(img)
    if gray.ndim == 3:
        gray = (
            cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            if gray.shape[2] == 3
            else gray[:, :, 0]
        )
    threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return int(threshold)


def global_threshold(img: NDArray[np.generic], threshold: float) -> Image:
    """Threshold grayscale intensity into a binary uint8 image.

    Args:
        img: uint8 grayscale or BGR image.
        threshold: Finite intensity cutoff; values strictly above become white.

    Returns:
        uint8 binary image with the original spatial dimensions.

    Syllabus: Unit 5 - global thresholding
    """
    image = _validate_image(img)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if image.ndim == 3:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.shape[2] == 3
            else image[:, :, 0]
        )
    else:
        gray = image
    return np.where(gray > threshold, 255, 0).astype(np.uint8)


def adaptive_threshold(
    img: NDArray[np.generic], method: ThresholdMethod, block: int, C: float
) -> Image:
    """Apply OpenCV adaptive mean or Gaussian thresholding.

    Args:
        img: uint8 grayscale or BGR image.
        method: Either ``"mean"`` or ``"gaussian"`` local threshold method.
        block: Odd neighborhood side length greater than one.
        C: Constant subtracted from each local threshold.

    Returns:
        uint8 binary image, with pixels above the local threshold white.

    Syllabus: Unit 5 - adaptive thresholding
    """
    image = _validate_image(img)
    if image.ndim == 3:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.shape[2] == 3
            else image[:, :, 0]
        )
    else:
        gray = image
    if method not in ("mean", "gaussian"):
        raise ValueError("method must be 'mean' or 'gaussian'.")
    if not isinstance(block, (int, np.integer)) or block <= 1 or block % 2 == 0:
        raise ValueError("block must be an odd integer greater than one.")
    if not np.isfinite(C):
        raise ValueError("C must be finite.")
    algorithm = (
        cv2.ADAPTIVE_THRESH_MEAN_C
        if method == "mean"
        else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )
    return cv2.adaptiveThreshold(
        gray, 255, algorithm, cv2.THRESH_BINARY, int(block), float(C)
    )


def sauvola(
    img: NDArray[np.generic], window: int, k: float
) -> Image:
    """Apply local Sauvola thresholding using scikit-image.

    Args:
        img: uint8 grayscale or BGR image.
        window: Odd local window length greater than one.
        k: Sauvola sensitivity parameter, commonly between 0.1 and 0.5.

    Returns:
        uint8 binary image, with pixels above the local threshold white.

    Syllabus: Unit 5 - Sauvola thresholding
    """
    gray = _gray_float(img)
    if not isinstance(window, (int, np.integer)) or window <= 1 or window % 2 == 0:
        raise ValueError("window must be an odd integer greater than one.")
    if not np.isfinite(k):
        raise ValueError("k must be finite.")
    threshold = threshold_sauvola(gray, window_size=int(window), k=float(k))
    return np.where(gray > threshold, 255, 0).astype(np.uint8)


def pixel_distance(
    p: Sequence[float],
    q: Sequence[float],
    metric: DistanceMetric = "euclidean",
) -> float:
    """Calculate a geometric distance between two equal-length coordinates.

    Args:
        p: First coordinate.
        q: Second coordinate.
        metric: ``"euclidean"``, ``"cityblock"``, or ``"chessboard"``.

    Returns:
        Distance as a float.

    Syllabus: Unit 5 - distance measures
    """
    first = np.asarray(p, dtype=np.float64)
    second = np.asarray(q, dtype=np.float64)
    if (
        first.ndim != 1
        or second.shape != first.shape
        or first.size == 0
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("p and q must be equal-length finite coordinate vectors.")
    difference = np.abs(first - second)
    if metric == "euclidean":
        return float(np.linalg.norm(difference))
    if metric == "cityblock":
        return float(difference.sum())
    if metric == "chessboard":
        return float(difference.max())
    raise ValueError("metric must be 'euclidean', 'cityblock', or 'chessboard'.")


def distance_transform(
    binary: NDArray[np.generic], metric: DistanceMetric = "euclidean"
) -> NDArray[np.float32]:
    """Compute foreground distances to the nearest zero pixel using OpenCV.

    Args:
        binary: Nonempty 2D uint8 or bool foreground mask.
        metric: ``"euclidean"``, ``"cityblock"``, or ``"chessboard"``.

    Returns:
        Float32 distance image, zero on background.

    Syllabus: Unit 5 - distance transforms
    """
    mask = np.asarray(binary)
    if mask.ndim != 2 or mask.size == 0 or mask.dtype not in (np.dtype("uint8"), np.dtype("bool")):
        raise ValueError("binary must be a nonempty 2D uint8 or bool mask.")
    if metric not in ("euclidean", "cityblock", "chessboard"):
        raise ValueError("unsupported distance metric.")
    distance_type = cv2.DIST_L2 if metric == "euclidean" else cv2.DIST_L1
    mask_size = cv2.DIST_MASK_PRECISE if metric == "euclidean" else 3
    if metric == "chessboard":
        # OpenCV's chessboard metric is the infinity norm via the C distance type.
        distance_type = cv2.DIST_C
    return cv2.distanceTransform(
        np.ascontiguousarray(mask != 0, dtype=np.uint8),
        distance_type,
        mask_size,
    ).astype(np.float32, copy=False)


def watershed_split(binary: NDArray[np.generic]) -> LabelImage:
    """Separate touching foreground objects using distance peaks and watershed.

    Args:
        binary: Two-dimensional uint8 or bool mask with foreground nonzero.

    Returns:
        int32 label map with background and watershed boundaries set to zero.

    Syllabus: Unit 5 - watershed segmentation
    """
    mask = np.asarray(binary)
    if mask.ndim != 2 or mask.size == 0 or mask.dtype not in (np.dtype("uint8"), np.dtype("bool")):
        raise ValueError("binary must be a nonempty 2D uint8 or bool mask.")
    foreground = mask != 0
    if not np.any(foreground):
        return np.zeros(mask.shape, dtype=np.int32)
    distance = distance_transform(foreground.astype(np.uint8))
    # Scale peak spacing with image size to avoid merging object centers on small masks.
    minimum_distance = max(1, min(mask.shape) // 30)
    peak_coordinates = peak_local_max(
        distance,
        min_distance=minimum_distance,
        labels=foreground.astype(np.uint8),
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, (row, column) in enumerate(peak_coordinates, start=2):
        markers[row, column] = index
    if not len(peak_coordinates):
        count, component_labels = cv2.connectedComponents(
            foreground.astype(np.uint8), connectivity=8
        )
        markers[foreground] = component_labels[foreground] + 1
        if count == 1:
            return np.zeros(mask.shape, dtype=np.int32)
    markers[~foreground] = 1
    watershed_markers = cv2.watershed(
        cv2.cvtColor((foreground.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR),
        markers,
    )
    result = np.where(watershed_markers >= 2, watershed_markers - 1, 0).astype(np.int32)
    result[~foreground] = 0
    return result


def graph_segmentation(
    img: NDArray[np.generic],
    scale: float = 100.0,
    sigma: float = 0.5,
    min_size: int = 20,
) -> LabelImage:
    """Segment an image with scikit-image's Felzenszwalb graph technique.

    Args:
        img: uint8 grayscale or BGR image.
        scale: Larger values favor larger regions.
        sigma: Positive Gaussian smoothing value.
        min_size: Minimum segment size in pixels.

    Returns:
        int32 segment-label image.

    Syllabus: Unit 5 - graph-based segmentation
    """
    image = _validate_image(img)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")
    if not isinstance(min_size, (int, np.integer)) or min_size <= 0:
        raise ValueError("min_size must be a positive integer.")
    if image.ndim == 3 and image.shape[2] == 3:
        source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3:
        source = image[:, :, 0]
    else:
        source = image
    labels = felzenszwalb(
        source,
        scale=float(scale),
        sigma=float(sigma),
        min_size=int(min_size),
        channel_axis=-1 if source.ndim == 3 else None,
    )
    return labels.astype(np.int32, copy=False)


def segment_text_blocks(
    binary: NDArray[np.generic],
) -> list[tuple[int, int, int, int]]:
    """Find line-sized text boxes, ordered top-to-bottom then left-to-right.

    Args:
        binary: Two-dimensional uint8 mask with text foreground nonzero.

    Returns:
        List of (x, y, width, height) boxes in reading order.

    Syllabus: Unit 3 - document layout segmentation
    """
    mask = np.asarray(binary)
    if mask.ndim != 2 or mask.dtype != np.uint8 or mask.size == 0:
        raise ValueError("binary must be a nonempty two-dimensional uint8 mask.")
    from docuvision.morphology import text_line_mask

    # Wide, shallow dilation bridges glyph gaps while keeping adjacent lines apart.
    line_mask = text_line_mask(mask, kw=21, kh=3)
    _, stats = label_components(line_mask, connectivity=8)
    boxes = [
        (int(x), int(y), int(width), int(height))
        for x, y, width, height, area in stats[1:]
        if area > 0
    ]
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def draw_boxes(
    img: NDArray[np.generic],
    boxes: Sequence[tuple[int, int, int, int]],
) -> Image:
    """Draw rectangular segmentation boxes on a copy of a grayscale or BGR image.

    Args:
        img: Nonempty uint8 grayscale or BGR image.
        boxes: Sequence of (x, y, width, height) rectangles.

    Returns:
        BGR uint8 image with green boxes drawn.

    Syllabus: Unit 3 - document layout visualization
    """
    image = _validate_image(img)
    output = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 2
        else image[:, :, 0].copy()
        if image.shape[2] == 1
        else image.copy()
    )
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        if len(box) != 4 or any(not isinstance(value, (int, np.integer)) for value in box):
            raise ValueError("each box must contain integer (x, y, width, height).")
        x, y, width, height = (int(value) for value in box)
        if width < 0 or height < 0:
            raise ValueError("box width and height must be nonnegative.")
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 1)
    return output

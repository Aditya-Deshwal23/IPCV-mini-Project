"""Classical feature detection and description algorithms.

SURF is patented and absent from standard OpenCV; this module uses SIFT and
ORB instead. SURF is discussed only in the report.

Syllabus: Unit 6 - image features and matching
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.feature import blob_doh, hog

Image = NDArray[np.uint8]
FloatImage = NDArray[np.float32]
PointArray = NDArray[np.float32]
DescriptorArray = NDArray[np.float32] | NDArray[np.uint8]
Pyramid = list[FloatImage]
ScaleSpace = list[list[FloatImage]]
KeypointMatches = list[cv2.DMatch]


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate a nonempty uint8 grayscale, singleton-channel, or BGR image.

    Args:
        img: Image array to validate.

    Returns:
        The input array without copying or mutating it.

    Syllabus: Unit 6 - image features and matching
    """
    image = np.asarray(img)
    valid_shape = image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] in (1, 3)
    )
    if image.dtype != np.uint8 or not valid_shape or image.size == 0:
        raise ValueError("Expected a nonempty uint8 grayscale or BGR image.")
    return image


def _gray_float(img: NDArray[np.generic]) -> FloatImage:
    """Convert a uint8 grayscale or BGR image to float32 grayscale.

    Args:
        img: Nonempty uint8 grayscale or BGR image.

    Returns:
        Grayscale float32 image in the original intensity range.

    Syllabus: Unit 6 - image features and matching
    """
    image = _validate_image(img)
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def _validate_levels(levels: int) -> int:
    """Validate a positive pyramid level count.

    Args:
        levels: Number of pyramid levels.

    Returns:
        The validated number of levels.

    Syllabus: Unit 6 - image pyramids
    """
    if not isinstance(levels, (int, np.integer)) or levels <= 0:
        raise ValueError("levels must be a positive integer.")
    return int(levels)


def harris_scratch(
    img: NDArray[np.generic], k: float = 0.04, sigma: float = 1.5
) -> FloatImage:
    """Calculate Harris corner response using Sobel derivatives and Gaussian sums.

    Args:
        img: uint8 grayscale or BGR image.
        k: Harris sensitivity coefficient, typically in [0.04, 0.06].
        sigma: Positive standard deviation of the second-moment Gaussian window.

    Returns:
        Float32 response map R = det(M) - k * trace(M)^2.

    Syllabus: Unit 6 - Harris corner detection
    """
    gray = _gray_float(img)
    if not np.isfinite(k) or not 0 < k < 0.25:
        raise ValueError("k must be finite and between zero and 0.25.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and greater than zero.")
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    # The Gaussian window integrates local gradient products into the structure tensor.
    ixx = cv2.GaussianBlur(ix * ix, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    iyy = cv2.GaussianBlur(iy * iy, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    ixy = cv2.GaussianBlur(ix * iy, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    determinant = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    return (determinant - float(k) * trace * trace).astype(np.float32, copy=False)


def non_max_suppression(
    response: NDArray[np.generic], thr_rel: float, min_dist: int
) -> PointArray:
    """Return spatially separated response maxima in (x, y) order.

    Args:
        response: Finite two-dimensional corner-response image.
        thr_rel: Relative threshold in [0, 1] of the strongest positive response.
        min_dist: Positive minimum Euclidean distance in pixels between points.

    Returns:
        Float32 array of keypoint coordinates with shape (N, 2), sorted by
        descending response. Empty output has shape (0, 2).

    Syllabus: Unit 6 - feature point selection
    """
    values = np.asarray(response)
    if (
        values.ndim != 2
        or values.size == 0
        or not np.issubdtype(values.dtype, np.number)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("response must be a nonempty finite 2D numeric array.")
    if not np.isfinite(thr_rel) or not 0 <= thr_rel <= 1:
        raise ValueError("thr_rel must be finite and in [0, 1].")
    if not isinstance(min_dist, (int, np.integer)) or min_dist <= 0:
        raise ValueError("min_dist must be a positive integer.")
    maximum = float(np.max(values))
    if maximum <= 0:
        return np.empty((0, 2), dtype=np.float32)
    threshold = maximum * float(thr_rel)
    local_max = cv2.dilate(
        values.astype(np.float32),
        np.ones((2 * int(min_dist) + 1, 2 * int(min_dist) + 1), dtype=np.uint8),
    )
    rows, columns = np.nonzero((values >= local_max) & (values >= threshold))
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float32)
    scores = values[rows, columns]
    order = np.argsort(scores, kind="stable")[::-1]
    selected: list[tuple[int, int]] = []
    squared_distance = int(min_dist) ** 2
    for index in order:
        point = (int(columns[index]), int(rows[index]))
        if all(
            (point[0] - x) ** 2 + (point[1] - y) ** 2 >= squared_distance
            for x, y in selected
        ):
            selected.append(point)
    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def harris_cv(
    img: NDArray[np.generic], block_size: int = 2, aperture_size: int = 3, k: float = 0.04
) -> FloatImage:
    """Calculate a Harris response map with OpenCV.

    Args:
        img: uint8 grayscale or BGR image.
        block_size: Positive neighborhood side length for the covariance matrix.
        aperture_size: Odd Sobel aperture size, 3, 5, or 7.
        k: Harris sensitivity coefficient.

    Returns:
        Float32 corner response map.

    Syllabus: Unit 6 - Harris corner detection
    """
    gray = _gray_float(img)
    if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if aperture_size not in (3, 5, 7):
        raise ValueError("aperture_size must be 3, 5, or 7.")
    if not np.isfinite(k) or not 0 < k < 0.25:
        raise ValueError("k must be finite and between zero and 0.25.")
    response = cv2.cornerHarris(
        gray,
        blockSize=int(block_size),
        ksize=int(aperture_size),
        k=float(k),
    )
    return response.astype(np.float32, copy=False)


def shi_tomasi(
    img: NDArray[np.generic],
    max_corners: int = 100,
    quality_level: float = 0.01,
    min_distance: float = 10.0,
) -> PointArray:
    """Detect Shi-Tomasi points with OpenCV goodFeaturesToTrack.

    Args:
        img: uint8 grayscale or BGR image.
        max_corners: Maximum number of points; zero requests all valid corners.
        quality_level: Positive relative quality cutoff in (0, 1].
        min_distance: Nonnegative minimum distance between detected corners.

    Returns:
        Float32 array of points in (x, y) order, shape (N, 2).

    Syllabus: Unit 6 - Shi-Tomasi feature detection
    """
    image = _validate_image(img)
    if not isinstance(max_corners, (int, np.integer)) or max_corners < 0:
        raise ValueError("max_corners must be a nonnegative integer.")
    if not np.isfinite(quality_level) or not 0 < quality_level <= 1:
        raise ValueError("quality_level must be in (0, 1].")
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError("min_distance must be finite and nonnegative.")
    gray = image if image.ndim == 2 else (
        image[:, :, 0] if image.shape[2] == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(max_corners),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance),
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32, copy=False)


def hessian_blobs(
    img: NDArray[np.generic],
    min_sigma: float = 1.0,
    max_sigma: float = 30.0,
    num_sigma: int = 10,
    threshold: float = 0.01,
) -> NDArray[np.float32]:
    """Detect determinant-of-Hessian blobs with scikit-image blob_doh.

    Args:
        img: uint8 grayscale or BGR image.
        min_sigma: Smallest positive scale to search.
        max_sigma: Largest scale to search, no smaller than min_sigma.
        num_sigma: Number of scales to examine.
        threshold: Positive response threshold for blob detection.

    Returns:
        Float32 array of blobs as (x, y, sigma).

    Syllabus: Unit 6 - Hessian blob detection
    """
    gray = _gray_float(img) / 255.0
    if (
        not np.isfinite(min_sigma)
        or not np.isfinite(max_sigma)
        or min_sigma <= 0
        or max_sigma < min_sigma
    ):
        raise ValueError("sigma bounds must be positive and max_sigma >= min_sigma.")
    if not isinstance(num_sigma, (int, np.integer)) or num_sigma <= 0:
        raise ValueError("num_sigma must be a positive integer.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive.")
    blobs = blob_doh(
        gray,
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        num_sigma=int(num_sigma),
        threshold=float(threshold),
    )
    if blobs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return blobs[:, [1, 0, 2]].astype(np.float32, copy=False)


def build_gaussian_pyramid(
    img: NDArray[np.generic], levels: int
) -> Pyramid:
    """Build a Gaussian pyramid containing the source and successive pyrDowns.

    Args:
        img: uint8 grayscale or BGR image.
        levels: Number of returned levels, including the source image.

    Returns:
        List of float32 pyramid levels, each with up to half the previous width
        and height, rounded upward for odd dimensions.

    Syllabus: Unit 6 - image pyramids
    """
    gray = _gray_float(img)
    count = _validate_levels(levels)
    pyramid: Pyramid = [gray]
    for _ in range(1, count):
        previous = pyramid[-1]
        if min(previous.shape) <= 1:
            break
        pyramid.append(cv2.pyrDown(previous))
    return pyramid


def build_laplacian_pyramid(
    img: NDArray[np.generic], levels: int
) -> Pyramid:
    """Build Laplacian bands and a coarsest Gaussian residual.

    Args:
        img: uint8 grayscale or BGR image.
        levels: Number of returned levels including the coarsest residual.

    Returns:
        Float32 arrays: one Laplacian band per Gaussian downsampling step,
        followed by the residual. The list has at most ``levels`` items.

    Syllabus: Unit 6 - image pyramids
    """
    gaussian = build_gaussian_pyramid(img, levels)
    laplacian: Pyramid = []
    for current, next_level in zip(gaussian[:-1], gaussian[1:]):
        upsampled = cv2.pyrUp(next_level, dstsize=(current.shape[1], current.shape[0]))
        laplacian.append((current - upsampled).astype(np.float32, copy=False))
    laplacian.append(gaussian[-1])
    return laplacian


def reconstruct_from_laplacian(pyramid: Sequence[NDArray[np.generic]]) -> FloatImage:
    """Reconstruct a Gaussian image from Laplacian bands and residual.

    Args:
        pyramid: Nonempty sequence returned by build_laplacian_pyramid.

    Returns:
        Float32 reconstruction at the original (finest-band) dimensions.

    Syllabus: Unit 6 - image pyramids
    """
    if len(pyramid) == 0:
        raise ValueError("pyramid must contain at least one level.")
    levels: list[FloatImage] = []
    for level in pyramid:
        values = np.asarray(level)
        if (
            values.ndim != 2
            or values.size == 0
            or not np.issubdtype(values.dtype, np.number)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("each pyramid level must be a finite nonempty 2D array.")
        levels.append(values.astype(np.float32, copy=True))
    reconstruction = levels[-1]
    for band in reversed(levels[:-1]):
        if band.shape[0] < reconstruction.shape[0] or band.shape[1] < reconstruction.shape[1]:
            raise ValueError("pyramid band dimensions must not be smaller than residual.")
        reconstruction = cv2.pyrUp(
            reconstruction, dstsize=(band.shape[1], band.shape[0])
        ) + band
    return reconstruction.astype(np.float32, copy=False)


def build_scale_space(
    img: NDArray[np.generic],
    octaves: int = 3,
    scales: int = 4,
    sigma0: float = 1.6,
) -> ScaleSpace:
    """Build Gaussian scale-space levels arranged by octave and scale.

    Args:
        img: uint8 grayscale or BGR image.
        octaves: Positive number of octave image sizes to create.
        scales: Positive number of Gaussian images per octave.
        sigma0: Positive base blur for the first scale in each octave.

    Returns:
        Nested list ``space[octave][scale]`` of float32 grayscale images.
        Each octave is half-resolution relative to the previous one.

    Syllabus: Unit 6 - scale-space feature detection
    """
    gray = _gray_float(img)
    if not isinstance(octaves, (int, np.integer)) or octaves <= 0:
        raise ValueError("octaves must be a positive integer.")
    if not isinstance(scales, (int, np.integer)) or scales < 3:
        raise ValueError("scales must be an integer of at least three.")
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("sigma0 must be finite and positive.")
    scale_space: ScaleSpace = []
    octave_base = gray
    scale_factor = 2.0 ** (1.0 / int(scales))
    for octave_index in range(int(octaves)):
        if min(octave_base.shape) < 8:
            break
        octave_levels: list[FloatImage] = []
        for scale_index in range(int(scales)):
            sigma = float(sigma0) * (scale_factor**scale_index)
            # Blur each resized octave base by its target absolute Gaussian sigma.
            octave_levels.append(
                cv2.GaussianBlur(
                    octave_base,
                    (0, 0),
                    sigmaX=sigma,
                    sigmaY=sigma,
                ).astype(np.float32, copy=False)
            )
        scale_space.append(octave_levels)
        if octave_index + 1 < int(octaves):
            octave_base = cv2.pyrDown(octave_levels[0])
    return scale_space


def build_dog(scale_space: ScaleSpace) -> ScaleSpace:
    """Create Difference-of-Gaussian images from adjacent scale-space levels.

    Args:
        scale_space: Nested octave/scale list from build_scale_space.

    Returns:
        Nested list with one fewer image per octave. DoG approximates the
        scale-normalized Laplacian of Gaussian (LoG).

    Syllabus: Unit 6 - scale-space feature detection
    """
    if len(scale_space) == 0 or any(len(octave) < 2 for octave in scale_space):
        raise ValueError("scale_space must contain octaves with at least two scales.")
    dog: ScaleSpace = []
    for octave in scale_space:
        differences: list[FloatImage] = []
        for lower, upper in zip(octave[:-1], octave[1:]):
            first = np.asarray(lower, dtype=np.float32)
            second = np.asarray(upper, dtype=np.float32)
            if first.shape != second.shape or first.ndim != 2:
                raise ValueError("adjacent scale-space images must have equal 2D shapes.")
            differences.append((second - first).astype(np.float32, copy=False))
        dog.append(differences)
    return dog


def find_dog_extrema(dog: ScaleSpace) -> NDArray[np.int32]:
    """Find strict local DoG extrema by comparing each sample to 26 neighbors.

    This is the first stage of SIFT keypoint detection: DoG approximates LoG,
    and extrema across position and scale become candidate keypoints.

    Args:
        dog: Nested octave list of at least three same-size DoG images.

    Returns:
        int32 keypoint coordinates with rows (x, y, octave, scale).

    Syllabus: Unit 6 - scale-space feature detection
    """
    keypoints: list[tuple[int, int, int, int]] = []
    for octave_index, octave in enumerate(dog):
        if len(octave) < 3:
            continue
        images = [np.asarray(level, dtype=np.float32) for level in octave]
        shape = images[0].shape
        if any(image.ndim != 2 or image.shape != shape for image in images):
            raise ValueError("each DoG octave must contain equal-size 2D images.")
        stack = np.stack(images)
        for scale_index in range(1, len(images) - 1):
            current = stack[scale_index]
            neighborhood_max = np.full(current.shape, -np.inf, dtype=np.float32)
            neighborhood_min = np.full(current.shape, np.inf, dtype=np.float32)
            for scale_neighbor in range(scale_index - 1, scale_index + 2):
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        if scale_neighbor == scale_index and row_offset == 0 and column_offset == 0:
                            continue
                        shifted = np.roll(
                            np.roll(stack[scale_neighbor], row_offset, axis=0),
                            column_offset,
                            axis=1,
                        )
                        neighborhood_max = np.maximum(neighborhood_max, shifted)
                        neighborhood_min = np.minimum(neighborhood_min, shifted)
            candidates = (
                (current > neighborhood_max)
                | (current < neighborhood_min)
            )
            candidates[:1, :] = False
            candidates[-1:, :] = False
            candidates[:, :1] = False
            candidates[:, -1:] = False
            rows, columns = np.nonzero(candidates)
            keypoints.extend(
                (int(column), int(row), octave_index, scale_index)
                for row, column in zip(rows, columns)
            )
    if not keypoints:
        return np.empty((0, 4), dtype=np.int32)
    return np.asarray(keypoints, dtype=np.int32).reshape(-1, 4)


def sift_detect(
    img: NDArray[np.generic],
    nfeatures: int = 0,
) -> tuple[list[cv2.KeyPoint], NDArray[np.float32] | None]:
    """Detect and describe local features using OpenCV SIFT.

    Args:
        img: uint8 grayscale or BGR image.
        nfeatures: Maximum requested feature count; zero lets SIFT choose.

    Returns:
        SIFT keypoints and a float32 descriptor matrix, or None if no features.

    Syllabus: Unit 6 - SIFT feature detection and description
    """
    image = _validate_image(img)
    if not isinstance(nfeatures, (int, np.integer)) or nfeatures < 0:
        raise ValueError("nfeatures must be a nonnegative integer.")
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide SIFT.")
    detector = cv2.SIFT_create(nfeatures=int(nfeatures))
    keypoints, descriptors = detector.detectAndCompute(image, None)
    if descriptors is not None:
        descriptors = descriptors.astype(np.float32, copy=False)
    return keypoints, descriptors


def orb_detect(
    img: NDArray[np.generic],
    nfeatures: int = 500,
) -> tuple[list[cv2.KeyPoint], NDArray[np.uint8] | None]:
    """Detect and describe local features using fast binary ORB descriptors.

    Args:
        img: uint8 grayscale or BGR image.
        nfeatures: Positive maximum feature count.

    Returns:
        ORB keypoints and uint8 descriptor matrix, or None if no features.

    Syllabus: Unit 6 - ORB feature detection and description
    """
    image = _validate_image(img)
    if not isinstance(nfeatures, (int, np.integer)) or nfeatures <= 0:
        raise ValueError("nfeatures must be a positive integer.")
    detector = cv2.ORB_create(nfeatures=int(nfeatures))
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors


def draw_keypoints(
    img: NDArray[np.generic], keypoints: Sequence[cv2.KeyPoint]
) -> Image:
    """Render feature locations on a BGR copy of an image.

    Args:
        img: uint8 grayscale or BGR image.
        keypoints: Sequence of OpenCV keypoints.

    Returns:
        BGR uint8 visualization; the input is never modified.

    Syllabus: Unit 6 - feature visualization
    """
    image = _validate_image(img)
    if image.ndim == 2:
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        color_image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        color_image = image.copy()
    return cv2.drawKeypoints(
        color_image,
        list(keypoints),
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def match_features(
    d1: NDArray[np.generic],
    d2: NDArray[np.generic],
    ratio: float = 0.75,
) -> KeypointMatches:
    """Match descriptor rows with brute-force k-NN and Lowe's ratio test.

    Args:
        d1: Nonempty 2D descriptor matrix for the first image.
        d2: Nonempty 2D descriptor matrix for the second image.
        ratio: Strict nearest-to-second-nearest distance ratio in (0, 1).

    Returns:
        List of accepted OpenCV DMatch objects, ordered by descriptor query.

    Syllabus: Unit 6 - feature matching
    """
    first = np.asarray(d1)
    second = np.asarray(d2)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[0] == 0
        or second.shape[0] < 2
        or first.shape[1] != second.shape[1]
        or first.dtype != second.dtype
    ):
        raise ValueError("descriptor arrays must be compatible 2D arrays; d2 needs >=2 rows.")
    if not np.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("ratio must be finite and in (0, 1).")
    if first.dtype == np.uint8:
        norm = cv2.NORM_HAMMING
    elif np.issubdtype(first.dtype, np.floating):
        norm = cv2.NORM_L2
    else:
        raise ValueError("descriptors must have uint8 or floating dtype.")
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    neighbors = matcher.knnMatch(first, second, k=2)
    return [
        pair[0]
        for pair in neighbors
        if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance
    ]


def estimate_homography(
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    ransac_threshold: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    """Estimate a RANSAC homography from matched OpenCV keypoints.

    Args:
        kp1: Keypoints detected in the source image.
        kp2: Keypoints detected in the destination image.
        matches: Descriptor matches linking indices in kp1 to kp2.
        ransac_threshold: Positive maximum RANSAC reprojection error in pixels.

    Returns:
        (3x3 float64 homography, uint8 inlier mask).

    Raises:
        ValueError: If fewer than four valid inlier correspondences exist.

    Syllabus: Unit 6 - geometric feature matching
    """
    if not np.isfinite(ransac_threshold) or ransac_threshold <= 0:
        raise ValueError("ransac_threshold must be finite and positive.")
    if len(matches) < 4:
        raise ValueError("at least four matches are required for a homography.")
    source_points: list[tuple[float, float]] = []
    destination_points: list[tuple[float, float]] = []
    for match in matches:
        if not (0 <= match.queryIdx < len(kp1) and 0 <= match.trainIdx < len(kp2)):
            raise ValueError("a match contains an out-of-range keypoint index.")
        source_points.append(kp1[match.queryIdx].pt)
        destination_points.append(kp2[match.trainIdx].pt)
    homography, inlier_mask = cv2.findHomography(
        np.asarray(source_points, dtype=np.float32),
        np.asarray(destination_points, dtype=np.float32),
        cv2.RANSAC,
        float(ransac_threshold),
    )
    if homography is None or inlier_mask is None:
        raise ValueError("RANSAC could not estimate a homography.")
    inliers = inlier_mask.reshape(-1).astype(np.uint8, copy=False)
    if int(inliers.sum()) < 4:
        raise ValueError("RANSAC found fewer than four homography inliers.")
    return homography.astype(np.float64, copy=False), inliers


def align_to_template(
    img: NDArray[np.generic], template: NDArray[np.generic]
) -> tuple[Image, int]:
    """Align an image to a template using SIFT matches and a RANSAC homography.

    Args:
        img: uint8 grayscale or BGR source image to warp.
        template: uint8 grayscale or BGR destination template.

    Returns:
        (aligned BGR uint8 image at template dimensions, RANSAC inlier count).

    Raises:
        ValueError: If reliable descriptors, matches, or a homography are absent.

    Syllabus: Unit 6 - image alignment and registration
    """
    source = _validate_image(img)
    target = _validate_image(template)
    keypoints1, descriptors1 = sift_detect(source)
    keypoints2, descriptors2 = sift_detect(target)
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("SIFT did not find enough features to align the images.")
    matches = match_features(descriptors1, descriptors2)
    homography, inliers = estimate_homography(keypoints1, keypoints2, matches)
    aligned = cv2.warpPerspective(
        source,
        homography,
        (target.shape[1], target.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if aligned.ndim == 2:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    elif aligned.shape[2] == 1:
        aligned = cv2.cvtColor(aligned[:, :, 0], cv2.COLOR_GRAY2BGR)
    return aligned.astype(np.uint8, copy=False), int(inliers.sum())


def compute_hog(
    img: NDArray[np.generic],
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> tuple[NDArray[np.float64], Image]:
    """Compute a HOG descriptor and a uint8 visualization image.

    Args:
        img: uint8 grayscale or BGR image.
        orientations: Positive number of unsigned gradient orientation bins.
        pixels_per_cell: Positive (row, column) cell dimensions.
        cells_per_block: Positive number of cells in each (row, column) block.

    Returns:
        (flattened float64 HOG descriptor, grayscale uint8 visualization).

    Syllabus: Unit 6 - histogram of oriented gradients
    """
    gray = _gray_float(img)
    if not isinstance(orientations, (int, np.integer)) or orientations <= 0:
        raise ValueError("orientations must be a positive integer.")
    if (
        len(pixels_per_cell) != 2
        or len(cells_per_block) != 2
        or any(value <= 0 for value in pixels_per_cell + cells_per_block)
    ):
        raise ValueError("cell and block dimensions must be positive integer pairs.")
    descriptor, visualization = hog(
        gray,
        orientations=int(orientations),
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        feature_vector=True,
    )
    visualization = np.asarray(visualization, dtype=np.float32)
    low, high = float(visualization.min()), float(visualization.max())
    if high > low:
        visualization = (visualization - low) * (255.0 / (high - low))
    else:
        visualization = np.zeros_like(visualization)
    return descriptor.astype(np.float64, copy=False), np.clip(
        np.rint(visualization), 0, 255
    ).astype(np.uint8)

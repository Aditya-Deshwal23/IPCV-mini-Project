"""Interactive geometry and local-feature laboratory.

Syllabus: Unit 4 - geometry and feature detection
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from numpy.typing import NDArray

from docuvision.features import (
    align_to_template,
    build_dog,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    build_scale_space,
    compute_hog,
    draw_keypoints,
    find_dog_extrema,
    harris_scratch,
    hessian_blobs,
    match_features,
    non_max_suppression,
    reconstruct_from_laplacian,
    shi_tomasi,
    sift_detect,
)
from docuvision.geometry import affine_transform, detect_document_corners
from docuvision.ui.common import gray, require_image, syllabus_info


def _small(image: NDArray[np.uint8], side: int = 256) -> NDArray[np.uint8]:
    """Resize large feature examples to the scratch-friendly display bound.

    Args:
        image: Grayscale or BGR uint8 source image.
        side: Maximum allowed output side length.

    Returns:
        Copy of the source or an area-resized image no larger than side pixels.

    Syllabus: Unit 4 - feature detection
    """
    height, width = image.shape[:2]
    scale = min(1.0, side / max(height, width))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def render(image: NDArray[np.generic] | None) -> None:
    """Render affine geometry, corner/blob features, registration, and HOG.

    Args:
        image: Shared grayscale or BGR source image or None.

    Returns:
        None.

    Syllabus: Unit 4 - geometry and feature detection
    """
    syllabus_info("affine geometry, Harris/Hessian, SIFT, pyramids, HOG", "Lab 6")
    source = require_image(image)
    if source is None:
        return
    working = _small(source)
    gray_image = gray(working)
    st.subheader("Geometry transform")
    translate_x = st.slider("Translate x", -50, 50, 0)
    translate_y = st.slider("Translate y", -50, 50, 0)
    angle = st.slider("Rotate (degrees)", -180, 180, 0)
    scale = st.slider("Scale", 0.25, 2.0, 1.0, 0.05)
    shear = st.slider("Shear x", -0.5, 0.5, 0.0, 0.05)

    transformed, matrix = affine_transform(
        gray_image,
        translation=(float(translate_x), float(translate_y)),
        rotation=float(angle),
        scale=float(scale),
        shear=float(shear),
        interpolation=cv2.INTER_LINEAR,
    )
    matrix_column, image_column = st.columns(2)
    matrix_column.code(np.array2string(matrix, precision=3, suppress_small=True))
    image_column.image(transformed, caption="Transformed image", use_container_width=True)
    st.caption("Interpolation comparison is provided by OpenCV's nearest, linear, and cubic sampling.")
    interpolation = st.selectbox("Interpolation", ("Nearest", "Bilinear", "Bicubic"))
    interpolation_map = {"Nearest": cv2.INTER_NEAREST, "Bilinear": cv2.INTER_LINEAR, "Bicubic": cv2.INTER_CUBIC}
    compared, _ = affine_transform(
        gray_image,
        translation=(float(translate_x), float(translate_y)),
        rotation=float(angle),
        scale=float(scale),
        shear=float(shear),
        interpolation=interpolation_map[interpolation],
    )
    st.image(compared, caption=f"{interpolation} interpolation", use_container_width=True)

    st.subheader("Document detection and feature points")
    corner_view = working.copy()
    try:
        corners = detect_document_corners(working)
    except ValueError:
        corners = np.empty((0, 2), dtype=np.float32)
    if len(corners) == 4:
        cv2.polylines(corner_view, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        st.caption("Four document corners detected.")
    else:
        st.caption("No four-corner page boundary was found.")
    harris_response = harris_scratch(gray_image)
    harris_points = non_max_suppression(harris_response, thr_rel=0.03, min_dist=5)[:200]
    harris_keypoints = [cv2.KeyPoint(float(x), float(y), 3) for x, y in harris_points]
    corner_column, harris_column = st.columns(2)
    corner_column.image(corner_view, caption="Document-corner debug view", use_container_width=True)
    harris_column.image(draw_keypoints(gray_image, harris_keypoints), caption=f"Harris ({len(harris_points)})", use_container_width=True)
    max_sigma = max(2.0, min(gray_image.shape) / 4.0)
    blobs = hessian_blobs(gray_image, min_sigma=1, max_sigma=max_sigma, num_sigma=5)
    blob_view = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for x, y, sigma in blobs[:200]:
        cv2.circle(blob_view, (round(float(x)), round(float(y))), max(1, round(float(sigma))), (0, 255, 255), 1)
    st.image(blob_view, caption=f"Hessian blobs ({len(blobs)})", use_container_width=True)
    shi_points = shi_tomasi(gray_image, max_corners=200)
    st.caption(f"Shi-Tomasi points: {len(shi_points)}")
    sift_points, _ = sift_detect(working, nfeatures=250)
    st.image(
        draw_keypoints(working, sift_points),
        caption=f"SIFT keypoints ({len(sift_points)})",
        use_container_width=True,
    )

    st.subheader("SIFT template alignment")
    template_upload = st.file_uploader("Optional template image", type=("png", "jpg", "jpeg", "bmp"), key="feature_template")
    if template_upload is not None:
        template = cv2.imdecode(np.frombuffer(template_upload.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if template is None:
            st.error("Could not decode the template image.")
        else:
            template = _small(template)
            keypoints_a, descriptors_a = sift_detect(working)
            keypoints_b, descriptors_b = sift_detect(template)
            if descriptors_a is None or descriptors_b is None or len(descriptors_b) < 2:
                st.info("Not enough SIFT features were found to match these images.")
            else:
                matches = match_features(descriptors_a, descriptors_b)
                match_image = cv2.drawMatches(working, keypoints_a, template, keypoints_b, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                st.image(match_image, caption=f"Top SIFT matches: {len(matches)}", use_container_width=True)
                try:
                    registered, inlier_count = align_to_template(working, template)
                    if inlier_count:
                        st.image(registered, caption=f"RANSAC template alignment ({inlier_count} inliers)", use_container_width=True)
                    else:
                        st.info("SIFT alignment did not produce a reliable homography.")
                except (ValueError, cv2.error) as error:
                    st.info(f"Template alignment failed: {error}")

    st.subheader("Pyramids, DoG scale space, and HOG")
    levels = st.slider("Pyramid levels", 1, 6, 4)
    pyramid = build_gaussian_pyramid(gray_image, levels)
    laplacian = build_laplacian_pyramid(gray_image, levels)
    reconstructed = np.clip(np.rint(reconstruct_from_laplacian(laplacian)), 0, 255).astype(np.uint8)
    pyramid_level = st.selectbox(
        "Gaussian pyramid level",
        range(len(pyramid)),
        format_func=lambda index: f"Level {index} · {pyramid[index].shape[1]}×{pyramid[index].shape[0]}",
    )
    st.image(
        [pyramid[pyramid_level].astype(np.uint8), reconstructed],
        caption=[f"Gaussian level {pyramid_level}", "Laplacian reconstruction"],
        use_container_width=True,
    )
    octave_count = st.slider("DoG octaves", 1, 3, 2)
    if min(gray_image.shape) >= 8:
        safe_octaves = min(octave_count, max(1, int(np.floor(np.log2(min(gray_image.shape)))) - 2))
        scale_space = build_scale_space(gray_image, octaves=safe_octaves, scales=4)
        dog = build_dog(scale_space)
        extrema = find_dog_extrema(dog)
        dog_octave = st.selectbox("DoG octave", range(len(dog)))
        dog_scale = st.selectbox("DoG scale", range(len(dog[dog_octave])))
        dog_display = cv2.normalize(
            dog[dog_octave][dog_scale], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        st.image(dog_display, caption="Selected Difference-of-Gaussian level", use_container_width=True)
        st.caption(f"DoG extrema candidates: {len(extrema)} · DoG approximates LoG and is the first stage of SIFT.")
    else:
        st.info("DoG scale space needs at least an 8×8 image.")
    if min(gray_image.shape) >= 16:
        descriptor, visualization = compute_hog(gray_image)
        st.image(visualization, caption=f"HOG visualization · {len(descriptor)} descriptor values", use_container_width=True)
    else:
        st.info("HOG needs a minimum image size of 16×16; use a larger source.")

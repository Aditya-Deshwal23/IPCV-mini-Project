"""Tests for deterministic evaluation data, metrics, and report generation.

Syllabus: Unit 1 - image processing evaluation
"""

import json
from pathlib import Path

import numpy as np

from docuvision.evaluation.degrade import (
    add_blur,
    add_noise,
    add_shadow,
    jpeg_artifacts,
    random_perspective,
    random_rotation,
    reduce_contrast,
)
from docuvision.evaluation.make_dataset import make_dataset
from docuvision.evaluation.metrics import (
    cer,
    corner_error,
    psnr,
    rms_contrast,
    sharpness,
    ssim,
    wer,
)
from docuvision.evaluation.run_eval import run_evaluation


def test_degradations_are_reproducible_from_seed() -> None:
    """Verify every stochastic degradation returns the same result for a seed.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 2 - reproducible image degradation
    """
    rows, columns = np.indices((90, 110))
    source = ((rows * 3 + columns * 2) % 256).astype(np.uint8)
    background = np.full((130, 150), 45, dtype=np.uint8)

    np.testing.assert_array_equal(
        add_shadow(source, 0.4, "linear", 5),
        add_shadow(source, 0.4, "linear", 5),
    )
    np.testing.assert_array_equal(
        add_shadow(source, 0.4, "radial", 7),
        add_shadow(source, 0.4, "radial", 7),
    )
    np.testing.assert_array_equal(
        add_noise(source, 8, 0.01, 11), add_noise(source, 8, 0.01, 11)
    )
    np.testing.assert_array_equal(
        add_blur(source, seed=13, motion_length=9),
        add_blur(source, seed=13, motion_length=9),
    )
    np.testing.assert_array_equal(
        random_rotation(source, 7, 17), random_rotation(source, 7, 17)
    )
    perspective1, corners1 = random_perspective(source, 0.1, background, 19)
    perspective2, corners2 = random_perspective(source, 0.1, background, 19)
    np.testing.assert_array_equal(perspective1, perspective2)
    np.testing.assert_array_equal(corners1, corners2)
    assert reduce_contrast(source, 0.8).shape == source.shape
    assert jpeg_artifacts(source, 75).shape == source.shape
    assert perspective1.shape == background.shape
    assert corners1.shape == (4, 2)


def test_metrics_handle_identical_images_and_normalize_text() -> None:
    """Check exact-match image metrics, OCR errors, and geometry/focus metrics.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 1 - image quality and OCR metrics
    """
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    corners = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], dtype=np.float32)

    assert np.isinf(psnr(image, image))
    assert ssim(image, image) == 1.0
    assert cer("Hello \n World", "hello world") == 0
    assert wer("Hello \n World", "hello world") == 0
    assert corner_error(corners, corners) == 0
    assert sharpness(image) >= 0
    assert rms_contrast(image) > 0


def test_make_dataset_outputs_two_aligned_sets_and_ground_truth(
    tmp_path: Path,
) -> None:
    """Generate one page twice and verify deterministic files and metadata.

    Args:
        tmp_path: Pytest-managed output root.

    Returns:
        None.

    Syllabus: Unit 1 - reproducible dataset generation
    """
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_dirs = make_dataset(1, first_root, seed=777)
    make_dataset(1, second_root, seed=777)

    clean = first_dirs["clean"] / "page_000.png"
    set_a = first_dirs["set_a"] / "page_000.png"
    set_b = first_dirs["set_b"] / "page_000.png"
    assert clean.exists() and set_a.exists() and set_b.exists()
    assert (first_dirs["clean"] / "page_000.txt").read_text(encoding="utf-8")
    assert (first_dirs["set_b"] / "page_000.json").exists()
    assert clean.read_bytes() == (second_root / "clean" / "page_000.png").read_bytes()
    assert set_a.read_bytes() == (second_root / "SET_A" / "page_000.png").read_bytes()
    corners = json.loads(
        (first_dirs["set_b"] / "page_000.json").read_text(encoding="utf-8")
    )["corners_xy"]
    assert np.asarray(corners).shape == (4, 2)


def test_run_eval_smoke_with_two_pages_and_no_tesseract(
    tmp_path: Path, monkeypatch
) -> None:
    """Run all evaluation groups on two pages and verify saved report artifacts.

    Args:
        tmp_path: Pytest-managed dataset and results directories.
        monkeypatch: Pytest helper to ensure OCR absence is deterministic.

    Returns:
        None.

    Syllabus: Unit 1 - image processing evaluation
    """
    from docuvision.evaluation import run_eval

    monkeypatch.setattr(run_eval, "tesseract_available", lambda: False)
    artifacts = run_evaluation(
        n=2,
        dataset_dir=tmp_path / "dataset",
        results_dir=tmp_path / "results",
        seed=404,
    )

    expected = {
        "set_a_metrics",
        "set_a_summary",
        "set_b_metrics",
        "set_b_summary",
        "ablation_metrics",
        "compression_codecs",
        "rate_distortion",
        "summary",
    }
    assert expected <= set(artifacts)
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts.values())
    import pandas as pd

    set_a = pd.read_csv(artifacts["set_a_metrics"])
    ablation = pd.read_csv(artifacts["ablation_metrics"])
    compression = pd.read_csv(artifacts["compression_codecs"])
    curve = pd.read_csv(artifacts["rate_distortion"])
    assert len(set_a) == 2 * 4
    assert len(ablation) == 2 * 6
    assert {"raw", "otsu_only", "adaptive_only", "full_pipeline"} == set(
        set_a["method"]
    )
    assert set(compression["codec"].str.lower())
    assert {"wht", "dct"} == set(curve["transform"])
    assert (compression["compressed_bytes"] > 0).all()
    assert (curve["CR"] > 0).all()

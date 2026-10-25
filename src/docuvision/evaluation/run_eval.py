"""Reproducible evaluation runner that writes measured results to CSV.

Syllabus: Unit 1 - image processing evaluation
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from docuvision.compression import block_transform_compress, compare_codecs
from docuvision.config import PipelineConfig
from docuvision.evaluation.make_dataset import make_dataset
from docuvision.evaluation.metrics import (
    cer as character_error_rate,
)
from docuvision.evaluation.metrics import (
    corner_error,
    psnr,
    sharpness,
    ssim,
    wer,
)
from docuvision.ocr import run_ocr, tesseract_available
from docuvision.pipeline import run_pipeline
from docuvision.segmentation import adaptive_threshold, global_threshold, otsu_cv


def _load_gray(path: Path) -> NDArray[np.uint8]:
    """Read an evaluation image as grayscale uint8."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not read evaluation image: {path}")
    return image


def _pipeline_config(
    method: str = "full",
    disabled_steps: tuple[str, ...] = (),
    allow_geometry: bool = False,
) -> PipelineConfig:
    """Construct a consistent evaluation config with selectable ablations."""
    baseline_disabled = (
        "document_rectification",
        "layout",
        "ocr",
    )
    if not allow_geometry:
        baseline_disabled += ("deskew",)
    if method == "otsu":
        return PipelineConfig.preset("baseline_otsu").replace(
            work_size=900,
            disabled_steps=PipelineConfig.preset("baseline_otsu").disabled_steps
            + tuple(step for step in disabled_steps if step != "binarize"),
        )
    if method == "adaptive":
        return PipelineConfig.preset("baseline_adaptive").replace(
            work_size=900,
            disabled_steps=PipelineConfig.preset("baseline_adaptive").disabled_steps
            + tuple(step for step in disabled_steps if step != "binarize"),
        )
    return PipelineConfig.preset("balanced").replace(
        work_size=900,
        disabled_steps=baseline_disabled + disabled_steps,
    )


def _method_image(
    path: Path,
    method: str,
    image: NDArray[np.uint8],
    enable_ocr: bool,
) -> tuple[NDArray[np.uint8], str, float, NDArray[np.uint8] | None]:
    """Run one SET_A method and return grayscale/binary result, OCR text, time."""
    started = time.perf_counter()
    binary: NDArray[np.uint8] | None = None
    if method == "raw":
        output = image.copy()
        text = run_ocr(output)[0] if enable_ocr else ""
    elif method == "otsu_only":
        output = global_threshold(image, otsu_cv(image))
        text = run_ocr(output)[0] if enable_ocr else ""
    elif method == "adaptive_only":
        block = max(3, min(31, (min(image.shape) // 24) | 1))
        output = adaptive_threshold(image, "gaussian", block, C=2.0)
        text = run_ocr(output)[0] if enable_ocr else ""
    elif method == "full_pipeline":
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise OSError(f"could not read evaluation image: {path}")
        result = run_pipeline(source, _pipeline_config())
        output = result.final_gray
        binary = result.final_binary
        text = run_ocr(output)[0] if enable_ocr else ""
    else:
        raise ValueError(f"unknown evaluation method: {method}")
    return output, text, time.perf_counter() - started, binary


def _group_metrics(
    rows: list[dict[str, Any]], group_column: str, metric_columns: list[str]
) -> pd.DataFrame:
    """Aggregate sample metrics using pandas mean and sample standard deviation."""
    frame = pd.DataFrame(rows)
    aggregates: list[dict[str, Any]] = []
    for group, group_frame in frame.groupby(group_column, sort=False):
        for metric in metric_columns:
            values = pd.to_numeric(group_frame[metric], errors="coerce").dropna()
            if values.empty:
                continue
            aggregates.append(
                {
                    group_column: group,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(values.size),
                }
            )
    return pd.DataFrame(
        aggregates, columns=[group_column, "metric", "mean", "std", "n"]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a self-contained Markdown table."""
    if frame.empty:
        return "_No measurements available._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, (float, np.floating)):
                cells.append("∞" if math.isinf(float(value)) else f"{float(value):.4f}")
            else:
                cells.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _save_quality_chart(summary: pd.DataFrame, path: Path) -> None:
    """Plot PSNR and SSIM means with measured standard-deviation error bars."""
    methods = list(summary["method"].drop_duplicates())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, label in zip(axes, ("psnr", "ssim"), ("PSNR (dB)", "SSIM")):
        subset = summary[summary["metric"] == metric].set_index("method")
        means = [float(subset.loc[item, "mean"]) for item in methods]
        errors = [float(subset.loc[item, "std"]) for item in methods]
        axis.bar(methods, means, yerr=errors, capsize=4, color="#3878a8")
        axis.set_title(label)
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_compression_chart(curve: pd.DataFrame, path: Path) -> None:
    """Plot measured mean PSNR against actual compression ratio."""
    figure, axis = plt.subplots(figsize=(7, 5))
    for transform, values in curve.groupby("transform"):
        grouped = values.groupby("parameter", as_index=False).agg(
            CR=("CR", "mean"),
            PSNR=("PSNR", "mean"),
            PSNR_std=("PSNR", "std"),
        )
        axis.errorbar(
            grouped["CR"],
            grouped["PSNR"],
            yerr=grouped["PSNR_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=transform.upper(),
        )
    axis.set_xlabel("Compression ratio (raw bytes / encoded bytes)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Measured block-transform rate-distortion")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_evaluation(
    n: int = 30,
    dataset_dir: str | Path = "data/evaluation",
    results_dir: str | Path = "results",
    seed: int = 2026,
) -> dict[str, Path]:
    """Generate datasets, evaluate methods, and write measured report artifacts.

    Args:
        n: Number of deterministic pages per dataset.
        dataset_dir: Dataset output location.
        results_dir: Directory for CSV, chart, and Markdown outputs.
        seed: Root dataset seed.

    Returns:
        Mapping of generated artifact names to output paths.

    Syllabus: Unit 1 - image processing evaluation
    """
    if not isinstance(n, int) or not 1 <= n <= 1000:
        raise ValueError("n must be an integer from 1 through 1000.")
    dataset_root = Path(dataset_dir)
    result_root = Path(results_dir)
    result_root.mkdir(parents=True, exist_ok=True)
    directories = make_dataset(n, dataset_root, seed)
    ocr_enabled = tesseract_available()
    print(f"Evaluating {n} pages per set; Tesseract available: {ocr_enabled}")

    methods = ("raw", "otsu_only", "adaptive_only", "full_pipeline")
    set_a_rows: list[dict[str, Any]] = []
    enhanced_binary_pages: list[tuple[str, NDArray[np.uint8], NDArray[np.uint8]]] = []
    for page_index in range(n):
        stem = f"page_{page_index:03d}"
        clean_path = directories["clean"] / f"{stem}.png"
        degraded_path = directories["set_a"] / f"{stem}.png"
        clean = _load_gray(clean_path)
        degraded = _load_gray(degraded_path)
        truth = (directories["clean"] / f"{stem}.txt").read_text(encoding="utf-8")
        for method in methods:
            output, recognized, elapsed, processed_binary = _method_image(
                degraded_path, method, degraded, ocr_enabled
            )
            if output.shape != clean.shape:
                output = cv2.resize(
                    output, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA
                )
                if processed_binary is not None:
                    processed_binary = cv2.resize(
                        processed_binary,
                        (clean.shape[1], clean.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
            row: dict[str, Any] = {
                "page": stem,
                "method": method,
                "psnr": psnr(clean, output),
                "ssim": ssim(clean, output),
                "seconds": elapsed,
            }
            if ocr_enabled:
                row["cer"] = character_error_rate(truth, recognized)
                row["wer"] = wer(truth, recognized)
            set_a_rows.append(row)
            if method == "full_pipeline":
                binary = (
                    processed_binary
                    if processed_binary is not None
                    else global_threshold(output, otsu_cv(output))
                )
                enhanced_binary_pages.append((stem, output, binary))
        print(f"SET_A {page_index + 1}/{n}")
    set_a_frame = pd.DataFrame(set_a_rows)
    set_a_frame.to_csv(result_root / "set_a_metrics.csv", index=False)
    metric_names = ["psnr", "ssim"] + (["cer", "wer"] if ocr_enabled else [])
    set_a_summary = _group_metrics(set_a_rows, "method", metric_names)
    set_a_summary.to_csv(result_root / "set_a_summary.csv", index=False)
    _save_quality_chart(set_a_summary, result_root / "set_a_quality.png")

    set_b_rows: list[dict[str, Any]] = []
    for page_index in range(n):
        stem = f"page_{page_index:03d}"
        image_path = directories["set_b"] / f"{stem}.png"
        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise OSError(f"could not read SET_B image: {image_path}")
        truth = (directories["set_b"] / f"{stem}.txt").read_text(encoding="utf-8")
        truth_corners_data = json.loads(
            (directories["set_b"] / f"{stem}.json").read_text(encoding="utf-8")
        )
        true_corners = np.asarray(truth_corners_data["corners_xy"], dtype=np.float32)
        raw_text = run_ocr(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))[0] if ocr_enabled else ""
        processed = run_pipeline(
            source, _pipeline_config(allow_geometry=True)
        )
        record: dict[str, Any] = {
            "page": stem,
            "corner_error_px": (
                corner_error(
                    processed.corners
                    / np.array(
                        [
                            source.shape[1] / max(processed.stages[0].image.shape[1], 1),
                            source.shape[0] / max(processed.stages[0].image.shape[0], 1),
                        ],
                        dtype=np.float32,
                    ),
                    true_corners,
                )
                if processed.corners is not None
                else float("nan")
            ),
            "sharpness": sharpness(source),
        }
        if ocr_enabled:
            pipeline_text = run_ocr(processed.final_gray)[0]
            record["raw_cer"] = character_error_rate(truth, raw_text)
            record["pipeline_cer"] = character_error_rate(truth, pipeline_text)
            record["raw_wer"] = wer(truth, raw_text)
            record["pipeline_wer"] = wer(truth, pipeline_text)
        set_b_rows.append(record)
        print(f"SET_B {page_index + 1}/{n}")
    set_b_frame = pd.DataFrame(set_b_rows)
    set_b_frame.to_csv(result_root / "set_b_metrics.csv", index=False)
    set_b_metric_columns = [
        column
        for column in set_b_frame.columns
        if column not in ("page",) and pd.api.types.is_numeric_dtype(set_b_frame[column])
    ]
    set_b_summary_rows = [
        {
            "metric": metric,
            "mean": float(set_b_frame[metric].mean()),
            "std": float(set_b_frame[metric].std(ddof=1))
            if set_b_frame[metric].count() > 1
            else 0.0,
            "n": int(set_b_frame[metric].count()),
        }
        for metric in set_b_metric_columns
    ]
    set_b_summary = pd.DataFrame(
        set_b_summary_rows, columns=["metric", "mean", "std", "n"]
    )
    set_b_summary.to_csv(result_root / "set_b_summary.csv", index=False)

    ablation_rows: list[dict[str, Any]] = []
    ablation_steps = (
        "illumination_correction",
        "denoise",
        "contrast",
        "sharpen",
        "binarize",
        "morphology_cleanup",
    )
    for page_index in range(n):
        stem = f"page_{page_index:03d}"
        path = directories["set_a"] / f"{stem}.png"
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise OSError(f"could not read SET_A image: {path}")
        clean = _load_gray(directories["clean"] / f"{stem}.png")
        truth = (directories["clean"] / f"{stem}.txt").read_text(encoding="utf-8")
        for step in ablation_steps:
            variant_cfg = _pipeline_config(disabled_steps=(step,))
            if step == "binarize":
                # Binarization is the final representation stage; the controlled
                # ablation replaces adaptive Sauvola with the Otsu baseline.
                variant_cfg = variant_cfg.replace(binarize="otsu")
            result = run_pipeline(source, variant_cfg)
            ablated = result.final_gray
            if ablated.shape != clean.shape:
                ablated = cv2.resize(
                    ablated, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA
                )
            row = {
                "page": stem,
                "omitted_stage": step,
                "psnr": psnr(clean, ablated),
                "ssim": ssim(clean, ablated),
            }
            if ocr_enabled:
                recognized = run_ocr(result.final_binary)[0]
                row["cer"] = character_error_rate(truth, recognized)
                row["wer"] = wer(truth, recognized)
            ablation_rows.append(row)
        print(f"ABLATION {page_index + 1}/{n}")
    ablation_frame = pd.DataFrame(ablation_rows)
    ablation_frame.to_csv(result_root / "ablation_metrics.csv", index=False)
    ablation_summary = _group_metrics(
        ablation_rows, "omitted_stage", metric_names
    )
    ablation_summary.to_csv(result_root / "ablation_summary.csv", index=False)

    compression_rows: list[dict[str, Any]] = []
    rate_distortion_rows: list[dict[str, Any]] = []
    transform_params: tuple[dict[str, int | float | None], ...] = (
        {"keep": 64},
        {"keep": 16},
        {"qstep": 8.0},
        {"qstep": 24.0},
    )
    for stem, enhanced_gray, enhanced_binary in enhanced_binary_pages:
        codec_table = compare_codecs(enhanced_gray, enhanced_binary)
        codec_table.insert(0, "page", stem)
        compression_rows.extend(codec_table.to_dict(orient="records"))
        # A moderate thumbnail bounds block-transform runtime on report generation.
        thumbnail = cv2.resize(
            enhanced_gray, (128, 128), interpolation=cv2.INTER_AREA
        )
        for transform in ("wht", "dct"):
            for parameters in transform_params:
                started = time.perf_counter()
                encoded, reconstructed = block_transform_compress(
                    thumbnail, transform=transform, **parameters
                )
                elapsed = time.perf_counter() - started
                parameter_name, parameter_value = next(iter(parameters.items()))
                rate_distortion_rows.append(
                    {
                        "page": stem,
                        "transform": transform,
                        "parameter": f"{parameter_name}={parameter_value}",
                        "CR": float(thumbnail.nbytes / len(encoded)),
                        "PSNR": psnr(thumbnail, reconstructed),
                        "SSIM": ssim(thumbnail, reconstructed),
                        "encode_seconds": elapsed,
                        "compressed_bytes": len(encoded),
                    }
                )
        print(f"COMPRESSION {stem}")
    compression_frame = pd.DataFrame(compression_rows)
    compression_frame.to_csv(result_root / "compression_codecs.csv", index=False)
    curve_frame = pd.DataFrame(rate_distortion_rows)
    curve_frame.to_csv(result_root / "rate_distortion.csv", index=False)
    if not curve_frame.empty:
        _save_compression_chart(curve_frame, result_root / "rate_distortion.png")

    compression_summary = (
        compression_frame.groupby("codec", as_index=False)
        .agg(
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            bytes_mean=("compressed_bytes", "mean"),
            lossless_fraction=("lossless", "mean"),
        )
        if not compression_frame.empty
        else pd.DataFrame()
    )
    compression_summary.to_csv(result_root / "compression_summary.csv", index=False)
    curve_summary = (
        curve_frame.groupby(["transform", "parameter"], as_index=False)
        .agg(
            CR_mean=("CR", "mean"),
            PSNR_mean=("PSNR", "mean"),
            PSNR_std=("PSNR", "std"),
            SSIM_mean=("SSIM", "mean"),
        )
        if not curve_frame.empty
        else pd.DataFrame()
    )
    curve_summary.to_csv(result_root / "rate_distortion_summary.csv", index=False)

    report = [
        "# Reproducible DocuVision Evaluation",
        "",
        f"- Pages per set: {n}",
        f"- Dataset seed: {seed}",
        f"- Tesseract available: {ocr_enabled}",
        "- Image metrics compare each photometric SET_A output with its pixel-aligned clean page.",
        "- Compression ratios use actual encoded byte streams, including codec headers.",
        "",
        "## SET_A quality and OCR metrics (mean and standard deviation)",
        "",
        _markdown_table(set_a_summary),
        "",
        "![SET_A quality](set_a_quality.png)",
        "",
        "## SET_B geometry and OCR",
        "",
        _markdown_table(set_b_summary),
        "",
        "## SET_A stage ablation",
        "",
        _markdown_table(ablation_summary),
        "",
        "## Actual-byte codec comparison",
        "",
        _markdown_table(compression_summary),
        "",
        "## WHT vs DCT rate-distortion",
        "",
        _markdown_table(curve_summary),
        "",
        "![Rate-distortion](rate_distortion.png)",
        "",
        "Raw per-page data is available in the adjacent CSV files.",
        "",
    ]
    summary_path = result_root / "summary.md"
    summary_path.write_text("\n".join(report), encoding="utf-8")
    return {
        "set_a_metrics": result_root / "set_a_metrics.csv",
        "set_a_summary": result_root / "set_a_summary.csv",
        "set_b_metrics": result_root / "set_b_metrics.csv",
        "set_b_summary": result_root / "set_b_summary.csv",
        "ablation_metrics": result_root / "ablation_metrics.csv",
        "compression_codecs": result_root / "compression_codecs.csv",
        "rate_distortion": result_root / "rate_distortion.csv",
        "summary": summary_path,
    }


def main(argv: list[str] | None = None) -> int:
    """Run evaluation from command-line arguments.

    Args:
        argv: Optional argument list; None parses sys.argv.

    Returns:
        Process exit status, zero on success.

    Syllabus: Unit 1 - image processing evaluation
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    run_evaluation(args.n, args.dataset, args.results, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

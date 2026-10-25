"""Creation of reproducible clean and degraded evaluation datasets.

Syllabus: Unit 1 - image processing evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from docuvision.evaluation.degrade import (
    add_blur,
    add_noise,
    add_shadow,
    random_perspective,
    reduce_contrast,
)
from docuvision.io_utils import save_image

PASSAGES = (
    "A careful reader can find useful details in every printed document. Clear text helps people understand instructions and keep accurate records.",
    "Libraries preserve knowledge and make it available to everyone. Readers can explore history, science, art, and new ideas at their own pace.",
    "Good notes are organized, brief, and easy to review. A clear heading helps readers find the information they need without delay.",
    "Small changes in light can make a page harder to read. Image processing improves contrast while preserving the original words and layout.",
    "A community garden brings neighbors together. People share tools, water the plants, and enjoy fresh vegetables throughout the summer.",
    "The weather report predicts a cool morning followed by bright sunshine. Remember to bring a light coat when you leave home.",
    "Every visitor should check in at the front desk. Please keep this pass visible and return it before leaving the building.",
    "A reliable map includes a title, a scale, and a simple legend. These details help travelers understand the route and its surroundings.",
    "Careful planning saves time during a busy workday. Set priorities, prepare materials early, and leave space for unexpected tasks.",
    "The museum opens at nine and closes at five. Guided tours begin every hour near the main entrance, beside the information counter.",
)


def _font_paths() -> tuple[str, ...]:
    """Return common platform font paths in preferred fallback order."""
    return (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a common TrueType font or Pillow's built-in fallback."""
    for path in _font_paths():
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_paragraph(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap one English passage to the requested pixel width."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_page(index: int, seed: int) -> tuple[NDArray[np.uint8], str]:
    """Render a deterministic 1240x1754 clean document and its ground truth.

    Args:
        index: Nonnegative page index.
        seed: Random seed selecting passages and typography.

    Returns:
        (RGB uint8 clean page, newline-separated ground-truth text).

    Syllabus: Unit 1 - reproducible image dataset generation
    """
    if not isinstance(index, int) or index < 0:
        raise ValueError("index must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    width, height = 1240, 1754
    page = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)
    margin = 92
    title_size = int(rng.integers(36, 49))
    title = f"Document Study {index + 1:02d}"
    title_font = _load_font(title_size)
    draw.text((margin, 96), title, fill=(18, 18, 18), font=title_font)
    title_bbox = draw.textbbox((margin, 96), title, font=title_font)
    draw.line(
        (margin, title_bbox[3] + 28, width - margin, title_bbox[3] + 28),
        fill=(70, 70, 70),
        width=2,
    )
    passages = list(PASSAGES)
    rng.shuffle(passages)
    paragraph_count = int(rng.integers(3, 5))
    selected = passages[:paragraph_count]
    sizes = rng.integers(25, 34, size=paragraph_count)
    y = title_bbox[3] + 70
    text_lines: list[str] = [title]
    for passage, size in zip(selected, sizes):
        font = _load_font(int(size))
        lines = _wrap_paragraph(draw, passage, font, width - 2 * margin)
        for line in lines:
            draw.text((margin, y), line, fill=(25, 25, 25), font=font)
            bounds = draw.textbbox((margin, y), line, font=font)
            y += (bounds[3] - bounds[1]) + 11
        text_lines.append(passage)
        y += int(rng.integers(24, 38))
    return np.asarray(page, dtype=np.uint8), "\n\n".join(text_lines)


def _make_background(seed: int, size: tuple[int, int]) -> NDArray[np.uint8]:
    """Create a deterministic dark textured BGR background canvas."""
    height, width = size
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 7.0, (height, width)).astype(np.float32)
    base = np.full((height, width), 48.0, dtype=np.float32) + noise
    texture = cv2.GaussianBlur(base, (0, 0), 5)
    canvas = np.clip(texture, 20, 80).astype(np.uint8)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _rotate_with_corners(
    img: NDArray[np.uint8],
    corners: NDArray[np.float32],
    max_angle: float,
    seed: int,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Rotate a canvas and transform its true quadrilateral coordinates."""
    angle = float(np.random.default_rng(seed).uniform(-max_angle, max_angle))
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    homogeneous = np.column_stack((corners, np.ones(4, dtype=np.float32)))
    transformed_corners = (matrix @ homogeneous.T).T.astype(np.float32)
    return rotated, transformed_corners


def make_dataset(
    n: int = 30,
    output_dir: str | Path = "data/evaluation",
    seed: int = 2026,
) -> dict[str, Path]:
    """Generate clean pages and aligned SET_A / geometric SET_B variants.

    Args:
        n: Number of pages per dataset, from one through one thousand.
        output_dir: Destination directory, created if needed.
        seed: Root seed; each page receives deterministic independent sub-seeds.

    Returns:
        Mapping of ``clean``, ``set_a``, and ``set_b`` to their directory paths.
        Every image has a matching .txt ground-truth file. SET_B also has a
        matching .json file containing true corners.

    Syllabus: Unit 1 - reproducible image dataset generation
    """
    if not isinstance(n, int) or not 1 <= n <= 1000:
        raise ValueError("n must be an integer from 1 through 1000.")
    root = Path(output_dir)
    directories = {
        "clean": root / "clean",
        "set_a": root / "SET_A",
        "set_b": root / "SET_B",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(n)
    manifest: list[dict[str, Any]] = []
    for index, child in enumerate(children):
        page_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        page_rgb, truth = render_page(index, page_seed)
        page_bgr = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR)
        stem = f"page_{index:03d}"
        clean_path = directories["clean"] / f"{stem}.png"
        save_image(clean_path, page_bgr)
        clean_text_path = directories["clean"] / f"{stem}.txt"
        clean_text_path.write_text(truth, encoding="utf-8")

        # SET_A deliberately changes only pixel values, preserving clean alignment.
        photometric = add_shadow(page_bgr, 0.35, "linear", page_seed + 1)
        photometric = add_noise(photometric, sigma=5.0, sp_prob=0.001, seed=page_seed + 2)
        photometric = add_blur(photometric, seed=page_seed + 3, sigma=0.55)
        photometric = reduce_contrast(photometric, alpha=0.78)
        set_a_path = directories["set_a"] / f"{stem}.png"
        save_image(set_a_path, photometric)
        (directories["set_a"] / f"{stem}.txt").write_text(truth, encoding="utf-8")

        # The larger textured canvas leaves room for projective and rotational motion.
        background = _make_background(page_seed + 4, (2050, 1550))
        perspective, true_corners = random_perspective(
            photometric,
            max_shift=0.045,
            bg=background,
            seed=page_seed + 5,
        )
        perspective, true_corners = _rotate_with_corners(
            perspective, true_corners, max_angle=4.0, seed=page_seed + 6
        )
        full_path = directories["set_b"] / f"{stem}.png"
        save_image(full_path, perspective)
        (directories["set_b"] / f"{stem}.txt").write_text(truth, encoding="utf-8")
        corners_path = directories["set_b"] / f"{stem}.json"
        corners_path.write_text(
            json.dumps(
                {"corners_xy": true_corners.round(4).tolist()},
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest.append(
            {
                "stem": stem,
                "seed": page_seed,
                "clean": str(clean_path.relative_to(root)),
                "set_a": str(set_a_path.relative_to(root)),
                "set_b": str(full_path.relative_to(root)),
                "ground_truth": truth,
            }
        )
        print(f"Generated {index + 1}/{n}: {stem}")
    (root / "manifest.json").write_text(
        json.dumps({"seed": seed, "count": n, "pages": manifest}, indent=2),
        encoding="utf-8",
    )
    return directories


def main(argv: list[str] | None = None) -> int:
    """Run dataset generation from command-line arguments.

    Args:
        argv: Optional argument list; None parses sys.argv.

    Returns:
        Process exit status, zero on success.

    Syllabus: Unit 1 - reproducible image dataset generation
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    make_dataset(args.n, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

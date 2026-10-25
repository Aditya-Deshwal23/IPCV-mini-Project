"""Optical character recognition adapters and text cleanup.

Syllabus: Unit 9 - character recognition
"""

from __future__ import annotations

import os
import re
from typing import TypeAlias

import cv2
import numpy as np
import pandas as pd
import pytesseract
from numpy.typing import NDArray
from rapidfuzz.distance import Levenshtein

Image: TypeAlias = NDArray[np.uint8]
OCR_COLUMNS = ["text", "left", "top", "width", "height", "confidence"]
_TESSERACT_MISSING = "Tesseract unavailable: install Tesseract OCR or set TESSERACT_CMD."


def _configure_tesseract() -> None:
    """Apply an optional environment-provided Tesseract executable path.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 9 - character recognition
    """
    command = os.environ.get("TESSERACT_CMD")
    if command:
        # pytesseract does not discover a Windows executable path automatically.
        pytesseract.pytesseract.tesseract_cmd = command


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate a nonempty uint8 grayscale, singleton-channel, or BGR image.

    Args:
        img: Image array to validate.

    Returns:
        The original array without mutation.

    Syllabus: Unit 9 - character recognition
    """
    image = np.asarray(img)
    valid_shape = image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] in (1, 3)
    )
    if image.dtype != np.uint8 or not valid_shape or image.size == 0:
        raise ValueError("Expected a nonempty uint8 grayscale or BGR image.")
    return image


def _grayscale(img: Image) -> Image:
    """Convert a validated uint8 image to grayscale."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def tesseract_available() -> bool:
    """Return whether the configured Tesseract executable responds to a version query.

    Args:
        None.

    Returns:
        True if pytesseract can start Tesseract and read its version; otherwise
        False. The TESSERACT_CMD environment variable overrides pytesseract's
        executable path, including on Windows.

    Syllabus: Unit 9 - character recognition
    """
    _configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
    except (pytesseract.TesseractNotFoundError, OSError, RuntimeError):
        return False
    return True


def run_ocr(
    img: NDArray[np.generic], psm: int = 6, lang: str = "eng"
) -> tuple[str, pd.DataFrame]:
    """Recognize text and return word boxes with Tesseract confidence values.

    Args:
        img: uint8 grayscale or BGR image.
        psm: Tesseract page-segmentation mode, from zero through thirteen.
        lang: Tesseract language identifier, such as ``"eng"``.

    Returns:
        (recognized text, DataFrame) where the DataFrame columns are text, left,
        top, width, height, and confidence. If Tesseract is missing, text is an
        explicit unavailable status and the DataFrame is empty.

    Syllabus: Unit 9 - character recognition
    """
    image = _grayscale(_validate_image(img))
    if not isinstance(psm, int) or not 0 <= psm <= 13:
        raise ValueError("psm must be an integer between 0 and 13.")
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError("lang must be a nonempty language identifier.")
    _configure_tesseract()
    empty = pd.DataFrame(
        {
            "text": pd.Series(dtype="object"),
            "left": pd.Series(dtype="int64"),
            "top": pd.Series(dtype="int64"),
            "width": pd.Series(dtype="int64"),
            "height": pd.Series(dtype="int64"),
            "confidence": pd.Series(dtype="float64"),
        }
    )
    if not tesseract_available():
        return _TESSERACT_MISSING, empty
    try:
        config = f"--psm {psm}"
        text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
        raw = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        return _TESSERACT_MISSING, empty
    except pytesseract.TesseractError as error:
        return f"OCR failed: {error}", empty
    words: list[dict[str, object]] = []
    for index, word in enumerate(raw.get("text", [])):
        cleaned = str(word).strip()
        if not cleaned:
            continue
        try:
            confidence = float(raw["conf"][index])
        except (ValueError, TypeError, IndexError):
            confidence = -1.0
        words.append(
            {
                "text": cleaned,
                "left": int(raw["left"][index]),
                "top": int(raw["top"][index]),
                "width": int(raw["width"][index]),
                "height": int(raw["height"][index]),
                "confidence": confidence,
            }
        )
    return text, pd.DataFrame(words, columns=OCR_COLUMNS)


def _normalize_text(text: str) -> str:
    """Lowercase text and collapse all whitespace to one ASCII space."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def cer(ref: str, hyp: str) -> float:
    """Calculate character error rate on normalized text.

    Args:
        ref: Reference transcription.
        hyp: Hypothesis transcription.

    Returns:
        Levenshtein character distance divided by reference character count.
        Empty reference and hypothesis gives zero; nonempty hypothesis against
        empty reference gives infinity.

    Syllabus: Unit 9 - OCR evaluation
    """
    reference = _normalize_text(ref)
    hypothesis = _normalize_text(hyp)
    if not reference:
        return 0.0 if not hypothesis else float("inf")
    return float(Levenshtein.distance(reference, hypothesis) / len(reference))


def wer(ref: str, hyp: str) -> float:
    """Calculate word error rate on normalized, whitespace-tokenized text.

    Args:
        ref: Reference transcription.
        hyp: Hypothesis transcription.

    Returns:
        Word-level Levenshtein distance divided by reference word count.
        Empty reference and hypothesis gives zero; nonempty hypothesis against
        empty reference gives infinity.

    Syllabus: Unit 9 - OCR evaluation
    """
    reference = _normalize_text(ref).split()
    hypothesis = _normalize_text(hyp).split()
    if not reference:
        return 0.0 if not hypothesis else float("inf")
    return float(Levenshtein.distance(reference, hypothesis) / len(reference))

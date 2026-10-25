"""Tests for byte-level lossless and lossy image codecs.

Syllabus: Unit 8 - image compression
"""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from docuvision.compression import (
    average_code_length,
    block_transform_compress,
    block_transform_decompress,
    compare_codecs,
    huffman_code_table,
    huffman_decode,
    huffman_encode,
    igs_quantize,
    lzw_decode,
    lzw_encode,
    rate_distortion_curve,
    rle_bitplanes,
    rle_bitplanes_decode,
    rle_decode,
    rle_encode,
    rle_gray,
    rle_gray_decode,
    shannon_entropy,
)


def _images() -> dict[str, NDArray[np.uint8]]:
    """Construct deterministic random, constant, gradient, and binary page inputs.

    Args:
        None.

    Returns:
        Mapping of synthetic image names to uint8 grayscale arrays.

    Syllabus: Unit 8 - codec test data
    """
    random = np.random.default_rng(812)
    noise = random.integers(0, 256, size=(48, 64), dtype=np.uint8)
    constant = np.full((48, 64), 127, dtype=np.uint8)
    gradient = np.tile(np.arange(64, dtype=np.uint8) * 4, (48, 1))
    page = np.full((128, 160), 255, dtype=np.uint8)
    cv2.rectangle(page, (8, 8), (151, 119), 0, 2)
    for line in range(5):
        y = 25 + line * 18
        cv2.putText(
            page,
            f"TEXT LINE {line}",
            (17, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            0,
            1,
            cv2.LINE_8,
        )
    binary_page = np.where(page < 128, 0, 255).astype(np.uint8)
    return {
        "random": noise,
        "constant": constant,
        "gradient": gradient,
        "binary_page": binary_page,
    }


@pytest.mark.parametrize("name", ["random", "constant", "gradient", "binary_page"])
def test_rle_and_huffman_lossless_roundtrips(name: str) -> None:
    """Assert byte codecs reconstruct each representative image exactly.

    Args:
        name: Synthetic fixture name from _images.

    Returns:
        None.

    Syllabus: Unit 8 - lossless compression
    """
    image = _images()[name]
    gray_stream = rle_gray(image)
    plane_stream = rle_bitplanes(image)
    huffman_stream = huffman_encode(image)

    np.testing.assert_array_equal(rle_gray_decode(gray_stream), image)
    np.testing.assert_array_equal(rle_bitplanes_decode(plane_stream), image)
    np.testing.assert_array_equal(huffman_decode(huffman_stream), image)
    assert isinstance(gray_stream, bytes)
    assert isinstance(huffman_stream, bytes)
    assert len(huffman_stream) > 0


def test_binary_rle_roundtrip_and_ratio_on_synthetic_page() -> None:
    """Verify binary scan run coding is lossless and yields a useful byte ratio.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 8 - binary run-length encoding
    """
    page = _images()["binary_page"]
    encoded = rle_encode(page)

    np.testing.assert_array_equal(rle_decode(encoded), page)
    assert page.nbytes / len(encoded) > 3.0


def test_lzw_roundtrip_and_compression_on_repeated_bytes() -> None:
    """Decode repeated byte data exactly and measure its serialized compression.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 8 - LZW coding
    """
    repeated = np.tile(np.arange(32, dtype=np.uint8), 128)
    encoded = lzw_encode(repeated)
    decoded = lzw_decode(encoded)

    np.testing.assert_array_equal(decoded, repeated)
    assert len(encoded) < repeated.nbytes


def test_lzw_dictionary_reset_preserves_long_stream() -> None:
    """Round-trip a long stream that fills and resets the 12-bit dictionary.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 8 - LZW coding
    """
    random = np.random.default_rng(112)
    source = random.integers(0, 256, size=80_000, dtype=np.uint8)

    np.testing.assert_array_equal(lzw_decode(lzw_encode(source)), source)


def test_huffman_code_length_bounds_and_single_symbol() -> None:
    """Check Huffman expected code length and constant-array edge behavior.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 8 - Huffman coding and information theory
    """
    image = _images()["gradient"]
    entropy = shannon_entropy(image)
    average = average_code_length(image)
    constant = np.full((7, 9), 42, dtype=np.uint8)

    assert entropy <= average < entropy + 1
    np.testing.assert_array_equal(huffman_decode(huffman_encode(constant)), constant)
    assert average_code_length(constant) == 0


def test_huffman_table_contains_canonical_codes_for_frequent_symbols() -> None:
    """Expose byte codes from the same canonical model used by the encoder.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 8 - Huffman coding
    """
    image = np.array([[0, 0, 0, 255, 255, 16]], dtype=np.uint8)
    table = huffman_code_table(image)
    assert list(table.columns) == ["symbol", "frequency", "code", "code_length"]
    assert table.iloc[0]["symbol"] == 0
    assert table.iloc[0]["frequency"] == 3
    assert set(table["code_length"]).issubset({1, 2})


def test_igs_quantization_uses_at_most_requested_gray_levels() -> None:
    """Check IGS output levels and compare its result to plain truncation.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 8 - IGS quantization
    """
    gradient = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
    igs = igs_quantize(gradient, bits=4)
    truncation = (gradient >> 4) << 4
    igs_psnr = _psnr(gradient, igs)
    truncation_psnr = _psnr(gradient, truncation)

    assert np.unique(igs).size <= 16
    assert np.count_nonzero(igs != truncation) > 0
    assert igs_psnr > truncation_psnr


def _psnr(
    reference: NDArray[np.uint8], image: NDArray[np.uint8]
) -> float:
    """Compute peak signal-to-noise ratio for codec assertions.

    Args:
        reference: Original uint8 image.
        image: Reconstructed uint8 image.

    Returns:
        PSNR in decibels.

    Syllabus: Unit 8 - rate-distortion measurement
    """
    mse = np.mean(
        (reference.astype(np.float64) - image.astype(np.float64)) ** 2
    )
    return float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse))


def test_transform_codec_all_coefficients_and_rate_distortion() -> None:
    """Verify full WHT reconstruction, actual bytes, and rate-distortion rows.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 8 - transform coding and rate-distortion
    """
    image = _images()["gradient"]
    encoded, reconstructed = block_transform_compress(image, "wht", keep=64)
    decoded = block_transform_decompress(encoded)
    curve = rate_distortion_curve(
        image,
        "dct",
        [{"keep": 64}, {"keep": 16}, {"qstep": 12.0}],
    )

    assert len(encoded) > 0
    assert np.max(np.abs(reconstructed.astype(np.int16) - image.astype(np.int16))) <= 1
    np.testing.assert_array_equal(decoded, reconstructed)
    assert list(curve.columns) == ["CR", "PSNR", "SSIM"]
    assert len(curve) == 3
    assert np.all(curve["CR"] > 0)


def test_compare_codecs_reports_actual_sizes_and_roundtrip_losslessness() -> None:
    """Check comparison fields use serialized stream lengths and actual decodes.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 8 - codec comparison
    """
    images = _images()
    table = compare_codecs(images["gradient"], images["binary_page"])

    assert {
        "codec",
        "original_bytes",
        "compressed_bytes",
        "ratio",
        "bits_per_pixel",
        "lossless",
        "encode_seconds",
    } <= set(table.columns)
    assert (table["compressed_bytes"] > 0).all()
    assert (table["ratio"] > 0).all()
    assert {"gray_png", "binary_png", "gray_jpeg_90", "binary_jpeg_50"} <= set(
        table["codec"]
    )
    png = table[table["codec"].str.endswith("_png")]
    jpeg = table[table["codec"].str.contains("jpeg")]
    assert png["lossless"].all()
    assert not jpeg["lossless"].all()

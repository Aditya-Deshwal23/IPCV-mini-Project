"""Byte-level lossless and lossy image compression algorithms.

Syllabus: Unit 8 - image compression
"""

from __future__ import annotations

import heapq
import math
import struct
import time
from collections import Counter
from collections.abc import Sequence
from itertools import count
from typing import Literal

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from skimage.metrics import structural_similarity

from docuvision.frequency import iwht2d, walsh_matrix, wht2d

Image = NDArray[np.uint8]
Transform = Literal["wht", "dct"]
CodecBytes = bytes


def _validate_image(img: NDArray[np.generic]) -> Image:
    """Validate a nonempty two-dimensional uint8 image.

    Args:
        img: Grayscale uint8 image.

    Returns:
        Original image array without copying or mutation.

    Syllabus: Unit 8 - image compression
    """
    image = np.asarray(img)
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty two-dimensional uint8 image.")
    return image


def _pack_varint(value: int) -> bytes:
    """Encode a nonnegative integer as an unsigned base-128 varint.

    Args:
        value: Nonnegative integer to encode.

    Returns:
        One or more bytes containing the varint.

    Syllabus: Unit 8 - run-length encoding
    """
    if value < 0:
        raise ValueError("varint values must be nonnegative.")
    packed = bytearray()
    while value >= 0x80:
        packed.append((value & 0x7F) | 0x80)
        value >>= 7
    packed.append(value)
    return bytes(packed)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode one unsigned base-128 varint.

    Args:
        data: Encoded byte stream.
        offset: Starting byte index.

    Returns:
        (decoded integer, first byte index after the integer).

    Raises:
        ValueError: If the stream ends before the varint terminates.

    Syllabus: Unit 8 - run-length encoding
    """
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("varint exceeds supported size.")
    raise ValueError("truncated varint.")


def _runs(values: NDArray[np.uint8]) -> list[tuple[int, int]]:
    """Convert a one-dimensional byte array into value/count runs."""
    if values.size == 0:
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate((np.array([0]), changes))
    ends = np.concatenate((changes, np.array([values.size])))
    return [
        (int(values[start]), int(end - start))
        for start, end in zip(starts, ends)
    ]


def rle_encode(img: NDArray[np.generic]) -> CodecBytes:
    """Encode a binary grayscale image with row-wise run lengths.

    Args:
        img: Nonempty 2D uint8 binary mask; pixels must be either 0 or 255.

    Returns:
        Bytes containing magic, height, width, first pixel value, and varint
        run lengths. Runs continue across row boundaries in row-major order.

    Syllabus: Unit 8 - binary run-length encoding
    """
    image = _validate_image(img)
    if not np.all((image == 0) | (image == 255)):
        raise ValueError("rle_encode accepts only binary pixels with values 0 or 255.")
    flattened = image.ravel()
    runs = _runs(flattened)
    encoded = bytearray(b"RLB1")
    encoded.extend(struct.pack(">II", image.shape[0], image.shape[1]))
    encoded.append(runs[0][0])
    for _, length in runs:
        encoded.extend(_pack_varint(length))
    return bytes(encoded)


def rle_decode(data: bytes) -> Image:
    """Decode the row-wise binary run-length format.

    Args:
        data: Bytes produced by rle_encode.

    Returns:
        Reconstructed uint8 binary image.

    Syllabus: Unit 8 - binary run-length encoding
    """
    if len(data) < 13 or data[:4] != b"RLB1":
        raise ValueError("invalid binary RLE stream.")
    height, width = struct.unpack(">II", data[4:12])
    first = data[12]
    if height <= 0 or width <= 0 or first not in (0, 255):
        raise ValueError("invalid binary RLE header.")
    total = height * width
    values: list[NDArray[np.uint8]] = []
    current = first
    offset = 13
    produced = 0
    while produced < total:
        length, offset = _read_varint(data, offset)
        if length <= 0 or produced + length > total:
            raise ValueError("invalid binary RLE run length.")
        values.append(np.full(length, current, dtype=np.uint8))
        produced += length
        current = 255 - current
    if offset != len(data):
        raise ValueError("binary RLE stream contains trailing bytes.")
    return np.concatenate(values).reshape(height, width)


def rle_gray(img: NDArray[np.generic]) -> CodecBytes:
    """Encode arbitrary grayscale uint8 pixels as value/count RLE pairs.

    Args:
        img: Nonempty 2D uint8 image.

    Returns:
        Bytes containing magic, image dimensions, then byte-value/varint-count
        pairs across row-major pixels.

    Syllabus: Unit 8 - grayscale run-length encoding
    """
    image = _validate_image(img)
    encoded = bytearray(b"RLG1")
    encoded.extend(struct.pack(">II", image.shape[0], image.shape[1]))
    for value, length in _runs(image.ravel()):
        encoded.append(value)
        encoded.extend(_pack_varint(length))
    return bytes(encoded)


def rle_gray_decode(data: bytes) -> Image:
    """Decode value/count grayscale RLE pairs.

    Args:
        data: Bytes produced by rle_gray.

    Returns:
        Reconstructed uint8 grayscale image.

    Syllabus: Unit 8 - grayscale run-length encoding
    """
    if len(data) < 12 or data[:4] != b"RLG1":
        raise ValueError("invalid grayscale RLE stream.")
    height, width = struct.unpack(">II", data[4:12])
    if height <= 0 or width <= 0:
        raise ValueError("invalid grayscale RLE dimensions.")
    total = height * width
    decoded: list[NDArray[np.uint8]] = []
    produced = 0
    offset = 12
    while produced < total:
        if offset >= len(data):
            raise ValueError("truncated grayscale RLE stream.")
        value = data[offset]
        length, offset = _read_varint(data, offset + 1)
        if length <= 0 or produced + length > total:
            raise ValueError("invalid grayscale RLE run length.")
        decoded.append(np.full(length, value, dtype=np.uint8))
        produced += length
    if offset != len(data):
        raise ValueError("grayscale RLE stream contains trailing bytes.")
    return np.concatenate(decoded).reshape(height, width)


def rle_bitplanes(img: NDArray[np.generic]) -> CodecBytes:
    """Encode each image bit plane independently with MSB-first binary RLE.

    Args:
        img: Nonempty 2D uint8 image.

    Returns:
        Self-describing byte stream containing eight binary RLE streams ordered
        from the most-significant plane to the least-significant plane.

    Syllabus: Unit 8 - bit-plane image compression
    """
    image = _validate_image(img)
    encoded = bytearray(b"RBP1")
    encoded.extend(struct.pack(">II", image.shape[0], image.shape[1]))
    for bit in range(7, -1, -1):
        plane = (((image >> bit) & 1) * 255).astype(np.uint8)
        stream = rle_encode(plane)
        encoded.extend(struct.pack(">I", len(stream)))
        encoded.extend(stream)
    return bytes(encoded)


def rle_bitplanes_decode(data: bytes) -> Image:
    """Decode an MSB-first bit-plane RLE image.

    Args:
        data: Bytes produced by rle_bitplanes.

    Returns:
        Reconstructed uint8 grayscale image.

    Syllabus: Unit 8 - bit-plane image compression
    """
    if len(data) < 12 or data[:4] != b"RBP1":
        raise ValueError("invalid bit-plane RLE stream.")
    height, width = struct.unpack(">II", data[4:12])
    if height <= 0 or width <= 0:
        raise ValueError("invalid bit-plane image dimensions.")
    output = np.zeros((height, width), dtype=np.uint8)
    offset = 12
    for bit in range(7, -1, -1):
        if offset + 4 > len(data):
            raise ValueError("truncated bit-plane RLE stream.")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        end = offset + length
        if length == 0 or end > len(data):
            raise ValueError("invalid bit-plane stream length.")
        plane = rle_decode(data[offset:end])
        if plane.shape != (height, width):
            raise ValueError("bit-plane dimensions do not match image header.")
        output |= ((plane != 0).astype(np.uint8) << bit)
        offset = end
    if offset != len(data):
        raise ValueError("bit-plane stream contains trailing bytes.")
    return output


def _huffman_lengths(data: bytes) -> dict[int, int]:
    """Build Huffman code lengths for byte symbols."""
    frequencies = Counter(data)
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        symbol = next(iter(frequencies))
        return {symbol: 0}
    serial = count()
    heap: list[tuple[int, int, int | tuple[object, object]]] = [
        (frequency, next(serial), symbol)
        for symbol, frequency in frequencies.items()
    ]
    heapq.heapify(heap)
    while len(heap) > 1:
        left_frequency, _, left = heapq.heappop(heap)
        right_frequency, _, right = heapq.heappop(heap)
        heapq.heappush(
            heap,
            (
                left_frequency + right_frequency,
                next(serial),
                (left, right),
            ),
        )
    root = heap[0][2]
    lengths: dict[int, int] = {}

    def visit(node: int | tuple[object, object], depth: int) -> None:
        """Record leaf depths in the Huffman tree."""
        if isinstance(node, int):
            lengths[node] = max(1, depth)
        else:
            left_node, right_node = node
            visit(left_node, depth + 1)  # type: ignore[arg-type]
            visit(right_node, depth + 1)  # type: ignore[arg-type]

    visit(root, 0)
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Generate canonical integer codes from symbol bit lengths."""
    ordered = sorted((length, symbol) for symbol, length in lengths.items())
    if len(ordered) == 1 and ordered[0][0] == 0:
        return {ordered[0][1]: (0, 0)}
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    previous_length = 0
    for length, symbol in ordered:
        if length <= 0 or length > 255:
            raise ValueError("invalid Huffman code length.")
        code <<= length - previous_length
        if code >= 1 << length:
            raise ValueError("invalid oversubscribed Huffman code lengths.")
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def _pack_codes(data: bytes, codes: dict[int, tuple[int, int]]) -> tuple[bytes, int]:
    """Pack canonical Huffman codes MSB-first into bytes."""
    output = bytearray()
    accumulator = 0
    bit_count = 0
    total_bits = 0
    for symbol in data:
        code, length = codes[symbol]
        accumulator = (accumulator << length) | code
        bit_count += length
        total_bits += length
        while bit_count >= 8:
            bit_count -= 8
            output.append((accumulator >> bit_count) & 0xFF)
            accumulator &= (1 << bit_count) - 1 if bit_count else 0
    if bit_count:
        output.append((accumulator << (8 - bit_count)) & 0xFF)
    return bytes(output), total_bits


def huffman_encode(img: NDArray[np.generic]) -> CodecBytes:
    """Encode any nonempty uint8 array with canonical Huffman bit codes.

    Args:
        img: Nonempty uint8 array of any dimensionality.

    Returns:
        Bytes with shape, canonical symbol/code-length table, bit count, and
        MSB-first bit-packed payload.

    Syllabus: Unit 8 - Huffman coding
    """
    array = np.asarray(img)
    if array.dtype != np.uint8 or array.size == 0 or array.ndim > 255:
        raise ValueError("img must be a nonempty uint8 array with at most 255 dimensions.")
    raw = array.tobytes(order="C")
    lengths = _huffman_lengths(raw)
    codes = _canonical_codes(lengths)
    payload, bit_length = _pack_codes(raw, codes)
    header = bytearray(b"HUF1")
    header.append(array.ndim)
    for dimension in array.shape:
        if dimension > 0xFFFFFFFF:
            raise ValueError("array dimension exceeds format limit.")
        header.extend(struct.pack(">I", dimension))
    header.extend(struct.pack(">H", len(lengths)))
    for symbol, length in sorted(lengths.items()):
        header.extend((symbol, length))
    header.extend(struct.pack(">Q", bit_length))
    header.extend(payload)
    return bytes(header)


def huffman_decode(data: bytes) -> Image | NDArray[np.uint8]:
    """Decode a canonical Huffman uint8 array.

    Args:
        data: Bytes produced by huffman_encode.

    Returns:
        uint8 ndarray with the original shape.

    Syllabus: Unit 8 - Huffman coding
    """
    if len(data) < 4 + 1 + 2 + 8 or data[:4] != b"HUF1":
        raise ValueError("invalid Huffman stream.")
    ndim = data[4]
    offset = 5
    dimension_bytes = 4 * ndim
    if offset + dimension_bytes + 2 + 8 > len(data):
        raise ValueError("truncated Huffman header.")
    shape = tuple(
        struct.unpack(">I", data[offset + 4 * index : offset + 4 * (index + 1)])[0]
        for index in range(ndim)
    )
    offset += dimension_bytes
    symbol_count = struct.unpack(">H", data[offset : offset + 2])[0]
    offset += 2
    if symbol_count == 0 or offset + 2 * symbol_count + 8 > len(data):
        raise ValueError("invalid Huffman symbol table.")
    lengths: dict[int, int] = {}
    for _ in range(symbol_count):
        symbol, length = data[offset], data[offset + 1]
        offset += 2
        if symbol in lengths:
            raise ValueError("duplicate symbol in Huffman table.")
        lengths[symbol] = length
    bit_length = struct.unpack(">Q", data[offset : offset + 8])[0]
    offset += 8
    payload = data[offset:]
    if bit_length > len(payload) * 8 or (bit_length + 7) // 8 != len(payload):
        raise ValueError("invalid Huffman payload bit length.")
    expected_size = math.prod(shape) if shape else 1
    codes = _canonical_codes(lengths)
    if len(lengths) == 1 and next(iter(lengths.values())) == 0:
        if bit_length != 0 or payload:
            raise ValueError("single-symbol Huffman stream must have an empty payload.")
        symbol = next(iter(lengths))
        return np.full(shape, symbol, dtype=np.uint8)
    decode_table = {(code, length): symbol for symbol, (code, length) in codes.items()}
    decoded = bytearray()
    code = 0
    current_length = 0
    consumed = 0
    for byte in payload:
        for bit_index in range(7, -1, -1):
            if consumed >= bit_length:
                break
            code = (code << 1) | ((byte >> bit_index) & 1)
            current_length += 1
            consumed += 1
            symbol = decode_table.get((code, current_length))
            if symbol is not None:
                decoded.append(symbol)
                code = 0
                current_length = 0
                if len(decoded) > expected_size:
                    raise ValueError("Huffman payload decodes to too many symbols.")
        if consumed >= bit_length:
            break
    if current_length != 0 or len(decoded) != expected_size:
        raise ValueError("invalid or incomplete Huffman payload.")
    return np.frombuffer(bytes(decoded), dtype=np.uint8).reshape(shape).copy()


def shannon_entropy(img: NDArray[np.generic]) -> float:
    """Compute empirical Shannon entropy in bits per byte.

    Args:
        img: Nonempty uint8 array.

    Returns:
        Shannon entropy in bits per symbol.

    Syllabus: Unit 8 - information theory and coding
    """
    array = np.asarray(img)
    if array.dtype != np.uint8 or array.size == 0:
        raise ValueError("img must be a nonempty uint8 array.")
    probabilities = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    probabilities /= array.size
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))


def huffman_code_table(img: NDArray[np.generic]) -> pd.DataFrame:
    """Return canonical Huffman codes for the most frequent image symbols.

    Args:
        img: Nonempty uint8 array whose byte histogram will be encoded.

    Returns:
        DataFrame with symbol, frequency, code, and code_length columns,
        ordered from most to least frequent symbol.

    Syllabus: Unit 8 - Huffman coding
    """
    array = np.asarray(img)
    if array.dtype != np.uint8 or array.size == 0:
        raise ValueError("img must be a nonempty uint8 array.")
    raw = array.tobytes(order="C")
    frequencies = Counter(raw)
    lengths = _huffman_lengths(raw)
    codes = _canonical_codes(lengths)
    rows = [
        {
            "symbol": symbol,
            "frequency": frequency,
            "code": format(codes[symbol][0], f"0{codes[symbol][1]}b")
            if codes[symbol][1]
            else "0",
            "code_length": codes[symbol][1],
        }
        for symbol, frequency in frequencies.most_common(10)
    ]
    return pd.DataFrame(rows, columns=["symbol", "frequency", "code", "code_length"])


def average_code_length(img: NDArray[np.generic]) -> float:
    """Compute the Huffman average code length for an array's byte histogram.

    Args:
        img: Nonempty uint8 array.

    Returns:
        Average number of encoded bits per input byte.

    Syllabus: Unit 8 - information theory and coding
    """
    array = np.asarray(img)
    if array.dtype != np.uint8 or array.size == 0:
        raise ValueError("img must be a nonempty uint8 array.")
    lengths = _huffman_lengths(array.tobytes(order="C"))
    counts = np.bincount(array.ravel(), minlength=256)
    return float(
        sum(int(counts[symbol]) * length for symbol, length in lengths.items())
        / array.size
    )


def _pack_12bit(codes: Sequence[int]) -> bytes:
    """Pack nonnegative 12-bit LZW codes MSB-first."""
    output = bytearray()
    accumulator = 0
    bits = 0
    for code in codes:
        if not 0 <= code <= 0xFFF:
            raise ValueError("LZW code is outside the 12-bit range.")
        accumulator = (accumulator << 12) | code
        bits += 12
        while bits >= 8:
            bits -= 8
            output.append((accumulator >> bits) & 0xFF)
            accumulator &= (1 << bits) - 1 if bits else 0
    if bits:
        output.append((accumulator << (8 - bits)) & 0xFF)
    return bytes(output)


def _unpack_12bit(data: bytes, count_codes: int) -> list[int]:
    """Unpack a fixed number of MSB-first 12-bit LZW codes."""
    codes: list[int] = []
    accumulator = 0
    bits = 0
    for byte in data:
        accumulator = (accumulator << 8) | byte
        bits += 8
        while bits >= 12 and len(codes) < count_codes:
            bits -= 12
            codes.append((accumulator >> bits) & 0xFFF)
            accumulator &= (1 << bits) - 1 if bits else 0
    if len(codes) != count_codes:
        raise ValueError("truncated LZW code payload.")
    return codes


def lzw_encode(data: bytes | NDArray[np.generic]) -> CodecBytes:
    """Encode a byte stream with 12-bit LZW codes and dictionary resets.

    Args:
        data: Input byte string or one-dimensional uint8 array. Image callers
            may pass flattened row-major pixel bytes.

    Returns:
        Bytes containing original length, code count, and fixed-width packed
        codes. Code 256 clears the dictionary; code 257 ends the stream.

    Syllabus: Unit 8 - LZW coding
    """
    raw = bytes(data) if isinstance(data, bytes) else np.asarray(data).tobytes()
    if not raw:
        raise ValueError("LZW input must not be empty.")
    if not isinstance(data, bytes):
        array = np.asarray(data)
        if array.dtype != np.uint8 or array.ndim != 1:
            raise ValueError("array LZW input must be a one-dimensional uint8 array.")
    clear_code, end_code, first_code = 256, 257, 258
    dictionary = {bytes((symbol,)): symbol for symbol in range(256)}
    next_code = first_code
    codes = [clear_code]
    prefix = bytes((raw[0],))
    for value in raw[1:]:
        character = bytes((value,))
        candidate = prefix + character
        if candidate in dictionary:
            prefix = candidate
            continue
        codes.append(dictionary[prefix])
        if next_code < 4096:
            dictionary[candidate] = next_code
            next_code += 1
        else:
            # The explicit clear marker keeps both encoder and decoder in sync.
            codes.append(clear_code)
            dictionary = {bytes((symbol,)): symbol for symbol in range(256)}
            next_code = first_code
        prefix = character
    codes.append(dictionary[prefix])
    codes.append(end_code)
    payload = _pack_12bit(codes)
    return b"LZW1" + struct.pack(">II", len(raw), len(codes)) + payload


def lzw_decode(data: bytes) -> NDArray[np.uint8]:
    """Decode an LZW stream into a one-dimensional uint8 byte array.

    Args:
        data: Bytes produced by lzw_encode.

    Returns:
        Decoded uint8 byte stream.

    Syllabus: Unit 8 - LZW coding
    """
    if len(data) < 12 or data[:4] != b"LZW1":
        raise ValueError("invalid LZW stream.")
    original_length, code_count = struct.unpack(">II", data[4:12])
    if original_length == 0 or code_count < 3:
        raise ValueError("invalid LZW stream header.")
    codes = _unpack_12bit(data[12:], code_count)
    clear_code, end_code, first_code = 256, 257, 258
    dictionary = {symbol: bytes((symbol,)) for symbol in range(256)}
    next_code = first_code
    previous: bytes | None = None
    output = bytearray()
    started = False
    ended = False
    for code in codes:
        if code == clear_code:
            dictionary = {symbol: bytes((symbol,)) for symbol in range(256)}
            next_code = first_code
            previous = None
            started = True
            continue
        if code == end_code:
            ended = True
            break
        if not started:
            raise ValueError("LZW stream does not begin with a clear code.")
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code and previous is not None:
            entry = previous + previous[:1]
        else:
            raise ValueError("invalid LZW code.")
        output.extend(entry)
        if previous is not None and next_code < 4096:
            dictionary[next_code] = previous + entry[:1]
            next_code += 1
        previous = entry
    if not ended or len(output) != original_length or len(data) != 12 + (code_count * 12 + 7) // 8:
        raise ValueError("invalid or incomplete LZW payload.")
    return np.frombuffer(bytes(output), dtype=np.uint8).copy()


def igs_quantize(img: NDArray[np.generic], bits: int = 4) -> Image:
    """Apply Improved Gray-Scale (IGS) quantization along each image row.

    IGS adds the previous pixel's accumulated low-order bits before truncation,
    except when the current sample's retained high bits are all ones. This
    breaks up false contours at the cost of a small dither-like variation.

    Args:
        img: Nonempty 2D uint8 grayscale image.
        bits: Number of retained high bits, from one through seven.

    Returns:
        Quantized uint8 image with no more than 2**bits distinct levels.

    Syllabus: Unit 8 - IGS quantization
    """
    image = _validate_image(img)
    if not isinstance(bits, (int, np.integer)) or not 1 <= bits <= 7:
        raise ValueError("bits must be an integer from 1 through 7.")
    shift = 8 - int(bits)
    low_mask = (1 << shift) - 1
    high_mask = (1 << int(bits)) - 1
    result = np.empty_like(image)
    for row_index, row in enumerate(image):
        carry = 0
        for column_index, value in enumerate(row):
            sample = int(value)
            adjusted = sample
            if (sample >> shift) != high_mask:
                adjusted = min(255, sample + carry)
            result[row_index, column_index] = (adjusted >> shift) << shift
            carry = adjusted & low_mask
    return result


def _zigzag_order(size: int) -> list[tuple[int, int]]:
    """Return matrix coordinates in the standard diagonal zigzag order."""
    coordinates: list[tuple[int, int]] = []
    for diagonal in range(2 * size - 1):
        positions = [
            (row, diagonal - row)
            for row in range(size)
            if 0 <= diagonal - row < size
        ]
        if diagonal % 2 == 0:
            positions.reverse()
        coordinates.extend(positions)
    return coordinates


def _coefficient_order(transform: Transform, size: int) -> list[tuple[int, int]]:
    """Choose a low-to-high frequency scan for transform coefficient selection."""
    if transform == "dct":
        return _zigzag_order(size)
    if transform == "wht":
        coordinates = [(row, column) for row in range(size) for column in range(size)]
        matrix = walsh_matrix(size)
        sequency = np.count_nonzero(matrix[:, 1:] != matrix[:, :-1], axis=1)
        return sorted(
            coordinates,
            key=lambda coordinate: (
                int(sequency[coordinate[0]]) + int(sequency[coordinate[1]]),
                coordinate[0] + coordinate[1],
                coordinate[0],
            ),
        )
    raise ValueError("transform must be 'wht' or 'dct'.")


def _transform_block(block: NDArray[np.float64], transform: Transform) -> NDArray[np.float64]:
    """Apply an orthonormal WHT or DCT to a single square block."""
    if transform == "dct":
        return cv2.dct(block.astype(np.float32)).astype(np.float64)
    if transform == "wht":
        return wht2d(block, ordering="walsh")
    raise ValueError("transform must be 'wht' or 'dct'.")


def _inverse_transform_block(block: NDArray[np.float64], transform: Transform) -> NDArray[np.float64]:
    """Invert a block transform."""
    if transform == "dct":
        return cv2.idct(block.astype(np.float32)).astype(np.float64)
    if transform == "wht":
        return iwht2d(block, ordering="walsh")
    raise ValueError("transform must be 'wht' or 'dct'.")


def block_transform_compress(
    img: NDArray[np.generic],
    transform: Transform,
    block: int = 8,
    keep: int | None = None,
    qstep: float | None = None,
) -> tuple[CodecBytes, Image]:
    """Compress image blocks with WHT/DCT, coefficient selection or quantization.

    Quantized signed coefficients are serialized as little-endian int16 bytes
    and encoded with the canonical Huffman format, so the returned byte count
    includes the actual coefficient, shape, transform, and entropy-code headers.

    Args:
        img: Nonempty 2D uint8 grayscale image.
        transform: ``"wht"`` Walsh-Hadamard or ``"dct"`` cosine transform.
        block: Positive block side length; currently required to be eight.
        keep: Optional number of lowest-frequency coefficients retained per
            block, from one through block**2.
        qstep: Optional positive uniform quantization step. With neither keep
            nor qstep supplied, step one is used.

    Returns:
        (self-describing compressed bytes, reconstructed uint8 image).

    Syllabus: Unit 8 - transform coding
    """
    image = _validate_image(img)
    if transform not in ("wht", "dct"):
        raise ValueError("transform must be 'wht' or 'dct'.")
    if not isinstance(block, (int, np.integer)) or block != 8:
        raise ValueError("block must be 8 for the supported 8x8 codec.")
    if keep is not None and (
        not isinstance(keep, (int, np.integer)) or not 1 <= keep <= block * block
    ):
        raise ValueError("keep must be between 1 and block**2.")
    if qstep is not None and (not np.isfinite(qstep) or qstep <= 0):
        raise ValueError("qstep must be finite and positive.")
    if keep is not None and qstep is not None:
        raise ValueError("choose either keep or qstep, not both.")
    step = 1.0 if qstep is None else float(qstep)
    height, width = image.shape
    padded_height = ((height + block - 1) // block) * block
    padded_width = ((width + block - 1) // block) * block
    padded = np.pad(
        image.astype(np.float64),
        ((0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )
    quantized = np.zeros((padded_height, padded_width), dtype=np.int16)
    reconstruction = np.zeros_like(padded, dtype=np.float64)
    order = _coefficient_order(transform, int(block))
    for row in range(0, padded_height, int(block)):
        for column in range(0, padded_width, int(block)):
            coefficients = _transform_block(
                padded[row : row + block, column : column + block], transform
            )
            if keep is not None:
                retained = set(order[: int(keep)])
                for coordinate in order:
                    if coordinate not in retained:
                        coefficients[coordinate] = 0.0
            quantized_block = np.rint(coefficients / step)
            if np.any((quantized_block < np.iinfo(np.int16).min) | (quantized_block > np.iinfo(np.int16).max)):
                raise ValueError("quantized transform coefficient exceeds int16 range.")
            quantized[row : row + block, column : column + block] = quantized_block.astype(
                np.int16
            )
            restored = _inverse_transform_block(quantized_block * step, transform)
            reconstruction[row : row + block, column : column + block] = restored
    coefficient_bytes = quantized.astype("<i2", copy=False).tobytes(order="C")
    entropy_stream = huffman_encode(np.frombuffer(coefficient_bytes, dtype=np.uint8))
    transform_code = 0 if transform == "wht" else 1
    keep_value = -1 if keep is None else int(keep)
    header = struct.pack(
        ">4sBIIIIHifI",
        b"BTC1",
        transform_code,
        height,
        width,
        padded_height,
        padded_width,
        int(block),
        keep_value,
        step,
        len(entropy_stream),
    )
    encoded = header + entropy_stream
    output = np.clip(np.rint(reconstruction[:height, :width]), 0, 255).astype(np.uint8)
    return encoded, output


def block_transform_decompress(data: bytes) -> Image:
    """Decode an 8x8 WHT/DCT coefficient stream into a uint8 image.

    Args:
        data: Bytes returned by block_transform_compress.

    Returns:
        Reconstructed uint8 grayscale image.

    Syllabus: Unit 8 - transform coding
    """
    header_format = ">4sBIIIIHifI"
    header_size = struct.calcsize(header_format)
    if len(data) < header_size:
        raise ValueError("truncated block transform stream.")
    (
        magic,
        transform_code,
        height,
        width,
        padded_height,
        padded_width,
        block,
        keep_value,
        step,
        entropy_length,
    ) = struct.unpack(header_format, data[:header_size])
    if (
        magic != b"BTC1"
        or transform_code not in (0, 1)
        or min(height, width, padded_height, padded_width) <= 0
        or block != 8
        or padded_height % block
        or padded_width % block
        or keep_value < -1
        or keep_value == 0
        or keep_value > block * block
        or not np.isfinite(step)
        or step <= 0
        or len(data) != header_size + entropy_length
    ):
        raise ValueError("invalid block transform header.")
    packed_coefficients = huffman_decode(data[header_size:])
    expected_bytes = padded_height * padded_width * 2
    if packed_coefficients.size != expected_bytes:
        raise ValueError("invalid transform coefficient byte count.")
    quantized = packed_coefficients.tobytes()
    coefficients = np.frombuffer(quantized, dtype="<i2").reshape(
        padded_height, padded_width
    )
    transform: Transform = "wht" if transform_code == 0 else "dct"
    reconstruction = np.zeros((padded_height, padded_width), dtype=np.float64)
    for row in range(0, padded_height, block):
        for column in range(0, padded_width, block):
            block_coefficients = coefficients[
                row : row + block, column : column + block
            ].astype(np.float64) * float(step)
            reconstruction[row : row + block, column : column + block] = (
                _inverse_transform_block(block_coefficients, transform)
            )
    return np.clip(np.rint(reconstruction[:height, :width]), 0, 255).astype(np.uint8)


def _psnr(reference: Image, reconstructed: Image) -> float:
    """Compute PSNR in dB, returning infinity for exact reconstruction."""
    error = np.mean(
        (reference.astype(np.float64) - reconstructed.astype(np.float64)) ** 2
    )
    return float("inf") if error == 0 else float(10.0 * np.log10(255.0**2 / error))


def rate_distortion_curve(
    img: NDArray[np.generic],
    transform: Transform,
    params: Sequence[dict[str, int | float | None]],
) -> pd.DataFrame:
    """Measure actual compressed rate and reconstruction quality for parameters.

    Args:
        img: Nonempty uint8 grayscale image.
        transform: ``"wht"`` or ``"dct"`` block transform.
        params: Sequence of keyword-argument mappings such as ``{"keep": 12}``
            or ``{"qstep": 8.0}``.

    Returns:
        DataFrame with columns CR, PSNR, and SSIM. CR is raw image bytes divided
        by actual compressed stream length, including all headers.

    Syllabus: Unit 8 - rate-distortion analysis
    """
    image = _validate_image(img)
    rows: list[dict[str, float]] = []
    for parameter_set in params:
        encoded, reconstruction = block_transform_compress(
            image, transform=transform, **parameter_set
        )
        rows.append(
            {
                "CR": float(image.nbytes / len(encoded)),
                "PSNR": _psnr(image, reconstruction),
                "SSIM": float(
                    structural_similarity(image, reconstruction, data_range=255)
                ),
            }
        )
    return pd.DataFrame(rows, columns=["CR", "PSNR", "SSIM"])


def compare_codecs(
    gray: NDArray[np.generic], binary: NDArray[np.generic]
) -> pd.DataFrame:
    """Compare real codec byte sizes, losslessness, and timing on two images.

    Args:
        gray: Nonempty 2D uint8 grayscale image.
        binary: Nonempty 2D uint8 0/255 binary image with identical dimensions.

    Returns:
        DataFrame columns codec, original_bytes, compressed_bytes, ratio,
        bits_per_pixel, lossless, and encode_seconds. PNG and JPEG at quality
        90/50/10 are included as image-codec baselines.

    Syllabus: Unit 8 - codec comparison
    """
    grayscale = _validate_image(gray)
    binary_image = _validate_image(binary)
    if not np.all((binary_image == 0) | (binary_image == 255)):
        raise ValueError("binary must contain only 0 and 255.")
    rows: list[dict[str, str | int | float | bool]] = []

    def record(name: str, source: Image, encoded: bytes, decoded: Image, elapsed: float) -> None:
        """Add a row using measured serialized size and actual round-trip loss."""
        original_bytes = int(source.nbytes)
        compressed_bytes = len(encoded)
        rows.append(
            {
                "codec": name,
                "original_bytes": original_bytes,
                "compressed_bytes": compressed_bytes,
                "ratio": float(original_bytes / compressed_bytes),
                "bits_per_pixel": float(compressed_bytes * 8 / source.size),
                "lossless": bool(np.array_equal(source, decoded)),
                "encode_seconds": float(elapsed),
            }
        )

    for name, source, encoder, decoder in (
        ("rle_binary", binary_image, rle_encode, rle_decode),
        ("rle_gray", grayscale, rle_gray, rle_gray_decode),
        ("rle_bitplanes", grayscale, rle_bitplanes, rle_bitplanes_decode),
        ("huffman_gray", grayscale, huffman_encode, huffman_decode),
        (
            "huffman_binary",
            binary_image,
            huffman_encode,
            huffman_decode,
        ),
        ("lzw_gray", grayscale, lambda image: lzw_encode(image.ravel()), lzw_decode),
        ("lzw_binary", binary_image, lambda image: lzw_encode(image.ravel()), lzw_decode),
    ):
        started = time.perf_counter()
        encoded = encoder(source)
        elapsed = time.perf_counter() - started
        decoded = decoder(encoded)
        if name.startswith("lzw_"):
            decoded = decoded.reshape(source.shape)
        record(name, source, encoded, decoded, elapsed)

    for source_name, source in (("gray", grayscale), ("binary", binary_image)):
        for extension, parameters, quality in (
            (".png", [], None),
            (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90], 90),
            (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 50], 50),
            (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 10], 10),
        ):
            started = time.perf_counter()
            success, buffer = cv2.imencode(extension, source, parameters)
            elapsed = time.perf_counter() - started
            if not success:
                raise RuntimeError(f"OpenCV could not encode {source_name} as {extension}.")
            encoded = buffer.tobytes()
            decoded = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if decoded is None:
                raise RuntimeError(f"OpenCV could not decode its {source_name} {extension} stream.")
            codec_name = (
                "png"
                if quality is None
                else f"jpeg_{quality}"
            )
            record(f"{source_name}_{codec_name}", source, encoded, decoded, elapsed)
    return pd.DataFrame(
        rows,
        columns=[
            "codec",
            "original_bytes",
            "compressed_bytes",
            "ratio",
            "bits_per_pixel",
            "lossless",
            "encode_seconds",
        ],
    )

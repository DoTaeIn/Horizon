"""Lossless container codec for 2D uint16 heightmaps.

Container layout (little-endian, 60-byte header followed by the payload):

    magic      4s   b"HRZN"
    version    u8   1
    method     u8   index into METHODS
    predictor  u8   0 = none, 1 = row-wise left-neighbour delta mod 65536
    reserved   u8   must be 0
    height     u32
    width      u32
    payload    u64  compressed payload length in bytes
    sha256     32s  digest of the original C-order <u2 bytes
    crc32      u32  CRC-32 of the preceding 56 header bytes

The CRC guards the shape fields, which the SHA-256 of the samples alone does
not cover (e.g. 8x8 vs 64x1 have identical bytes).
"""

import hashlib
import struct
import zlib

import lz4.frame
import numpy as np
import zstandard

METHODS = ("zstd", "lz4", "delta_zstd")

MAGIC = b"HRZN"
VERSION = 1
MAX_RAW_BYTES = 64 * 1024 * 1024
ZSTD_LEVEL = 3

_HEADER = struct.Struct("<4sBBBBIIQ32s")
_CRC = struct.Struct("<I")
HEADER_SIZE = _HEADER.size + _CRC.size

_PRED_NONE = 0
_PRED_ROW_DELTA = 1
_PREDICTORS = {"zstd": _PRED_NONE, "lz4": _PRED_NONE, "delta_zstd": _PRED_ROW_DELTA}
_DTYPE = np.dtype("<u2")
_UNDELTA_BATCH = 1 << 16  # samples per uint32 accumulation batch


def _check_dims(height, width):
    if height < 1 or width < 1:
        raise ValueError("heightmap dimensions must be positive")
    if height * width * _DTYPE.itemsize > MAX_RAW_BYTES:
        raise ValueError("heightmap exceeds the 64 MiB raw size limit")


def _delta(arr):
    out = arr.copy()
    out[:, 1:] = arr[:, 1:] - arr[:, :-1]  # uint16 arithmetic wraps mod 65536
    return out


def _undelta(arr):
    height, width = arr.shape
    # Small maps keep the direct uint16 accumulate: microbenchmarks showed the
    # uint32 scratch + copy overhead outweighs any gain at this size. Rows wider
    # than the batch bound cannot fit a single complete row in scratch.
    if arr.size <= _UNDELTA_BATCH or width > _UNDELTA_BATCH:
        return np.add.accumulate(arr, axis=1, dtype=_DTYPE)
    out = np.empty((height, width), dtype=_DTYPE)
    rows = min(height, _UNDELTA_BATCH // width)
    scratch = np.empty((rows, width), dtype=np.uint32)
    for start in range(0, height, rows):
        stop = min(start + rows, height)
        buf = scratch[:stop - start]
        np.add.accumulate(arr[start:stop], axis=1, dtype=np.uint32, out=buf)
        np.copyto(out[start:stop], buf, casting="unsafe")
    return out

def _zstd_compress(raw):
    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_content_size=True)
    return cctx.compress(raw)


def _zstd_decompress(payload, size):
    try:
        # decompress() allocates whatever the frame header claims, so check it first.
        params = zstandard.get_frame_parameters(payload)
        if params.content_size != size:
            raise ValueError("zstd frame content size mismatch")
        if params.window_size > MAX_RAW_BYTES:
            raise ValueError("zstd frame window exceeds the 64 MiB limit")
        return zstandard.ZstdDecompressor().decompress(
            payload, max_output_size=size, allow_extra_data=False
        )
    except ValueError:
        raise
    except zstandard.ZstdError as exc:
        raise ValueError(f"corrupt zstd payload: {exc}") from None


def _lz4_compress(raw):
    return lz4.frame.compress(raw, store_size=True)


def _lz4_decompress(payload, size):
    try:
        if lz4.frame.get_frame_info(payload).get("content_size") != size:
            raise ValueError("lz4 frame content size mismatch")
        dctx = lz4.frame.LZ4FrameDecompressor()
        # One spare byte lets the frame end marker be consumed while still
        # bounding output; any overrun is rejected below.
        out = dctx.decompress(payload, max_length=size + 1)
    except ValueError:
        raise
    except RuntimeError as exc:
        raise ValueError(f"corrupt lz4 payload: {exc}") from None
    if not dctx.eof or dctx.unused_data or len(out) != size:
        raise ValueError("corrupt lz4 payload")
    return out


def encode(heightmap, method="delta_zstd"):
    """Encode a nonempty 2D little-endian uint16 array into a container."""
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not isinstance(heightmap, np.ndarray):
        raise ValueError("heightmap must be a numpy.ndarray")
    if heightmap.dtype != _DTYPE:
        raise ValueError(f"heightmap dtype must be <u2, got {heightmap.dtype.str}")
    if heightmap.ndim != 2:
        raise ValueError(f"heightmap must be 2D, got {heightmap.ndim}D")
    height, width = heightmap.shape
    _check_dims(height, width)

    arr = np.ascontiguousarray(heightmap)
    raw = arr.tobytes()
    if method == "zstd":
        payload = _zstd_compress(raw)
    elif method == "lz4":
        payload = _lz4_compress(raw)
    else:
        payload = _zstd_compress(_delta(arr).tobytes())

    header = _HEADER.pack(
        MAGIC,
        VERSION,
        METHODS.index(method),
        _PREDICTORS[method],
        0,
        height,
        width,
        len(payload),
        hashlib.sha256(raw).digest(),
    )
    return header + _CRC.pack(zlib.crc32(header)) + payload


def decode(blob):
    """Decode a container produced by encode() back into a 2D <u2 array."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError("blob must be bytes")
    blob = bytes(blob)
    if len(blob) < HEADER_SIZE:
        raise ValueError("truncated header")
    magic, version, method_id, predictor, reserved, height, width, plen, digest = (
        _HEADER.unpack_from(blob)
    )
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    (crc,) = _CRC.unpack_from(blob, _HEADER.size)
    if crc != zlib.crc32(blob[: _HEADER.size]):
        raise ValueError("header CRC mismatch")
    if method_id >= len(METHODS):
        raise ValueError(f"unknown method id {method_id}")
    method = METHODS[method_id]
    if predictor != _PREDICTORS[method] or reserved != 0:
        raise ValueError("invalid predictor or reserved field")
    _check_dims(height, width)
    if plen != len(blob) - HEADER_SIZE:
        raise ValueError("payload length mismatch (truncated or trailing data)")

    size = height * width * _DTYPE.itemsize
    payload = blob[HEADER_SIZE:]
    if method == "lz4":
        raw = _lz4_decompress(payload, size)
    else:
        raw = _zstd_decompress(payload, size)

    result = np.frombuffer(raw, dtype=_DTYPE).reshape(height, width)
    if predictor == _PRED_ROW_DELTA:
        result = _undelta(result)
    else:
        result = result.copy()
    # result is C-contiguous, so its buffer is exactly the tobytes() bytes.
    if hashlib.sha256(result).digest() != digest:
        raise ValueError("checksum mismatch")
    return result

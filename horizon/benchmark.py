'''Whole-map container benchmark for the uint16 heightmap codec.

Timings cover complete encode()/decode() calls on in-memory maps. They measure
compression container cost only, not streaming and not engine performance.
'''

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from horizon.codec import HEADER_SIZE, MAX_RAW_BYTES, METHODS, VERSION, ZSTD_LEVEL, decode, encode

MIN_REPEATS = 5
WARMUP = 1
SEED = 20240611
SCHEMA_VERSION = 1
_DTYPE = np.dtype('<u2')


def _int_arg(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return int(value)


def validate_heightmap(arr, label='input'):
    '''Raise ValueError unless arr is a nonempty 2D little-endian uint16 map.'''
    if not isinstance(arr, np.ndarray):
        raise ValueError(f'{label}: expected a numpy array')
    if arr.ndim != 2:
        raise ValueError(f'{label}: expected a 2D array, got {arr.ndim}D')
    if arr.dtype != _DTYPE:
        raise ValueError(f'{label}: expected little-endian uint16, got {arr.dtype}')
    if arr.size == 0:
        raise ValueError(f'{label}: array is empty')
    if arr.nbytes > MAX_RAW_BYTES:
        raise ValueError(
            f'{label}: {arr.nbytes} raw bytes exceeds the {MAX_RAW_BYTES} byte limit'
        )


def load_heightmap(path):
    '''Load and validate a .npy heightmap (no pickle).'''
    path = Path(path)
    try:
        with path.open('rb') as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in ((2, 0), (3, 0)):
                # The accepted plain uint16 descriptor and shape are ASCII in both versions.
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f'unsupported NPY version {version}')
            if len(shape) != 2 or min(shape) <= 0 or dtype != _DTYPE:
                raise ValueError('expected a nonempty 2D little-endian uint16 array')
            raw_size = shape[0] * shape[1] * _DTYPE.itemsize
            if raw_size > MAX_RAW_BYTES:
                raise ValueError('heightmap exceeds the 64 MiB raw size limit')
            expected_end = handle.tell() + raw_size
            if handle.seek(0, os.SEEK_END) != expected_end:
                raise ValueError('NPY payload is truncated or has trailing data')
            # Read only after checking the header's claimed allocation and actual file length.
            handle.seek(0)
            obj = np.load(handle, allow_pickle=False)
    except (ValueError, EOFError) as exc:
        raise ValueError(f'{path.name}: not a loadable .npy file ({exc})') from exc
    validate_heightmap(obj, path.name)
    return obj


def make_fixtures(size):
    '''Deterministic synthetic maps as (name, source, array) tuples.'''
    idx = np.arange(size, dtype=np.uint32)
    y = idx[:, None]
    x = idx[None, :]

    flat = np.full((size, size), 32768, dtype=_DTYPE)
    plane = ((y * 7 + x * 13) & 0xFFFF).astype(_DTYPE)

    t = np.linspace(0.0, 1.0, size)
    wave = (
        np.sin(2.0 * np.pi * 2.0 * t)[:, None]
        + np.cos(2.0 * np.pi * 3.0 * t)[None, :]
        + 0.5 * np.sin(2.0 * np.pi * 5.0 * (t[:, None] + t[None, :]))
    )
    lo = float(wave.min())
    hi = float(wave.max())
    span = hi - lo if hi > lo else 1.0
    smooth = np.rint((wave - lo) / span * 60000.0 + 2000.0).astype(_DTYPE)

    rng = np.random.default_rng(SEED)
    noise = rng.integers(0, 65536, size=(size, size), dtype=np.uint16).astype(_DTYPE)

    return [
        ('flat', 'synthetic', flat),
        ('plane', 'synthetic', plane),
        ('smooth_terrain', 'synthetic', smooth),
        ('random', 'synthetic', noise),
    ]


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'


def _matches(decoded, arr, expected):
    return (
        isinstance(decoded, np.ndarray)
        and decoded.shape == arr.shape
        and decoded.dtype == arr.dtype
        and decoded.tobytes() == expected
    )


def _bench_one(name, source, arr, expected, digest, method, repeats):
    blob = encode(arr, method=method)
    if not isinstance(blob, bytes):
        raise RuntimeError(f'{name}/{method}: encode did not return bytes')
    # Warmup decode, also the first exactness check.
    if not _matches(decode(blob), arr, expected):
        raise RuntimeError(f'{name}/{method}: decode mismatch during warmup')

    enc_ns = []
    dec_ns = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        again = encode(arr, method=method)
        t1 = time.perf_counter_ns()
        decoded = decode(blob)
        t2 = time.perf_counter_ns()
        enc_ns.append(t1 - t0)
        dec_ns.append(t2 - t1)
        # Comparisons stay outside the timed regions.
        if again != blob:
            raise RuntimeError(f'{name}/{method}: encode output is not deterministic')
        if not _matches(decoded, arr, expected):
            raise RuntimeError(f'{name}/{method}: decode mismatch')

    raw_bytes = len(expected)
    return {
        'map': name,
        'source': source,
        'method': method,
        'sha256': digest,
        'shape': [int(n) for n in arr.shape],
        'raw_bytes': raw_bytes,
        'container_bytes': len(blob),
        'ratio': round(len(blob) / raw_bytes, 6),
        'encode_median_ms': round(statistics.median(enc_ns) / 1e6, 4),
        'decode_median_ms': round(statistics.median(dec_ns) / 1e6, 4),
        'decode_p95_ms': round(float(np.percentile(dec_ns, 95, method='linear')) / 1e6, 4),
        'exact_match': True,
    }


def run_benchmark(size=256, repeats=30, inputs=()):
    '''Benchmark every codec method on synthetic fixtures and optional inputs.

    inputs holds .npy paths. Everything is validated
    before any timing starts.
    '''
    size = _int_arg('size', size, 1)
    repeats = _int_arg('repeats', repeats, MIN_REPEATS)
    if size * size * 2 > MAX_RAW_BYTES:
        raise ValueError(f'size {size} exceeds the 64 MiB raw limit')

    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    user_maps = []
    for item in inputs:
        label = Path(item).name
        user_maps.append((label, 'file', load_heightmap(item)))

    maps = make_fixtures(size) + user_maps

    results = []
    for name, source, arr in maps:
        expected = np.ascontiguousarray(arr).tobytes()
        digest = hashlib.sha256(expected).hexdigest()
        for method in METHODS:
            results.append(
                _bench_one(name, source, arr, expected, digest, method, repeats)
            )

    return {
        'schema_version': SCHEMA_VERSION,
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'numpy': np.__version__,
            'zstandard': _version('zstandard'),
            'lz4': _version('lz4'),
        },
        'settings': {
            'size': size,
            'repeats': repeats,
            'warmup': WARMUP,
            'seed': SEED,
            'methods': list(METHODS),
            'container_version': VERSION,
            'container_header_bytes': HEADER_SIZE,
            'codec_settings': {
                'zstd': f'level={ZSTD_LEVEL}, write_content_size=True',
                'lz4': 'lz4.frame defaults, store_size=True',
                'delta_zstd': f'row-left delta modulo 65536, Zstd level={ZSTD_LEVEL}',
            },
            'p95_method': 'numpy.percentile, linear interpolation',
            'max_raw_bytes': MAX_RAW_BYTES,
            'input_count': len(user_maps),
        },
        'results': results,
    }

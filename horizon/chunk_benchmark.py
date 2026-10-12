'''Whole-map vs. independent-chunk container benchmark for the codec.

Timings cover in-memory, already-warm encode()/decode() calls only: every
timed code path is exercised once, untimed, before the repeats loop (using
the identical helper function the loop calls), and a fresh encode's output is
only compared for determinism after its timer stops. There is no disk I/O and
no claim about any storage/streaming engine.

For each chunk size, the heightmap is cut into a grid of independent chunks
(edge chunks are shrunk to fit when the dimensions do not divide evenly), each
chunk is encoded on its own with the existing container codec, and a small
binary index (coords, shape, offset, length per chunk) is packed in front of
the concatenated chunk blobs. That index plus the chunk headers/payloads is
compared against the single whole-map container, for every entry in
horizon.codec.METHODS.

Random-region retrieval compares an O(1) index-entry lookup + container slice
+ single chunk decode against decoding the whole map and copying the same
region out of it, across a deterministic schedule of distinct grid chunks
(shared across methods, logged and verified per repeat). Execution order of
each measured pair alternates by repeat parity to reduce ordering bias.

The index/container format here is private to this benchmark, not a
production format: see the note above _unpack_index for what it does not
defend against.
'''

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import statistics
import struct
import time

import numpy as np

from horizon.benchmark import validate_heightmap
from horizon.codec import HEADER_SIZE, METHODS, decode, encode

MIN_REPEATS = 5
WARMUP = 1
SCHEMA_VERSION = 2
_DTYPE = np.dtype('<u2')
_U32_MAX = 0xFFFFFFFF

_INDEX_MAGIC = b'HCIX'
_INDEX_VERSION = 1
# magic, version, method_id, reserved, chunk_size, grid_rows, grid_cols, chunk_count
_INDEX_HEADER = struct.Struct('<4sBBHIIII')
# row, col, height, width, offset, length
_INDEX_ENTRY = struct.Struct('<IIIIQQ')

ORDER_POLICY = (
    'per repeat, swap execution order of each measured pair (whole-map vs. '
    'chunked) by repeat index parity: even repeats run whole-first, odd '
    'repeats run chunk-first; applied independently to the encode pair, the '
    'decode/reconstruct pair, and the region-retrieval pair'
)
REGION_SCHEDULE_POLICY = (
    'deterministic schedule of grid chunk indices built from cyclic random '
    'permutations (numpy Generator.permutation) of the shared seeded rng, '
    'generated once per chunk_size before the method loop and reused '
    'unchanged for every codec method; each full cycle visits every grid '
    'chunk exactly once'
)
CONTAINER_FORMAT_NOTE = (
    'private benchmark container, not a production format; the index '
    'parser below trusts its own just-built output and performs no '
    'adversarial validation (no bounds/overflow checks on offset or length, '
    'no checksum) because it only ever parses data this module itself '
    'produced, never untrusted input'
)


def _int_arg(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return int(value)


def _validate_chunk_sizes(chunk_sizes):
    if isinstance(chunk_sizes, (int, np.integer)) and not isinstance(chunk_sizes, bool):
        chunk_sizes = (chunk_sizes,)
    try:
        values = list(chunk_sizes)
    except TypeError:
        raise ValueError('chunk_sizes must be an iterable of positive integers') from None
    if not values:
        raise ValueError('chunk_sizes must not be empty')
    out = []
    for v in values:
        n = _int_arg('chunk_sizes element', v, 1)
        if n > _U32_MAX:
            raise ValueError(
                f'chunk_sizes element {n} exceeds the u32 index field limit {_U32_MAX}'
            )
        out.append(n)
    if len(set(out)) != len(out):
        raise ValueError('chunk_sizes must not contain duplicate values')
    return tuple(out)


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'


def _grid_dims(height, width, chunk_size):
    rows = (height + chunk_size - 1) // chunk_size
    cols = (width + chunk_size - 1) // chunk_size
    return rows, cols


def _chunk_entries(height, width, chunk_size):
    '''List (row, col, height, width) of every grid cell; edge cells shrink to fit.'''
    entries = []
    for row in range(0, height, chunk_size):
        h = min(chunk_size, height - row)
        for col in range(0, width, chunk_size):
            w = min(chunk_size, width - col)
            entries.append((row, col, h, w))
    return entries


def _region_schedule(rng, chunk_count, repeats):
    '''Deterministic repeats-long schedule of chunk indices; each full cycle
    is a random permutation covering every chunk once, so coverage stays
    diverse even when repeats exceeds chunk_count.'''
    schedule = []
    while len(schedule) < repeats:
        schedule.extend(int(x) for x in rng.permutation(chunk_count))
    return schedule[:repeats]


def _matches_bytes(decoded, shape, dtype, expected_bytes):
    return (
        isinstance(decoded, np.ndarray)
        and decoded.shape == shape
        and decoded.dtype == dtype
        and decoded.tobytes() == expected_bytes
    )


def _crop_copy(decoded, row, col, h, w):
    '''Materialize an independently owned region copy, not a view, so this
    arm's result is owned the same way decode() always owns its output.'''
    return decoded[row:row + h, col:col + w].copy()


def _encode_chunks(arr, entries, method):
    return [encode(arr[row:row + h, col:col + w], method=method) for row, col, h, w in entries]


def _pack_index(method, chunk_size, grid_rows, grid_cols, packed_entries):
    header = _INDEX_HEADER.pack(
        _INDEX_MAGIC, _INDEX_VERSION, METHODS.index(method), 0,
        chunk_size, grid_rows, grid_cols, len(packed_entries),
    )
    body = b''.join(
        _INDEX_ENTRY.pack(row, col, h, w, offset, length)
        for row, col, h, w, offset, length in packed_entries
    )
    return header + body


# Private benchmark-only container format, not a production format. The
# parsing below trusts its own just-built output: it does not defend against
# adversarial or corrupted offsets/lengths (no bounds checks, no checksum)
# because this module only ever parses data it produced itself.
def _unpack_index(container):
    magic, version, method_id, _reserved, chunk_size, grid_rows, grid_cols, count = (
        _INDEX_HEADER.unpack_from(container)
    )
    if magic != _INDEX_MAGIC:
        raise ValueError('bad chunk index magic')
    if version != _INDEX_VERSION:
        raise ValueError(f'unsupported chunk index version {version}')
    if method_id >= len(METHODS):
        raise ValueError(f'unknown chunk index method id {method_id}')
    offset = _INDEX_HEADER.size
    entries = []
    for _ in range(count):
        row, col, h, w, chunk_offset, length = _INDEX_ENTRY.unpack_from(container, offset)
        entries.append((row, col, h, w, chunk_offset, length))
        offset += _INDEX_ENTRY.size
    return METHODS[method_id], chunk_size, grid_rows, grid_cols, entries, offset


def _lookup_entry(container, chunk_count, chunk_index):
    '''O(1) index entry lookup: direct offset arithmetic + a single
    unpack_from, no scan over the other entries. Returns an absolute slice
    bound into `container` ready to pass straight to decode().'''
    payload_base = _INDEX_HEADER.size + chunk_count * _INDEX_ENTRY.size
    entry_offset = _INDEX_HEADER.size + chunk_index * _INDEX_ENTRY.size
    row, col, h, w, offset, length = _INDEX_ENTRY.unpack_from(container, entry_offset)
    return row, col, h, w, payload_base + offset, length


def _build_container(method, chunk_size, grid_rows, grid_cols, entries, blobs):
    packed_entries = []
    offset = 0
    for (row, col, h, w), blob in zip(entries, blobs):
        packed_entries.append((row, col, h, w, offset, len(blob)))
        offset += len(blob)
    index_bytes = _pack_index(method, chunk_size, grid_rows, grid_cols, packed_entries)
    return index_bytes + b''.join(blobs), index_bytes


def _reconstruct_container(container, height, width):
    _method, _chunk_size, _grid_rows, _grid_cols, entries, base = _unpack_index(container)
    out = np.empty((height, width), dtype=_DTYPE)
    for row, col, h, w, chunk_offset, length in entries:
        blob = container[base + chunk_offset: base + chunk_offset + length]
        out[row:row + h, col:col + w] = decode(blob)
    return out


def _stats(samples_ns):
    ms = [n / 1e6 for n in samples_ns]
    return {
        'raw_ms': [round(x, 6) for x in ms],
        'median_ms': round(statistics.median(ms), 6),
        'p95_ms': round(float(np.percentile(ms, 95, method='linear')), 6),
    }


def _ordered_pair(swapped, first, second):
    '''Run two zero-arg callables in parity-swapped order, but always return
    (first_result, second_result) so callers don't care which ran first.'''
    if swapped:
        b = second()
        a = first()
    else:
        a = first()
        b = second()
    return a, b


def _time_whole_encode(arr, method, blob_whole, chunk_size):
    t0 = time.perf_counter_ns()
    again = encode(arr, method=method)
    t1 = time.perf_counter_ns()
    if again != blob_whole:
        raise RuntimeError(f'{method}/chunk={chunk_size}: whole-map encode is not deterministic')
    return t1 - t0


def _time_chunked_encode(arr, entries, method, chunk_size, grid_rows, grid_cols, container):
    t0 = time.perf_counter_ns()
    blobs = _encode_chunks(arr, entries, method)
    built, _index_bytes = _build_container(method, chunk_size, grid_rows, grid_cols, entries, blobs)
    t1 = time.perf_counter_ns()
    if built != container:
        raise RuntimeError(f'{method}/chunk={chunk_size}: chunked encode is not deterministic')
    return t1 - t0


def _time_whole_decode(blob_whole, shape, dtype, expected_bytes, method, chunk_size):
    t0 = time.perf_counter_ns()
    decoded = decode(blob_whole)
    t1 = time.perf_counter_ns()
    if not _matches_bytes(decoded, shape, dtype, expected_bytes):
        raise RuntimeError(f'{method}/chunk={chunk_size}: whole-map decode mismatch')
    return t1 - t0


def _time_chunked_reconstruct(container, height, width, shape, dtype, expected_bytes, method, chunk_size):
    t0 = time.perf_counter_ns()
    reconstructed = _reconstruct_container(container, height, width)
    t1 = time.perf_counter_ns()
    if not _matches_bytes(reconstructed, shape, dtype, expected_bytes):
        raise RuntimeError(f'{method}/chunk={chunk_size}: chunked reconstruct mismatch')
    return t1 - t0


def _time_region_chunk_decode(
    container, chunk_count, chunk_index, region_shape, dtype, expected_bytes, method, chunk_size
):
    t0 = time.perf_counter_ns()
    _row, _col, _h, _w, abs_offset, length = _lookup_entry(container, chunk_count, chunk_index)
    blob = container[abs_offset: abs_offset + length]
    decoded = decode(blob)
    t1 = time.perf_counter_ns()
    if not _matches_bytes(decoded, region_shape, dtype, expected_bytes):
        raise RuntimeError(f'{method}/chunk={chunk_size}: region chunk decode mismatch')
    return t1 - t0


def _time_region_full_decode_crop(blob_whole, row, col, h, w, dtype, expected_bytes, method, chunk_size):
    t0 = time.perf_counter_ns()
    full_decoded = decode(blob_whole)
    cropped = _crop_copy(full_decoded, row, col, h, w)
    t1 = time.perf_counter_ns()
    if not _matches_bytes(cropped, (h, w), dtype, expected_bytes):
        raise RuntimeError(f'{method}/chunk={chunk_size}: region full-decode-crop mismatch')
    return t1 - t0


def _bench_chunk_config(
    arr, height, width, chunk_size, grid_rows, grid_cols, entries,
    schedule, expected_whole_bytes, method, repeats,
):
    dtype = arr.dtype
    chunk_count = len(entries)

    # Bootstrap the ground-truth blob/container this config will time against.
    # Not timed: establishing a reference has no timed counterpart to bias.
    blob_whole = encode(arr, method=method)
    chunk_blobs = _encode_chunks(arr, entries, method)
    container, index_bytes = _build_container(method, chunk_size, grid_rows, grid_cols, entries, chunk_blobs)

    region_expected_bytes = []
    for chunk_index in schedule:
        row, col, h, w = entries[chunk_index]
        region_expected_bytes.append(arr[row:row + h, col:col + w].tobytes())

    # Untimed warmup: call every timed helper once, discarding its elapsed
    # time, so each timed path (including the one about to run `repeats`
    # times) has already run and been verified at least once beforehand.
    _time_whole_decode(blob_whole, arr.shape, dtype, expected_whole_bytes, method, chunk_size)
    _time_whole_encode(arr, method, blob_whole, chunk_size)
    _time_chunked_reconstruct(
        container, height, width, arr.shape, dtype, expected_whole_bytes, method, chunk_size
    )
    _time_chunked_encode(arr, entries, method, chunk_size, grid_rows, grid_cols, container)
    first_chunk_index = schedule[0]
    first_row, first_col, first_h, first_w = entries[first_chunk_index]
    _time_region_chunk_decode(
        container, chunk_count, first_chunk_index, (first_h, first_w), dtype,
        region_expected_bytes[0], method, chunk_size,
    )
    _time_region_full_decode_crop(
        blob_whole, first_row, first_col, first_h, first_w, dtype,
        region_expected_bytes[0], method, chunk_size,
    )

    whole_encode_ns = []
    chunked_encode_ns = []
    whole_decode_ns = []
    chunked_reconstruct_ns = []
    region_chunk_decode_ns = []
    region_full_decode_crop_ns = []
    region_descriptors = []

    for i in range(repeats):
        chunk_index = schedule[i]
        row, col, h, w = entries[chunk_index]
        expected_region = region_expected_bytes[i]
        swapped = bool(i % 2)

        whole_ns, chunk_ns = _ordered_pair(
            swapped,
            lambda: _time_whole_encode(arr, method, blob_whole, chunk_size),
            lambda: _time_chunked_encode(arr, entries, method, chunk_size, grid_rows, grid_cols, container),
        )
        whole_encode_ns.append(whole_ns)
        chunked_encode_ns.append(chunk_ns)

        whole_dec_ns, chunk_rec_ns = _ordered_pair(
            swapped,
            lambda: _time_whole_decode(blob_whole, arr.shape, dtype, expected_whole_bytes, method, chunk_size),
            lambda: _time_chunked_reconstruct(
                container, height, width, arr.shape, dtype, expected_whole_bytes, method, chunk_size
            ),
        )
        whole_decode_ns.append(whole_dec_ns)
        chunked_reconstruct_ns.append(chunk_rec_ns)

        region_chunk_ns, region_crop_ns = _ordered_pair(
            swapped,
            lambda: _time_region_chunk_decode(
                container, chunk_count, chunk_index, (h, w), dtype, expected_region, method, chunk_size
            ),
            lambda: _time_region_full_decode_crop(
                blob_whole, row, col, h, w, dtype, expected_region, method, chunk_size
            ),
        )
        region_chunk_decode_ns.append(region_chunk_ns)
        region_full_decode_crop_ns.append(region_crop_ns)

        region_descriptors.append({
            'repeat': i,
            'row': row,
            'col': col,
            'height': h,
            'width': w,
            'chunk_index': chunk_index,
            'verified': True,
        })

    chunk_payload_bytes = sum(len(b) for b in chunk_blobs)
    chunked_container_bytes = len(index_bytes) + chunk_payload_bytes

    return {
        'method': method,
        'chunk_size': chunk_size,
        'grid': [grid_rows, grid_cols],
        'chunk_count': chunk_count,
        'whole_container_bytes': len(blob_whole),
        'chunk_index_bytes': len(index_bytes),
        'chunk_payload_bytes': chunk_payload_bytes,
        'chunked_container_bytes': chunked_container_bytes,
        'size_ratio_chunked_to_whole': round(chunked_container_bytes / len(blob_whole), 6),
        'region_schedule': region_descriptors,
        'whole_encode': _stats(whole_encode_ns),
        'chunked_encode': _stats(chunked_encode_ns),
        'whole_decode': _stats(whole_decode_ns),
        'chunked_reconstruct': _stats(chunked_reconstruct_ns),
        'region_chunk_decode': _stats(region_chunk_decode_ns),
        'region_full_decode_crop': _stats(region_full_decode_crop_ns),
        'exact_match': True,
    }


def run_chunk_benchmark(arr, chunk_sizes=(64, 128, 256, 512), repeats=30, seed=20261012):
    '''Benchmark whole-map vs. independent-chunk containers for every codec method.

    In-memory only: no disk I/O, no storage/streaming engine involved. Every
    timed path is warmed up once (untimed) before its `repeats` timed calls,
    and a timed encode's output is only compared for determinism after its
    timer has stopped. Random-region retrieval walks a deterministic,
    diverse schedule of grid chunks (shared across methods, verified every
    repeat) instead of hammering a single region. Execution order of each
    measured pair alternates by repeat parity (see ORDER_POLICY) to reduce
    ordering bias. The chunk index/container is a private benchmark format,
    not a production one (see CONTAINER_FORMAT_NOTE).
    '''
    validate_heightmap(arr, label='arr')
    chunk_sizes = _validate_chunk_sizes(chunk_sizes)
    repeats = _int_arg('repeats', repeats, MIN_REPEATS)
    seed = _int_arg('seed', seed, 0)

    arr = np.ascontiguousarray(arr)
    height, width = arr.shape
    expected_whole_bytes = arr.tobytes()
    input_sha256 = hashlib.sha256(expected_whole_bytes).hexdigest()

    rng = np.random.default_rng(seed)

    results = []
    for chunk_size in chunk_sizes:
        entries = _chunk_entries(height, width, chunk_size)
        grid_rows, grid_cols = _grid_dims(height, width, chunk_size)
        schedule = _region_schedule(rng, len(entries), repeats)

        for method in METHODS:
            results.append(
                _bench_chunk_config(
                    arr, height, width, chunk_size, grid_rows, grid_cols, entries,
                    schedule, expected_whole_bytes, method, repeats,
                )
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
        'input': {
            'shape': [height, width],
            'dtype': str(_DTYPE),
            'raw_bytes': arr.nbytes,
            'sha256': input_sha256,
        },
        'settings': {
            'chunk_sizes': [int(c) for c in chunk_sizes],
            'repeats': repeats,
            'warmup': WARMUP,
            'seed': seed,
            'methods': list(METHODS),
            'p95_method': 'numpy.percentile, linear interpolation',
            'container_header_bytes': HEADER_SIZE,
            'chunk_index_header_bytes': _INDEX_HEADER.size,
            'chunk_index_entry_bytes': _INDEX_ENTRY.size,
            'order_policy': ORDER_POLICY,
            'region_schedule_policy': REGION_SCHEDULE_POLICY,
            'container_format': CONTAINER_FORMAT_NOTE,
            'scope': 'in-memory warm benchmark only; no disk or engine I/O measured',
        },
        'results': results,
    }

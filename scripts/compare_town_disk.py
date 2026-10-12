#!/usr/bin/env python3
'''Real disk-load benchmark: whole vs chunked containers of the synthetic town map.

Generates one deterministic heightmap via horizon.town_map.make_town_map,
prepares on-disk containers via horizon.town_disk.prepare_case (whole +
chunk_size=256 chunked, per codec method), and times two workloads per
method/mode from real files on disk:

  - full:    load_whole(region=None) vs load_chunked(region=None) -- both
             reconstruct the entire map.
  - partial: load_whole(region=chunk) [whole decode + owned crop] vs
             load_chunked(region=chunk) [true single-chunk disk read] --
             region is always one exact, grid-aligned chunk.

This module does not implement horizon.town_disk; it only calls the
documented API (prepare_case, load_whole, load_chunked). See
docs/town-disk-method.md for full scope, I/O-boundary notes and caveats.
'''
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from horizon.benchmark import validate_heightmap  # noqa: E402
from horizon.codec import METHODS  # noqa: E402
from horizon.town_map import MAX_SIZE, MIN_SIZE, make_town_map  # noqa: E402

DEFAULT_SIZE = 1024
DEFAULT_SEED = 20261012
DEFAULT_REPEATS = 50
MIN_REPEATS = 30
WARMUPS = 2
CHUNK_SIZE = 256  # fixed; horizon.town_disk.prepare_case default
SEED_EXCLUSIVE_MAX = 2 ** 64

ALL_MODES = ('warm_buffered', 'macos_nocache')
_DTYPE = np.dtype('<u2')

WORKLOADS = ('full', 'partial')

ORDER_POLICY = (
    'per repeat, swap execution order of the measured pair (whole vs. '
    'chunked) by repeat index parity: even repeats run whole-first, odd '
    'repeats run chunk-first; applied independently to the full workload '
    'and the partial workload. Mode and method ordering is static (outer '
    'loops: method, then mode), not alternated -- recorded here as a '
    'caveat rather than implemented, since both are set once per run from '
    '--modes/METHODS and the per-repeat whole/chunk swap already covers '
    'the dominant ordering bias within a (method, mode) cell.'
)
REGION_SCHEDULE_POLICY = (
    'per repeat, a pixel coordinate (row_px, col_px) is drawn uniformly at '
    'random from the seeded RNG via Generator.integers(0, size) '
    'independently per axis -- every pixel in [0, size-1], including the '
    'last row/column, is reachable with equal probability. That pixel is '
    'mapped to its exact, grid-aligned chunk_size=256 chunk (edge chunks '
    'shrink to fit, same convention as horizon.chunk_benchmark). One '
    'schedule entry is drawn per repeat (length == --repeats, no '
    'truncation), and the same per-repeat schedule of pixels/chunks is '
    'reused for every method and mode. Both the drawn pixel and the '
    'resulting region are recorded per repeat.'
)
PRIMING_POLICY = (
    'warm_buffered: before every measured load_whole/load_chunked call, '
    'this script does one untimed ordinary buffered read of BOTH the whole '
    'container file and the chunked container file for that method, '
    'regardless of which one is about to be measured, so both files are '
    'symmetrically warmed in the OS page cache. macos_nocache: no priming '
    'is performed by this script; this is NOT a claim of a physically cold '
    'read. horizon.town_disk only requests fcntl(fd, F_NOCACHE, 1) on the '
    'reading descriptor -- it has not been verified to evict any page, and '
    'it has not been verified to reach the SSD/controller/drive cache. It '
    'is also not a claim that the OS page cache never serves this read from '
    'pages already resident from an earlier warmup, repeat, or process: '
    'F_NOCACHE is only documented to stop this descriptor from retaining '
    'newly read data in the cache, not to force every read to bypass '
    'already-cached pages.'
)


def _log(message):
    print(message, file=sys.stderr, flush=True)


class ValidationError(RuntimeError):
    pass


def _positive_int(name, text, minimum, maximum=None):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{name} must be an integer') from None
    if value < minimum:
        raise argparse.ArgumentTypeError(f'{name} must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise argparse.ArgumentTypeError(f'{name} must be <= {maximum}')
    return value


def _size_type(text):
    return _positive_int('size', text, MIN_SIZE, MAX_SIZE)


def _seed_type(text):
    return _positive_int('seed', text, 0, SEED_EXCLUSIVE_MAX - 1)


def _repeats_type(text):
    return _positive_int('repeats', text, MIN_REPEATS)


def _default_modes():
    return 'warm_buffered,macos_nocache' if sys.platform == 'darwin' else 'warm_buffered'


def _modes_type(text):
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('--modes must list at least one mode')
    seen = set()
    out = []
    for p in parts:
        if p not in ALL_MODES:
            raise argparse.ArgumentTypeError(f'--modes: unknown mode {p!r}; expected one of {ALL_MODES}')
        if p in seen:
            raise argparse.ArgumentTypeError(f'--modes: duplicate mode {p!r}')
        seen.add(p)
        out.append(p)
    return tuple(out)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Real disk-load comparison of whole vs chunked containers for the '
                    'synthetic town map (single seed per run).')
    parser.add_argument('--size', type=_size_type, default=DEFAULT_SIZE)
    parser.add_argument('--seed', type=_seed_type, default=DEFAULT_SEED)
    parser.add_argument('--repeats', type=_repeats_type, default=DEFAULT_REPEATS)
    parser.add_argument('--output', required=True, type=Path,
                         help='new JSON report path; never overwritten')
    parser.add_argument('--containers-dir', required=True, type=Path,
                         help='new directory for prepared container files; must not exist')
    parser.add_argument('--modes', type=_modes_type, default=_default_modes())
    return parser.parse_args(argv)


def _import_town_disk():
    '''Lazy import: horizon.town_disk is owned/built separately. No stub is
    fabricated here -- if it is not yet present this raises a clear error
    instead of a silent fallback.'''
    try:
        return importlib.import_module('horizon.town_disk')
    except ModuleNotFoundError as exc:
        raise ValidationError(
            'horizon.town_disk is not available yet; this script only consumes its '
            'documented API (prepare_case/load_whole/load_chunked) and does not provide '
            'a fallback implementation'
        ) from exc


def check_free(path, label):
    if path.exists() or path.is_symlink():
        raise ValidationError(f'{label} already exists, refusing to overwrite: {path}')


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _source_files_sha256(town_disk_module):
    pkg_dir = (REPO_ROOT / 'horizon').resolve()
    hashes = {'scripts/compare_town_disk.py': _sha256_file(Path(__file__).resolve())}
    for p in sorted(pkg_dir.glob('*.py')):
        hashes[p.relative_to(REPO_ROOT).as_posix()] = _sha256_file(p)
    return hashes


def _versions():
    out = {}
    for dist in ('numpy', 'zstandard', 'lz4'):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _environment():
    return {
        'python': sys.version,
        'python_implementation': platform.python_implementation(),
        'os': platform.platform(),
        'os_version': platform.version(),
        'machine': platform.machine(),
        'cpu': platform.processor(),
        'cpu_count': os.cpu_count(),
        'libraries': _versions(),
    }


def _chunk_entries(height, width, chunk_size):
    '''List (row, col, h, w) of every grid cell; edge cells shrink to fit --
    same convention as horizon.chunk_benchmark._chunk_entries.'''
    entries = []
    for row in range(0, height, chunk_size):
        h = min(chunk_size, height - row)
        for col in range(0, width, chunk_size):
            w = min(chunk_size, width - col)
            entries.append((row, col, h, w))
    return entries


def _chunk_for_pixel(entries, chunk_size, grid_cols, row_px, col_px):
    row_block = row_px // chunk_size
    col_block = col_px // chunk_size
    return entries[row_block * grid_cols + col_block]


def _region_schedule(rng, size, entries, chunk_size, grid_cols, repeats):
    '''One (pixel, region) pair per repeat; pixel coords are uniform over the
    full [0, size-1] range on each axis (rng.integers' upper bound is
    exclusive, so size-1 itself is reachable), independently per axis.'''
    schedule = []
    for _ in range(repeats):
        row_px, col_px = (int(v) for v in rng.integers(0, size, size=2))
        region = _chunk_for_pixel(entries, chunk_size, grid_cols, row_px, col_px)
        schedule.append({'row_px': row_px, 'col_px': col_px, 'region': region})
    return schedule


def _validate_loaded(arr, expected_shape, expected_bytes, label):
    if type(arr) is not np.ndarray:
        raise ValidationError(f'{label}: expected ndarray, got {type(arr).__name__}')
    if arr.shape != expected_shape:
        raise ValidationError(f'{label}: shape mismatch {arr.shape} != {expected_shape}')
    if arr.dtype != _DTYPE:
        raise ValidationError(f'{label}: dtype mismatch {arr.dtype} != {_DTYPE}')
    if arr.tobytes() != expected_bytes:
        raise ValidationError(f'{label}: byte content mismatch')


def _read_buffered_discard(path):
    with open(path, 'rb') as f:
        while f.read(1 << 20):
            pass


def _prime(mode, paths):
    if mode != 'warm_buffered':
        return
    for path in paths:
        _read_buffered_discard(path)


def _ordered_pair(swapped, first, second):
    if swapped:
        b = second()
        a = first()
    else:
        a = first()
        b = second()
    return a, b


def _stats(samples_ns):
    ms = [n / 1e6 for n in samples_ns]
    return {
        'raw_ns': list(samples_ns),
        'median_ms': round(statistics.median(ms), 6),
        'p95_ms': round(float(np.percentile(ms, 95, method='linear')), 6),
    }


def _metric_block(metric_dicts):
    return {
        'total': _stats([m['total_ns'] for m in metric_dicts]),
        'read': _stats([m['read_ns'] for m in metric_dicts]),
        'decode': _stats([m['decode_ns'] for m in metric_dicts]),
        'bytes_read': [int(m['bytes_read']) for m in metric_dicts],
        'bytes_requested': [int(m.get('bytes_requested', m['bytes_read'])) for m in metric_dicts],
        'read_calls': [int(m['read_calls']) for m in metric_dicts],
        'seek_calls': [int(m.get('seek_calls', 0)) for m in metric_dicts],
    }


def _run_full(town_disk, whole_path, chunked_path, mode, expected_bytes, expected_shape, swapped):
    whole_paths = (whole_path, chunked_path)

    def do_whole():
        _prime(mode, whole_paths)
        arr, metrics = town_disk.load_whole(whole_path, mode=mode, region=None)
        _validate_loaded(arr, expected_shape, expected_bytes, 'full/whole')
        return metrics

    def do_chunked():
        _prime(mode, whole_paths)
        arr, metrics = town_disk.load_chunked(chunked_path, mode=mode, region=None)
        _validate_loaded(arr, expected_shape, expected_bytes, 'full/chunked')
        return metrics

    return _ordered_pair(swapped, do_whole, do_chunked)


def _run_partial(town_disk, whole_path, chunked_path, mode, arr_source, region, swapped):
    row, col, h, w = region
    expected_shape = (h, w)
    expected_bytes = np.ascontiguousarray(arr_source[row:row + h, col:col + w]).tobytes()
    paths = (whole_path, chunked_path)

    def do_whole_crop():
        _prime(mode, paths)
        cropped, metrics = town_disk.load_whole(whole_path, mode=mode, region=region)
        _validate_loaded(cropped, expected_shape, expected_bytes, 'partial/whole_decode_crop')
        return metrics

    def do_chunk_read():
        _prime(mode, paths)
        chunk, metrics = town_disk.load_chunked(chunked_path, mode=mode, region=region)
        _validate_loaded(chunk, expected_shape, expected_bytes, 'partial/single_chunk_read')
        return metrics

    return _ordered_pair(swapped, do_whole_crop, do_chunk_read)


def _benchmark_case(town_disk, arr, case, mode, repeats, entries, schedule, grid_cols):
    whole_path = Path(case['whole_path'])
    chunked_path = Path(case['chunked_path'])
    height, width = arr.shape

    for _ in range(WARMUPS):
        _run_full(town_disk, whole_path, chunked_path, mode, arr.tobytes(), arr.shape, False)
        _run_partial(town_disk, whole_path, chunked_path, mode, arr, entries[0], False)

    full_whole, full_chunk = [], []
    partial_whole, partial_chunk = [], []
    region_descriptors = []
    expected_whole_bytes = arr.tobytes()

    for i in range(repeats):
        swapped = bool(i % 2)
        w_metrics, c_metrics = _run_full(
            town_disk, whole_path, chunked_path, mode, expected_whole_bytes, arr.shape, swapped
        )
        full_whole.append(w_metrics)
        full_chunk.append(c_metrics)

        entry = schedule[i]
        region = entry['region']
        pw_metrics, pc_metrics = _run_partial(town_disk, whole_path, chunked_path, mode, arr, region, swapped)
        partial_whole.append(pw_metrics)
        partial_chunk.append(pc_metrics)
        row, col, h, w = region
        region_descriptors.append({
            'repeat': i,
            'row_px': entry['row_px'],
            'col_px': entry['col_px'],
            'row': row,
            'col': col,
            'height': h,
            'width': w,
        })

    return {
        'full': {'whole': _metric_block(full_whole), 'chunked': _metric_block(full_chunk)},
        'partial': {
            'whole_decode_crop': _metric_block(partial_whole),
            'single_chunk_read': _metric_block(partial_chunk),
        },
        'region_schedule': region_descriptors,
    }


def _file_hash_snapshot(case):
    return {
        'whole': _sha256_file(case['whole_path']),
        'chunked': _sha256_file(case['chunked_path']),
    }


SCOPE = (
    'Real disk reads of prepared per-method container files (one whole, one '
    'chunk_size=256 chunked, per codec method) followed by horizon.town_disk '
    'decode, on this machine only. "full" times reconstructing the entire map '
    'from each container; "partial" times retrieving one exact, grid-aligned '
    'chunk via a true single-chunk read against decoding the whole container '
    'and cropping an owned copy out of it. Chunk-index/header overhead is '
    'included in the chunked timings and byte counts as read off disk -- it is '
    'not subtracted out. There is no multi-chunk arbitrary-ROI engine here: '
    'region() only ever addresses one exact chunk on the fixed chunk_size=256 '
    'grid, never an arbitrary rectangle spanning chunk boundaries. '
    'warm_buffered is explicitly primed by this script (see priming_policy). '
    'macos_nocache only requests fcntl(F_NOCACHE) on the reading handle (see '
    'priming_policy for what that does and does not establish); it is not a '
    'verified physically cold read. bytes_read/bytes_requested/read_calls '
    'are the application-level read() syscall counts horizon.town_disk\'s '
    'reader itself tallies, not a measurement of physical SSD/controller '
    'transfer size.'
)


def _run(args, town_disk, cases):
    '''cases is populated in place (even if this raises partway through) so
    the caller can clean up exactly the files this run created.'''
    arr = make_town_map(size=args.size, seed=args.seed)
    validate_heightmap(arr, 'make_town_map output')
    arr = np.ascontiguousarray(arr)
    height, width = arr.shape

    entries = _chunk_entries(height, width, CHUNK_SIZE)
    grid_cols = -(-width // CHUNK_SIZE)
    rng = np.random.default_rng(args.seed)
    schedule = _region_schedule(rng, args.size, entries, CHUNK_SIZE, grid_cols, args.repeats)

    for method in METHODS:
        _log(f'preparing case: method={method}')
        # prepare_case creates its own target subdirectory exclusively inside
        # the already-created --containers-dir root; each method gets its own.
        case = town_disk.prepare_case(arr, args.containers_dir / method, method, chunk_size=CHUNK_SIZE)
        case['method'] = method
        # Register immediately, before anything that can still fail (the key
        # check, hashing): cleanup must be able to find these real files even
        # if a later step in this loop iteration raises.
        cases.append(case)
        for key in ('whole_path', 'chunked_path'):
            if key not in case:
                raise ValidationError(f'prepare_case({method!r}) missing {key!r}')
        case['hashes_before'] = _file_hash_snapshot(case)

    for case in cases:
        method = case['method']
        case['modes'] = {}
        for mode in args.modes:
            _log(f'benchmarking method={method} mode={mode}')
            case['modes'][mode] = _benchmark_case(
                town_disk, arr, case, mode, args.repeats, entries, schedule, grid_cols
            )
        case['hashes_after'] = _file_hash_snapshot(case)
        if case['hashes_after'] != case['hashes_before']:
            raise ValidationError(f'container file(s) changed during benchmark for method={method}')

    report = {
        'schema_version': 1,
        'scope': SCOPE,
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'source_files_sha256': _source_files_sha256(town_disk),
        'environment': _environment(),
        'map': {
            'source': 'horizon.town_map.make_town_map (same original synthetic map as '
                      'scripts/compare_town.py)',
            'shape': [int(n) for n in arr.shape],
            'dtype': arr.dtype.str,
            'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
        },
        'settings': {
            'size': args.size,
            'seed': args.seed,
            'repeats': args.repeats,
            'warmups': WARMUPS,
            'chunk_size': CHUNK_SIZE,
            'methods': list(METHODS),
            'modes': list(args.modes),
            'workloads': list(WORKLOADS),
            'p95_method': 'numpy.percentile, linear interpolation',
            'order_policy': ORDER_POLICY,
            'region_schedule_policy': REGION_SCHEDULE_POLICY,
            'priming_policy': PRIMING_POLICY,
        },
        'cases': cases,
    }
    return report


def _cleanup_own_containers(containers_dir, created_dir, cases):
    '''Remove only files/directories this run itself wrote: each method's
    whole/chunked container files and prepare_case's per-method subdirectory
    (removed only if now empty), then --containers-dir itself -- but ONLY if
    `created_dir` says this run created that root. A pre-existing
    --containers-dir (e.g. left over from a run that failed before reaching
    this one, or simply present from something else) must never be removed
    by this script, empty or not.'''
    for case in cases:
        for key in ('whole_path', 'chunked_path'):
            path = case.get(key)
            if path:
                with contextlib.suppress(OSError):
                    Path(path).unlink()
        method = case.get('method')
        if method:
            with contextlib.suppress(OSError):
                (containers_dir / method).rmdir()  # only succeeds if empty
    if created_dir:
        with contextlib.suppress(OSError):
            containers_dir.rmdir()  # only succeeds if empty


def main(argv=None):
    args = parse_args(argv)
    output = args.output
    containers_dir = args.containers_dir
    cases = []
    created_dir = False

    try:
        check_free(output, 'output')
        check_free(containers_dir, 'containers-dir')
        if 'macos_nocache' in args.modes and sys.platform != 'darwin':
            raise ValidationError('macos_nocache mode requires Darwin (macOS)')

        town_disk = _import_town_disk()

        # Create the containers root ourselves, exclusively, only now that
        # every preflight guard has passed, and remember that we did: cleanup
        # must only ever rmdir this root if `created_dir` is True, never a
        # pre-existing (e.g. already-empty) directory. town_disk.prepare_case
        # then creates each method's own exclusive subdirectory inside it.
        containers_dir.mkdir(parents=True)
        created_dir = True

        report = _run(args, town_disk, cases)

        check_free(output, 'output')  # narrow the race right before writing
        output.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                output.unlink()
            raise
    except Exception as exc:
        # Caught broadly (not a narrow type whitelist): any failure from
        # horizon.town_disk, numpy, or this script -- not just the types we
        # happen to anticipate -- must still trigger cleanup of exactly what
        # this run created before the CLI exits non-zero.
        _log(f'error: {type(exc).__name__}: {exc}')
        _cleanup_own_containers(containers_dir, created_dir, cases)
        return 1

    _log(f'wrote {output}')
    _log(f'containers written to {containers_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

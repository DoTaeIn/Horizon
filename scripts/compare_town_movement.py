#!/usr/bin/env python3
'''Real disk-load benchmark: whole-container ROI crop vs. multi-chunk ROI
reads, driven by deterministic simulated-character viewport trajectories
over the synthetic town map.

Generates one deterministic heightmap via horizon.town_map.make_town_map,
prepares on-disk containers via horizon.town_disk.prepare_case (one whole
container, one chunk_size=256 chunked container, per codec method), and
times two arms per frame of each trajectory, for each method/mode:

  - whole:  horizon.town_disk.load_whole(whole_path, mode=mode, region=ROI)
            -- decodes the entire whole container, then crops an owned copy.
  - region: horizon.town_region.load_region(chunked_path, ROI, mode=mode)
            -- reads only the chunks the 256x256 viewport ROI overlaps from
            the chunked container; unlike horizon.town_disk.load_chunked,
            the ROI need not be aligned to or equal to one grid chunk, so a
            viewport that straddles a chunk boundary spans more than one
            chunk read.

Both loaders do real file I/O every call: open, seek, read, close, decode.
Nothing is cached by this script or by either module between calls -- there
is no application-level cache for either arm (see "no_appcache_note" in the
report). This module does not implement horizon.town_disk or
horizon.town_region; it only calls their documented APIs. See
docs/town-movement-method.md for full scope, trajectory construction,
priming/ordering policy and caveats.
'''
from __future__ import annotations

import argparse
import collections
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
from horizon.town_map import MAX_SIZE, make_town_map  # noqa: E402

DEFAULT_SIZE = 1024
DEFAULT_SEED = 20261012
DEFAULT_STEPS = 120
MIN_STEPS = 30
DEFAULT_REPEATS = 3
MIN_REPEATS = 1
CHUNK_SIZE = 256  # fixed; horizon.town_disk.prepare_case default
VIEWPORT = 256  # fixed 256x256 character viewport
# The viewport must be able to straddle at least one interior chunk_size=256
# grid line in both directions (so "crossing many boundaries" trajectories
# are meaningful) while never needing to shrink below VIEWPORT -- 513 is the
# smallest size where a 256-wide viewport, clamped into [0, size-256], still
# has room either side of the first interior boundary at row/col 256.
MIN_SIZE = 513
SEED_EXCLUSIVE_MAX = 2 ** 64

ALL_MODES = ('warm_buffered', 'macos_nocache')
TRAJECTORY_KINDS = ('smooth_walk', 'boundary_backtrack', 'teleport')
_DTYPE = np.dtype('<u2')

ORDER_POLICY = (
    'within each (trajectory_kind, repeat) continuous run, a starting parity '
    'is tracked per (method, mode) cell: frame 0 of that run uses the '
    'current parity, frame 1 the opposite, and so on (swapped = '
    '(start_parity + frame_index) % 2), so the measured pair (whole vs. '
    'region) alternates every frame. The starting parity itself is flipped '
    'once per (trajectory_kind, repeat) run (before frame 0 of the next '
    'run), so it also alternates across repeats and across trajectory '
    'kinds -- "alternate each frame and trajectory" is satisfied by the '
    'combination of the two. Method and mode ordering is not alternated: '
    'the outer loops are method, then mode, in the fixed order '
    'horizon.codec.METHODS and --modes were given in.'
)
TRAJECTORY_POLICY = (
    'Three deterministic trajectory kinds are generated once per run (from '
    'the seeded RNG, one independent stream per kind) and reused, unchanged, '
    'across every method and mode (same schedule everywhere, like '
    'scripts/compare_town_disk.py\'s region_schedule): '
    '(1) smooth_walk -- a momentum random walk (acceleration ~ Normal(0, '
    '8.0), velocity clipped to +/-24 px/step) over `--steps` steps, '
    'wandering continuously and crossing many chunk_size=256 boundaries as '
    'it drifts across a size>=513 map; '
    '(2) boundary_backtrack -- one axis and one interior chunk_size=256 '
    'grid line are chosen once, then the viewport center oscillates back '
    'and forth across that exact line every step (alternating sign, growing '
    'amplitude), repeatedly crossing and re-crossing the same boundary; '
    '(3) teleport -- `--steps` independent uniform draws over the full map, '
    'no correlation between consecutive positions, the opposite of a smooth '
    'walk. Each kind\'s center-point sequence is mapped to a 256x256 '
    'viewport region via clamping (see viewport_policy), then that exact, '
    'already-computed region sequence is replayed `--repeats` times per '
    '(method, mode) cell for the timing statistics; the positions '
    'themselves are never regenerated or perturbed between repeats.'
)
VIEWPORT_POLICY = (
    'A trajectory center (center_row, center_col) is mapped to a region '
    '(row, col, 256, 256) by row = clamp(center_row - 128, 0, size - 256), '
    'col = clamp(center_col - 128, 0, size - 256). With --size >= 513 this '
    'never shrinks the viewport (size - 256 >= 257 > 0 always holds), so '
    'every frame\'s region is exactly 256x256; only its row/col offset is '
    'clamped at the map edges.'
)
PRIMING_POLICY = (
    'warm_buffered: before the first frame of each (trajectory_kind, '
    'repeat) continuous run -- i.e. once per run, not once per frame -- '
    'this script does one untimed ordinary buffered read of BOTH the whole '
    'container file and the chunked container file for that method, '
    'discarding the bytes, so both files are symmetrically warmed before '
    'timing starts. It is never repeated mid-run: once a run\'s frames start '
    'being timed, no further priming read happens until the next '
    '(trajectory_kind, repeat) run begins. macos_nocache: no priming is '
    'performed by this script at all; see no_appcache_note.'
)
VALIDATION_POLICY = (
    'Every loaded array, from both arms, on every frame, is checked (exact '
    'ndarray type, shape, dtype, byte content against a crop of the '
    'original in-memory map) immediately after the call returns -- but '
    'after metrics/timing for that call have already been captured by the '
    'library function itself, so validation work is never inside a timed '
    'span.'
)
NO_APPCACHE_NOTE = (
    'Neither arm is backed by any application-level cache in this script, '
    'horizon.town_disk, or horizon.town_region: every frame, for both arms, '
    'opens its container file fresh, reads exactly the bytes it needs, '
    'closes it, and decodes -- nothing from one frame (or one repeat, or '
    'one trajectory) is reused by the next. "warm_buffered" vs '
    '"macos_nocache" is a request about the OS page cache, not a claim '
    'about this script\'s own state, and "warm_buffered" is not a claim of '
    'a physically cold read either way: it only means this script primed '
    'the OS page cache by reading both files once before timing; '
    '"macos_nocache" only means horizon.town_disk/horizon.town_region asked '
    'the kernel (fcntl F_NOCACHE) not to retain newly read pages for that '
    'descriptor -- neither mode is a verified cold-hardware read (see '
    'horizon.town_disk.CACHE_NOTE).'
)
SCOPE = (
    'Real disk reads of prepared per-method container files (one whole, '
    'one chunk_size=256 chunked, written by horizon.town_disk.prepare_case) '
    'followed by decode, on this machine only, replaying three deterministic '
    '256x256-viewport trajectories (smooth_walk, boundary_backtrack, '
    'teleport) over the synthetic town map. "whole" times '
    'horizon.town_disk.load_whole(region=ROI) (decode the whole container, '
    'crop an owned copy); "region" times horizon.town_region.load_region '
    '(ROI) (read and decode only the chunks the ROI overlaps). The ROI is a '
    'real, possibly multi-chunk, viewport window -- not restricted to one '
    'exact grid chunk the way horizon.town_disk.load_chunked is. There is '
    'no application cache for either arm (see no_appcache_note). '
    'bytes_read/bytes_requested/read_calls/seek_calls are the '
    'application-level read()/seek() syscall counts the readers themselves '
    'tally, not a measurement of physical SSD/controller transfer size.'
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


def _steps_type(text):
    return _positive_int('steps', text, MIN_STEPS)


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
        description='Real disk-load comparison of whole-container ROI crop vs. '
                    'multi-chunk region reads, driven by simulated character '
                    'viewport trajectories over the synthetic town map.')
    parser.add_argument('--size', type=_size_type, default=DEFAULT_SIZE)
    parser.add_argument('--seed', type=_seed_type, default=DEFAULT_SEED)
    parser.add_argument('--steps', type=_steps_type, default=DEFAULT_STEPS)
    parser.add_argument('--repeats', type=_repeats_type, default=DEFAULT_REPEATS)
    parser.add_argument('--output', required=True, type=Path,
                         help='new JSON report path; never overwritten')
    parser.add_argument('--containers-dir', required=True, type=Path,
                         help='new directory for prepared container files; must not exist')
    parser.add_argument('--modes', type=_modes_type, default=_default_modes())
    return parser.parse_args(argv)


def _import_town_disk():
    try:
        return importlib.import_module('horizon.town_disk')
    except ModuleNotFoundError as exc:
        raise ValidationError(
            'horizon.town_disk is not available; this script only consumes its '
            'documented API (prepare_case/load_whole) and does not provide a '
            'fallback implementation'
        ) from exc


def _import_town_region():
    try:
        return importlib.import_module('horizon.town_region')
    except ModuleNotFoundError as exc:
        raise ValidationError(
            'horizon.town_region is not available; this script only consumes its '
            'documented API (load_region) and does not provide a fallback '
            'implementation'
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


def _source_files_sha256():
    pkg_dir = (REPO_ROOT / 'horizon').resolve()
    hashes = {'scripts/compare_town_movement.py': _sha256_file(Path(__file__).resolve())}
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


# --------------------------------------------------------------------------- #
# trajectories
# --------------------------------------------------------------------------- #

def _smooth_walk(rng, steps, size):
    pos = rng.integers(0, size, size=2).astype(np.float64)
    vel = np.zeros(2, dtype=np.float64)
    positions = []
    for _ in range(steps):
        accel = rng.normal(0.0, 8.0, size=2)
        vel = np.clip(vel + accel, -24.0, 24.0)
        pos = np.clip(pos + vel, 0, size - 1)
        positions.append((int(round(pos[0])), int(round(pos[1]))))
    return positions


def _boundary_backtrack(rng, steps, size, chunk_size=CHUNK_SIZE, viewport=VIEWPORT):
    '''Oscillate the viewport ORIGIN (not the center) across one interior
    chunk_size grid line every step, so the selected chunk set actually
    changes almost every frame.

    A trajectory position here is a *center*; _viewport_region later turns
    it into an origin via `clamp(center - viewport // 2, 0, max_origin)`.
    To make the origin itself land exactly on alternating sides of a grid
    boundary -- with no clamping distortion -- the boundary candidates are
    restricted to 0 < boundary < max_origin (strictly interior to the
    origin's own valid range, never the last grid line, which would have no
    room above it and would clamp to the same origin every step), and the
    oscillation amplitude is capped to min(room, 12) -- the room available
    on the tighter side of that boundary, further capped at 12px so a large
    room (e.g. a central boundary on a big map) cannot produce swings far
    wider than one chunk; with the per-step +0..2 growth below, the largest
    single-step jump is 2 * min(room, 14) <= 28px, keeping this a tight
    backtrack rather than a long-range hop.
    '''
    half = viewport // 2
    max_origin = size - viewport
    candidates = [b for b in range(chunk_size, size, chunk_size) if 0 < b < max_origin]
    axis = int(rng.integers(0, 2))  # 0: oscillate the row, 1: oscillate the col
    boundary = int(rng.choice(candidates))
    room = min(boundary, max_origin - boundary)
    amplitude_cap = min(room, 12)
    amplitude_base = int(rng.integers(1, amplitude_cap + 1))
    other_origin = int(rng.integers(0, max_origin + 1))
    other_fixed = other_origin + half
    positions = []
    for t in range(steps):
        delta = min(room, amplitude_base + (t % 3))
        sign = 1 if t % 2 == 0 else -1
        coord = boundary + half + sign * delta
        positions.append((coord, other_fixed) if axis == 0 else (other_fixed, coord))
    return positions


def _teleport(rng, steps, size):
    return [tuple(int(v) for v in rng.integers(0, size, size=2)) for _ in range(steps)]


_TRAJECTORY_BUILDERS = {
    'smooth_walk': _smooth_walk,
    'boundary_backtrack': _boundary_backtrack,
    'teleport': _teleport,
}


def _build_trajectories(seed, steps, size):
    '''One independent RNG stream per kind, seeded from the run seed, so every
    kind is deterministic on its own and unaffected by the others' draws.'''
    trajectories = {}
    for index, kind in enumerate(TRAJECTORY_KINDS):
        rng = np.random.default_rng([seed, index])
        trajectories[kind] = _TRAJECTORY_BUILDERS[kind](rng, steps, size)
    return trajectories


def _viewport_region(center_row, center_col, size, viewport=VIEWPORT):
    row = max(0, min(center_row - viewport // 2, size - viewport))
    col = max(0, min(center_col - viewport // 2, size - viewport))
    return row, col, viewport, viewport


# --------------------------------------------------------------------------- #
# validation / priming / timing helpers
# --------------------------------------------------------------------------- #

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


def _run_frame(town_disk, town_region, whole_path, chunked_path, mode, region, arr, swapped):
    row, col, h, w = region
    expected_shape = (h, w)
    expected_bytes = np.ascontiguousarray(arr[row:row + h, col:col + w]).tobytes()

    def do_whole():
        arr_out, metrics = town_disk.load_whole(whole_path, mode=mode, region=region)
        _validate_loaded(arr_out, expected_shape, expected_bytes, 'whole')
        return metrics

    def do_region():
        arr_out, metrics = town_region.load_region(chunked_path, region, mode=mode)
        _validate_loaded(arr_out, expected_shape, expected_bytes, 'region')
        for key in ('chunks_loaded', 'chunk_coordinates'):
            if key not in metrics:
                raise ValidationError(f'load_region metrics missing {key!r}')
        return metrics

    return _ordered_pair(swapped, do_whole, do_region)


def _stats(samples_ns):
    ms = [n / 1e6 for n in samples_ns]
    return {
        'median_ms': round(statistics.median(ms), 6),
        'p95_ms': round(float(np.percentile(ms, 95, method='linear')), 6),
        'max_ms': round(max(ms), 6),
    }


def _metric_block(metric_dicts, with_chunks):
    block = {
        'total': _stats([m['total_ns'] for m in metric_dicts]),
        'read': _stats([m['read_ns'] for m in metric_dicts]),
        'decode': _stats([m['decode_ns'] for m in metric_dicts]),
        'bytes_read': [int(m['bytes_read']) for m in metric_dicts],
        'bytes_requested': [int(m.get('bytes_requested', m['bytes_read'])) for m in metric_dicts],
        'read_calls': [int(m['read_calls']) for m in metric_dicts],
        'seek_calls': [int(m.get('seek_calls', 0)) for m in metric_dicts],
    }
    if with_chunks:
        chunks_loaded = [int(m['chunks_loaded']) for m in metric_dicts]
        block['chunks_loaded'] = chunks_loaded
        block['chunk_coordinates'] = [list(m['chunk_coordinates']) for m in metric_dicts]
        block['chunk_count_distribution'] = {
            str(k): v for k, v in sorted(collections.Counter(chunks_loaded).items())
        }
    return block


def _totals_stats(per_repeat_sums):
    return _stats(per_repeat_sums)


def _totals_bytes(per_repeat_sums):
    return {
        'median': round(statistics.median(per_repeat_sums), 3),
        'p95': round(float(np.percentile(per_repeat_sums, 95, method='linear')), 3),
        'max': round(max(per_repeat_sums), 3),
    }


# --------------------------------------------------------------------------- #
# benchmark core
# --------------------------------------------------------------------------- #

def _benchmark_trajectory(town_disk, town_region, arr, whole_path, chunked_path, mode,
                           positions, repeats, order_parity):
    size = arr.shape[0]
    repeats_out = []
    whole_frames_all = []
    region_frames_all = []
    whole_totals_ns = []
    region_totals_ns = []
    whole_totals_bytes = []
    region_totals_bytes = []

    for repeat_index in range(repeats):
        start_parity = order_parity % 2
        order_parity += 1

        _prime(mode, (whole_path, chunked_path))

        frames = []
        whole_total_ns = 0
        region_total_ns = 0
        whole_total_bytes = 0
        region_total_bytes = 0

        for frame_index, (center_row, center_col) in enumerate(positions):
            region = _viewport_region(center_row, center_col, size)
            swapped = bool((start_parity + frame_index) % 2)
            whole_metrics, region_metrics = _run_frame(
                town_disk, town_region, whole_path, chunked_path, mode, region, arr, swapped
            )
            whole_frames_all.append(whole_metrics)
            region_frames_all.append(region_metrics)
            whole_total_ns += whole_metrics['total_ns']
            region_total_ns += region_metrics['total_ns']
            whole_total_bytes += whole_metrics['bytes_read']
            region_total_bytes += region_metrics['bytes_read']
            frames.append({
                'frame': frame_index,
                'center_row': center_row,
                'center_col': center_col,
                'roi': list(region),
                'swapped': swapped,
                'whole': whole_metrics,
                'region': region_metrics,
            })

        whole_totals_ns.append(whole_total_ns)
        region_totals_ns.append(region_total_ns)
        whole_totals_bytes.append(whole_total_bytes)
        region_totals_bytes.append(region_total_bytes)
        repeats_out.append({'repeat': repeat_index, 'frames': frames})

    aggregate = {
        'whole': _metric_block(whole_frames_all, with_chunks=False),
        'region': _metric_block(region_frames_all, with_chunks=True),
        'trajectory_totals': {
            'whole': {'time_ms': _totals_stats(whole_totals_ns), 'bytes': _totals_bytes(whole_totals_bytes)},
            'region': {'time_ms': _totals_stats(region_totals_ns), 'bytes': _totals_bytes(region_totals_bytes)},
        },
    }
    return {'repeats': repeats_out, 'aggregate': aggregate}, order_parity


def _file_hash_snapshot(case):
    return {
        'whole': _sha256_file(case['whole_path']),
        'chunked': _sha256_file(case['chunked_path']),
    }


def _run(args, town_disk, town_region, cases):
    '''cases is populated in place (even if this raises partway through) so
    the caller can clean up exactly the files this run created.'''
    arr = make_town_map(size=args.size, seed=args.seed)
    validate_heightmap(arr, 'make_town_map output')
    arr = np.ascontiguousarray(arr)

    trajectories = _build_trajectories(args.seed, args.steps, args.size)

    for method in METHODS:
        _log(f'preparing case: method={method}')
        case = town_disk.prepare_case(arr, args.containers_dir / method, method, chunk_size=CHUNK_SIZE)
        case['method'] = method
        cases.append(case)
        for key in ('whole_path', 'chunked_path'):
            if key not in case:
                raise ValidationError(f'prepare_case({method!r}) missing {key!r}')
        case['hashes_before'] = _file_hash_snapshot(case)

    for case in cases:
        method = case['method']
        whole_path = Path(case['whole_path'])
        chunked_path = Path(case['chunked_path'])
        case['modes'] = {}
        for mode in args.modes:
            case['modes'][mode] = {}
            order_parity = 0
            for kind in TRAJECTORY_KINDS:
                _log(f'benchmarking method={method} mode={mode} trajectory={kind}')
                result, order_parity = _benchmark_trajectory(
                    town_disk, town_region, arr, whole_path, chunked_path, mode,
                    trajectories[kind], args.repeats, order_parity
                )
                case['modes'][mode][kind] = result
        case['hashes_after'] = _file_hash_snapshot(case)
        if case['hashes_after'] != case['hashes_before']:
            raise ValidationError(f'container file(s) changed during benchmark for method={method}')

    report = {
        'schema_version': 1,
        'scope': SCOPE,
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'source_files_sha256': _source_files_sha256(),
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
            'steps': args.steps,
            'repeats': args.repeats,
            'chunk_size': CHUNK_SIZE,
            'viewport': VIEWPORT,
            'methods': list(METHODS),
            'modes': list(args.modes),
            'trajectory_kinds': list(TRAJECTORY_KINDS),
            'p95_method': 'numpy.percentile, linear interpolation',
            'order_policy': ORDER_POLICY,
            'trajectory_policy': TRAJECTORY_POLICY,
            'viewport_policy': VIEWPORT_POLICY,
            'priming_policy': PRIMING_POLICY,
            'validation_policy': VALIDATION_POLICY,
            'no_appcache_note': NO_APPCACHE_NOTE,
        },
        'trajectories': {
            kind: [list(pos) for pos in positions] for kind, positions in trajectories.items()
        },
        'cases': cases,
    }
    return report


def _cleanup_own_containers(containers_dir, created_dir, cases):
    '''Remove only files/directories this run itself wrote: each method's
    whole/chunked container files and prepare_case's per-method subdirectory
    (removed only if now empty), then --containers-dir itself -- but ONLY if
    `created_dir` says this run created that root. A pre-existing
    --containers-dir must never be removed by this script, empty or not.'''
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
        town_region = _import_town_region()

        containers_dir.mkdir(parents=True)
        created_dir = True

        report = _run(args, town_disk, town_region, cases)

        check_free(output, 'output')  # narrow the race right before writing
        output.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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
        _log(f'error: {type(exc).__name__}: {exc}')
        _cleanup_own_containers(containers_dir, created_dir, cases)
        return 1

    _log(f'wrote {output}')
    _log(f'containers written to {containers_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

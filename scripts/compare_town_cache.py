#!/usr/bin/env python3
'''Application-level cache comparison, replaying an existing town-movement
schedule file's recorded trajectories verbatim.

This script does not generate a new map or new trajectories: it loads an
existing `scripts/compare_town_movement.py` report (e.g.
benchmarks/town-movement-1024-v1.json or the seed2 variant) via --schedule,
re-derives the map from its recorded (size, seed) and asserts the sha256
matches, asserts that EVERY file the schedule recorded a hash for -- including
scripts/compare_town_movement.py itself -- is still present and byte-identical
on disk, and then replays its recorded trajectories[kind] positions exactly --
never regenerating, clipping or perturbing them.

Verification is strict by design: a recorded source file that is missing, or
whose bytes changed, is a rejected run, not a skipped entry. A schedule whose
settings, map block or trajectories are the wrong shape/type/range is rejected
before any container is written.

Four arms are compared per (method, mode, trajectory kind, repeat), all
driven by the same positions:

  stateless      horizon.town_region.load_region(chunked_path, roi, mode) --
                 no cache, real I/O every call.
  lru{N}         horizon.town_cache.ChunkRegionCache(chunked_path,
                 max_chunks=N, mode) -- one per --cache-sizes entry.
  whole_resident horizon.town_cache.WholeMapResident(whole_path, mode).

The recorded schedules this was built for use chunk_size=256 and a 256x256
viewport (DOCUMENTED_CHUNK_SIZE/DOCUMENTED_VIEWPORT, and every published
number in benchmarks/town-cache-audit.md is at those values), but nothing here
is hard-coded to 256: any positive chunk_size/viewport that fits the recorded
map is accepted and replayed, which is what makes small fixture schedules
usable in tests.

Timing output: raw per-frame and per-repeat values are integer nanoseconds
(`*_ns`); every aggregated statistic is in milliseconds and named `*_ms`
inside its stat block, so no aggregate is labelled with a unit it is not in.

This script implements neither town_disk/town_region's container parsing
nor town_cache's cache policy; it only calls their documented APIs. See
benchmarks/town-cache-audit.md for full scope and caveats.
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
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from horizon.town_map import make_town_map  # noqa: E402

DEFAULT_REPEATS = 3
MIN_REPEATS = 1
DEFAULT_MEMORY_REPEATS = 3
DEFAULT_CACHE_SIZES = (4, 8)
ALL_MODES = ('warm_buffered', 'macos_nocache')
DOCUMENTED_CHUNK_SIZE = 256
DOCUMENTED_VIEWPORT = 256
MEMORY_SCRIPT = REPO_ROOT / 'scripts' / 'measure_town_cache_memory.py'
MEMORY_SCRIPT_REL = MEMORY_SCRIPT.relative_to(REPO_ROOT).as_posix()
_DTYPE = np.dtype('<u2')
_HEX64 = re.compile(r'\A[0-9a-f]{64}\Z')

#: metrics keys this script reads out of the loader/cache metrics dicts; a
#: missing one is a contract break in the library, reported as such.
_REQUIRED_METRIC_KEYS = (
    'total_ns', 'read_ns', 'decode_ns', 'bytes_read', 'bytes_requested',
    'read_calls', 'seek_calls',
)
_REQUIRED_FRAME_KEYS = _REQUIRED_METRIC_KEYS + (
    'resident_bytes', 'chunks_requested', 'chunks_loaded', 'cache_hits',
    'cache_misses', 'evictions',
)

ROTATION_POLICY = (
    'Arm execution order within a frame is rotated by '
    '(trajectory_index + repeat + frame_index) % num_arms; session '
    'construction order within a repeat is rotated by '
    '(trajectory_index + repeat) % num_arms. Both realized orders are '
    'recorded per frame/repeat so no arm is systematically first or last.'
)
PRIMING_POLICY = (
    'warm_buffered: once per (trajectory_kind, repeat) run, before any arm '
    'is constructed or timed, this script does one untimed ordinary '
    'buffered read of BOTH the whole and chunked container files for that '
    'method, discarding the bytes. macos_nocache: no priming is performed.'
)
SESSION_POLICY = (
    'Every arm gets a freshly constructed session (stateless: none, by '
    'definition; lru{N}: an empty ChunkRegionCache; whole_resident: a new '
    'WholeMapResident) once per (method, mode, trajectory_kind, repeat), '
    'never reused across repeats. Each repeat records that session\'s own raw '
    'startup_metrics verbatim under repeats[].startup_metrics[arm], and '
    'first_viewport_ns = that startup total_ns + the repeat\'s frame 0 '
    'total_ns. If any arm\'s construction fails, the sessions already built '
    'for that repeat are closed before the error propagates.'
)
VALIDATION_POLICY = (
    'Every array returned by every arm, every frame, is checked (ndarray '
    'type, exact (height, width) shape, dtype, byte content against a crop '
    'of the in-memory original map) strictly after the timed call returns.'
)
TIMING_UNIT_POLICY = (
    'Raw values are integer nanoseconds and carry an _ns suffix: '
    'repeats[].frames[].arms[arm].total_ns, '
    'repeats[].startup_metrics[arm].total_ns and '
    'repeats[].trajectory_ns[arm]. Every aggregate is a stat block of '
    'median_ms/p95_ms/max_ms computed in milliseconds, so aggregate names '
    'are never given an _ns suffix.'
)
TRAJECTORY_TOTAL_POLICY = (
    'repeats[].trajectory_ns[arm] records, per repeat, that arm\'s raw '
    'startup_total_ns, frames_total_ns (sum of all frame total_ns) and '
    'total_ns_including_startup = startup + frames. The aggregates '
    'trajectory_total_including_startup and trajectory_total_movement_only '
    'are stats over those per-repeat totals. They are loader time only: not '
    'wall-clock frames, and they exclude rendering, physics, validation and '
    'report writing.'
)
WHOLE_RESIDENT_NOTE = (
    'whole_resident has zero I/O/decode cost per .load() by construction '
    'in this benchmark (the map is already decoded and resident); this is '
    'not evidence that whole-map residency is universally the best choice '
    '-- it ignores memory cost and multi-map scenarios.'
)
MEMORY_POLICY = (
    f'--memory-repeats > 0 shells out once to {MEMORY_SCRIPT_REL} with '
    '--schedule/--containers-dir/--cache-sizes/--repeats/--output, so memory '
    'is measured in fresh processes that never inherit this process\'s '
    'resident map. That script owns its own schema; it is folded in verbatim '
    'under report["memory"]["report"] with its path and sha256. Its existence '
    'is checked, and its output path refused if taken, before any container '
    'is written. Container hashes are re-verified AFTER it runs, so its reads '
    'are covered too.'
)


def _log(message):
    print(message, file=sys.stderr, flush=True)


class ValidationError(RuntimeError):
    pass


def _positive_int(name, text, minimum):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{name} must be an integer') from None
    if value < minimum:
        raise argparse.ArgumentTypeError(f'{name} must be >= {minimum}')
    return value


def _repeats_type(text):
    return _positive_int('repeats', text, MIN_REPEATS)


def _memory_repeats_type(text):
    return _positive_int('memory-repeats', text, 0)


def _cache_sizes_type(text):
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('--cache-sizes must list at least one size')
    sizes = []
    for p in parts:
        n = _positive_int('cache-sizes', p, 1)
        if n in sizes:
            raise argparse.ArgumentTypeError(f'--cache-sizes: duplicate size {n}')
        sizes.append(n)
    return tuple(sizes)


def _modes_type(text):
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('--modes must list at least one mode')
    out = []
    for p in parts:
        if p not in ALL_MODES:
            raise argparse.ArgumentTypeError(f'--modes: unknown mode {p!r}; expected one of {ALL_MODES}')
        if p in out:
            raise argparse.ArgumentTypeError(f'--modes: duplicate mode {p!r}')
        out.append(p)
    return tuple(out)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Application-level cache comparison (stateless vs. '
                    'LRU-bounded chunk cache vs. whole-map resident) over an '
                    'existing town-movement schedule file, replayed verbatim.')
    parser.add_argument('--schedule', required=True, type=Path,
                         help='existing town-movement-*.json report to replay trajectories from')
    parser.add_argument('--repeats', type=_repeats_type, default=DEFAULT_REPEATS)
    parser.add_argument('--cache-sizes', type=_cache_sizes_type, default=DEFAULT_CACHE_SIZES)
    parser.add_argument('--modes', type=_modes_type, default=None,
                         help='default: the schedule\'s own recorded modes')
    parser.add_argument('--memory-repeats', type=_memory_repeats_type, default=DEFAULT_MEMORY_REPEATS,
                         help=f'0 disables the memory sub-measurement (fast smoke runs); '
                              f'any other value requires {MEMORY_SCRIPT_REL}')
    parser.add_argument('--output', required=True, type=Path,
                         help='new JSON report path; never overwritten')
    parser.add_argument('--containers-dir', required=True, type=Path,
                         help='new directory for prepared container files; must not exist')
    return parser.parse_args(argv)


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
    '''This script, the memory worker it drives (when present) and every
    horizon/*.py module, so a report pins the exact code that produced it.'''
    pkg_dir = (REPO_ROOT / 'horizon').resolve()
    hashes = {'scripts/compare_town_cache.py': _sha256_file(Path(__file__).resolve())}
    if MEMORY_SCRIPT.is_file():
        hashes[MEMORY_SCRIPT_REL] = _sha256_file(MEMORY_SCRIPT)
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


def _import(name, used_for):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise ValidationError(
            f'{name} is not available; this script only consumes its documented '
            f'API ({used_for}) and does not provide a fallback implementation'
        ) from exc


# --------------------------------------------------------------------------- #
# schedule loading / strict verification
# --------------------------------------------------------------------------- #

def _load_schedule(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValidationError(f'--schedule is not valid JSON: {path}: {exc}') from None


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValidationError(f'schedule {label} must be an object, got {type(value).__name__}')
    return value


def _sched_int(settings, key, minimum, maximum=None):
    if key not in settings:
        raise ValidationError(f'schedule settings.{key} is missing')
    value = settings[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f'schedule settings.{key} must be an integer, got {value!r}')
    if value < minimum:
        raise ValidationError(f'schedule settings.{key} must be >= {minimum}, got {value}')
    if maximum is not None and value > maximum:
        raise ValidationError(f'schedule settings.{key} must be <= {maximum}, got {value}')
    return value


def _sched_str_list(settings, key, allowed=None):
    if key not in settings:
        raise ValidationError(f'schedule settings.{key} is missing')
    value = settings[key]
    if not isinstance(value, list) or not value:
        raise ValidationError(f'schedule settings.{key} must be a non-empty list')
    out = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValidationError(f'schedule settings.{key} must contain non-empty strings, got {item!r}')
        if item in out:
            raise ValidationError(f'schedule settings.{key} contains duplicate entry {item!r}')
        if allowed is not None and item not in allowed:
            raise ValidationError(
                f'schedule settings.{key} entry {item!r} is not supported; expected one of {allowed}'
            )
        out.append(item)
    return tuple(out)


def _validate_settings(schedule):
    settings = _mapping(schedule.get('settings'), 'settings')
    size = _sched_int(settings, 'size', 1)
    checked = {
        'size': size,
        'seed': _sched_int(settings, 'seed', 0),
        'steps': _sched_int(settings, 'steps', 1),
        # Not pinned to 256: any chunking/viewport that fits the recorded map
        # replays correctly. 256/256 is what the published runs used.
        'chunk_size': _sched_int(settings, 'chunk_size', 1, size),
        'viewport': _sched_int(settings, 'viewport', 1, size),
        'methods': _sched_str_list(settings, 'methods'),
        'modes': _sched_str_list(settings, 'modes', allowed=ALL_MODES),
        'trajectory_kinds': _sched_str_list(settings, 'trajectory_kinds'),
    }
    return checked


def _validate_map_block(schedule, size):
    block = _mapping(schedule.get('map'), 'map')
    sha = block.get('sha256')
    if not isinstance(sha, str) or not _HEX64.match(sha):
        raise ValidationError(f'schedule map.sha256 must be a 64-char lowercase hex digest, got {sha!r}')
    shape = block.get('shape')
    if shape is not None and [int(n) for n in shape] != [size, size]:
        raise ValidationError(
            f'schedule map.shape {shape} disagrees with settings.size {size}; '
            'this script replays square maps regenerated from (size, seed)'
        )
    dtype = block.get('dtype')
    if dtype is not None and dtype != _DTYPE.str:
        raise ValidationError(f'schedule map.dtype {dtype!r} is not {_DTYPE.str!r}')
    source = block.get('source')
    if source is not None and not isinstance(source, str):
        raise ValidationError('schedule map.source must be a string when present')
    return block, sha


def _validate_positions(schedule, kinds, steps, size, viewport):
    '''Read the recorded centers verbatim: validated for type/length/range,
    never regenerated, clipped or reordered.'''
    trajectories = _mapping(schedule.get('trajectories'), 'trajectories')
    out = {}
    for kind in kinds:
        if kind not in trajectories:
            raise ValidationError(f'schedule trajectories is missing recorded kind {kind!r}')
        positions = trajectories[kind]
        if not isinstance(positions, list):
            raise ValidationError(f'schedule trajectories[{kind!r}] must be a list')
        if len(positions) != steps:
            raise ValidationError(
                f'schedule trajectories[{kind!r}] has {len(positions)} positions '
                f'but settings.steps is {steps}'
            )
        checked = []
        for index, point in enumerate(positions):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValidationError(
                    f'schedule trajectories[{kind!r}][{index}] must be a [row, col] pair, got {point!r}'
                )
            row, col = point
            for name, value in (('row', row), ('col', col)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValidationError(
                        f'schedule trajectories[{kind!r}][{index}] {name} must be an integer, got {value!r}'
                    )
                if not 0 <= value < size:
                    raise ValidationError(
                        f'schedule trajectories[{kind!r}][{index}] {name}={value} is outside '
                        f'the recorded {size}x{size} map'
                    )
            checked.append((int(row), int(col)))
        out[kind] = tuple(checked)
    if viewport > size:  # already enforced by settings validation; kept explicit
        raise ValidationError(f'viewport {viewport} does not fit in a {size}x{size} map')
    return out


def _verify_recorded_sources(schedule):
    '''Strict: every file the schedule recorded a hash for must still exist and
    still hash identically -- including scripts/compare_town_movement.py, the
    script that produced the schedule. A missing file is a rejection, not a
    skip: replaying a schedule whose recorded code is gone cannot be claimed
    to be the same experiment.'''
    recorded = _mapping(schedule.get('source_files_sha256'), 'source_files_sha256')
    if not recorded:
        raise ValidationError('schedule source_files_sha256 is empty; nothing to verify against')
    verified = []
    for relpath, expected_sha in sorted(recorded.items()):
        if not isinstance(relpath, str) or not relpath:
            raise ValidationError(f'schedule source_files_sha256 has a non-path key {relpath!r}')
        if not isinstance(expected_sha, str) or not _HEX64.match(expected_sha):
            raise ValidationError(
                f'schedule source_files_sha256[{relpath!r}] must be a 64-char lowercase '
                f'hex digest, got {expected_sha!r}'
            )
        path = Path(relpath)
        if path.is_absolute() or '..' in path.parts:
            raise ValidationError(
                f'schedule source_files_sha256 key {relpath!r} must be a repo-relative path'
            )
        full = REPO_ROOT / path
        if not full.is_file():
            raise ValidationError(
                f'{relpath} is recorded in the schedule but is not present in this working '
                f'tree; refusing to replay a schedule whose recorded source is missing'
            )
        actual_sha = _sha256_file(full)
        if actual_sha != expected_sha:
            raise ValidationError(
                f'{relpath} sha256 mismatch vs schedule: expected {expected_sha}, got {actual_sha} '
                '(recorded source changed since the schedule was recorded)'
            )
        verified.append(relpath)
    return verified


def _verify_schedule(schedule, schedule_path):
    '''Validate the schedule's shape, regenerate the recorded map and re-hash
    every recorded source file; raise loudly on any mismatch. Trajectories are
    read verbatim.'''
    _mapping(schedule, 'report')
    settings = _validate_settings(schedule)
    size = settings['size']
    _map_block, expected_map_sha = _validate_map_block(schedule, size)
    positions = _validate_positions(
        schedule, settings['trajectory_kinds'], settings['steps'], size, settings['viewport']
    )

    arr = np.ascontiguousarray(make_town_map(size=size, seed=settings['seed']))
    if arr.shape != (size, size) or arr.dtype != _DTYPE:
        raise ValidationError(
            f'regenerated map is {arr.shape} {arr.dtype}, expected ({size}, {size}) {_DTYPE}'
        )
    actual_map_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    if actual_map_sha != expected_map_sha:
        raise ValidationError(
            f'schedule map sha256 mismatch: expected {expected_map_sha}, got {actual_map_sha}'
        )

    verified = _verify_recorded_sources(schedule)
    verification = {
        'schedule_path': str(schedule_path),
        'schedule_sha256': _sha256_file(schedule_path),
        'map_sha256_verified': True,
        'source_files_verified': verified,
        'source_files_policy': (
            'strict: every file recorded in the schedule\'s source_files_sha256 '
            '(scripts/compare_town_movement.py included) must be present and '
            'byte-identical; a missing or changed file rejects the run'
        ),
    }
    return arr, settings, positions, verification


# --------------------------------------------------------------------------- #
# validation / priming / timing helpers
# --------------------------------------------------------------------------- #

def _viewport_region(center_row, center_col, size, viewport):
    row = max(0, min(center_row - viewport // 2, size - viewport))
    col = max(0, min(center_col - viewport // 2, size - viewport))
    return row, col, viewport, viewport


def _read_buffered_discard(path):
    with open(path, 'rb') as f:
        while f.read(1 << 20):
            pass


def _prime(mode, paths):
    if mode != 'warm_buffered':
        return
    for path in paths:
        _read_buffered_discard(path)


def _validate_loaded(arr, expected_shape, expected_bytes, label):
    if type(arr) is not np.ndarray:
        raise ValidationError(f'{label}: expected ndarray, got {type(arr).__name__}')
    if arr.shape != expected_shape:
        raise ValidationError(f'{label}: shape mismatch {arr.shape} != {expected_shape}')
    if arr.dtype != _DTYPE:
        raise ValidationError(f'{label}: dtype mismatch {arr.dtype} != {_DTYPE}')
    if arr.tobytes() != expected_bytes:
        raise ValidationError(f'{label}: byte content mismatch')


def _validate_metrics(metrics, keys, label):
    if not isinstance(metrics, dict):
        raise ValidationError(f'{label}: expected a metrics dict, got {type(metrics).__name__}')
    for key in keys:
        if key not in metrics:
            raise ValidationError(f'{label}: metrics is missing required key {key!r}')
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(f'{label}: metrics[{key!r}] must be an integer, got {value!r}')
        if value < 0:
            raise ValidationError(f'{label}: metrics[{key!r}] is negative ({value})')
    return metrics


def _stats(samples_ns):
    '''Stat block for a list of raw nanosecond samples. Every value is in
    milliseconds and says so in its key.'''
    ms = [n / 1e6 for n in samples_ns]
    return {
        'median_ms': round(statistics.median(ms), 6),
        'p95_ms': round(float(np.percentile(ms, 95, method='linear')), 6),
        'max_ms': round(max(ms), 6),
    }


def _normalize_metrics(arm, raw):
    '''Stateless's load_region metrics lack cache concepts; fill them in
    honestly as "every chunk touched is a miss, nothing is retained".
    lru{N}/whole_resident metrics from horizon.town_cache already carry the
    full key set, including resident_bytes, and are passed through as-is.'''
    if arm != 'stateless':
        return dict(raw)
    _validate_metrics(raw, _REQUIRED_METRIC_KEYS + ('chunks_loaded',), arm)
    chunks_loaded = list(raw.get('chunk_coordinates', []))
    out = dict(raw)
    out.update({
        'chunks_requested': int(raw['chunks_loaded']),
        'chunks_loaded': int(raw['chunks_loaded']),
        'cache_hits': 0,
        'cache_misses': int(raw['chunks_loaded']),
        'evictions': 0,
        'chunk_coordinates': chunks_loaded,
        'hit_coordinates': [],
        'miss_coordinates': chunks_loaded,
        'evicted_coordinates': [],
        'resident_bytes': 0,
    })
    return out


def _arm_names(cache_sizes):
    return ['stateless'] + [f'lru{n}' for n in cache_sizes] + ['whole_resident']


def _rotate(items, by):
    by %= len(items)
    return items[by:] + items[:by]


def _make_session(arm, cache_sizes, town_cache, whole_path, chunked_path, mode):
    if arm == 'stateless':
        return None
    if arm == 'whole_resident':
        return town_cache.WholeMapResident(whole_path, mode=mode)
    n = int(arm[len('lru'):])
    if n not in cache_sizes:
        raise ValidationError(f'internal: arm {arm!r} is not one of --cache-sizes {cache_sizes}')
    return town_cache.ChunkRegionCache(chunked_path, max_chunks=n, mode=mode)


def _close_sessions(sessions):
    for session in sessions.values():
        if session is not None:
            with contextlib.suppress(Exception):
                session.close()


def _startup_metrics(arm, session):
    '''The arm's own raw constructor metrics, recorded verbatim. stateless has
    no session and therefore a genuine all-zero startup.'''
    if arm == 'stateless':
        return {k: 0 for k in _REQUIRED_METRIC_KEYS}
    return _validate_metrics(dict(session.startup_metrics), _REQUIRED_METRIC_KEYS,
                              f'{arm} startup_metrics')


def _load_once(arm, session, town_region, chunked_path, mode, roi):
    if arm == 'stateless':
        return town_region.load_region(chunked_path, roi, mode=mode)
    return session.load(roi)


# --------------------------------------------------------------------------- #
# benchmark core
# --------------------------------------------------------------------------- #

def _benchmark_run(town_cache, town_region, arr, whole_path, chunked_path, mode,
                    positions, repeats, cache_sizes, viewport, trajectory_index):
    size = arr.shape[0]
    arms = _arm_names(cache_sizes)
    repeats_out = []
    per_arm = {a: {'frames': [], 'startup': [], 'first_viewport_ns': [],
                   'trajectory_ns': [], 'movement_ns': []} for a in arms}

    for repeat_index in range(repeats):
        _prime(mode, (whole_path, chunked_path))

        construct_order = _rotate(arms, trajectory_index + repeat_index)
        sessions = {}
        startup = {}
        try:
            for arm in construct_order:
                sessions[arm] = _make_session(
                    arm, cache_sizes, town_cache, whole_path, chunked_path, mode
                )
                startup[arm] = _startup_metrics(arm, sessions[arm])
        except BaseException:
            # Partial construction: close what was built before propagating,
            # so a failed repeat leaves no open handle or resident array.
            _close_sessions(sessions)
            raise

        frames = []
        try:
            for frame_index, (center_row, center_col) in enumerate(positions):
                roi = _viewport_region(center_row, center_col, size, viewport)
                row, col, h, w = roi
                expected_bytes = np.ascontiguousarray(arr[row:row + h, col:col + w]).tobytes()
                order = _rotate(arms, trajectory_index + repeat_index + frame_index)

                frame_record = {'frame': frame_index, 'roi': list(roi), 'arm_order': order, 'arms': {}}
                for arm in order:
                    out, raw_metrics = _load_once(
                        arm, sessions[arm], town_region, chunked_path, mode, roi
                    )
                    _validate_loaded(out, (h, w), expected_bytes, arm)
                    metrics = _validate_metrics(
                        _normalize_metrics(arm, raw_metrics), _REQUIRED_FRAME_KEYS, arm
                    )
                    frame_record['arms'][arm] = metrics
                    per_arm[arm]['frames'].append(metrics)
                frames.append(frame_record)
        finally:
            _close_sessions(sessions)

        trajectory_ns = {}
        for arm in arms:
            frame_ns = [f['arms'][arm]['total_ns'] for f in frames]
            startup_ns = int(startup[arm]['total_ns'])
            frames_ns = int(sum(frame_ns))
            trajectory_ns[arm] = {
                'startup_total_ns': startup_ns,
                'frames_total_ns': frames_ns,
                'total_ns_including_startup': startup_ns + frames_ns,
            }
            per_arm[arm]['startup'].append(startup[arm])
            per_arm[arm]['first_viewport_ns'].append(startup_ns + frame_ns[0])
            per_arm[arm]['trajectory_ns'].append(startup_ns + frames_ns)
            per_arm[arm]['movement_ns'].append(frames_ns)

        repeats_out.append({
            'repeat': repeat_index,
            'construct_order': construct_order,
            'startup_metrics': {a: startup[a] for a in arms},
            'trajectory_ns': trajectory_ns,
            'frames': frames,
        })

    aggregate = {arm: _aggregate_arm(per_arm[arm]) for arm in arms}
    return {'repeats': repeats_out, 'aggregate': aggregate}


def _aggregate_arm(collected):
    '''Aggregate one arm. Stat blocks are milliseconds (median_ms/p95_ms/
    max_ms); only raw counters keep integer units in their names.'''
    frames = collected['frames']
    startups = collected['startup']
    total_ns = [m['total_ns'] for m in frames]
    hit_only = [m['total_ns'] for m in frames if m['cache_misses'] == 0]
    miss_frames = [m['total_ns'] for m in frames if m['cache_misses'] > 0]
    bytes_read_total = (sum(m['bytes_read'] for m in frames)
                        + sum(s['bytes_read'] for s in startups))
    chunks_requested_total = sum(m['chunks_requested'] for m in frames)
    cache_hits_total = sum(m['cache_hits'] for m in frames)
    resident_series = [m['resident_bytes'] for m in frames]
    startup_total_ns = [s['total_ns'] for s in startups]
    return {
        'startup_total': _stats(startup_total_ns) if startup_total_ns else None,
        'frame_total': _stats(total_ns) if total_ns else None,
        'frame_total_hit_only': _stats(hit_only) if hit_only else None,
        'frame_total_miss_involving': _stats(miss_frames) if miss_frames else None,
        'first_viewport': _stats(collected['first_viewport_ns']) if collected['first_viewport_ns'] else None,
        'trajectory_total_including_startup': (
            _stats(collected['trajectory_ns']) if collected['trajectory_ns'] else None),
        'trajectory_total_movement_only': (
            _stats(collected['movement_ns']) if collected['movement_ns'] else None),
        'bytes_read_total_including_startup': int(bytes_read_total),
        'hit_rate': (cache_hits_total / chunks_requested_total) if chunks_requested_total else None,
        'evictions_total': sum(m['evictions'] for m in frames),
        'resident_bytes_max': max(resident_series) if resident_series else 0,
        'resident_bytes_end': resident_series[-1] if resident_series else 0,
        'hit_only_frame_count': len(hit_only),
        'miss_involving_frame_count': len(miss_frames),
    }


def _file_hash_snapshot(case):
    return {'whole': _sha256_file(case['whole_path']), 'chunked': _sha256_file(case['chunked_path'])}


def _resolve_modes(requested, schedule_modes):
    '''Explicit --modes is honoured or rejected, never silently rewritten; the
    schedule's own modes may be filtered down to what this platform supports.'''
    if requested is not None:
        unsupported = [m for m in requested if m == 'macos_nocache' and sys.platform != 'darwin']
        if unsupported:
            raise ValidationError(
                f'--modes requested {unsupported[0]!r}, which requires Darwin (F_NOCACHE); '
                f'this platform is {sys.platform!r}. Re-run without it instead of having '
                'the run silently measure a different mode than you asked for.'
            )
        return tuple(requested)
    modes = tuple(m for m in schedule_modes
                  if m != 'macos_nocache' or sys.platform == 'darwin')
    if not modes:
        raise ValidationError(
            f'the schedule records only {list(schedule_modes)}, none of which this '
            f'platform ({sys.platform!r}) supports; pass --modes explicitly to choose '
            'a supported mode'
        )
    if modes != tuple(schedule_modes):
        dropped = [m for m in schedule_modes if m not in modes]
        _log(f'note: dropping schedule mode(s) {dropped} unsupported on {sys.platform!r}')
    return modes


def _run(args, town_disk, town_cache, town_region, arr, schedule, settings, positions,
         verification, modes, cases, memory_sidecar):
    chunk_size = settings['chunk_size']
    viewport = settings['viewport']
    methods = settings['methods']
    trajectory_kinds = settings['trajectory_kinds']

    for method in methods:
        _log(f'preparing case: method={method}')
        case = town_disk.prepare_case(arr, args.containers_dir / method, method,
                                       chunk_size=chunk_size)
        case['method'] = method
        cases.append(case)
        case['hashes_before'] = _file_hash_snapshot(case)
        recorded = case.get('hashes') or {}
        if (case['hashes_before']['whole'] != recorded.get('whole_sha256')
                or case['hashes_before']['chunked'] != recorded.get('chunked_sha256')):
            raise ValidationError(
                f'prepared container bytes for method={method} do not match the hashes '
                'prepare_case reported for them'
            )

    for case in cases:
        method = case['method']
        whole_path = Path(case['whole_path'])
        chunked_path = Path(case['chunked_path'])
        case['modes'] = {}
        for mode in modes:
            case['modes'][mode] = {}
            for trajectory_index, kind in enumerate(trajectory_kinds):
                _log(f'comparing cache arms: method={method} mode={mode} trajectory={kind}')
                case['modes'][mode][kind] = _benchmark_run(
                    town_cache, town_region, arr, whole_path, chunked_path, mode,
                    positions[kind], args.repeats, args.cache_sizes, viewport, trajectory_index,
                )

    memory_report = None
    if args.memory_repeats > 0:
        memory_report = _run_memory_subprocess(args, memory_sidecar)

    # Containers are re-hashed only now: the memory sub-measurement reads the
    # same files, so "unchanged during the benchmark" has to cover it too.
    for case in cases:
        case['hashes_after'] = _file_hash_snapshot(case)
        if case['hashes_after'] != case['hashes_before']:
            raise ValidationError(
                f'container file(s) changed during the run for method={case["method"]}'
            )

    report = {
        'schema_version': 1,
        'scope': 'Application-level cache comparison replaying an existing '
                 'town-movement schedule\'s recorded trajectories verbatim; see '
                 'benchmarks/town-cache-audit.md.',
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'source_files_sha256': _source_files_sha256(),
        'environment': _environment(),
        'map': {
            'source': schedule['map'].get('source'),
            'shape': [int(n) for n in arr.shape],
            'dtype': arr.dtype.str,
            'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
        },
        'verification': verification,
        'settings': {
            'schedule': str(args.schedule),
            'size': settings['size'],
            'seed': settings['seed'],
            'steps': settings['steps'],
            'repeats': args.repeats,
            'chunk_size': chunk_size,
            'viewport': viewport,
            'documented_chunk_size': DOCUMENTED_CHUNK_SIZE,
            'documented_viewport': DOCUMENTED_VIEWPORT,
            'methods': list(methods),
            'modes': list(modes),
            'cache_sizes': list(args.cache_sizes),
            'trajectory_kinds': list(trajectory_kinds),
            'memory_repeats': args.memory_repeats,
            'p95_method': 'numpy.percentile, linear interpolation',
            'json_format': 'compact separators, no indentation (reports are large)',
            'rotation_policy': ROTATION_POLICY,
            'priming_policy': PRIMING_POLICY,
            'session_policy': SESSION_POLICY,
            'validation_policy': VALIDATION_POLICY,
            'timing_unit_policy': TIMING_UNIT_POLICY,
            'trajectory_total_policy': TRAJECTORY_TOTAL_POLICY,
            'memory_policy': MEMORY_POLICY,
            'whole_resident_note': WHOLE_RESIDENT_NOTE,
        },
        'cases': cases,
        'memory': memory_report,
    }
    return report


# --------------------------------------------------------------------------- #
# memory sub-measurement (separate processes, separate script)
# --------------------------------------------------------------------------- #

def _memory_sidecar_path(output):
    sidecar = output.with_suffix('.memory.json')
    if sidecar == output:
        raise ValidationError(f'--output {output} leaves no distinct memory sidecar path')
    return sidecar


def _preflight_memory(args, memory_sidecar):
    '''Everything that can be known about the memory step before a single
    container is written: the worker exists, and its output path is free.'''
    if args.memory_repeats <= 0:
        return
    if not MEMORY_SCRIPT.is_file():
        raise ValidationError(
            f'--memory-repeats {args.memory_repeats} needs {MEMORY_SCRIPT_REL}, which does '
            'not exist in this working tree; run with --memory-repeats 0 to skip the '
            'memory sub-measurement, or provide that script'
        )
    memory_sidecar['path'] = _memory_sidecar_path(args.output)
    check_free(memory_sidecar['path'], 'memory output')


def _run_memory_subprocess(args, memory_sidecar):
    memory_output = memory_sidecar['path']
    check_free(memory_output, 'memory output')  # narrow the race before launching
    cmd = [
        sys.executable, str(MEMORY_SCRIPT),
        '--schedule', str(args.schedule),
        '--containers-dir', str(args.containers_dir),
        '--cache-sizes', ','.join(str(n) for n in args.cache_sizes),
        '--repeats', str(args.memory_repeats),
        '--output', str(memory_output),
    ]
    _log(f'running memory sub-measurement: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # The worker may or may not have created the sidecar before failing, so
        # this run never deletes it: a file we cannot prove we own is left for
        # inspection rather than removed.
        raise ValidationError(
            f'{MEMORY_SCRIPT_REL} failed (exit {result.returncode}): {result.stderr.strip()}'
            + (f' [left any partial output at {memory_output} untouched]'
               if memory_output.exists() else '')
        )
    if not memory_output.is_file():
        raise ValidationError(
            f'{MEMORY_SCRIPT_REL} exited 0 but wrote no report at {memory_output}'
        )
    with open(memory_output, 'r', encoding='utf-8') as f:
        try:
            memory_report = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValidationError(f'{MEMORY_SCRIPT_REL} wrote invalid JSON: {exc}') from None
    # Identity + digest of the file this invocation's worker created, so a
    # later failure can clean up exactly this sidecar and nothing else.
    st = os.stat(memory_output)
    memory_sidecar['owned'] = True
    memory_sidecar['identity'] = (st.st_dev, st.st_ino)
    memory_sidecar['sha256'] = _sha256_file(memory_output)
    return {
        'output_path': str(memory_output),
        'output_sha256': memory_sidecar['sha256'],
        'worker': MEMORY_SCRIPT_REL,
        'worker_sha256': _sha256_file(MEMORY_SCRIPT),
        'repeats': args.memory_repeats,
        'note': MEMORY_POLICY,
        'report': memory_report,
    }


# --------------------------------------------------------------------------- #
# cleanup of this run's own artifacts
# --------------------------------------------------------------------------- #

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


def _cleanup_own_memory_sidecar(memory_sidecar):
    '''Delete the memory sidecar only if this invocation's worker created it
    AND the path still holds that exact file (same dev/inode, same digest).
    A pre-existing file, a replaced file or a file we cannot prove we own is
    left alone.'''
    if not memory_sidecar.get('owned'):
        return
    path = memory_sidecar.get('path')
    if path is None:
        return
    try:
        st = os.stat(path)
    except OSError:
        return
    if (st.st_dev, st.st_ino) != memory_sidecar.get('identity'):
        _log(f'note: leaving {path} alone; it is no longer the file this run wrote')
        return
    try:
        if _sha256_file(path) != memory_sidecar.get('sha256'):
            _log(f'note: leaving {path} alone; its contents changed since this run wrote it')
            return
    except OSError:
        return
    with contextlib.suppress(OSError):
        Path(path).unlink()
        memory_sidecar['owned'] = False


def main(argv=None):
    args = parse_args(argv)
    output = args.output
    containers_dir = args.containers_dir
    cases = []
    created_dir = False
    memory_sidecar = {'path': None, 'owned': False}

    try:
        check_free(output, 'output')
        check_free(containers_dir, 'containers-dir')
        if not args.schedule.is_file():
            raise ValidationError(f'--schedule does not exist or is not a file: {args.schedule}')
        _preflight_memory(args, memory_sidecar)

        town_disk = _import('horizon.town_disk', 'prepare_case/load_whole/load_chunked')
        town_cache = _import('horizon.town_cache', 'ChunkRegionCache/WholeMapResident')
        town_region = _import('horizon.town_region', 'load_region')

        schedule = _load_schedule(args.schedule)
        arr, settings, positions, verification = _verify_schedule(schedule, args.schedule)
        modes = _resolve_modes(args.modes, settings['modes'])

        containers_dir.mkdir(parents=True)
        created_dir = True

        report = _run(args, town_disk, town_cache, town_region, arr, schedule, settings,
                      positions, verification, modes, cases, memory_sidecar)

        check_free(output, 'output')  # narrow the race right before writing
        output.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json.dumps(report, separators=(',', ':'), allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                output.unlink()
            raise
    except Exception as exc:
        _log(f'error: {type(exc).__name__}: {exc}')
        _cleanup_own_memory_sidecar(memory_sidecar)
        _cleanup_own_containers(containers_dir, created_dir, cases)
        return 1

    _log(f'wrote {output}')
    _log(f'containers written to {containers_dir}')
    if memory_sidecar.get('owned'):
        _log(f'memory sub-measurement written to {memory_sidecar["path"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

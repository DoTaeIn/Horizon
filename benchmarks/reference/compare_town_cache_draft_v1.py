#!/usr/bin/env python3
'''Application-level cache comparison, replaying an existing town-movement
schedule file's recorded trajectories verbatim.

This script does not generate a new map or new trajectories: it loads an
existing `scripts/compare_town_movement.py` report (e.g.
benchmarks/town-movement-1024-v1.json or the seed2 variant) via --schedule,
re-derives the map from its recorded (size, seed) and asserts the sha256
matches, asserts every horizon/*.py file the schedule recorded a hash for is
still byte-identical on disk, and then replays its recorded
trajectories[kind] positions exactly -- never regenerating or perturbing
them.

Four arms are compared per (method, mode, trajectory kind, repeat), all
driven by the same positions:

  stateless      horizon.town_region.load_region(chunked_path, roi, mode) --
                 no cache, real I/O every call.
  lru{N}         horizon.town_cache.ChunkRegionCache(chunked_path,
                 max_chunks=N, mode) -- one per --cache-sizes entry.
  whole_resident horizon.town_cache.WholeMapResident(whole_path, mode).

This script implements neither town_disk/town_region's container parsing
nor town_cache's cache policy; it only calls their documented APIs. See
docs/town-cache-method.md for full scope and caveats.
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
_DTYPE = np.dtype('<u2')

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
    'never reused across repeats. first_viewport_ns = that session\'s own '
    'startup_metrics total_ns + its first frame\'s total_ns.'
)
VALIDATION_POLICY = (
    'Every array returned by every arm, every frame, is checked (ndarray '
    'type, shape, dtype, byte content against a crop of the in-memory '
    'original map) strictly after the timed call returns.'
)
WHOLE_RESIDENT_NOTE = (
    'whole_resident has zero I/O/decode cost per .load() by construction '
    'in this benchmark (the map is already decoded and resident); this is '
    'not evidence that whole-map residency is universally the best choice '
    '-- it ignores memory cost and multi-map scenarios.'
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
        if p not in out:
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
                         help='0 disables the memory sub-measurement (fast smoke runs)')
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
    pkg_dir = (REPO_ROOT / 'horizon').resolve()
    hashes = {'scripts/compare_town_cache.py': _sha256_file(Path(__file__).resolve())}
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
# schedule loading / verification
# --------------------------------------------------------------------------- #

def _load_schedule(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _verify_schedule(schedule, schedule_path):
    '''Regenerate the recorded map and re-hash every still-existing source
    file the schedule recorded a hash for; raise loudly on any mismatch.
    Never regenerates or perturbs trajectories -- those are read verbatim.'''
    settings = schedule['settings']
    arr = make_town_map(size=settings['size'], seed=settings['seed'])
    arr = np.ascontiguousarray(arr)
    actual_map_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    expected_map_sha = schedule['map']['sha256']
    if actual_map_sha != expected_map_sha:
        raise ValidationError(
            f'schedule map sha256 mismatch: expected {expected_map_sha}, got {actual_map_sha}'
        )
    checked = []
    for relpath, expected_sha in schedule['source_files_sha256'].items():
        full = REPO_ROOT / relpath
        if relpath == 'scripts/compare_town_movement.py' or not full.exists():
            continue  # that script and any not-present file are out of this script's scope
        actual_sha = _sha256_file(full)
        if actual_sha != expected_sha:
            raise ValidationError(
                f'{relpath} sha256 mismatch vs schedule: expected {expected_sha}, got {actual_sha} '
                '(existing module changed since the schedule was recorded)'
            )
        checked.append(relpath)
    return arr, {
        'schedule_path': str(schedule_path),
        'schedule_sha256': _sha256_file(schedule_path),
        'map_sha256_verified': True,
        'source_files_verified': checked,
    }


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


def _validate_loaded(arr, expected_bytes, label):
    if type(arr) is not np.ndarray:
        raise ValidationError(f'{label}: expected ndarray, got {type(arr).__name__}')
    if arr.dtype != _DTYPE:
        raise ValidationError(f'{label}: dtype mismatch {arr.dtype} != {_DTYPE}')
    if arr.tobytes() != expected_bytes:
        raise ValidationError(f'{label}: byte content mismatch')


def _stats(samples_ns):
    ms = [n / 1e6 for n in samples_ns]
    return {
        'median_ms': round(statistics.median(ms), 6),
        'p95_ms': round(float(np.percentile(ms, 95, method='linear')), 6),
        'max_ms': round(max(ms), 6),
    }


_CHUNK_FIELDS = (
    'chunks_requested', 'chunks_loaded', 'cache_hits', 'cache_misses', 'evictions',
    'chunk_coordinates', 'hit_coordinates', 'miss_coordinates', 'evicted_coordinates',
)


def _normalize_metrics(arm, raw):
    '''Stateless's load_region metrics lack cache concepts; fill them in
    honestly as "every chunk touched is a miss, nothing is retained".
    lru{N}/whole_resident metrics from horizon.town_cache already carry the
    full key set, including resident_bytes, and are passed through as-is.'''
    if arm != 'stateless':
        return dict(raw)
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
    assert n in cache_sizes
    return town_cache.ChunkRegionCache(chunked_path, max_chunks=n, mode=mode)


def _startup_metrics(arm, session):
    zero = {k: 0 for k in ('total_ns', 'read_ns', 'decode_ns', 'bytes_read',
                            'bytes_requested', 'read_calls', 'seek_calls')}
    if arm == 'stateless':
        return zero
    return dict(session.startup_metrics)


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
    per_arm_frames = {a: [] for a in arms}
    per_arm_startup = {a: [] for a in arms}

    for repeat_index in range(repeats):
        _prime(mode, (whole_path, chunked_path))

        construct_order = _rotate(arms, trajectory_index + repeat_index)
        sessions = {}
        startup = {}
        for arm in construct_order:
            session = _make_session(arm, cache_sizes, town_cache, whole_path, chunked_path, mode)
            sessions[arm] = session
            startup[arm] = _startup_metrics(arm, session)
            per_arm_startup[arm].append(startup[arm])

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
                    _validate_loaded(out, expected_bytes, arm)
                    metrics = _normalize_metrics(arm, raw_metrics)
                    frame_record['arms'][arm] = metrics
                    per_arm_frames[arm].append(metrics)
                frames.append(frame_record)
        finally:
            for arm, session in sessions.items():
                if session is not None:
                    session.close()
        repeats_out.append({'repeat': repeat_index, 'construct_order': construct_order, 'frames': frames})

    aggregate = {
        arm: _aggregate_arm(per_arm_startup[arm], per_arm_frames[arm])
        for arm in arms
    }
    return {'repeats': repeats_out, 'aggregate': aggregate}


def _aggregate_arm(startups, frames):
    total_ns = [m['total_ns'] for m in frames]
    hit_only = [m['total_ns'] for m in frames if m['cache_misses'] == 0]
    miss_frames = [m['total_ns'] for m in frames if m['cache_misses'] > 0]
    bytes_read_total = sum(m['bytes_read'] for m in frames) + sum(s['bytes_read'] for s in startups)
    chunks_requested_total = sum(m['chunks_requested'] for m in frames)
    cache_hits_total = sum(m['cache_hits'] for m in frames)
    resident_series = [m['resident_bytes'] for m in frames]
    startup_total_ns = [s['total_ns'] for s in startups]
    first_viewport_ns = [
        s['total_ns'] + frames[i * (len(frames) // max(len(startups), 1))]['total_ns']
        for i, s in enumerate(startups)
    ] if startups and frames else []
    return {
        'startup_total_ns': _stats(startup_total_ns) if startup_total_ns else None,
        'frame_total': _stats(total_ns) if total_ns else None,
        'frame_total_hit_only': _stats(hit_only) if hit_only else None,
        'frame_total_miss_involving': _stats(miss_frames) if miss_frames else None,
        'first_viewport_ns': _stats(first_viewport_ns) if first_viewport_ns else None,
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


def _run(args, town_disk, town_cache, town_region, arr, schedule, cases):
    settings = schedule['settings']
    chunk_size = settings['chunk_size']
    viewport = settings['viewport']
    methods = settings['methods']
    trajectory_kinds = settings['trajectory_kinds']
    trajectories = schedule['trajectories']
    modes = args.modes or tuple(
        m for m in settings['modes'] if m in ALL_MODES
    )
    if 'macos_nocache' in modes and sys.platform != 'darwin':
        modes = tuple(m for m in modes if m != 'macos_nocache')
        if not modes:
            modes = ('warm_buffered',)

    for method in methods:
        _log(f'preparing case: method={method}')
        case = town_disk.prepare_case(arr, args.containers_dir / method, method, chunk_size=chunk_size)
        case['method'] = method
        cases.append(case)
        case['hashes_before'] = _file_hash_snapshot(case)

    for case in cases:
        method = case['method']
        whole_path = Path(case['whole_path'])
        chunked_path = Path(case['chunked_path'])
        case['modes'] = {}
        for mode in modes:
            case['modes'][mode] = {}
            for trajectory_index, kind in enumerate(trajectory_kinds):
                _log(f'comparing cache arms: method={method} mode={mode} trajectory={kind}')
                positions = trajectories[kind]
                case['modes'][mode][kind] = _benchmark_run(
                    town_cache, town_region, arr, whole_path, chunked_path, mode,
                    positions, args.repeats, args.cache_sizes, viewport, trajectory_index,
                )
        case['hashes_after'] = _file_hash_snapshot(case)
        if case['hashes_after'] != case['hashes_before']:
            raise ValidationError(f'container file(s) changed during benchmark for method={method}')

    memory_report = None
    if args.memory_repeats > 0:
        memory_report = _run_memory_subprocess(args, methods)

    report = {
        'schema_version': 1,
        'scope': 'Application-level cache comparison replaying an existing '
                 'town-movement schedule\'s recorded trajectories verbatim; see '
                 'docs/town-cache-method.md.',
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'source_files_sha256': _source_files_sha256(),
        'environment': _environment(),
        'map': {
            'source': schedule['map']['source'],
            'shape': [int(n) for n in arr.shape],
            'dtype': arr.dtype.str,
            'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
        },
        'verification': args.verification,
        'settings': {
            'schedule': str(args.schedule),
            'size': settings['size'],
            'seed': settings['seed'],
            'steps': settings['steps'],
            'repeats': args.repeats,
            'chunk_size': chunk_size,
            'viewport': viewport,
            'methods': list(methods),
            'modes': list(modes),
            'cache_sizes': list(args.cache_sizes),
            'trajectory_kinds': list(trajectory_kinds),
            'memory_repeats': args.memory_repeats,
            'p95_method': 'numpy.percentile, linear interpolation',
            'rotation_policy': ROTATION_POLICY,
            'priming_policy': PRIMING_POLICY,
            'session_policy': SESSION_POLICY,
            'validation_policy': VALIDATION_POLICY,
            'whole_resident_note': WHOLE_RESIDENT_NOTE,
        },
        'cases': cases,
        'memory': memory_report,
    }
    return report


def _run_memory_subprocess(args, methods):
    memory_output = args.output.with_suffix('.memory.json')
    check_free(memory_output, 'memory output')
    cmd = [
        sys.executable, str(REPO_ROOT / 'scripts' / 'measure_town_cache_memory.py'),
        '--schedule', str(args.schedule),
        '--containers-dir', str(args.containers_dir),
        '--cache-sizes', ','.join(str(n) for n in args.cache_sizes),
        '--repeats', str(args.memory_repeats),
        '--output', str(memory_output),
    ]
    _log(f'running memory sub-measurement: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValidationError(
            f'measure_town_cache_memory.py failed (exit {result.returncode}): {result.stderr.strip()}'
        )
    with open(memory_output, 'r', encoding='utf-8') as f:
        memory_report = json.load(f)
    return {'output_path': str(memory_output), 'output_sha256': _sha256_file(memory_output),
            'report': memory_report}


def _cleanup_own_containers(containers_dir, created_dir, cases):
    for case in cases:
        for key in ('whole_path', 'chunked_path'):
            path = case.get(key)
            if path:
                with contextlib.suppress(OSError):
                    Path(path).unlink()
        method = case.get('method')
        if method:
            with contextlib.suppress(OSError):
                (containers_dir / method).rmdir()
    if created_dir:
        with contextlib.suppress(OSError):
            containers_dir.rmdir()


def main(argv=None):
    args = parse_args(argv)
    output = args.output
    containers_dir = args.containers_dir
    cases = []
    created_dir = False

    try:
        check_free(output, 'output')
        check_free(containers_dir, 'containers-dir')
        if not args.schedule.exists():
            raise ValidationError(f'--schedule does not exist: {args.schedule}')

        town_disk = _import('horizon.town_disk', 'prepare_case/load_whole/load_chunked')
        town_cache = _import('horizon.town_cache', 'ChunkRegionCache/WholeMapResident')
        town_region = _import('horizon.town_region', 'load_region')

        schedule = _load_schedule(args.schedule)
        arr, verification = _verify_schedule(schedule, args.schedule)
        args.verification = verification

        containers_dir.mkdir(parents=True)
        created_dir = True

        report = _run(args, town_disk, town_cache, town_region, arr, schedule, cases)

        check_free(output, 'output')
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

#!/usr/bin/env python3
'''Generic memory sub-measurement CLI for the town-cache arms.

Standalone contract consumed by scripts/compare_town_cache.py's
--memory-repeats path (see that script and benchmarks/town-cache-audit.md),
but runnable on its own: --schedule an existing town-movement-style schedule
JSON, --containers-dir an existing directory already holding, for every
method the schedule names, <containers-dir>/<method>/whole.hrzn and
chunked.hcix (as written by horizon.town_disk.prepare_case) -- this script
never writes, prepares or mutates a container, it only reads and re-hashes
them.

For every (method, trajectory_kind, arm, repeat) this process -- the
"parent" -- regenerates the map from the schedule's recorded (size, seed)
and asserts its sha256, derives that trajectory's viewport ROI sequence from
the schedule's recorded positions, and hashes the expected crop sequence.
It then hands a *small* request (just the arm name, the two container
paths, the ROI list and the expected sha256 -- never the full map array and
never the full schedule) to a brand-new `python -c` subprocess, which
imports horizon.town_cache/horizon.town_region fresh, builds exactly one
arm instance (or none, for stateless), replays the ROIs, and reports its own
memory metrics before exiting. No state is carried between measurements:
every (method, trajectory_kind, arm, repeat) tuple gets its own process, so
a parent's already-decoded map, previous reports or other arms are never
visible to it. Each reported sample includes the child's own pid, so two
measurements never collide.

Memory is reported on three distinct, non-interchangeable axes, and this
script never conflates them:

  resident_decoded_bytes_{peak,end}  the *exact* retained-array accounting
      each arm already exposes via `.resident_bytes` (sum of retained
      decoded chunks' own .nbytes for lru{N}, the resident map's .nbytes for
      whole_resident, always 0 for stateless). This excludes the index,
      stat snapshot, metrics dicts and every array already handed back to a
      caller -- it is the cache policy's own ledger, not a memory-profiler
      estimate.
  tracemalloc_*_bytes   traced Python/NumPy allocator activity only (started
      after imports and a gc.collect(), so neither is included), covering
      everything live on the Python heap -- cache state, output arrays,
      this probe's own small bookkeeping -- but never native allocations
      tracemalloc cannot see (e.g. inside a native codec's own buffers).
      startup_* covers the constructor only (peak reset to 0 immediately
      after); movement_* covers the replay loop only.
  process_rss_*_hwm_bytes   the OS's absolute resident-set high-water mark
      for the whole process (resource.getrusage(RUSAGE_SELF).ru_maxrss),
      including the interpreter, every import, and anything else the
      process ever touched -- a ceiling, not an attributable cache cost,
      and monotonically non-decreasing within one process (a "baseline"
      reading is therefore also already a high-water mark, not a current
      value). Units differ by platform: Darwin reports bytes directly;
      Linux reports KiB and is converted to bytes here; on any other
      platform (or if the `resource` module is unavailable) this script
      reports the value as unavailable rather than guess a unit.

warm_buffered is the only mode this script measures, and it describes the
OS page cache only: this process does one untimed buffered read-and-discard
of both container files before any arm is built, but neither that nor any
reading here is evidence of, or makes any claim about, a physically cold
SSD/controller read -- see horizon.town_disk.CACHE_NOTE.
'''
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from horizon.codec import METHODS as ALLOWED_METHODS  # noqa: E402
from horizon.town_map import make_town_map  # noqa: E402

DEFAULT_REPEATS = 3
MIN_REPEATS = 1
DEFAULT_CACHE_SIZES = (4, 8)
WHOLE_NAME = 'whole.hrzn'
CHUNKED_NAME = 'chunked.hcix'
_DTYPE = np.dtype('<u2')

MEMORY_SCOPE = (
    'Independent memory sub-measurement: each (method, trajectory_kind, arm, repeat) '
    'is measured in its own fresh Python subprocess that imports horizon.town_cache / '
    'horizon.town_region fresh and is handed only a small request (arm name, two '
    'container paths, one trajectory\'s ROI list, one expected sha256) -- never the '
    'full map array and never the full schedule file. warm_buffered only. '
    'resident_decoded_bytes_* is the arm\'s own exact retained-array ledger '
    '(.resident_bytes); tracemalloc_*_bytes is traced Python/NumPy heap activity only, '
    'measured after imports+gc.collect(), split into startup (constructor) and '
    'movement (replay loop) phases with peak reset between them; process_rss_*_hwm_bytes '
    'is the absolute, monotonically non-decreasing OS RSS high-water mark for the whole '
    'process, not an attributable cache cost. None of this is a claim about physically '
    'cold SSD/controller reads; warm_buffered only describes this process priming the OS '
    'page cache once before each repeat. Only one output array is ever live per load step.'
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Fresh-process memory measurement of the town-cache arms '
                    '(stateless, lru{N}, whole_resident) over an existing '
                    'town-movement-style schedule, replayed verbatim against '
                    'existing containers.')
    parser.add_argument('--schedule', required=True, type=Path,
                         help='existing town-movement-style schedule JSON to replay trajectories from')
    parser.add_argument('--containers-dir', required=True, type=Path,
                         help='existing directory holding <method>/whole.hrzn and chunked.hcix '
                              'for every method the schedule names; never written to')
    parser.add_argument('--cache-sizes', type=_cache_sizes_type, default=DEFAULT_CACHE_SIZES)
    parser.add_argument('--repeats', type=_repeats_type, default=DEFAULT_REPEATS)
    parser.add_argument('--output', required=True, type=Path,
                         help='new JSON report path; never overwritten')
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
    hashes = {'scripts/measure_town_cache_memory.py': _sha256_file(Path(__file__).resolve())}
    for p in sorted(pkg_dir.glob('*.py')):
        hashes[p.relative_to(REPO_ROOT).as_posix()] = _sha256_file(p)
    return hashes


def _environment():
    return {
        'python': sys.version,
        'python_implementation': platform.python_implementation(),
        'os': platform.platform(),
        'machine': platform.machine(),
    }


def _load_schedule(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _require_positive_int(settings, key, label):
    value = settings.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f'schedule settings.{key} must be a positive integer, got {value!r} ({label})')
    return value


def _require_nonnegative_int(settings, key, label):
    value = settings.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f'schedule settings.{key} must be a non-negative integer, got {value!r} ({label})')
    return value


def _require_nonempty_unique_str_list(settings, key):
    value = settings.get(key)
    if not isinstance(value, list) or not value:
        raise ValidationError(f'schedule settings.{key} must be a non-empty list')
    if not all(isinstance(v, str) and v for v in value):
        raise ValidationError(f'schedule settings.{key} entries must all be non-empty strings')
    if len(set(value)) != len(value):
        raise ValidationError(f'schedule settings.{key} must not contain duplicate entries')
    return value


_SHA256_HEX_RE = re.compile(r'^[0-9a-f]{64}$')


def _is_valid_repo_relpath(relpath):
    '''A plain repo-relative POSIX path: a string, not absolute, and never
    containing a ".." component (which would let a recorded hash entry walk
    outside REPO_ROOT).'''
    if not isinstance(relpath, str) or not relpath:
        return False
    if relpath.startswith('/') or relpath.startswith('\\'):
        return False
    pure = PurePosixPath(relpath)
    if pure.is_absolute():
        return False
    return '..' not in pure.parts


def _validate_schedule_shape(schedule):
    '''Validate every dimension/structural field this script depends on,
    strictly and before any work (map regeneration, container access, or
    subprocess spawning) is attempted. Rejects missing/wrong-typed fields,
    an out-of-range viewport, empty or duplicate methods/trajectory_kinds,
    an unwhitelisted method (methods are used to build a filesystem path
    under --containers-dir, so an unwhitelisted value here is a path
    traversal vector, e.g. "../../etc"), a trajectory_kind with no matching
    entry in schedule["trajectories"] or whose recorded position count
    disagrees with settings.steps, any position whose coordinates are not
    plain in-bounds integers (repeated positions -- standing still,
    backtracking, a teleport landing twice on the same cell -- are valid
    and not rejected), and a malformed/empty source_files_sha256 map.'''
    if not isinstance(schedule, dict) or not isinstance(schedule.get('settings'), dict):
        raise ValidationError('schedule must be a JSON object with a "settings" object')
    settings = schedule['settings']

    size = _require_positive_int(settings, 'size', 'map side length')
    _require_nonnegative_int(settings, 'seed', 'map seed')
    chunk_size = _require_positive_int(settings, 'chunk_size', 'chunk side length')
    viewport = _require_positive_int(settings, 'viewport', 'viewport side length')
    steps = _require_positive_int(settings, 'steps', 'trajectory step count')
    if viewport > size:
        raise ValidationError(f'schedule settings.viewport ({viewport}) must not exceed settings.size ({size})')
    if chunk_size > size:
        raise ValidationError(f'schedule settings.chunk_size ({chunk_size}) must not exceed settings.size ({size})')

    methods = _require_nonempty_unique_str_list(settings, 'methods')
    for method in methods:
        if method not in ALLOWED_METHODS:
            raise ValidationError(
                f'schedule settings.methods: {method!r} is not a whitelisted codec method '
                f'(expected one of {ALLOWED_METHODS}); unwhitelisted method names are refused '
                'outright because they are used to build a path under --containers-dir'
            )

    kinds = _require_nonempty_unique_str_list(settings, 'trajectory_kinds')
    trajectories = schedule.get('trajectories')
    if not isinstance(trajectories, dict):
        raise ValidationError('schedule["trajectories"] must be an object')
    for kind in kinds:
        positions = trajectories.get(kind)
        if not isinstance(positions, list) or not positions:
            raise ValidationError(f'schedule["trajectories"][{kind!r}] must be a non-empty list')
        if len(positions) != steps:
            raise ValidationError(
                f'schedule["trajectories"][{kind!r}] has {len(positions)} positions, '
                f'expected settings.steps ({steps})'
            )
        for position in positions:
            if (not isinstance(position, (list, tuple)) or len(position) != 2
                    or not all(isinstance(c, int) and not isinstance(c, bool) for c in position)):
                raise ValidationError(
                    f'schedule["trajectories"][{kind!r}] entries must be [row, col] integer pairs, '
                    f'got {position!r}'
                )
            row, col = position
            if not (0 <= row < size and 0 <= col < size):
                raise ValidationError(
                    f'schedule["trajectories"][{kind!r}] entry ({row}, {col}) is out of the '
                    f'0..{size - 1} map bounds'
                )
            # Repeated positions within or across a trajectory are valid
            # (standing still, backtracking, a teleport landing on the same
            # cell twice) and are deliberately not rejected here.

    source_hashes = schedule.get('source_files_sha256')
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValidationError('schedule["source_files_sha256"] must be a non-empty object')
    for relpath, sha in source_hashes.items():
        if not _is_valid_repo_relpath(relpath):
            raise ValidationError(
                f'schedule["source_files_sha256"] key {relpath!r} is not a plain repo-relative '
                'path (absolute paths and ".." components are refused)'
            )
        if not isinstance(sha, str) or not _SHA256_HEX_RE.match(sha):
            raise ValidationError(
                f'schedule["source_files_sha256"][{relpath!r}] must be a 64-character lowercase '
                f'hex sha256 digest, got {sha!r}'
            )

    map_info = schedule.get('map')
    if isinstance(map_info, dict):
        if 'shape' in map_info and list(map_info['shape']) != [size, size]:
            raise ValidationError(
                f'schedule["map"]["shape"] {map_info["shape"]!r} disagrees with settings.size {size}'
            )
        if 'dtype' in map_info and map_info['dtype'] != _DTYPE.str:
            raise ValidationError(
                f'schedule["map"]["dtype"] {map_info["dtype"]!r} disagrees with the expected {_DTYPE.str!r}'
            )


def _verify_schedule(schedule, schedule_path):
    '''Regenerate the recorded map and strictly re-hash every single source
    file recorded in schedule["source_files_sha256"] -- including this
    script's own file, if the schedule happens to record one for it, and
    the other scripts that produced/consumed this schedule -- raising
    loudly on any mismatch or on any recorded file that no longer exists.
    Nothing is skipped as "out of scope": a replay can never confirm a file
    it does not re-hash, and comparing a recorded sha256 against this
    script's own on-disk bytes is an ordinary file check, not a cyclical
    import. Never regenerates or perturbs trajectories -- those are read
    verbatim.'''
    settings = schedule['settings']
    arr = np.ascontiguousarray(make_town_map(size=settings['size'], seed=settings['seed']))
    actual_map_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    expected_map_sha = schedule['map']['sha256']
    if actual_map_sha != expected_map_sha:
        raise ValidationError(
            f'schedule map sha256 mismatch: expected {expected_map_sha}, got {actual_map_sha}'
        )
    checked = []
    for relpath, expected_sha in schedule['source_files_sha256'].items():
        full = REPO_ROOT / relpath
        if not full.exists():
            # A schedule naming a source file that no longer exists can never be
            # confirmed to still match what it recorded -- "not present" is not
            # "out of scope" and must fail loudly rather than be silently skipped.
            raise ValidationError(
                f'{relpath} is recorded in the schedule\'s source_files_sha256 but no '
                'longer exists on disk; cannot confirm it still matches'
            )
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


def _rois_and_expected_sha(arr, positions, size, viewport):
    rois = [list(_viewport_region(r, c, size, viewport)) for r, c in positions]
    digest = hashlib.sha256()
    for row, col, h, w in rois:
        digest.update(memoryview(np.ascontiguousarray(arr[row:row + h, col:col + w])))
    return rois, digest.hexdigest()


def _arm_names(cache_sizes):
    return ['stateless'] + [f'lru{n}' for n in cache_sizes] + ['whole_resident']


def _container_paths(containers_dir, method):
    directory = containers_dir / method
    whole_path = directory / WHOLE_NAME
    chunked_path = directory / CHUNKED_NAME
    if not whole_path.is_file():
        raise ValidationError(f'missing existing container: {whole_path}')
    if not chunked_path.is_file():
        raise ValidationError(f'missing existing container: {chunked_path}')
    return whole_path, chunked_path


def _hash_containers(whole_path, chunked_path):
    return {'whole': _sha256_file(whole_path), 'chunked': _sha256_file(chunked_path)}


def _read_buffered_discard(path):
    with open(path, 'rb') as f:
        while f.read(1 << 20):
            pass


# --------------------------------------------------------------------------- #
# the fresh-process probe
# --------------------------------------------------------------------------- #

_PROBE = r'''
import gc, hashlib, json, os, platform, sys, tracemalloc
req = json.load(sys.stdin)


def _rss_hwm():
    try:
        import resource
    except ImportError:
        return None, 'unavailable (resource module not present on this platform)'
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system = platform.system()
    if system == 'Darwin':
        return int(value), 'bytes (Darwin ru_maxrss)'
    if system == 'Linux':
        return int(value) * 1024, 'bytes (converted from Linux ru_maxrss KiB)'
    return None, f'unavailable (unverified ru_maxrss unit on platform {system!r})'


from horizon.town_cache import ChunkRegionCache, WholeMapResident
from horizon.town_region import load_region

gc.collect()
baseline_rss, rss_units = _rss_hwm()
digest = hashlib.sha256()
obj = None
tracemalloc.start()

if req['arm'] == 'whole_resident':
    obj = WholeMapResident(req['whole_path'], mode='warm_buffered')
elif req['arm'].startswith('lru'):
    obj = ChunkRegionCache(req['chunked_path'], max_chunks=int(req['arm'][3:]), mode='warm_buffered')

startup_current, startup_peak = tracemalloc.get_traced_memory()
startup_rss, _ = _rss_hwm()
tracemalloc.reset_peak()

retained_peak = 0 if obj is None else obj.resident_bytes
output_peak = 0
read_bytes = 0 if obj is None else obj.startup_metrics['bytes_read']
hits = 0
misses = 0

for roi in req['rois']:
    if obj is None:
        out, metrics = load_region(req['chunked_path'], roi, mode='warm_buffered')
    else:
        out, metrics = obj.load(roi)
    assert type(out).__name__ == 'ndarray'
    assert out.shape == tuple(roi[2:])
    assert out.dtype.str == '<u2'
    assert out.flags.owndata and out.flags.c_contiguous
    digest.update(memoryview(out))
    output_peak = max(output_peak, out.nbytes)
    retained_peak = max(retained_peak, 0 if obj is None else obj.resident_bytes)
    read_bytes += metrics['bytes_read']
    hits += metrics.get('cache_hits', 0)
    misses += metrics.get('cache_misses', metrics['chunks_loaded'])
    del out, metrics

movement_current, movement_peak = tracemalloc.get_traced_memory()
process_rss, _ = _rss_hwm()
retained_final = 0 if obj is None else obj.resident_bytes
tracemalloc.stop()
if obj is not None:
    obj.close()

actual = digest.hexdigest()
assert actual == req['expected_sha256'], (actual, req['expected_sha256'])

print(json.dumps({
    'pid': os.getpid(),
    'output_sequence_sha256': actual,
    'resident_decoded_bytes_peak': retained_peak,
    'resident_decoded_bytes_end': retained_final,
    'one_output_bytes_peak': output_peak,
    'tracemalloc_startup_current_bytes': startup_current,
    'tracemalloc_startup_peak_bytes': startup_peak,
    'tracemalloc_movement_current_bytes': movement_current,
    'tracemalloc_movement_peak_bytes': movement_peak,
    'tracemalloc_overall_peak_bytes': max(startup_peak, movement_peak),
    'process_rss_baseline_hwm_bytes': baseline_rss,
    'process_rss_startup_hwm_bytes': startup_rss,
    'process_rss_overall_hwm_bytes': process_rss,
    'process_rss_units': rss_units,
    'bytes_read_including_startup': read_bytes,
    'cache_hits': hits,
    'cache_misses': misses,
}))
'''


def _run_probe(arm, whole_path, chunked_path, rois, expected_sha256):
    request = {
        'arm': arm,
        'whole_path': str(whole_path),
        'chunked_path': str(chunked_path),
        'rois': rois,
        'expected_sha256': expected_sha256,
    }
    done = subprocess.run(
        [sys.executable, '-c', _PROBE],
        input=json.dumps(request), text=True, capture_output=True, cwd=REPO_ROOT,
    )
    if done.returncode != 0:
        raise ValidationError(
            f'memory probe subprocess failed for arm={arm} (exit {done.returncode}): '
            f'{done.stderr.strip()}'
        )
    return json.loads(done.stdout)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def _run(args, arr, schedule):
    settings = schedule['settings']
    size = settings['size']
    viewport = settings['viewport']
    methods = settings['methods']
    trajectory_kinds = settings['trajectory_kinds']
    trajectories = schedule['trajectories']
    arms = _arm_names(args.cache_sizes)

    cases = []
    container_hashes = {}
    for method in methods:
        whole_path, chunked_path = _container_paths(args.containers_dir, method)
        before = _hash_containers(whole_path, chunked_path)

        for kind in trajectory_kinds:
            _log(f'measuring memory: method={method} trajectory={kind}')
            positions = trajectories[kind]
            rois, expected_sha256 = _rois_and_expected_sha(arr, positions, size, viewport)

            for arm in arms:
                samples = []
                for repeat in range(args.repeats):
                    # Symmetric untimed priming of both container files, once
                    # per repeat, before the timed/measured probe subprocess
                    # starts -- matches scripts/compare_town_cache.py's own
                    # warm_buffered priming policy.
                    _read_buffered_discard(whole_path)
                    _read_buffered_discard(chunked_path)
                    sample = _run_probe(arm, whole_path, chunked_path, rois, expected_sha256)
                    sample['repeat'] = repeat
                    samples.append(sample)
                cases.append({
                    'method': method,
                    'trajectory': kind,
                    'arm': arm,
                    'samples': samples,
                })

        after = _hash_containers(whole_path, chunked_path)
        if after != before:
            raise ValidationError(
                f'container file(s) changed during memory measurement for method={method}'
            )
        container_hashes[method] = {'before': before, 'after': after}

    return {
        'schema_version': 1,
        'scope': MEMORY_SCOPE,
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'source_files_sha256': _source_files_sha256(),
        'environment': _environment(),
        'map': {
            'source': schedule.get('map', {}).get('source', 'horizon.town_map.make_town_map'),
            'shape': [int(n) for n in arr.shape],
            'dtype': arr.dtype.str,
            'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
        },
        'verification': args.verification,
        'containers': container_hashes,
        'settings': {
            'schedule': str(args.schedule),
            'containers_dir': str(args.containers_dir),
            'size': size,
            'seed': settings['seed'],
            'viewport': viewport,
            'methods': list(methods),
            'trajectory_kinds': list(trajectory_kinds),
            'cache_sizes': list(args.cache_sizes),
            'repeats': args.repeats,
            'mode': 'warm_buffered',
        },
        'cases': cases,
    }


def main(argv=None):
    args = parse_args(argv)
    output = args.output

    try:
        check_free(output, 'output')
        if not args.schedule.exists():
            raise ValidationError(f'--schedule does not exist: {args.schedule}')
        if not args.containers_dir.is_dir():
            raise ValidationError(f'--containers-dir does not exist: {args.containers_dir}')

        schedule = _load_schedule(args.schedule)
        _validate_schedule_shape(schedule)
        arr, verification = _verify_schedule(schedule, args.schedule)
        args.verification = verification

        report = _run(args, arr, schedule)

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
        return 1

    _log(f'wrote {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

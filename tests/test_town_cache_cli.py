'''Focused tests for scripts/compare_town_cache.py (the CLI, not
horizon.town_cache/horizon.town_disk/horizon.town_region themselves), plus
the scripts/measure_town_cache_memory.py worker it drives.

This CLI never generates a new map or new trajectories: it loads an existing
town-movement-style schedule file, re-derives the map from the schedule's
recorded (size, seed) and asserts the sha256 matches, re-hashes every
recorded source file (strict: missing or changed is a rejected run, never a
silently skipped entry -- see scripts/compare_town_cache.py's own module
docstring), and then replays the schedule's recorded trajectory positions
verbatim across four cache arms (stateless, lru{N}..., whole_resident).

Covers:
  * a small, real (not mocked) town-movement-style schedule fixture at the
    documented chunk_size/viewport (256/256), whose map/source hashes are
    regenerated from the actual repo state, and confirming the CLI accepts
    and replays it.
  * default-enabled memory sub-measurement: a real subprocess integration
    with --memory-repeats 1 and few steps, folded into report["memory"].
  * raw per-repeat startup_metrics (verbatim session constructor metrics)
    and trajectory_ns (startup/frames/total, in nanoseconds) shape and
    arithmetic identity, plus an independent recomputation of every
    millisecond aggregate stat block from those same raw per-repeat values.
  * strict schedule validation: missing map hash match, mismatched recorded
    source file hash, a schedule naming a source file that no longer exists
    (strict: rejected, not skipped), and the scripts/compare_town_movement.py
    entry specifically (missing and mismatched).
  * output / containers-dir pre-existence is preserved on a refused run.
  * invalid arguments and an unsupported --modes entry.
  * the memory sub-measurement run standalone (scripts/measure_town_cache_memory.py
    invoked directly): fresh-process-per-sample pid isolation, and that a
    missing or mismatched hash for ANY recorded source file -- including
    scripts/compare_town_movement.py itself -- is rejected there too (its
    own _verify_schedule documents that nothing recorded is out of scope).
'''
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / 'scripts' / 'compare_town_cache.py'
MEMORY_SCRIPT = REPO_ROOT / 'scripts' / 'measure_town_cache_memory.py'
MOVEMENT_SCRIPT_REL = 'scripts/compare_town_movement.py'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from horizon.town_map import make_town_map  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location('compare_town_cache_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()

# The published runs and this CLI's own defaults use chunk_size=viewport=256
# (mod.DOCUMENTED_CHUNK_SIZE / mod.DOCUMENTED_VIEWPORT); the fixture matches
# that so it exercises the real, documented grid shape rather than an
# arbitrary smaller one, while staying small enough (a 2x2 chunk grid, few
# steps) to replay quickly.
FIXTURE_SIZE = 512
FIXTURE_SEED = 42
FIXTURE_CHUNK_SIZE = 256
FIXTURE_VIEWPORT = 256
FIXTURE_STEPS = 6


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _real_source_files_sha256():
    '''The actual, current sha256 of every horizon/*.py file plus
    scripts/compare_town_movement.py, so a fixture schedule built from this
    verifies cleanly against today's repo state. Mirrors the shape a real
    scripts/compare_town_movement.py report records (it hashes itself and
    every horizon/*.py module), computed independently here rather than
    imported from the module under test.'''
    pkg_dir = REPO_ROOT / 'horizon'
    hashes = {
        p.relative_to(REPO_ROOT).as_posix(): _sha256_file(p)
        for p in sorted(pkg_dir.glob('*.py'))
    }
    hashes[MOVEMENT_SCRIPT_REL] = _sha256_file(REPO_ROOT / MOVEMENT_SCRIPT_REL)
    return hashes


def _fixture_trajectories(seed, steps, size, viewport, kinds=('smooth_walk', 'teleport')):
    '''Small, deterministic center-point sequences, fully in bounds for a
    `viewport`x`viewport` ROI at `size`. Not meant to reproduce
    scripts/compare_town_movement.py's own trajectory generators -- only to
    give compare_town_cache.py something real and in-bounds to replay.'''
    out = {}
    for index, kind in enumerate(kinds):
        rng = np.random.default_rng([seed, index])
        lo, hi = viewport // 2, size - viewport // 2
        rows = rng.integers(lo, hi, size=steps).tolist()
        cols = rng.integers(lo, hi, size=steps).tolist()
        out[kind] = [[int(r), int(c)] for r, c in zip(rows, cols)]
    return out


def build_fixture_schedule(size=FIXTURE_SIZE, seed=FIXTURE_SEED, steps=FIXTURE_STEPS,
                            chunk_size=FIXTURE_CHUNK_SIZE, viewport=FIXTURE_VIEWPORT,
                            methods=('zstd',), modes=('warm_buffered',),
                            kinds=('smooth_walk', 'teleport')):
    '''A real, regenerated-from-current-source town-movement-style schedule
    dict, small enough to replay quickly but exercising the real map
    generator, real source hashing and real trajectory replay path.'''
    arr = np.ascontiguousarray(make_town_map(size=size, seed=seed))
    map_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    return {
        'schema_version': 1,
        'settings': {
            'size': size,
            'seed': seed,
            'steps': steps,
            'chunk_size': chunk_size,
            'viewport': viewport,
            'methods': list(methods),
            'modes': list(modes),
            'trajectory_kinds': list(kinds),
        },
        'map': {'source': 'horizon.town_map.make_town_map', 'sha256': map_sha},
        'source_files_sha256': _real_source_files_sha256(),
        'trajectories': _fixture_trajectories(seed, steps, size, viewport, kinds=kinds),
    }, arr


def write_fixture_schedule(path, **kwargs):
    schedule, arr = build_fixture_schedule(**kwargs)
    path.write_text(json.dumps(schedule), encoding='utf-8')
    return schedule, arr


class _RunTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.schedule_path = self.dir / 'schedule.json'
        self.output = self.dir / 'out.json'
        self.containers = self.dir / 'containers'

    def base_argv(self, schedule=None, repeats=1, cache_sizes='2', memory_repeats=0,
                  modes=None, output=None, containers=None):
        argv = [
            '--schedule', str(schedule or self.schedule_path),
            '--repeats', str(repeats),
            '--cache-sizes', cache_sizes,
            '--memory-repeats', str(memory_repeats),
            '--output', str(output or self.output),
            '--containers-dir', str(containers or self.containers),
        ]
        if modes is not None:
            argv += ['--modes', modes]
        return argv

    def run_main_quiet(self, argv):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return mod.main(argv)

    def read_report(self):
        return json.loads(self.output.read_text())


# --------------------------------------------------------------------------- #
# fixture schedule verification: regenerated map/source hashes
# --------------------------------------------------------------------------- #

class FixtureScheduleVerificationTests(_RunTestCase):
    def test_small_real_fixture_schedule_is_accepted_and_replayed(self):
        write_fixture_schedule(self.schedule_path)
        rc = self.run_main_quiet(self.base_argv())
        self.assertEqual(rc, 0, 'a freshly regenerated, correctly hashed fixture must be accepted')
        report = self.read_report()
        self.assertTrue(report['verification']['map_sha256_verified'])
        verified = report['verification']['source_files_verified']
        self.assertIn('horizon/town_cache.py', verified)
        self.assertIn(MOVEMENT_SCRIPT_REL, verified, 'the movement script\'s own hash must be '
                      'verified too -- compare_town_cache.py exempts no recorded file')
        case, = report['cases']
        for kind in ('smooth_walk', 'teleport'):
            for mode_name in report['settings']['modes']:
                self.assertIn(kind, case['modes'][mode_name])


# --------------------------------------------------------------------------- #
# default-enabled memory sub-measurement: subprocess integration
# --------------------------------------------------------------------------- #

class DefaultMemorySubprocessIntegrationTests(_RunTestCase):
    def test_memory_repeats_one_runs_measure_town_cache_memory_subprocess(self):
        '''--memory-repeats 1 with a few steps must shell out to
        scripts/measure_town_cache_memory.py and fold its report in under
        report["memory"]["report"].'''
        write_fixture_schedule(self.schedule_path, steps=3)
        rc = self.run_main_quiet(self.base_argv(memory_repeats=1))
        self.assertEqual(
            rc, 0,
            'memory sub-measurement must succeed end to end; if '
            'scripts/measure_town_cache_memory.py is missing or broken this is a real '
            'production failure, not a test bug'
        )
        report = self.read_report()
        self.assertIsNotNone(report['memory'])
        self.assertIn('output_path', report['memory'])
        self.assertIn('output_sha256', report['memory'])
        memory_output = Path(report['memory']['output_path'])
        self.assertTrue(memory_output.exists())
        self.assertEqual(report['memory']['output_sha256'], _sha256_file(memory_output))
        self.assertIn('report', report['memory'])


# --------------------------------------------------------------------------- #
# raw per-repeat startup_metrics / trajectory_ns, and independent aggregate
# recomputation
# --------------------------------------------------------------------------- #

class RawStartupAndTrajectoryTotalsShapeTests(_RunTestCase):
    def _run_report(self, repeats=2, steps=4, cache_sizes='2'):
        write_fixture_schedule(self.schedule_path, steps=steps)
        rc = self.run_main_quiet(self.base_argv(repeats=repeats, cache_sizes=cache_sizes))
        self.assertEqual(rc, 0)
        return self.read_report()

    def test_each_repeat_records_raw_startup_metrics_and_trajectory_ns_per_arm(self):
        '''repeats[].startup_metrics[arm] must be the session's own raw
        constructor metrics (not a derived/aggregated value), and
        repeats[].trajectory_ns[arm] must be exactly startup_total_ns +
        frames_total_ns, where frames_total_ns is the real sum of that
        repeat's own frame total_ns values for that arm -- recomputed here
        independently from the raw per-frame records rather than trusted.'''
        report = self._run_report()
        case, = report['cases']
        arms = mod._arm_names((2,))
        for by_kind in case['modes'].values():
            for run in by_kind.values():
                for repeat in run['repeats']:
                    for arm in arms:
                        startup = repeat['startup_metrics'][arm]
                        for key in mod._REQUIRED_METRIC_KEYS:
                            self.assertIn(key, startup, f'{arm} startup_metrics missing {key!r}')
                        self.assertEqual(
                            startup['total_ns'], startup['read_ns'] + startup['decode_ns'],
                            f'{arm} startup total_ns must equal read_ns + decode_ns'
                        )

                        frame_total_ns_values = [
                            frame['arms'][arm]['total_ns'] for frame in repeat['frames']
                        ]
                        traj = repeat['trajectory_ns'][arm]
                        self.assertEqual(traj['startup_total_ns'], startup['total_ns'])
                        self.assertEqual(traj['frames_total_ns'], sum(frame_total_ns_values))
                        self.assertEqual(
                            traj['total_ns_including_startup'],
                            traj['startup_total_ns'] + traj['frames_total_ns'],
                        )

    def test_stateless_arm_startup_metrics_are_genuinely_all_zero(self):
        '''stateless has no session, so its recorded startup_metrics for
        every repeat must be the exact all-zero dict documented in
        scripts/compare_town_cache.py (_startup_metrics), not merely
        "small" or "mostly zero".'''
        report = self._run_report(steps=3)
        case, = report['cases']
        for by_kind in case['modes'].values():
            for run in by_kind.values():
                for repeat in run['repeats']:
                    startup = repeat['startup_metrics']['stateless']
                    for key in mod._REQUIRED_METRIC_KEYS:
                        self.assertEqual(startup[key], 0, f'stateless startup_metrics[{key!r}] != 0')
                    for frame in repeat['frames']:
                        stateless = frame['arms']['stateless']
                        self.assertEqual(stateless['cache_hits'], 0)
                        self.assertEqual(stateless['cache_misses'], stateless['chunks_requested'])
                        self.assertEqual(stateless['evictions'], 0)
                        self.assertEqual(stateless['resident_bytes'], 0)

    def test_aggregate_ms_stat_blocks_match_independent_recomputation(self):
        '''Every millisecond aggregate stat block must be exactly
        mod._stats() applied to the same raw nanosecond samples this test
        pulls directly out of repeats[], recomputed here rather than trusted
        from the report -- a genuine independent check, not a shape probe.'''
        report = self._run_report()
        case, = report['cases']
        arms = mod._arm_names((2,))
        for by_kind in case['modes'].values():
            for run in by_kind.values():
                repeats = run['repeats']
                aggregate = run['aggregate']
                for arm in arms:
                    startup_ns = [r['startup_metrics'][arm]['total_ns'] for r in repeats]
                    first_viewport_ns = [
                        r['trajectory_ns'][arm]['startup_total_ns']
                        + r['frames'][0]['arms'][arm]['total_ns']
                        for r in repeats
                    ]
                    trajectory_total_ns = [r['trajectory_ns'][arm]['total_ns_including_startup']
                                            for r in repeats]
                    movement_only_ns = [r['trajectory_ns'][arm]['frames_total_ns'] for r in repeats]
                    all_frame_total_ns = [
                        f['arms'][arm]['total_ns'] for r in repeats for f in r['frames']
                    ]

                    block = aggregate[arm]
                    self.assertEqual(block['startup_total'], mod._stats(startup_ns))
                    self.assertEqual(block['first_viewport'], mod._stats(first_viewport_ns))
                    self.assertEqual(
                        block['trajectory_total_including_startup'],
                        mod._stats(trajectory_total_ns),
                    )
                    self.assertEqual(
                        block['trajectory_total_movement_only'], mod._stats(movement_only_ns)
                    )
                    self.assertEqual(block['frame_total'], mod._stats(all_frame_total_ns))

    def test_aggregate_startup_and_first_viewport_stats_present_per_arm(self):
        report = self._run_report()
        case, = report['cases']
        arms = mod._arm_names((2,))
        for by_kind in case['modes'].values():
            for run in by_kind.values():
                aggregate = run['aggregate']
                self.assertEqual(sorted(aggregate), sorted(arms))
                for arm in arms:
                    block = aggregate[arm]
                    for key in ('startup_total', 'frame_total', 'first_viewport',
                                'trajectory_total_including_startup',
                                'trajectory_total_movement_only'):
                        self.assertIn(key, block)
                    self.assertIsNotNone(block['frame_total'])
                    for stat_block in (block['startup_total'], block['first_viewport']):
                        self.assertIsNotNone(stat_block)
                        self.assertEqual(sorted(stat_block), ['max_ms', 'median_ms', 'p95_ms'])


# --------------------------------------------------------------------------- #
# strict schedule validation: missing / mismatched source and map hashes
# --------------------------------------------------------------------------- #

class ScheduleValidationTests(_RunTestCase):
    def test_map_sha256_mismatch_rejected(self):
        schedule, _arr = build_fixture_schedule()
        schedule['map']['sha256'] = 'f' * 64
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        rc = self.run_main_quiet(self.base_argv())
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.containers.exists())

    def test_mismatched_recorded_source_hash_rejected(self):
        '''A still-present horizon/*.py file whose recorded hash in the
        schedule disagrees with its current on-disk hash must fail loudly --
        exactly "existing module changed since the schedule was recorded".'''
        schedule, _arr = build_fixture_schedule()
        schedule['source_files_sha256']['horizon/town_cache.py'] = '0' * 64
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        rc = self.run_main_quiet(self.base_argv())
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())

    def test_recorded_source_file_no_longer_present_is_rejected(self):
        '''compare_town_cache.py's verification is strict: a schedule naming
        a source file that no longer exists on disk can never be confirmed
        to still match what it recorded, so "not present" must reject the
        run rather than being silently treated as out of scope.'''
        schedule, _arr = build_fixture_schedule()
        schedule['source_files_sha256']['horizon/this_file_does_not_exist.py'] = 'a' * 64
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        rc = self.run_main_quiet(self.base_argv())
        self.assertEqual(
            rc, 1,
            'a schedule naming a source file that no longer exists must be rejected, '
            'not silently treated as out of scope'
        )
        self.assertFalse(self.output.exists())
        self.assertFalse(self.containers.exists())

    def test_movement_script_hash_mismatch_rejected(self):
        '''scripts/compare_town_cache.py's module docstring is explicit that
        EVERY recorded file, "including scripts/compare_town_movement.py
        itself", must still be present and byte-identical. A schedule
        recording a wrong hash for that one file must be rejected exactly
        like any other mismatched source file -- it must not be carved out
        as a special exception the way the pre-audit draft treated it.'''
        schedule, _arr = build_fixture_schedule()
        schedule['source_files_sha256'][MOVEMENT_SCRIPT_REL] = '1' * 64
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        rc = self.run_main_quiet(self.base_argv())
        self.assertEqual(rc, 1, 'a mismatched scripts/compare_town_movement.py hash must reject')
        self.assertFalse(self.output.exists())

    def test_movement_script_hash_missing_is_rejected(self):
        '''Same as above, but for "recorded, and no longer present" rather
        than "recorded, and changed". Verified directly against
        _verify_recorded_sources under a substitute REPO_ROOT that has every
        other recorded file (symlinked to the real ones, so their hashes
        still match) except scripts/compare_town_movement.py, so this
        isolates exactly the file under test rather than tripping on some
        other missing file first.'''
        schedule, _arr = build_fixture_schedule()
        fake_root = self.dir / 'fake_repo'
        for relpath in schedule['source_files_sha256']:
            if relpath == MOVEMENT_SCRIPT_REL:
                continue
            target = fake_root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(REPO_ROOT / relpath, target)
        with mock.patch.object(mod, 'REPO_ROOT', fake_root):
            with self.assertRaises(mod.ValidationError):
                mod._verify_recorded_sources(schedule)

    def test_missing_schedule_file_rejected(self):
        rc = self.run_main_quiet(self.base_argv(schedule=self.dir / 'nope.json'))
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.containers.exists())


# --------------------------------------------------------------------------- #
# output / containers-dir pre-existence preservation
# --------------------------------------------------------------------------- #

class ExclusiveGuardTests(_RunTestCase):
    def test_existing_output_refused_and_preserved(self):
        write_fixture_schedule(self.schedule_path)
        self.output.write_bytes(b'keep-me')
        rc = self.run_main_quiet(self.base_argv())
        self.assertEqual(rc, 1)
        self.assertEqual(self.output.read_bytes(), b'keep-me')
        self.assertFalse(self.containers.exists())

    def test_existing_containers_dir_refused_and_preserved(self):
        write_fixture_schedule(self.schedule_path)
        self.containers.mkdir()
        (self.containers / 'someone-elses-file').write_bytes(b'not mine')
        rc = self.run_main_quiet(self.base_argv())
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.containers.exists())
        self.assertEqual((self.containers / 'someone-elses-file').read_bytes(), b'not mine')

    def test_second_run_refuses_to_overwrite_a_real_report(self):
        write_fixture_schedule(self.schedule_path)
        rc = self.run_main_quiet(self.base_argv())
        self.assertEqual(rc, 0)
        before = self.output.read_bytes()
        rc2 = self.run_main_quiet(self.base_argv(containers=self.dir / 'containers2'))
        self.assertEqual(rc2, 1)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertFalse((self.dir / 'containers2').exists())


# --------------------------------------------------------------------------- #
# invalid arguments / unsupported mode
# --------------------------------------------------------------------------- #

class ArgValidationTests(_RunTestCase):
    def test_rejects_non_integer_and_zero_repeats(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(repeats='0'))
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(repeats='nope'))

    def test_rejects_duplicate_cache_sizes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(cache_sizes='4,4'))

    def test_rejects_non_positive_cache_size(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(cache_sizes='0'))

    def test_rejects_unknown_mode(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(modes='bogus_mode'))

    def test_rejects_empty_modes_string(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(modes=''))

    def test_rejects_negative_memory_repeats(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(memory_repeats=-1))

    def test_requesting_macos_nocache_off_darwin_is_rejected_not_silently_dropped(self):
        '''An explicit --modes request that names an unsupported mode on
        this platform must be refused outright (ValidationError at runtime),
        never silently rewritten to a different mode set than what was
        asked for -- only the schedule's own *default* mode list may be
        filtered.'''
        write_fixture_schedule(self.schedule_path)
        if sys.platform != 'darwin':
            rc = self.run_main_quiet(self.base_argv(modes='macos_nocache'))
            self.assertEqual(rc, 1)
            self.assertFalse(self.output.exists())

    def test_schedule_default_modes_filtered_on_unsupported_platform(self):
        '''Unlike an explicit --modes request, the schedule's own recorded
        default mode list may be filtered down to what this platform
        supports (dropping macos_nocache off Darwin), as long as at least
        one supported mode remains.'''
        write_fixture_schedule(self.schedule_path, modes=('warm_buffered', 'macos_nocache'))
        rc = self.run_main_quiet(self.base_argv())
        if sys.platform != 'darwin':
            self.assertEqual(rc, 0)
            report = self.read_report()
            self.assertEqual(report['settings']['modes'], ['warm_buffered'])


# --------------------------------------------------------------------------- #
# memory sub-measurement run standalone, and fresh-process isolation
# --------------------------------------------------------------------------- #

class MemoryStandaloneTests(_RunTestCase):
    '''Exercises scripts/measure_town_cache_memory.py directly, bypassing
    compare_town_cache.py's subprocess wrapper. Written against the contract
    documented in that script's own module docstring: one fresh Python
    subprocess per (method, trajectory_kind, arm, repeat) sample, so a
    parent's resident state is never inherited, and three distinct memory
    axes (resident_decoded_bytes_*, tracemalloc_*_bytes,
    process_rss_*_hwm_bytes) are never conflated.'''

    def _prepared_containers(self, **schedule_kwargs):
        schedule, arr = build_fixture_schedule(methods=('zstd',), **schedule_kwargs)
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        from horizon import town_disk
        case = town_disk.prepare_case(
            arr, self.containers / 'zstd', 'zstd', chunk_size=FIXTURE_CHUNK_SIZE
        )
        return schedule, case

    def _run_memory_cli(self, output=None, cache_sizes='2', repeats='1', schedule_path=None):
        cmd = [
            sys.executable, str(MEMORY_SCRIPT),
            '--schedule', str(schedule_path or self.schedule_path),
            '--containers-dir', str(self.containers),
            '--cache-sizes', cache_sizes,
            '--repeats', repeats,
            '--output', str(output or (self.dir / 'memory.json')),
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_memory_script_exists(self):
        self.assertTrue(
            MEMORY_SCRIPT.exists(),
            f'{MEMORY_SCRIPT} is required by scripts/compare_town_cache.py\'s '
            '--memory-repeats path and by docs/town-cache-method.md; its absence is a '
            'real production gap to report, not something for this test suite to skip',
        )

    def test_standalone_invocation_reports_fresh_process_isolation_metadata(self):
        '''Each (method, trajectory_kind, arm, repeat) sample is documented
        to run in its own fresh Python subprocess, so no two samples should
        report the same OS pid.'''
        self._prepared_containers()
        memory_output = self.dir / 'memory.json'
        result = self._run_memory_cli(output=memory_output, repeats='2')
        self.assertEqual(
            result.returncode, 0,
            f'measure_town_cache_memory.py failed (this is a production failure to '
            f'report, not a test bug): stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        report = json.loads(memory_output.read_text())
        self.assertTrue(report['cases'])
        pids = [sample['pid'] for case in report['cases'] for sample in case['samples']]
        self.assertTrue(pids, 'memory report must record per-sample process identity')
        self.assertEqual(len(pids), len(set(pids)), 'every sample must run in a fresh process')

    def test_standalone_report_distinguishes_the_three_memory_axes(self):
        '''resident_decoded_bytes_* (exact retained-array ledger),
        tracemalloc_*_bytes (traced Python heap activity) and
        process_rss_*_hwm_bytes (OS RSS high-water mark) must all be present,
        as distinct keys, on every sample -- never collapsed into one
        number.'''
        self._prepared_containers()
        memory_output = self.dir / 'memory.json'
        result = self._run_memory_cli(output=memory_output, repeats='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(memory_output.read_text())
        required = (
            'resident_decoded_bytes_peak', 'resident_decoded_bytes_end',
            'tracemalloc_startup_peak_bytes', 'tracemalloc_movement_peak_bytes',
            'process_rss_overall_hwm_bytes',
        )
        for case in report['cases']:
            for sample in case['samples']:
                for key in required:
                    self.assertIn(key, sample, f'sample missing {key!r}: {sample!r}')
                if case['arm'] == 'stateless':
                    self.assertEqual(sample['resident_decoded_bytes_peak'], 0)
                    self.assertEqual(sample['resident_decoded_bytes_end'], 0)

    def test_preexisting_output_refused_and_preserved(self):
        self._prepared_containers()
        memory_output = self.dir / 'memory.json'
        memory_output.write_bytes(b'keep-me')
        result = self._run_memory_cli(output=memory_output, repeats='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(memory_output.read_bytes(), b'keep-me')

    def test_mismatched_recorded_source_hash_rejected_standalone(self):
        '''Strict verification applies here too, for any recorded file that
        is not one of this script's own three documented exemptions
        (scripts/compare_town_movement.py, scripts/compare_town_cache.py,
        scripts/measure_town_cache_memory.py): a changed horizon/*.py hash
        must still reject the run.'''
        schedule, _case = self._prepared_containers()
        schedule['source_files_sha256']['horizon/town_cache.py'] = '0' * 64
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        result = self._run_memory_cli(repeats='1')
        self.assertNotEqual(result.returncode, 0)

    def test_recorded_source_file_no_longer_present_is_rejected_standalone(self):
        schedule, _case = self._prepared_containers()
        schedule['source_files_sha256']['horizon/this_file_does_not_exist.py'] = 'a' * 64
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        result = self._run_memory_cli(repeats='1')
        self.assertNotEqual(
            result.returncode, 0,
            'a schedule naming a source file that no longer exists must be rejected here '
            'too, not silently treated as out of scope'
        )

    def test_movement_script_hash_mismatch_rejected_standalone(self):
        '''measure_town_cache_memory.py's own _verify_schedule docstring is
        explicit that nothing recorded in source_files_sha256 is "out of
        scope" -- not even the driver scripts that produced/consumed the
        schedule. A wrong hash for scripts/compare_town_movement.py must
        reject the run exactly like any other mismatched source file.'''
        schedule, _case = self._prepared_containers()
        schedule['source_files_sha256'][MOVEMENT_SCRIPT_REL] = '1' * 64
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        result = self._run_memory_cli(repeats='1')
        self.assertNotEqual(result.returncode, 0, 'a mismatched movement-script hash must reject')

    def test_movement_script_hash_missing_is_rejected_standalone(self):
        schedule, _case = self._prepared_containers()
        schedule['source_files_sha256'][MOVEMENT_SCRIPT_REL + '.does-not-exist'] = 'a' * 64
        self.schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
        result = self._run_memory_cli(repeats='1')
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()

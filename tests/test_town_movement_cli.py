'''Focused tests for scripts/compare_town_movement.py (the CLI, not
horizon.town_disk / horizon.town_region themselves).

Covers:
  * trajectory semantics -- in particular whether the boundary_backtrack
    kind actually changes the *set of chunks the viewport intersects* from
    step to step, which is the only thing that makes a "crosses chunk
    boundaries repeatedly" trajectory meaningful for the region arm. A
    character center crossing a grid line while the 256x256 viewport keeps
    straddling the same line is not a chunk-set change and does not exercise
    anything different in the region loader.
  * the shared, deterministic trajectory schedules: same seed -> same
    positions, one independent stream per kind, every ROI exactly 256x256
    and fully in bounds, and the same schedule replayed unchanged across
    every method, mode and repeat.
  * paired-order parity: the measured pair alternates every frame within a
    run, the starting parity flips once per (trajectory_kind, repeat) run,
    and the recorded `swapped` flag matches the real call order.
  * priming: warm_buffered primes both container files exactly once per
    (trajectory_kind, repeat) run and never mid-run; macos_nocache primes
    nothing at all.
  * metric aggregation: every aggregate block recomputes exactly from the
    per-frame metrics the report records.
  * the exclusive-output / exclusive-containers-dir guards, that a
    pre-existing report or containers dir is preserved on failure, and that
    cleanup-on-failure removes only containers this run itself created.
  * one true end-to-end smoke run against the real horizon.town_disk and
    horizon.town_region at --size 513 --steps 30 --repeats 1.
'''
import collections
import contextlib
import importlib.util
import io
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / 'scripts' / 'compare_town_movement.py'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from horizon import town_disk as real_town_disk  # noqa: E402
from horizon import town_region as real_town_region  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location('compare_town_movement_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()

SMOKE_SIZE = 513
SMOKE_STEPS = mod.MIN_STEPS  # 30


def chunk_set(region, chunk_size=mod.CHUNK_SIZE):
    '''Grid cells a (row, col, h, w) ROI intersects, by grid arithmetic.

    Anchored against the real loader's own chunk_coordinates by
    ChunkSetHelperTests below, so the trajectory tests can reason about chunk
    sets without doing I/O per step.
    '''
    row, col, h, w = region
    return frozenset(
        (grid_row, grid_col)
        for grid_row in range(row // chunk_size, (row + h - 1) // chunk_size + 1)
        for grid_col in range(col // chunk_size, (col + w - 1) // chunk_size + 1)
    )


def regions_for(kind, seed, steps, size):
    positions = mod._build_trajectories(seed, steps, size)[kind]
    return [mod._viewport_region(r, c, size) for r, c in positions]


# --------------------------------------------------------------------------- #
# chunk-set helper, anchored on the real loader
# --------------------------------------------------------------------------- #

class ChunkSetHelperTests(unittest.TestCase):
    '''chunk_set() must agree with what horizon.town_region.load_region
    actually reads, otherwise every chunk-set assertion below is vacuous.'''

    def test_helper_matches_real_load_region_chunk_coordinates(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        arr = mod.make_town_map(size=SMOKE_SIZE, seed=mod.DEFAULT_SEED)
        case = real_town_disk.prepare_case(
            np.ascontiguousarray(arr), Path(tmp.name) / 'case', 'zstd',
            chunk_size=mod.CHUNK_SIZE,
        )
        rois = [
            (0, 0, mod.VIEWPORT, mod.VIEWPORT),
            (1, 1, mod.VIEWPORT, mod.VIEWPORT),
            (128, 128, mod.VIEWPORT, mod.VIEWPORT),
            (257, 257, mod.VIEWPORT, mod.VIEWPORT),
            (0, 257, mod.VIEWPORT, mod.VIEWPORT),
        ]
        for roi in rois:
            with self.subTest(roi=roi):
                _out, metrics = real_town_region.load_region(case['chunked_path'], roi)
                actual = frozenset(tuple(c) for c in metrics['chunk_coordinates'])
                self.assertEqual(actual, chunk_set(roi))
                self.assertEqual(metrics['chunks_loaded'], len(actual))


# --------------------------------------------------------------------------- #
# trajectory semantics
# --------------------------------------------------------------------------- #

class TrajectorySemanticsTests(unittest.TestCase):

    def test_boundary_backtrack_origin_lands_on_both_sides_of_a_grid_line(self):
        '''Precondition for the regression test below.

        What has to alternate is the viewport *origin*, not the center: the
        ROI is [origin, origin + 256), so the chunk set only changes when the
        origin itself moves across a chunk_size grid line. (A center that
        crosses a grid line while the origin stays put leaves the viewport
        straddling the same two chunk columns/rows every frame.)
        '''
        for size in (SMOKE_SIZE, 1024):
            for seed in (mod.DEFAULT_SEED, 1, 7):
                with self.subTest(size=size, seed=seed):
                    regions = regions_for('boundary_backtrack', seed, 40, size)
                    rows = [r for r, _c, _h, _w in regions]
                    cols = [c for _r, c, _h, _w in regions]
                    moving = rows if len(set(rows)) > 1 else cols
                    sides = {v // mod.CHUNK_SIZE for v in moving}
                    self.assertGreater(
                        len(sides), 1,
                        'viewport origin never lands on both sides of a chunk grid line',
                    )

    def test_boundary_backtrack_changes_viewport_chunk_set_between_neighboring_steps(self):
        '''Regression test for the trajectory's stated purpose.

        boundary_backtrack is documented as "repeatedly crossing and
        re-crossing the same boundary", and the region arm's whole point is
        that a boundary-straddling viewport spans more than one chunk. For
        that to be measured, consecutive steps must actually *change* the set
        of chunks the 256x256 viewport intersects -- a center that hops over
        a grid line while the viewport keeps covering the exact same chunks
        exercises identical reads every frame.
        '''
        for size in (SMOKE_SIZE, 1024, 2048):
            for seed in (mod.DEFAULT_SEED, 1, 7):
                with self.subTest(size=size, seed=seed):
                    regions = regions_for('boundary_backtrack', seed, 40, size)
                    sets = [chunk_set(r) for r in regions]
                    changes = sum(1 for a, b in zip(sets, sets[1:]) if a != b)
                    self.assertGreater(
                        changes, 0,
                        f'the viewport chunk set never changes between neighboring steps '
                        f'(all {len(sets)} steps intersect {sorted(sets[0])}); the center '
                        f'crosses the grid line but the 256x256 viewport straddles it the '
                        f'whole time, so every frame reads the same chunks',
                    )
                    self.assertGreater(
                        changes, len(sets) // 4,
                        'chunk-set changes should happen repeatedly, not once',
                    )

    def test_smooth_walk_and_teleport_do_change_the_chunk_set(self):
        for kind in ('smooth_walk', 'teleport'):
            with self.subTest(kind=kind):
                sets = [chunk_set(r) for r in regions_for(kind, mod.DEFAULT_SEED, 40, 1024)]
                changes = sum(1 for a, b in zip(sets, sets[1:]) if a != b)
                self.assertGreater(changes, 0)


# --------------------------------------------------------------------------- #
# deterministic, shared schedules
# --------------------------------------------------------------------------- #

class SharedScheduleTests(unittest.TestCase):

    def test_same_seed_reproduces_every_trajectory_exactly(self):
        a = mod._build_trajectories(mod.DEFAULT_SEED, 35, SMOKE_SIZE)
        b = mod._build_trajectories(mod.DEFAULT_SEED, 35, SMOKE_SIZE)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a), sorted(mod.TRAJECTORY_KINDS))
        for kind in mod.TRAJECTORY_KINDS:
            self.assertEqual(len(a[kind]), 35)

    def test_different_seed_changes_trajectories(self):
        a = mod._build_trajectories(mod.DEFAULT_SEED, 35, SMOKE_SIZE)
        b = mod._build_trajectories(mod.DEFAULT_SEED + 1, 35, SMOKE_SIZE)
        self.assertNotEqual(a, b)

    def test_each_kind_has_an_independent_stream(self):
        '''Dropping a kind must not shift another kind's draws.'''
        full = mod._build_trajectories(mod.DEFAULT_SEED, 35, SMOKE_SIZE)
        for index, kind in enumerate(mod.TRAJECTORY_KINDS):
            with self.subTest(kind=kind):
                rng = np.random.default_rng([mod.DEFAULT_SEED, index])
                alone = mod._TRAJECTORY_BUILDERS[kind](rng, 35, SMOKE_SIZE)
                self.assertEqual(alone, full[kind])

    def test_every_roi_is_exactly_the_viewport_and_fully_inbounds(self):
        for size in (mod.MIN_SIZE, 600, 1024):
            trajectories = mod._build_trajectories(mod.DEFAULT_SEED, 40, size)
            for kind, positions in trajectories.items():
                for center_row, center_col in positions:
                    with self.subTest(size=size, kind=kind, center=(center_row, center_col)):
                        self.assertTrue(0 <= center_row < size)
                        self.assertTrue(0 <= center_col < size)
                        row, col, h, w = mod._viewport_region(center_row, center_col, size)
                        self.assertEqual((h, w), (mod.VIEWPORT, mod.VIEWPORT))
                        self.assertGreaterEqual(row, 0)
                        self.assertGreaterEqual(col, 0)
                        self.assertLessEqual(row + h, size)
                        self.assertLessEqual(col + w, size)


# --------------------------------------------------------------------------- #
# run harness
# --------------------------------------------------------------------------- #

class _Loaders:
    '''Delegates to the real modules, records the per-frame arm call order,
    and can inject a failure at a chosen prepare_case call or after N loads.'''

    def __init__(self, prepare_case_override=None, load_fail_after=None):
        self._prepare_case_override = prepare_case_override
        self._load_fail_after = load_fail_after
        self.prepare_calls = 0
        self.load_calls = 0
        self.call_order = []  # 'whole' / 'region', in real invocation order
        self.MODES = real_town_disk.MODES

    # -- town_disk surface -------------------------------------------------- #
    def prepare_case(self, arr, directory, method, chunk_size=256):
        self.prepare_calls += 1
        if self._prepare_case_override is not None:
            return self._prepare_case_override(
                self.prepare_calls, real_town_disk, arr, directory, method, chunk_size
            )
        return real_town_disk.prepare_case(arr, directory, method, chunk_size=chunk_size)

    def _count_load(self, label):
        self.load_calls += 1
        self.call_order.append(label)
        if self._load_fail_after is not None and self.load_calls > self._load_fail_after:
            raise RuntimeError('simulated mid-benchmark load failure')

    def load_whole(self, path, mode='warm_buffered', region=None):
        self._count_load('whole')
        return real_town_disk.load_whole(path, mode=mode, region=region)

    # -- town_region surface ------------------------------------------------ #
    def load_region(self, path, region, mode='warm_buffered'):
        self._count_load('region')
        return real_town_region.load_region(path, region, mode=mode)


class _RunTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.output = self.dir / 'out.json'
        self.containers = self.dir / 'containers'

    def base_argv(self, size=SMOKE_SIZE, steps=SMOKE_STEPS, repeats=1,
                  modes='warm_buffered', output=None, containers=None):
        return [
            '--output', str(output or self.output),
            '--containers-dir', str(containers or self.containers),
            '--size', str(size), '--steps', str(steps), '--repeats', str(repeats),
            '--modes', modes,
        ]

    def run_main_quiet(self, argv, loaders=None, methods=None, patches=()):
        loaders = loaders if loaders is not None else _Loaders()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(mod, '_import_town_disk', return_value=loaders))
        stack.enter_context(mock.patch.object(mod, '_import_town_region', return_value=loaders))
        if methods is not None:
            stack.enter_context(mock.patch.object(mod, 'METHODS', tuple(methods)))
        for patch in patches:
            stack.enter_context(patch)
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return mod.main(argv), loaders

    def read_report(self):
        return json.loads(self.output.read_text())


# --------------------------------------------------------------------------- #
# order parity
# --------------------------------------------------------------------------- #

class OrderParityTests(_RunTestCase):
    '''Documented policy: swapped = (start_parity + frame_index) % 2, with
    start_parity flipped once per (trajectory_kind, repeat) run and reset per
    (method, mode) cell.'''

    def test_parity_alternates_per_frame_and_flips_per_run(self):
        repeats = 2
        rc, loaders = self.run_main_quiet(
            self.base_argv(repeats=repeats), methods=('zstd',)
        )
        self.assertEqual(rc, 0)
        report = self.read_report()
        case, = report['cases']
        flat_expected = []
        for kind_index, kind in enumerate(mod.TRAJECTORY_KINDS):
            result = case['modes']['warm_buffered'][kind]
            self.assertEqual(len(result['repeats']), repeats)
            for repeat_index, repeat in enumerate(result['repeats']):
                start_parity = (kind_index * repeats + repeat_index) % 2
                self.assertEqual(repeat['repeat'], repeat_index)
                for frame_index, frame in enumerate(repeat['frames']):
                    with self.subTest(kind=kind, repeat=repeat_index, frame=frame_index):
                        self.assertEqual(frame['frame'], frame_index)
                        self.assertEqual(
                            frame['swapped'], bool((start_parity + frame_index) % 2)
                        )
                    flat_expected.append(frame['swapped'])
                parities = [f['swapped'] for f in repeat['frames']]
                self.assertEqual(len(set(parities)), 2, 'both orders must occur in a run')

        # The starting parity must differ between consecutive runs.
        starts = []
        for kind in mod.TRAJECTORY_KINDS:
            for repeat in case['modes']['warm_buffered'][kind]['repeats']:
                starts.append(repeat['frames'][0]['swapped'])
        for a, b in zip(starts, starts[1:]):
            self.assertNotEqual(a, b, 'start parity must flip once per (kind, repeat) run')

        # The recorded flag must match what was really called first.
        self.assertEqual(len(loaders.call_order), 2 * len(flat_expected))
        for index, swapped in enumerate(flat_expected):
            first = loaders.call_order[2 * index]
            second = loaders.call_order[2 * index + 1]
            with self.subTest(frame=index):
                self.assertEqual((first, second),
                                 ('region', 'whole') if swapped else ('whole', 'region'))

    def test_parity_resets_per_mode_cell(self):
        '''order_parity is reset to 0 for each (method, mode) cell, so the
        first frame of the first run of every cell has the same parity.'''
        rc, _loaders = self.run_main_quiet(
            self.base_argv(repeats=1), methods=('zstd', 'lz4')
        )
        self.assertEqual(rc, 0)
        report = self.read_report()
        firsts = {
            case['method']: case['modes']['warm_buffered'][
                mod.TRAJECTORY_KINDS[0]]['repeats'][0]['frames'][0]['swapped']
            for case in report['cases']
        }
        self.assertEqual(set(firsts.values()), {False})


# --------------------------------------------------------------------------- #
# priming
# --------------------------------------------------------------------------- #

class PrimingTests(_RunTestCase):

    def _primed_paths(self, argv, methods, loaders=None):
        calls = []
        patch = mock.patch.object(mod, '_read_buffered_discard', side_effect=calls.append)
        rc, loaders = self.run_main_quiet(argv, loaders=loaders, methods=methods,
                                          patches=(patch,))
        return rc, calls

    def test_warm_buffered_primes_both_files_once_per_trajectory_repeat(self):
        repeats = 2
        rc, calls = self._primed_paths(
            self.base_argv(repeats=repeats), methods=('zstd',)
        )
        self.assertEqual(rc, 0)
        runs = len(mod.TRAJECTORY_KINDS) * repeats
        # Two files (whole + chunked) primed per run, and nothing else.
        self.assertEqual(len(calls), 2 * runs)
        counted = collections.Counter(Path(p).name for p in calls)
        self.assertEqual(counted, collections.Counter({
            real_town_disk.WHOLE_NAME: runs,
            real_town_disk.CHUNKED_NAME: runs,
        }))
        for a, b in zip(calls[0::2], calls[1::2]):
            self.assertEqual(Path(a).name, real_town_disk.WHOLE_NAME)
            self.assertEqual(Path(b).name, real_town_disk.CHUNKED_NAME)

    def test_warm_buffered_does_not_prime_mid_run(self):
        '''All priming for a run must happen before that run's first timed
        frame: priming calls must be strictly interleaved as
        [2 primes][N frames][2 primes][N frames]...'''
        events = []
        prime_patch = mock.patch.object(
            mod, '_read_buffered_discard', side_effect=lambda p: events.append('prime')
        )

        class _Recording(_Loaders):
            def _count_load(self, label):
                events.append('load')
                return super()._count_load(label)

        rc, _loaders = self.run_main_quiet(
            self.base_argv(repeats=2), loaders=_Recording(), methods=('zstd',),
            patches=(prime_patch,),
        )
        self.assertEqual(rc, 0)
        runs = len(mod.TRAJECTORY_KINDS) * 2
        blocks = ''.join('P' if e == 'prime' else 'L' for e in events)
        expected = ('PP' + 'L' * (2 * SMOKE_STEPS)) * runs
        self.assertEqual(blocks, expected)

    @unittest.skipUnless(sys.platform == 'darwin', 'macos_nocache requires Darwin')
    def test_macos_nocache_never_primes(self):
        rc, calls = self._primed_paths(
            self.base_argv(modes='macos_nocache'), methods=('zstd',)
        )
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])

    def test_prime_helper_is_a_noop_for_non_warm_modes(self):
        with mock.patch.object(mod, '_read_buffered_discard') as reader:
            mod._prime('macos_nocache', ('a', 'b'))
            reader.assert_not_called()
            mod._prime('warm_buffered', ('a', 'b'))
            self.assertEqual(reader.call_count, 2)


# --------------------------------------------------------------------------- #
# metric aggregation
# --------------------------------------------------------------------------- #

class MetricAggregationTests(_RunTestCase):

    def test_aggregates_recompute_from_recorded_frames(self):
        repeats = 2
        rc, _loaders = self.run_main_quiet(
            self.base_argv(repeats=repeats), methods=('zstd',)
        )
        self.assertEqual(rc, 0)
        report = self.read_report()
        case, = report['cases']
        for kind in mod.TRAJECTORY_KINDS:
            result = case['modes']['warm_buffered'][kind]
            whole_frames = []
            region_frames = []
            whole_totals_ns = []
            region_totals_ns = []
            whole_totals_bytes = []
            region_totals_bytes = []
            for repeat in result['repeats']:
                frames = repeat['frames']
                whole_frames.extend(f['whole'] for f in frames)
                region_frames.extend(f['region'] for f in frames)
                whole_totals_ns.append(sum(f['whole']['total_ns'] for f in frames))
                region_totals_ns.append(sum(f['region']['total_ns'] for f in frames))
                whole_totals_bytes.append(sum(f['whole']['bytes_read'] for f in frames))
                region_totals_bytes.append(sum(f['region']['bytes_read'] for f in frames))

            aggregate = result['aggregate']
            with self.subTest(kind=kind):
                self.assertEqual(
                    aggregate['whole'], mod._metric_block(whole_frames, with_chunks=False)
                )
                self.assertEqual(
                    aggregate['region'], mod._metric_block(region_frames, with_chunks=True)
                )
                totals = aggregate['trajectory_totals']
                # trajectory_totals time stats are reported in ms (_stats
                # converts ns samples to ms), hence the 'time_ms' key.
                self.assertEqual(totals['whole']['time_ms'], mod._totals_stats(whole_totals_ns))
                self.assertEqual(totals['region']['time_ms'], mod._totals_stats(region_totals_ns))
                self.assertEqual(totals['whole']['bytes'], mod._totals_bytes(whole_totals_bytes))
                self.assertEqual(totals['region']['bytes'], mod._totals_bytes(region_totals_bytes))

                # Independent spot-check of the stats themselves.
                medians = statistics.median([n / 1e6 for n in whole_totals_ns])
                self.assertAlmostEqual(totals['whole']['time_ms']['median_ms'], medians, places=5)

                # Per-frame metric identity and chunk bookkeeping.
                for frame_metrics in region_frames:
                    self.assertEqual(
                        frame_metrics['total_ns'],
                        frame_metrics['read_ns'] + frame_metrics['decode_ns'],
                    )
                    self.assertEqual(
                        frame_metrics['chunks_loaded'],
                        len(frame_metrics['chunk_coordinates']),
                    )
                distribution = aggregate['region']['chunk_count_distribution']
                self.assertEqual(
                    sum(distribution.values()), len(region_frames)
                )
                self.assertEqual(
                    {int(k) for k in distribution},
                    set(aggregate['region']['chunks_loaded']),
                )

    def test_stats_block_shape(self):
        samples = [1_000_000, 2_000_000, 3_000_000, 4_000_000]
        stats = mod._stats(samples)
        self.assertEqual(sorted(stats), ['max_ms', 'median_ms', 'p95_ms'])
        self.assertAlmostEqual(stats['median_ms'], 2.5)
        self.assertAlmostEqual(stats['max_ms'], 4.0)
        self.assertAlmostEqual(
            stats['p95_ms'], float(np.percentile([1.0, 2.0, 3.0, 4.0], 95, method='linear'))
        )


# --------------------------------------------------------------------------- #
# exclusivity guards and cleanup
# --------------------------------------------------------------------------- #

class ExclusiveGuardTests(_RunTestCase):

    def test_existing_output_refused_and_preserved(self):
        self.output.write_bytes(b'keep-me')
        rc, loaders = self.run_main_quiet(self.base_argv(), methods=('zstd',))
        self.assertEqual(rc, 1)
        self.assertEqual(self.output.read_bytes(), b'keep-me')
        self.assertFalse(self.containers.exists())
        self.assertEqual(loaders.prepare_calls, 0, 'guard must run before any preparation')

    def test_existing_containers_dir_refused_and_preserved(self):
        self.containers.mkdir()
        (self.containers / 'someone-elses-file').write_bytes(b'not mine')
        rc, loaders = self.run_main_quiet(self.base_argv(), methods=('zstd',))
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.containers.exists(), 'pre-existing dir must survive')
        self.assertEqual(
            (self.containers / 'someone-elses-file').read_bytes(), b'not mine'
        )
        self.assertEqual(loaders.prepare_calls, 0)

    def test_preexisting_empty_containers_dir_survives_a_failure(self):
        '''Cleanup may only rmdir --containers-dir when this run created it.'''
        self.output.write_bytes(b'keep-me')
        self.containers.mkdir()
        rc, _loaders = self.run_main_quiet(self.base_argv(), methods=('zstd',))
        self.assertEqual(rc, 1)
        self.assertTrue(self.containers.exists())
        self.assertEqual(list(self.containers.iterdir()), [])

    def test_second_run_refuses_to_overwrite_a_real_report(self):
        rc, _ = self.run_main_quiet(self.base_argv(), methods=('zstd',))
        self.assertEqual(rc, 0)
        before = self.output.read_bytes()
        rc2, _ = self.run_main_quiet(
            self.base_argv(containers=self.dir / 'containers2'), methods=('zstd',)
        )
        self.assertEqual(rc2, 1)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertFalse((self.dir / 'containers2').exists())


class CleanupOnFailureTests(_RunTestCase):

    def test_preparation_failure_removes_already_prepared_containers(self):
        def override(call_number, real, arr, directory, method, chunk_size):
            if call_number == 2:
                raise RuntimeError('simulated prepare_case failure')
            return real.prepare_case(arr, directory, method, chunk_size=chunk_size)

        rc, loaders = self.run_main_quiet(
            self.base_argv(), loaders=_Loaders(prepare_case_override=override),
            methods=('zstd', 'lz4'),
        )
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(loaders.prepare_calls, 2)
        self.assertFalse(
            self.containers.exists(),
            "the first method's containers and the run-created root must be removed",
        )

    def test_mid_benchmark_failure_removes_all_own_containers(self):
        rc, loaders = self.run_main_quiet(
            self.base_argv(), loaders=_Loaders(load_fail_after=5), methods=('zstd', 'lz4'),
        )
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(loaders.load_calls, 6)
        self.assertFalse(self.containers.exists())

    def test_unrecorded_partial_container_is_left_alone(self):
        '''If prepare_case fails after writing only one of its files, this
        script was never told those paths; it must clean up every method it
        does have paths for, leave the orphan alone, and not crash.'''
        def override(call_number, real, arr, directory, method, chunk_size):
            if call_number == 2:
                directory = Path(directory)
                directory.mkdir(parents=True)
                (directory / real_town_disk.WHOLE_NAME).write_bytes(b'partial-orphan')
                raise RuntimeError('simulated failure after the first file')
            return real.prepare_case(arr, directory, method, chunk_size=chunk_size)

        rc, _loaders = self.run_main_quiet(
            self.base_argv(), loaders=_Loaders(prepare_case_override=override),
            methods=('zstd', 'lz4'),
        )
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        remaining = sorted(p.name for p in self.containers.iterdir()) \
            if self.containers.exists() else []
        self.assertEqual(remaining, ['lz4'])
        self.assertEqual(
            (self.containers / 'lz4' / real_town_disk.WHOLE_NAME).read_bytes(),
            b'partial-orphan',
        )

    def test_missing_town_region_fails_without_writing_anything(self):
        with mock.patch.object(
            mod, '_import_town_region',
            side_effect=mod.ValidationError('horizon.town_region is not available yet'),
        ):
            with mock.patch.object(mod, '_import_town_disk', return_value=_Loaders()):
                with contextlib.redirect_stderr(io.StringIO()):
                    rc = mod.main(self.base_argv())
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.containers.exists())


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #

class ArgValidationTests(_RunTestCase):

    def test_rejects_sizes_below_the_boundary_straddling_minimum(self):
        for bad in ('512', '256', '0'):
            with self.subTest(size=bad):
                with self.assertRaises(SystemExit):
                    with contextlib.redirect_stderr(io.StringIO()):
                        mod.parse_args(self.base_argv(size=bad))

    def test_rejects_too_few_steps_and_repeats(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(steps=mod.MIN_STEPS - 1))
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(repeats=0))

    def test_rejects_unknown_and_duplicate_modes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(modes='nope'))
            with self.assertRaises(SystemExit):
                mod.parse_args(self.base_argv(modes='warm_buffered,warm_buffered'))


# --------------------------------------------------------------------------- #
# end-to-end smoke
# --------------------------------------------------------------------------- #

class TrueSmokeTest(_RunTestCase):

    def test_real_modules_size_513_steps_30_repeats_1_warm(self):
        argv = self.base_argv(size=SMOKE_SIZE, steps=SMOKE_STEPS, repeats=1,
                              modes='warm_buffered')
        with contextlib.redirect_stderr(io.StringIO()):
            rc = mod.main(argv)
        self.assertEqual(rc, 0)
        self.assertTrue(self.output.exists())
        report = self.read_report()

        self.assertEqual(report['schema_version'], 1)
        self.assertEqual(report['settings']['size'], SMOKE_SIZE)
        self.assertEqual(report['settings']['steps'], SMOKE_STEPS)
        self.assertEqual(report['settings']['repeats'], 1)
        self.assertEqual(report['settings']['chunk_size'], mod.CHUNK_SIZE)
        self.assertEqual(report['settings']['viewport'], mod.VIEWPORT)
        self.assertEqual(report['map']['shape'], [SMOKE_SIZE, SMOKE_SIZE])
        self.assertEqual(len(report['cases']), len(mod.METHODS))

        shared = report['trajectories']
        self.assertEqual(sorted(shared), sorted(mod.TRAJECTORY_KINDS))
        for kind, positions in shared.items():
            self.assertEqual(len(positions), SMOKE_STEPS)

        for case in report['cases']:
            self.assertEqual(case['hashes_before'], case['hashes_after'])
            modes = case['modes']
            self.assertEqual(sorted(modes), ['warm_buffered'])
            for kind in mod.TRAJECTORY_KINDS:
                repeat, = modes['warm_buffered'][kind]['repeats']
                frames = repeat['frames']
                self.assertEqual(len(frames), SMOKE_STEPS)
                # The same schedule, replayed unchanged everywhere.
                self.assertEqual(
                    [[f['center_row'], f['center_col']] for f in frames], shared[kind]
                )
                for frame in frames:
                    row, col, h, w = frame['roi']
                    self.assertEqual((h, w), (mod.VIEWPORT, mod.VIEWPORT))
                    self.assertGreaterEqual(row, 0)
                    self.assertGreaterEqual(col, 0)
                    self.assertLessEqual(row + h, SMOKE_SIZE)
                    self.assertLessEqual(col + w, SMOKE_SIZE)
                    self.assertEqual(
                        tuple(frame['roi']),
                        mod._viewport_region(frame['center_row'], frame['center_col'],
                                             SMOKE_SIZE),
                    )
                    region_metrics = frame['region']
                    self.assertEqual(
                        frozenset(tuple(c) for c in region_metrics['chunk_coordinates']),
                        chunk_set(tuple(frame['roi'])),
                    )
                    # A 256x256 viewport can only ever touch a 2x2 block of
                    # chunks, and the region arm never reads more than the
                    # chunked container holds. (At --size 513 the whole
                    # container is itself only ~4 chunks' worth, so "region
                    # reads fewer bytes than whole" is not expected here.)
                    self.assertGreaterEqual(region_metrics['chunks_loaded'], 1)
                    self.assertLessEqual(region_metrics['chunks_loaded'], 4)
                    self.assertGreater(region_metrics['bytes_read'], 0)
                    self.assertLessEqual(
                        region_metrics['bytes_read'], case['chunked_bytes']
                    )


if __name__ == '__main__':
    unittest.main()

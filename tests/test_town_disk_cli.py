'''Focused tests for scripts/compare_town_disk.py (the CLI, not horizon.town_disk itself).

Covers: the uniform pixel-selection fix (last pixel must be reachable),
the output-exclusive guard, that cleanup-on-failure only removes files/dirs
this script itself created (never an arbitrary delete), at three failure
points (case preparation, mid-benchmark, and a prepare_case that fails
after writing only its first file), and one true end-to-end smoke run
against the real horizon.town_disk at size=64.
'''
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / 'scripts' / 'compare_town_disk.py'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from horizon import town_disk as real_town_disk  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location('compare_town_disk_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


class _StubRNG:
    '''Replays a fixed list of (row, col) pairs from integers(0, size, size=2).'''

    def __init__(self, pairs):
        self._pairs = list(pairs)
        self.calls = 0

    def integers(self, low, high, size=2):
        pair = self._pairs[self.calls]
        self.calls += 1
        return np.array(pair)


class RegionScheduleUniformityTests(unittest.TestCase):
    '''The schedule used to draw from Beta(0.5, 0.5) scaled by (size - 1), which
    (being supported on the open interval (0, 1)) could essentially never
    produce the last pixel. It must now be reachable with a plain uniform draw.
    '''

    def test_last_pixel_is_reachable_and_recorded(self):
        size = 64
        entries = mod._chunk_entries(size, size, 256)  # single chunk: chunk_size > size
        grid_cols = 1
        rng = _StubRNG([(63, 63), (0, 0)])

        schedule = mod._region_schedule(rng, size, entries, 256, grid_cols, repeats=2)

        self.assertEqual(len(schedule), 2)
        self.assertEqual(schedule[0]['row_px'], 63)
        self.assertEqual(schedule[0]['col_px'], 63)
        self.assertEqual(schedule[0]['region'], (0, 0, 64, 64))
        self.assertEqual(schedule[1]['row_px'], 0)
        self.assertEqual(schedule[1]['col_px'], 0)

    def test_last_pixel_maps_to_last_grid_chunk(self):
        size = 1024
        chunk_size = 256
        entries = mod._chunk_entries(size, size, chunk_size)
        grid_cols = 4
        rng = _StubRNG([(1023, 1023)])

        schedule = mod._region_schedule(rng, size, entries, chunk_size, grid_cols, repeats=1)

        self.assertEqual(schedule[0]['region'], entries[-1])
        self.assertEqual(schedule[0]['region'], (768, 768, 256, 256))

    def test_schedule_length_always_equals_repeats(self):
        size = 1024
        entries = mod._chunk_entries(size, size, 256)
        rng = np.random.default_rng(12345)
        for repeats in (1, 30, 50):
            with self.subTest(repeats=repeats):
                schedule = mod._region_schedule(rng, size, entries, 256, 4, repeats)
                self.assertEqual(len(schedule), repeats)


class _Wrapped:
    '''Delegates to the real horizon.town_disk by default; lets individual
    tests inject a failure at a specific prepare_case call or after a
    specific number of load_* calls, without touching the real module.'''

    def __init__(self, real, prepare_case_override=None, load_fail_after=None):
        self._real = real
        self._prepare_case_override = prepare_case_override
        self._load_fail_after = load_fail_after
        self.prepare_calls = 0
        self.load_calls = 0
        self.MODES = real.MODES

    def prepare_case(self, arr, directory, method, chunk_size=256):
        self.prepare_calls += 1
        if self._prepare_case_override is not None:
            return self._prepare_case_override(self.prepare_calls, self._real, arr, directory, method, chunk_size)
        return self._real.prepare_case(arr, directory, method, chunk_size=chunk_size)

    def _maybe_fail_load(self):
        self.load_calls += 1
        if self._load_fail_after is not None and self.load_calls > self._load_fail_after:
            raise RuntimeError('simulated mid-benchmark load failure')

    def load_whole(self, path, mode='warm_buffered', region=None):
        self._maybe_fail_load()
        return self._real.load_whole(path, mode=mode, region=region)

    def load_chunked(self, path, mode='warm_buffered', region=None):
        self._maybe_fail_load()
        return self._real.load_chunked(path, mode=mode, region=region)


class _TmpDirTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.output = self.dir / 'out.json'
        self.containers = self.dir / 'containers'

    def run_main_quiet(self, argv, wrapped):
        with mock.patch.object(mod, '_import_town_disk', return_value=wrapped):
            with contextlib.redirect_stderr(io.StringIO()):
                return mod.main(argv)

    def base_argv(self, size=64, repeats=30, modes='warm_buffered'):
        return ['--output', str(self.output), '--containers-dir', str(self.containers),
                '--size', str(size), '--repeats', str(repeats), '--modes', modes]


class OutputExclusiveGuardTests(_TmpDirTestCase):
    def test_second_run_refuses_to_overwrite_existing_output(self):
        wrapped = _Wrapped(real_town_disk)
        rc = self.run_main_quiet(self.base_argv(), wrapped)
        self.assertEqual(rc, 0)
        self.assertTrue(self.output.exists())
        before = self.output.read_bytes()

        rc2 = self.run_main_quiet(
            ['--output', str(self.output), '--containers-dir', str(self.dir / 'containers2'),
             '--size', '64', '--repeats', '30', '--modes', 'warm_buffered'],
            _Wrapped(real_town_disk),
        )
        self.assertEqual(rc2, 1)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertFalse((self.dir / 'containers2').exists())

    def test_existing_containers_dir_refused_before_any_write(self):
        self.containers.mkdir()
        rc = self.run_main_quiet(self.base_argv(), _Wrapped(real_town_disk))
        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        # The pre-existing (empty) containers dir must survive untouched --
        # this run never created it, so cleanup must never rmdir it.
        self.assertTrue(self.containers.exists())
        self.assertEqual(list(self.containers.iterdir()), [])

    def test_existing_output_with_preexisting_empty_containers_dir_preserved(self):
        '''Regression test: _cleanup_own_containers used to unconditionally
        rmdir --containers-dir on any failure, even when this run never
        created it (e.g. a pre-existing empty dir, with --output already
        existing causing the very first guard to fail before anything is
        written). That deleted a directory this script had no business
        touching.'''
        self.output.write_bytes(b'keep-me')
        self.containers.mkdir()

        rc = self.run_main_quiet(self.base_argv(), _Wrapped(real_town_disk))

        self.assertEqual(rc, 1)
        self.assertEqual(self.output.read_bytes(), b'keep-me')
        self.assertTrue(self.containers.exists(), 'pre-existing containers dir must survive')
        self.assertEqual(list(self.containers.iterdir()), [])


class RollbackTests(_TmpDirTestCase):
    def test_preparation_failure_cleans_up_already_prepared_methods(self):
        def override(call_number, real, arr, directory, method, chunk_size):
            if call_number == 2:
                raise RuntimeError('simulated prepare_case failure')
            return real.prepare_case(arr, directory, method, chunk_size=chunk_size)

        wrapped = _Wrapped(real_town_disk, prepare_case_override=override)
        rc = self.run_main_quiet(self.base_argv(), wrapped)

        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.containers.exists(),
                          'first method\'s real files/dir must be removed, and the now-empty '
                          'containers dir removed with them')

    def test_mid_benchmark_failure_cleans_up_all_prepared_methods(self):
        # All 3 methods prepare successfully; fail partway through loading.
        wrapped = _Wrapped(real_town_disk, load_fail_after=3)
        rc = self.run_main_quiet(self.base_argv(), wrapped)

        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.containers.exists(),
                          'every already-prepared method is recorded in cases, so cleanup '
                          'must remove all of them')

    def test_prepare_case_failing_after_first_file_leaves_orphan_for_that_method_only(self):
        '''If prepare_case itself fails after writing only one of its two files
        (its own rollback is that module's responsibility, not this script's),
        this script must not guess at or delete paths it was never told about.
        It must still fully clean up every *other* method it does have a
        recorded path for, and must not crash.'''
        def override(call_number, real, arr, directory, method, chunk_size):
            if call_number == 2:
                directory = Path(directory)
                directory.mkdir(parents=True)
                (directory / 'whole.hrzn').write_bytes(b'partial-orphan')
                raise RuntimeError('simulated failure before writing the second file')
            return real.prepare_case(arr, directory, method, chunk_size=chunk_size)

        wrapped = _Wrapped(real_town_disk, prepare_case_override=override)
        rc = self.run_main_quiet(self.base_argv(), wrapped)

        self.assertEqual(rc, 1)
        self.assertFalse(self.output.exists())
        # The first method's recorded files/dir were cleaned up by this script.
        method_dirs = sorted(p.name for p in self.containers.iterdir()) if self.containers.exists() else []
        # Only the orphan from the un-recorded failed method may remain.
        self.assertLessEqual(len(method_dirs), 1)
        if method_dirs:
            orphan_dir = self.containers / method_dirs[0]
            self.assertTrue((orphan_dir / 'whole.hrzn').exists())


class TrueSmokeTest(_TmpDirTestCase):
    def test_real_town_disk_size_64_end_to_end(self):
        rc = self.run_main_quiet(self.base_argv(size=64, repeats=30), _Wrapped(real_town_disk))
        self.assertEqual(rc, 0)
        self.assertTrue(self.output.exists())
        self.assertTrue(self.containers.exists())

        import json
        report = json.loads(self.output.read_text())
        self.assertEqual(report['schema_version'], 1)
        self.assertEqual(len(report['cases']), 3)
        for case in report['cases']:
            self.assertEqual(case['hashes_before'], case['hashes_after'])
            modes = case['modes']
            self.assertIn('warm_buffered', modes)
            region_schedule = modes['warm_buffered']['region_schedule']
            self.assertEqual(len(region_schedule), 30)
            for entry in region_schedule:
                self.assertIn('row_px', entry)
                self.assertIn('col_px', entry)
                self.assertGreaterEqual(entry['row_px'], 0)
                self.assertLess(entry['row_px'], 64)


if __name__ == '__main__':
    unittest.main()

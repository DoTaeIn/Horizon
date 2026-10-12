import contextlib
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from horizon.town_map import MAX_SIZE as REAL_MAX_SIZE
from horizon.town_map import MIN_SIZE as REAL_MIN_SIZE

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'compare_town.py'


def _load():
    spec = importlib.util.spec_from_file_location('compare_town_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()

BASE_ARGS = ['--size', '64', '--seed', '0', '--repeats', str(mod.MIN_REPEATS), '--chunk-sizes', '64']


def _stub_plan(size=1024, seed=20261012):
    '''Minimal town_plan()-shaped dict: just enough for render_layout_svg.'''
    return {
        'river': {'t': [0.0, 0.5, 1.0], 'x': [0.4, 0.5, 0.6]},
        'routes': [
            {'name': 'north_route', 'points': [(0.5, 0.6), (0.5, 0.1)]},
        ],
        'flats': [
            {'name': 'plaza', 'center': (0.5, 0.6), 'radii': (0.08, 0.06)},
        ],
        'crossings': [
            {'kind': 'bridge', 'u': 0.5, 'v': 0.4},
        ],
    }


def install_stub_modules(test_case, with_town_plan=False):
    '''Replace horizon.town_map/horizon.chunk_benchmark with trivial stand-ins for the
    duration of one test, so CLI/plumbing tests never run the real codec benchmark.
    with_town_plan=True also stubs town_plan(), to exercise the layout-overlay path
    without the real (heavier) geometry generator.'''
    saved = {name: sys.modules.get(name) for name in ('horizon.town_map', 'horizon.chunk_benchmark')}

    town_map_stub = types.ModuleType('horizon.town_map')

    def make_town_map(size=1024, seed=20261012):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 65536, size=(size, size), dtype=np.uint32).astype('<u2')

    town_map_stub.make_town_map = make_town_map
    if with_town_plan:
        town_map_stub.town_plan = _stub_plan
    sys.modules['horizon.town_map'] = town_map_stub

    chunk_benchmark_stub = types.ModuleType('horizon.chunk_benchmark')

    def run_chunk_benchmark(arr, chunk_sizes=(64, 128, 256, 512), repeats=30, seed=20261012):
        return {
            'schema_version': 1,
            'settings': {'chunk_sizes': list(chunk_sizes), 'repeats': repeats, 'seed': seed},
            'results': [{'chunk_size': int(c)} for c in chunk_sizes],
        }

    chunk_benchmark_stub.run_chunk_benchmark = run_chunk_benchmark
    sys.modules['horizon.chunk_benchmark'] = chunk_benchmark_stub

    def restore():
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    test_case.addCleanup(restore)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def make_file(self, name, data=b'x'):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def try_symlink(self, link, target):
        try:
            os.symlink(str(target), str(link))
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f'symlink creation unavailable: {exc}')


class ParseArgsTests(unittest.TestCase):
    def parse_quiet(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            return mod.parse_args(argv)

    def test_defaults(self):
        ns = self.parse_quiet(['--output', 'out.json'])
        self.assertEqual(ns.size, mod.DEFAULT_SIZE)
        self.assertEqual(ns.seed, mod.DEFAULT_SEED)
        self.assertEqual(ns.repeats, mod.DEFAULT_REPEATS)
        self.assertEqual(ns.chunk_sizes, mod.DEFAULT_CHUNK_SIZES)
        self.assertIsNone(ns.preview_dir)

    def test_size_bounds_match_town_map_contract(self):
        self.assertEqual(mod.MIN_SIZE, REAL_MIN_SIZE)
        self.assertEqual(mod.MAX_SIZE, REAL_MAX_SIZE)
        for bad in ('0', '1', str(mod.MIN_SIZE - 1), str(mod.MAX_SIZE + 1)):
            with self.subTest(size=bad):
                with self.assertRaises(SystemExit):
                    self.parse_quiet(['--size', bad, '--output', 'out.json'])
        for good in (mod.MIN_SIZE, mod.MAX_SIZE):
            ns = self.parse_quiet(['--size', str(good), '--output', 'out.json'])
            self.assertEqual(ns.size, good)

    def test_seed_bounds_match_town_map_contract(self):
        ns = self.parse_quiet(['--seed', '0', '--output', 'out.json'])
        self.assertEqual(ns.seed, 0)
        ns2 = self.parse_quiet(['--seed', str(mod.SEED_EXCLUSIVE_MAX - 1), '--output', 'out.json'])
        self.assertEqual(ns2.seed, mod.SEED_EXCLUSIVE_MAX - 1)
        for bad in ('-1', str(mod.SEED_EXCLUSIVE_MAX)):
            with self.subTest(seed=bad):
                with self.assertRaises(SystemExit):
                    self.parse_quiet(['--seed', bad, '--output', 'out.json'])

    def test_repeats_minimum(self):
        with self.assertRaises(SystemExit):
            self.parse_quiet(['--repeats', str(mod.MIN_REPEATS - 1), '--output', 'out.json'])

    def test_chunk_sizes_rejects_nonpositive_and_duplicates(self):
        for bad in ('0,64', '64,-1', '64,64'):
            with self.subTest(chunk_sizes=bad):
                with self.assertRaises(SystemExit):
                    self.parse_quiet(['--chunk-sizes', bad, '--output', 'out.json'])

    def test_output_required(self):
        with self.assertRaises(SystemExit):
            self.parse_quiet([])


class WriteAllExclusiveTests(TempDirTestCase):
    def test_writes_every_file_on_success(self):
        a = self.dir / 'a.bin'
        b = self.dir / 'sub' / 'b.bin'
        mod.write_all_exclusive([(a, b'111'), (b, b'222')])
        self.assertEqual(a.read_bytes(), b'111')
        self.assertEqual(b.read_bytes(), b'222')

    def test_all_or_nothing_rollback_on_mid_write_failure(self):
        first = self.dir / 'a.bin'
        blocker = self.make_file('blocked', b'not a directory')
        second = blocker / 'b.bin'  # parent exists as a plain file: mkdir/open must fail

        with self.assertRaises(OSError):
            mod.write_all_exclusive([(first, b'111'), (second, b'222')])

        self.assertFalse(first.exists())

    def test_rollback_does_not_touch_unrelated_files(self):
        unrelated = self.make_file('keep.txt', b'keep')
        first = self.dir / 'a.bin'
        blocker = self.make_file('blocked', b'not a directory')
        second = blocker / 'b.bin'

        with self.assertRaises(OSError):
            mod.write_all_exclusive([(first, b'111'), (second, b'222')])

        self.assertEqual(unrelated.read_bytes(), b'keep')


class CliTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        install_stub_modules(self)

    def run_main(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            return mod.main(argv)

    def test_successful_run_then_rerun_refused_without_touching_files(self):
        output = self.dir / 'out.json'
        preview_dir = self.dir / 'preview'
        argv = BASE_ARGS + ['--output', str(output), '--preview-dir', str(preview_dir)]

        rc = self.run_main(list(argv))
        self.assertEqual(rc, 0)
        svg_path = preview_dir / 'town_64_0.svg'
        npy_path = preview_dir / 'town_64_0.npy'
        self.assertTrue(output.exists())
        self.assertTrue(svg_path.exists())
        self.assertTrue(npy_path.exists())
        report = json.loads(output.read_text())
        self.assertEqual(report['settings']['size'], 64)
        self.assertEqual(report['settings']['seed'], 0)

        svg_before = svg_path.read_bytes()
        npy_before = npy_path.read_bytes()
        rc2 = self.run_main(list(argv))
        self.assertEqual(rc2, 1)
        self.assertEqual(svg_path.read_bytes(), svg_before)
        self.assertEqual(npy_path.read_bytes(), npy_before)

    def test_existing_output_refused_and_unchanged(self):
        output = self.make_file('out.json', b'original')
        argv = BASE_ARGS + ['--output', str(output)]
        rc = self.run_main(argv)
        self.assertEqual(rc, 1)
        self.assertEqual(output.read_bytes(), b'original')

    def test_output_symlink_to_existing_file_refused(self):
        real = self.make_file('real.json', b'original')
        link = self.dir / 'link.json'
        self.try_symlink(link, real)
        argv = BASE_ARGS + ['--output', str(link)]
        rc = self.run_main(argv)
        self.assertEqual(rc, 1)
        self.assertEqual(real.read_bytes(), b'original')

    def test_output_dangling_symlink_refused_and_target_not_created(self):
        missing = self.dir / 'missing.json'
        link = self.dir / 'dangling.json'
        self.try_symlink(link, missing)
        argv = BASE_ARGS + ['--output', str(link)]
        rc = self.run_main(argv)
        self.assertEqual(rc, 1)
        self.assertFalse(missing.exists())

    def test_preexisting_preview_file_blocks_everything_including_output(self):
        preview_dir = self.dir / 'preview'
        preview_dir.mkdir()
        npy_path = preview_dir / 'town_64_0.npy'
        npy_path.write_bytes(b'existing')
        output = self.dir / 'out.json'
        argv = BASE_ARGS + ['--output', str(output), '--preview-dir', str(preview_dir)]

        rc = self.run_main(argv)
        self.assertEqual(rc, 1)
        self.assertFalse(output.exists())
        self.assertFalse((preview_dir / 'town_64_0.svg').exists())
        self.assertEqual(npy_path.read_bytes(), b'existing')

    def test_preview_svg_symlink_refused_and_npy_not_created(self):
        preview_dir = self.dir / 'preview'
        preview_dir.mkdir()
        real = self.make_file('elsewhere.svg', b'original')
        svg_link = preview_dir / 'town_64_0.svg'
        self.try_symlink(svg_link, real)
        output = self.dir / 'out.json'
        argv = BASE_ARGS + ['--output', str(output), '--preview-dir', str(preview_dir)]

        rc = self.run_main(argv)
        self.assertEqual(rc, 1)
        self.assertFalse(output.exists())
        self.assertFalse((preview_dir / 'town_64_0.npy').exists())
        self.assertEqual(real.read_bytes(), b'original')

    def test_layout_svg_skipped_when_town_plan_unavailable(self):
        # install_stub_modules(self) in setUp stubs town_map without town_plan.
        preview_dir = self.dir / 'preview'
        argv = BASE_ARGS + ['--output', str(self.dir / 'out.json'), '--preview-dir', str(preview_dir)]
        rc = self.run_main(argv)
        self.assertEqual(rc, 0)
        self.assertTrue((preview_dir / 'town_64_0.svg').exists())
        self.assertFalse((preview_dir / 'town_64_0_layout.svg').exists())


class CliWithTownPlanTests(TempDirTestCase):
    '''Exercises the layout-overlay path with a stubbed town_plan(), so these tests
    stay fast and independent of the real geometry generator.'''

    def setUp(self):
        super().setUp()
        install_stub_modules(self, with_town_plan=True)

    def run_main(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            return mod.main(argv)

    def test_layout_svg_created_well_formed_with_overlay_elements(self):
        preview_dir = self.dir / 'preview'
        argv = BASE_ARGS + ['--output', str(self.dir / 'out.json'), '--preview-dir', str(preview_dir)]
        rc = self.run_main(argv)
        self.assertEqual(rc, 0)

        layout_path = preview_dir / 'town_64_0_layout.svg'
        self.assertTrue(layout_path.exists())
        svg = layout_path.read_text(encoding='utf-8')
        ET.fromstring(svg)  # well-formed XML
        self.assertIn('North Route', svg)
        self.assertIn('Plaza', svg)
        self.assertIn('Schematic metadata', svg)  # disclaimer present

    def test_layout_svg_collision_refused_and_nothing_written(self):
        preview_dir = self.dir / 'preview'
        preview_dir.mkdir()
        layout_path = preview_dir / 'town_64_0_layout.svg'
        layout_path.write_bytes(b'existing')
        output = self.dir / 'out.json'
        argv = BASE_ARGS + ['--output', str(output), '--preview-dir', str(preview_dir)]

        rc = self.run_main(argv)
        self.assertEqual(rc, 1)
        self.assertFalse(output.exists())
        self.assertFalse((preview_dir / 'town_64_0.svg').exists())
        self.assertFalse((preview_dir / 'town_64_0.npy').exists())
        self.assertEqual(layout_path.read_bytes(), b'existing')


class RealTownMapSmokeTest(TempDirTestCase):
    '''Only runs once horizon.town_map is delivered; tiny size/repeats so this stays a
    correctness smoke check, not a performance run.'''

    def setUp(self):
        super().setUp()
        try:
            importlib.import_module('horizon.town_map')
        except ModuleNotFoundError:
            self.skipTest('horizon.town_map is not available yet')

    def test_minimum_size_smoke_run(self):
        output = self.dir / 'smoke.json'
        argv = ['--size', str(mod.MIN_SIZE), '--seed', '0', '--repeats', str(mod.MIN_REPEATS),
                 '--chunk-sizes', str(mod.MIN_SIZE), '--output', str(output)]
        with contextlib.redirect_stderr(io.StringIO()):
            rc = mod.main(argv)
        self.assertEqual(rc, 0)
        report = json.loads(output.read_text())
        self.assertEqual(report['map']['shape'], [mod.MIN_SIZE, mod.MIN_SIZE])
        self.assertEqual(report['settings']['size'], mod.MIN_SIZE)


if __name__ == '__main__':
    unittest.main()

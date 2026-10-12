import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from horizon import benchmark
from horizon.codec import METHODS

ROOT = Path(__file__).resolve().parents[1]

RESULT_KEYS = {
    'map',
    'source',
    'method',
    'sha256',
    'shape',
    'raw_bytes',
    'container_bytes',
    'ratio',
    'encode_median_ms',
    'decode_median_ms',
    'decode_p95_ms',
    'exact_match',
}


def run_cli(*args):
    return subprocess.run(
        [sys.executable, '-m', 'horizon', *[str(a) for a in args]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


class ReportTests(unittest.TestCase):
    def test_schema_and_values(self):
        report = benchmark.run_benchmark(size=8, repeats=5)
        self.assertEqual(report['schema_version'], 1)
        for key in ('python', 'platform', 'numpy', 'zstandard', 'lz4'):
            self.assertIn(key, report['environment'])
        self.assertEqual(report['settings']['repeats'], 5)
        self.assertEqual(report['settings']['size'], 8)
        self.assertEqual(report['settings']['seed'], benchmark.SEED)
        self.assertEqual(set(report['settings']['codec_settings']), set(METHODS))
        self.assertIn('linear', report['settings']['p95_method'])
        results = report['results']
        self.assertEqual(len(results), 4 * len(METHODS))
        for row in results:
            self.assertEqual(set(row), RESULT_KEYS)
            self.assertTrue(row['exact_match'])
            self.assertEqual(row['source'], 'synthetic')
            self.assertEqual(row['shape'], [8, 8])
            self.assertEqual(row['raw_bytes'], 128)
            self.assertEqual(len(row['sha256']), 64)
            self.assertGreater(row['container_bytes'], 0)
            self.assertAlmostEqual(row['ratio'], row['container_bytes'] / 128, places=5)
            self.assertGreaterEqual(row['decode_p95_ms'], 0)

    def test_fixtures_deterministic(self):
        a = benchmark.make_fixtures(16)
        b = benchmark.make_fixtures(16)
        self.assertEqual([n for n, _, _ in a], ['flat', 'plane', 'smooth_terrain', 'random'])
        for (_, _, x), (_, _, y) in zip(a, b):
            self.assertEqual(x.dtype, np.dtype('<u2'))
            np.testing.assert_array_equal(x, y)

    def test_file_inputs(self):
        arr = (np.arange(35, dtype=np.uint16) * 3).reshape(5, 7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'real.npy'
            np.save(path, arr)
            report = benchmark.run_benchmark(
                size=4, repeats=5, inputs=[path]
            )
        sources = [r['source'] for r in report['results']]
        self.assertEqual(sources.count('file'), len(METHODS))
        self.assertEqual(report['settings']['input_count'], 1)


class ValidationTests(unittest.TestCase):
    def test_bad_size_and_repeats(self):
        for size in (0, -1, True, 1.5, 5793):
            with self.assertRaises(ValueError, msg=repr(size)):
                benchmark.run_benchmark(size=size, repeats=5)
        for repeats in (4, 0, False, 'x'):
            with self.assertRaises(ValueError, msg=repr(repeats)):
                benchmark.run_benchmark(size=4, repeats=repeats)

    def test_bad_inputs_rejected_before_timing(self):
        bad = [
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.int32),
            np.zeros((4, 4), dtype='>u2'),
            np.zeros(4, dtype=np.uint16),
            np.zeros((0, 3), dtype=np.uint16),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(benchmark, 'encode', side_effect=AssertionError):
                for index, arr in enumerate(bad):
                    path = Path(tmp) / f'bad_{index}.npy'
                    np.save(path, arr)
                    with self.assertRaises(ValueError):
                        benchmark.run_benchmark(size=4, repeats=5, inputs=[path])

    def test_forged_npy_shape_rejected_before_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'huge.npy'
            with path.open('wb') as handle:
                np.lib.format.write_array_header_1_0(handle, {
                    'descr': '<u2', 'fortran_order': False, 'shape': (2**30, 2**30)
                })
            with self.assertRaises(ValueError):
                benchmark.load_heightmap(path)

    def test_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            junk = Path(tmp) / 'junk.npy'
            junk.write_bytes(b'not a npy file')
            pickled = Path(tmp) / 'obj.npy'
            np.save(pickled, np.array([{'a': 1}], dtype=object))
            for path in (junk, pickled):
                with self.assertRaises(ValueError):
                    benchmark.run_benchmark(size=4, repeats=5, inputs=[path])

    def test_npy_versions_fortran_order_and_trailing_bytes(self):
        arr = np.asfortranarray(np.arange(12, dtype='<u2').reshape(3, 4))
        with tempfile.TemporaryDirectory() as tmp:
            for version in ((1, 0), (2, 0), (3, 0)):
                path = Path(tmp) / f'version_{version[0]}.npy'
                with path.open('wb') as handle:
                    np.lib.format.write_array(handle, arr, version=version, allow_pickle=False)
                restored = benchmark.load_heightmap(path)
                self.assertEqual(restored.shape, arr.shape)
                self.assertEqual(restored.tobytes(), arr.tobytes())
                with path.open('ab') as handle:
                    handle.write(b'extra')
                with self.assertRaises(ValueError):
                    benchmark.load_heightmap(path)


class CliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_pack_unpack_roundtrip_and_no_overwrite(self):
        arr = np.random.default_rng(1).integers(0, 65536, size=(5, 7), dtype=np.uint16)
        src = self.dir / 'in.npy'
        np.save(src, arr)
        packed = self.dir / 'out.hzn'
        restored = self.dir / 'back.npy'

        r = run_cli('pack', src, packed)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = run_cli('unpack', packed, restored)
        self.assertEqual(r.returncode, 0, r.stderr)
        back = np.load(restored, allow_pickle=False)
        self.assertEqual(back.dtype, arr.dtype)
        self.assertEqual(back.shape, arr.shape)
        np.testing.assert_array_equal(back, arr)

        before = packed.read_bytes()
        r = run_cli('pack', src, packed, '--method', 'zstd')
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn('Traceback', r.stderr)
        self.assertEqual(packed.read_bytes(), before)

        r = run_cli('unpack', packed, restored)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn('Traceback', r.stderr)

    def test_invalid_inputs_create_no_output(self):
        bad = self.dir / 'float.npy'
        np.save(bad, np.zeros((3, 3), dtype=np.float32))
        out = self.dir / 'bad.hzn'
        r = run_cli('pack', bad, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn('Traceback', r.stderr)
        self.assertFalse(out.exists())

        corrupt = self.dir / 'corrupt.hzn'
        corrupt.write_bytes(b'garbage bytes')
        out2 = self.dir / 'corrupt.npy'
        r = run_cli('unpack', corrupt, out2)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn('Traceback', r.stderr)
        self.assertFalse(out2.exists())

    def test_benchmark_output_and_refusal(self):
        report_path = self.dir / 'report.json'
        r = run_cli('benchmark', '--size', 8, '--repeats', 5, '--output', report_path)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(report_path.read_text(encoding='utf-8'))
        self.assertEqual(len(report['results']), 4 * len(METHODS))

        before = report_path.read_bytes()
        r = run_cli('benchmark', '--size', 8, '--repeats', 5, '--output', report_path)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn('Traceback', r.stderr)
        self.assertEqual(report_path.read_bytes(), before)

    def test_benchmark_stdout_and_bad_args(self):
        r = run_cli('benchmark', '--size', 4, '--repeats', 5)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['settings']['size'], 4)

        r = run_cli('benchmark', '--size', 4, '--repeats', 4)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn('Traceback', r.stderr)


if __name__ == '__main__':
    unittest.main()

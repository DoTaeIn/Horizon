import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'compare_disk.py'


def _load():
    spec = importlib.util.spec_from_file_location('compare_disk_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()
IS_WINDOWS = sys.platform == 'win32'
BOUNDARY_SIZES = (1, 4095, 4096, 4097, 8193)
LARGE_SIZE = 3 * 1024 * 1024 + 123


def payload(size):
    return random.Random(size).randbytes(size)


def file_state(path):
    st = os.stat(path)
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), st.st_size, st.st_mtime_ns


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def make_file(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path


class ConstantsTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(mod.MAX_FILE_BYTES, 65 * 1024 * 1024)
        self.assertEqual(tuple(mod.METHODS), ('zstd', 'delta_zstd'))
        self.assertEqual(tuple(mod.MODES), ('warm_buffered', 'os_cache_bypass'))


class ReadBufferedTests(TempDirTestCase):
    def test_boundary_sizes_exact_bytes_and_file_unchanged(self):
        for size in BOUNDARY_SIZES + (LARGE_SIZE,):
            with self.subTest(size=size):
                data = payload(size)
                path = self.make_file('f%d.bin' % size, data)
                before = file_state(path)
                out = mod.read_buffered(str(path))
                self.assertIsInstance(out, bytes)
                self.assertEqual(out, data)
                self.assertEqual(file_state(path), before)

    def test_empty_file_rejected(self):
        path = self.make_file('empty.bin', b'')
        with self.assertRaises(ValueError):
            mod.read_buffered(str(path))

    def test_cap_boundary(self):
        with mock.patch.object(mod, 'MAX_FILE_BYTES', 10):
            ok = self.make_file('ok.bin', payload(10))
            self.assertEqual(mod.read_buffered(str(ok)), payload(10))
            big = self.make_file('big.bin', payload(11))
            with self.assertRaises(ValueError):
                mod.read_buffered(str(big))


    def test_read_requests_actual_size_plus_one(self):
        import builtins

        real_open = builtins.open
        reads = []

        class _SpyFile:
            def __init__(self, f):
                self._f = f

            def __enter__(self):
                self._f.__enter__()
                return self

            def __exit__(self, *exc):
                return self._f.__exit__(*exc)

            def fileno(self):
                return self._f.fileno()

            def read(self, n=-1):
                reads.append(n)
                return self._f.read(n)

        def spy_open(*args, **kwargs):
            return _SpyFile(real_open(*args, **kwargs))

        data = b"hello"
        path = self.make_file("small.bin", data)
        with mock.patch.object(mod, "MAX_FILE_BYTES", 64), \
                mock.patch.object(mod, "open", spy_open, create=True):
            result = mod.read_buffered(path)

        self.assertEqual(result, data)
        self.assertEqual(reads, [os.path.getsize(path) + 1])

    def test_oversized_file_rejected_before_read(self):
        import builtins

        real_open = builtins.open
        reads = []

        class _SpyFile:
            def __init__(self, f):
                self._f = f

            def __enter__(self):
                self._f.__enter__()
                return self

            def __exit__(self, *exc):
                return self._f.__exit__(*exc)

            def fileno(self):
                return self._f.fileno()

            def read(self, n=-1):
                reads.append(n)
                return self._f.read(n)

        def spy_open(*args, **kwargs):
            return _SpyFile(real_open(*args, **kwargs))

        path = self.make_file("big.bin", b"x" * 16)
        with mock.patch.object(mod, "MAX_FILE_BYTES", 8), \
                mock.patch.object(mod, "open", spy_open, create=True):
            with self.assertRaises(Exception):
                mod.read_buffered(path)

        self.assertEqual(reads, [])

@unittest.skipUnless(IS_WINDOWS, 'native unbuffered reads are Windows-only')
class ReadUnbufferedNativeTests(TempDirTestCase):
    def test_boundary_sizes_match_buffered_and_file_unchanged(self):
        for size in BOUNDARY_SIZES + (LARGE_SIZE,):
            with self.subTest(size=size):
                data = payload(size)
                path = self.make_file('n%d.bin' % size, data)
                before = file_state(path)
                out = mod.read_unbuffered(str(path))
                self.assertIsInstance(out, bytes)
                self.assertEqual(len(out), size)
                self.assertEqual(out, data)
                self.assertEqual(out, mod.read_buffered(str(path)))
                self.assertEqual(file_state(path), before)

    def test_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            mod.read_unbuffered(str(self.dir / 'does_not_exist.bin'))

    def test_empty_file_rejected(self):
        path = self.make_file('empty.bin', b'')
        with self.assertRaises(ValueError):
            mod.read_unbuffered(str(path))

    def test_cap_boundary(self):
        with mock.patch.object(mod, 'MAX_FILE_BYTES', 4096):
            ok = self.make_file('ok.bin', payload(4096))
            self.assertEqual(mod.read_unbuffered(str(ok)), payload(4096))
            big = self.make_file('big.bin', payload(4097))
            with self.assertRaises(ValueError):
                mod.read_unbuffered(str(big))


@unittest.skipIf(IS_WINDOWS, 'unsupported-OS check applies to non-Windows only')
class ReadUnbufferedUnsupportedOsTests(TempDirTestCase):
    def test_rejects_on_non_windows(self):
        path = self.make_file('f.bin', payload(4096))
        with self.assertRaises(Exception):
            mod.read_unbuffered(str(path))


class ValidateResultTests(unittest.TestCase):
    def setUp(self):
        self.expected = np.arange(12, dtype=np.uint16).reshape(3, 4)

    def test_accepts_equal_array(self):
        self.assertIsNone(mod.validate_result(self.expected.copy(), self.expected))

    def test_rejects_non_ndarray(self):
        with self.assertRaises(ValueError):
            mod.validate_result(self.expected.tolist(), self.expected)
        with self.assertRaises(ValueError):
            mod.validate_result(bytes(24), self.expected)

    def test_rejects_wrong_dtype(self):
        for dtype in (np.uint32, np.int16, '>u2'):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    mod.validate_result(self.expected.astype(dtype), self.expected)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            mod.validate_result(self.expected.reshape(4, 3).copy(), self.expected)
        with self.assertRaises(ValueError):
            mod.validate_result(self.expected.reshape(-1).copy(), self.expected)

    def test_rejects_wrong_values(self):
        actual = self.expected.copy()
        actual[2, 3] += 1
        with self.assertRaises(ValueError):
            mod.validate_result(actual, self.expected)

    def test_rejects_non_c_contiguous(self):
        actual = np.asfortranarray(self.expected)
        self.assertFalse(actual.flags['C_CONTIGUOUS'])
        with self.assertRaises(ValueError):
            mod.validate_result(actual, self.expected)

    def test_rejects_read_only(self):
        actual = self.expected.copy()
        actual.setflags(write=False)
        with self.assertRaises(ValueError):
            mod.validate_result(actual, self.expected)


class MeasureOnceTests(unittest.TestCase):
    def setUp(self):
        self.path = 'unused-path.bin'
        self.expected = np.arange(8, dtype=np.uint16)
        self.raw = b'raw-timed-bytes'

    def run_mode(self, mode, decoded=None, ticks=(100, 250, 700), unbuffered_error=None):
        events = []
        decode_calls = []
        paths = {'buffered': [], 'unbuffered': []}
        pending = list(ticks)
        real_validate = mod.validate_result
        raw = self.raw
        expected = self.expected

        def tick():
            events.append('tick')
            return pending.pop(0) if pending else 10 ** 15

        def buffered(path):
            events.append('read_buffered')
            paths['buffered'].append(path)
            return raw if events.count('read_buffered') > 1 else b'primed-bytes'

        def unbuffered(path):
            events.append('read_unbuffered')
            paths['unbuffered'].append(path)
            if unbuffered_error is not None:
                raise unbuffered_error
            return raw

        def decode(*args, **kwargs):
            events.append('decode')
            decode_calls.append((args, kwargs))
            return expected.copy() if decoded is None else decoded

        def validate(actual, exp):
            events.append('validate')
            return real_validate(actual, exp)

        with mock.patch.object(mod, 'read_buffered', buffered), \
                mock.patch.object(mod, 'read_unbuffered', unbuffered), \
                mock.patch.object(mod, 'decode', decode), \
                mock.patch.object(mod, 'validate_result', validate), \
                mock.patch.object(time, 'perf_counter_ns', tick):
            try:
                result = mod.measure_once(self.path, expected, mode)
            except BaseException as exc:
                return exc, events, decode_calls, paths
        return result, events, decode_calls, paths

    def decode_got_raw(self, decode_calls):
        self.assertEqual(len(decode_calls), 1)
        args, kwargs = decode_calls[0]
        values = list(args) + list(kwargs.values())
        self.assertTrue(any(v == self.raw for v in values))
        self.assertFalse(any(v == b'primed-bytes' for v in values))

    def test_warm_buffered_order_timing_and_data(self):
        result, events, calls, paths = self.run_mode('warm_buffered')
        self.assertIsInstance(result, dict)
        self.assertEqual(events[:6], ['read_buffered', 'tick', 'read_buffered', 'tick', 'decode', 'tick'])
        self.assertIn('validate', events[6:])
        self.assertNotIn('validate', events[:6])
        self.assertEqual(paths['buffered'], [self.path, self.path])
        self.assertEqual(paths['unbuffered'], [])
        self.assertEqual(result['read_ns'], 150)
        self.assertEqual(result['decode_ns'], 450)
        self.assertEqual(result['total_ns'], 600)
        for key in ('total_ns', 'read_ns', 'decode_ns'):
            self.assertIsInstance(result[key], int)
        self.decode_got_raw(calls)

    def test_os_cache_bypass_order_timing_and_data(self):
        result, events, calls, paths = self.run_mode('os_cache_bypass')
        self.assertIsInstance(result, dict)
        self.assertEqual(events[:5], ['tick', 'read_unbuffered', 'tick', 'decode', 'tick'])
        self.assertIn('validate', events[5:])
        self.assertNotIn('read_buffered', events)
        self.assertEqual(paths['unbuffered'], [self.path])
        self.assertEqual(result['read_ns'], 150)
        self.assertEqual(result['decode_ns'], 450)
        self.assertEqual(result['total_ns'], result['read_ns'] + result['decode_ns'])
        self.decode_got_raw(calls)

    def test_unknown_mode_rejected(self):
        result, events, calls, paths = self.run_mode('not_a_mode')
        self.assertIsInstance(result, ValueError)
        self.assertEqual(calls, [])

    def test_bypass_error_propagates_without_fallback(self):
        err = OSError('native read failed')
        result, events, calls, paths = self.run_mode('os_cache_bypass', unbuffered_error=err)
        self.assertIsInstance(result, OSError)
        self.assertNotIn('read_buffered', events)
        self.assertNotIn('decode', events)
        self.assertEqual(paths['buffered'], [])

    def test_invalid_decode_output_rejected_in_both_modes(self):
        readonly = self.expected.copy()
        readonly.setflags(write=False)
        bad_outputs = {
            'values': np.zeros_like(self.expected),
            'dtype': self.expected.astype(np.uint32),
            'shape': self.expected.reshape(2, 4).copy(),
            'readonly': readonly,
            'not_array': self.expected.tolist(),
        }
        for mode in ('warm_buffered', 'os_cache_bypass'):
            for name, bad in bad_outputs.items():
                with self.subTest(mode=mode, bad=name):
                    result, events, calls, paths = self.run_mode(mode, decoded=bad)
                    self.assertIsInstance(result, ValueError)


class SummarizeTests(unittest.TestCase):
    @staticmethod
    def make(totals, reads, decodes):
        ms = 10 ** 6
        return [{'total_ns': t * ms, 'read_ns': r * ms, 'decode_ns': d * ms}
                for t, r, d in zip(totals, reads, decodes)]

    def test_known_percentiles_and_samples_preserved(self):
        samples = self.make([1, 2, 3, 4, 5], [10, 10, 10, 10, 10], [5, 1, 4, 2, 3])
        snapshot = copy.deepcopy(samples)
        out = mod.summarize(samples)
        self.assertEqual(out['repeats'], 5)
        self.assertAlmostEqual(out['total']['median_ms'], 3.0)
        self.assertAlmostEqual(out['total']['p95_ms'], 4.8)
        self.assertAlmostEqual(out['read']['median_ms'], 10.0)
        self.assertAlmostEqual(out['read']['p95_ms'], 10.0)
        self.assertAlmostEqual(out['decode']['median_ms'], 3.0)
        self.assertAlmostEqual(out['decode']['p95_ms'], 4.8)
        self.assertEqual(out['samples_ns'], snapshot)
        self.assertEqual(samples, snapshot)

    def test_even_count_median_and_linear_p95(self):
        out = mod.summarize(self.make([4, 1, 3, 2], [1, 1, 1, 1], [1, 1, 1, 1]))
        self.assertAlmostEqual(out['total']['median_ms'], 2.5)
        self.assertAlmostEqual(out['total']['p95_ms'], 3.85)

    def test_single_sample(self):
        out = mod.summarize(self.make([7], [3], [4]))
        self.assertEqual(out['repeats'], 1)
        self.assertAlmostEqual(out['total']['median_ms'], 7.0)
        self.assertAlmostEqual(out['total']['p95_ms'], 7.0)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            mod.summarize([])


class WriteReportTests(TempDirTestCase):
    def try_symlink(self, link, target):
        try:
            os.symlink(str(target), str(link))
        except (OSError, NotImplementedError) as exc:
            self.skipTest('symlink creation unavailable: %s' % exc)

    def test_writes_json(self):
        report = {'a': 1, 'b': [1.5, 2.5], 'c': {'d': 'text'}}
        target = self.dir / 'report.json'
        mod.write_report(target, report)
        self.assertEqual(json.loads(target.read_text(encoding='utf-8')), report)

    def test_existing_target_refused_and_unchanged(self):
        target = self.make_file('report.json', b'original')
        with self.assertRaises((OSError, ValueError)):
            mod.write_report(target, {'a': 1})
        self.assertEqual(target.read_bytes(), b'original')

    def test_symlink_to_existing_file_refused(self):
        real = self.make_file('real.json', b'original')
        link = self.dir / 'link.json'
        self.try_symlink(link, real)
        with self.assertRaises((OSError, ValueError)):
            mod.write_report(link, {'a': 1})
        self.assertEqual(real.read_bytes(), b'original')

    def test_dangling_symlink_refused_and_target_not_created(self):
        missing = self.dir / 'missing.json'
        link = self.dir / 'dangling.json'
        self.try_symlink(link, missing)
        with self.assertRaises((OSError, ValueError)):
            mod.write_report(link, {'a': 1})
        self.assertFalse(missing.exists())

    def test_nan_rejected_and_other_files_untouched(self):
        other = self.make_file('other.txt', b'keep')
        with self.assertRaises(Exception):
            mod.write_report(self.dir / 'nan.json', {'x': float('nan')})
        self.assertEqual(other.read_bytes(), b'keep')


class ParseArgsTests(unittest.TestCase):
    BASE = ['--input', 'a.bin', '--input', 'b.bin',
            '--containers-dir', 'containers', '--output', 'out.json']

    def parse_quiet(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            return mod.parse_args(argv)

    def test_defaults_and_repeatable_input(self):
        ns = self.parse_quiet(self.BASE)
        self.assertEqual([str(p) for p in ns.input], ['a.bin', 'b.bin'])
        self.assertEqual(str(ns.containers_dir), 'containers')
        self.assertEqual(str(ns.output), 'out.json')
        self.assertEqual(ns.repeats, 50)

    def test_repeats_minimum(self):
        ns = self.parse_quiet(self.BASE + ['--repeats', '30'])
        self.assertEqual(ns.repeats, 30)
        for bad in ('29', '0', '-5', 'abc'):
            with self.subTest(repeats=bad):
                with self.assertRaises(SystemExit):
                    self.parse_quiet(self.BASE + ['--repeats', bad])

    def test_required_arguments(self):
        missing_sets = [
            ['--containers-dir', 'c', '--output', 'o.json'],
            ['--input', 'a.bin', '--output', 'o.json'],
            ['--input', 'a.bin', '--containers-dir', 'c'],
        ]
        for argv in missing_sets:
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    self.parse_quiet(argv)

    def test_main_rejects_invalid_repeats(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                mod.main(self.BASE + ['--repeats', '5'])


if __name__ == '__main__':
    unittest.main()

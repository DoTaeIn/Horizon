import contextlib
import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location('_script_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = load_script('prepare_terrain')
cd = load_script('compare_decode')

NAME = 'N37E127'
VALUES = [-32768, -1, 0, 1, 100, 32767, -500, 8848, 12]
RAW = np.asarray(VALUES, dtype='>i2').tobytes()
EXPECTED_U = (np.asarray(VALUES, dtype=np.int32) + 32768).astype('<u2').reshape(3, 3)
NOW = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)


def make_fetch(raw=RAW, headers=None):
    body = gzip.compress(raw, mtime=0)

    def fetch(url):
        fetch.calls.append(url)
        return body, dict(headers or {})

    fetch.calls = []
    fetch.body = body
    return fetch


class FakeResponse:
    def __init__(self, url, body, headers=None):
        self._url = url
        self._body = body
        self.headers = dict(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def geturl(self):
        return self._url

    def read(self, n=-1):
        return self._body if n is None or n < 0 else self._body[:n]


class NormalizeTests(unittest.TestCase):
    def test_known_values_and_inverse(self):
        raw = np.asarray([-32768, -1, 0, 32767], dtype='>i2').tobytes()
        u = prep.normalize_hgt(raw, side=2)
        self.assertEqual(u.dtype, np.dtype('<u2'))
        self.assertEqual(u.tolist(), [[0, 32767], [32768, 65535]])
        self.assertEqual(prep.denormalize_hgt(u), raw)

    def test_side_three_round_trip(self):
        u = prep.normalize_hgt(RAW, side=3)
        self.assertEqual(u.shape, (3, 3))
        self.assertTrue(u.flags.c_contiguous)
        self.assertTrue(np.array_equal(u, EXPECTED_U))
        self.assertEqual(prep.denormalize_hgt(u), RAW)

    def test_full_int16_range_is_a_bijection(self):
        values = np.arange(-32768, 32768, dtype=np.int32)
        raw = values.astype('>i2').tobytes()
        u = prep.normalize_hgt(raw, side=256)
        self.assertEqual(u.reshape(-1).tolist(), list(range(65536)))
        self.assertEqual(prep.denormalize_hgt(u), raw)

    def test_wrong_length_rejected(self):
        for size in (0, 17, 19):
            with self.assertRaises(prep.TerrainError):
                prep.normalize_hgt(b'\x00' * size, side=3)

    def test_nodata_is_kept_and_excluded_from_range(self):
        s = prep.summarize_values(prep.normalize_hgt(RAW, side=3))
        self.assertEqual(s['nodata_count'], 1)
        self.assertEqual(s['valid_count'], 8)
        self.assertEqual(s['valid_min_m'], -500)
        self.assertEqual(s['valid_max_m'], 32767)

    def test_all_nodata_has_no_valid_range(self):
        raw = np.full(4, -32768, dtype='>i2').tobytes()
        s = prep.summarize_values(prep.normalize_hgt(raw, side=2))
        self.assertEqual(s['nodata_count'], 4)
        self.assertIsNone(s['valid_min_m'])
        self.assertIsNone(s['valid_max_m'])


class GunzipAndDownloadTests(unittest.TestCase):
    def test_expected_hgt_size(self):
        self.assertEqual(prep.SIDE * prep.SIDE * 2, 25934402)

    def test_exact_size_accepted(self):
        self.assertEqual(prep.gunzip_hgt(gzip.compress(RAW), side=3), RAW)

    def test_too_long_and_too_short_rejected(self):
        for size in (17, 19, 100000):
            with self.assertRaises(prep.TerrainError):
                prep.gunzip_hgt(gzip.compress(b'\x00' * size), side=3)

    def test_corrupt_and_truncated_gzip_rejected(self):
        with self.assertRaises(prep.TerrainError):
            prep.gunzip_hgt(b'definitely not gzip', side=3)
        with self.assertRaises(prep.TerrainError):
            prep.gunzip_hgt(gzip.compress(RAW)[:-5], side=3)

    def test_non_allowlisted_url_never_fetched(self):
        with mock.patch.object(prep.urllib.request, 'urlopen') as urlopen:
            with self.assertRaises(prep.TerrainError):
                prep.download_limited('https://example.com/N37E127.hgt.gz')
            with self.assertRaises(prep.TerrainError):
                prep.download_limited(prep.TILE_URLS[NAME].replace('https://', 'http://'))
        urlopen.assert_not_called()

    def test_download_size_cap(self):
        url = prep.TILE_URLS[NAME]
        with mock.patch.object(prep.urllib.request, 'urlopen',
                               return_value=FakeResponse(url, b'x' * 11)):
            with self.assertRaises(prep.TerrainError):
                prep.download_limited(url, max_bytes=10)
        headers = {'X-Amz-Meta-X-Imagery-Sources': 'abc', 'Set-Cookie': 'secret'}
        with mock.patch.object(prep.urllib.request, 'urlopen',
                               return_value=FakeResponse(url, b'x' * 10, headers)):
            data, kept = prep.download_limited(url, max_bytes=10)
        self.assertEqual(data, b'x' * 10)
        self.assertEqual(kept.get('x-amz-meta-x-imagery-sources'), 'abc')
        self.assertNotIn('set-cookie', kept)

    def test_redirect_rejected(self):
        url = prep.TILE_URLS[NAME]
        with mock.patch.object(prep.urllib.request, 'urlopen',
                               return_value=FakeResponse('https://example.com/x', b'x')):
            with self.assertRaises(prep.TerrainError):
                prep.download_limited(url)


class PrepareTileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / 'terrain'

    def prepare(self, fetch=None):
        return prep.prepare_tile(NAME, self.out, fetch=fetch or make_fetch(), now=NOW, side=3)

    def snapshot(self):
        if not self.out.exists():
            return {}
        return {p.name: p.read_bytes() for p in sorted(self.out.iterdir())}

    def test_invalid_tile(self):
        with self.assertRaises(ValueError):
            prep.prepare_tile('N00E000', self.out, fetch=make_fetch(), side=3)
        self.assertEqual(self.snapshot(), {})
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                prep.parse_args(['--tile', 'N00E000'])

    def test_default_and_repeated_tiles(self):
        self.assertIsNone(prep.parse_args([]).tile)
        self.assertEqual(prep.parse_args(['--tile', 'N46E007', '--tile', 'N37E127']).tile,
                         ['N46E007', 'N37E127'])

    def test_fresh_download_writes_three_files(self):
        fetch = make_fetch(headers={'x-amz-meta-x-imagery-sources': 'abc'})
        meta, reused = self.prepare(fetch)
        self.assertFalse(reused)
        self.assertEqual(fetch.calls, [prep.TILE_URLS[NAME]])
        snap = self.snapshot()
        self.assertEqual(set(snap), {NAME + '.hgt.gz', NAME + '.npy', NAME + '.json'})
        self.assertEqual(snap[NAME + '.hgt.gz'], fetch.body)
        arr = np.load(io.BytesIO(snap[NAME + '.npy']), allow_pickle=False)
        self.assertEqual(arr.dtype, np.dtype('<u2'))
        self.assertTrue(np.array_equal(arr, EXPECTED_U))
        self.assertEqual(json.loads(snap[NAME + '.json']), meta)
        self.assertEqual(meta['source_url'], prep.TILE_URLS[NAME])
        self.assertEqual(meta['shape'], [3, 3])
        self.assertTrue(meta['fetched_utc'].startswith('2026-01-02T03:04:05'))
        self.assertEqual(meta['sha256']['gzip'], hashlib.sha256(fetch.body).hexdigest())
        self.assertEqual(meta['sha256']['raw_hgt'], hashlib.sha256(RAW).hexdigest())
        self.assertEqual(meta['sha256']['normalized_uint16_le'],
                         hashlib.sha256(EXPECTED_U.tobytes()).hexdigest())
        self.assertEqual(meta['sha256']['npy_file'], hashlib.sha256(snap[NAME + '.npy']).hexdigest())
        self.assertEqual(meta['nodata_count'], 1)
        self.assertEqual((meta['valid_min_m'], meta['valid_max_m']), (-500, 32767))
        self.assertEqual(meta['source_imagery_headers'],
                         {'x-amz-meta-x-imagery-sources': 'abc'})

    def test_metadata_links_and_attribution(self):
        meta, _ = self.prepare()
        self.assertEqual(meta['source_imagery_headers'], {})
        self.assertEqual(meta['registry_url'], 'https://registry.opendata.aws/terrain-tiles/')
        self.assertIn('data-sources.md', meta['source_docs_url'])
        self.assertIn('attribution.md', meta['attribution_url'])
        for text in ('Mapzen / Tilezen',
                     'Terrain data courtesy of the U.S. Geological Survey.',
                     'Produced using Copernicus data and information funded by the European Union - EU-DEM layers.',
                     'NOAA / ETOPO1'):
            self.assertIn(text, meta['attribution'])

    def test_complete_cache_is_verified_and_reused(self):
        meta, _ = self.prepare()
        before = self.snapshot()
        refetch = mock.Mock(side_effect=AssertionError('must not download'))
        meta2, reused = self.prepare(refetch)
        self.assertTrue(reused)
        self.assertEqual(meta2, meta)
        self.assertEqual(self.snapshot(), before)

    def test_partial_cache_is_an_error_and_untouched(self):
        for key in ('.hgt.gz', '.npy', '.json'):
            with self.subTest(missing=key):
                self.setUp()
                self.prepare()
                (self.out / (NAME + key)).unlink()
                before = self.snapshot()
                refetch = mock.Mock(side_effect=AssertionError('must not download'))
                with self.assertRaises(prep.TerrainError):
                    self.prepare(refetch)
                self.assertEqual(self.snapshot(), before)

    def test_single_existing_file_is_never_overwritten(self):
        self.out.mkdir(parents=True)
        (self.out / (NAME + '.json')).write_bytes(b'keep')
        with self.assertRaises(prep.TerrainError):
            self.prepare()
        self.assertEqual(self.snapshot(), {NAME + '.json': b'keep'})

    def test_corrupt_caches_are_errors_and_untouched(self):
        other = (np.asarray(VALUES, dtype=np.int32)[::-1] + 32768).astype('<u2').reshape(3, 3)
        buf = io.BytesIO()
        np.save(buf, other, allow_pickle=False)

        def tamper_npy():
            (self.out / (NAME + '.npy')).write_bytes(buf.getvalue())

        def tamper_gz():
            (self.out / (NAME + '.hgt.gz')).write_bytes(gzip.compress(RAW[::-1], mtime=0))

        def garbage_gz():
            (self.out / (NAME + '.hgt.gz')).write_bytes(b'garbage')

        def bad_json():
            meta_path = self.out / (NAME + '.json')
            meta_path.write_text('not json')

        def bad_hash():
            meta_path = self.out / (NAME + '.json')
            data = json.loads(meta_path.read_text())
            data['sha256']['raw_hgt'] = '0' * 64
            meta_path.write_text(json.dumps(data))

        def bad_nodata_count():
            meta_path = self.out / (NAME + '.json')
            data = json.loads(meta_path.read_text())
            data['nodata_count'] = 0
            meta_path.write_text(json.dumps(data))

        for fn in (tamper_npy, tamper_gz, garbage_gz, bad_json, bad_hash, bad_nodata_count):
            with self.subTest(case=fn.__name__):
                self.setUp()
                self.prepare()
                fn()
                before = self.snapshot()
                with self.assertRaises(prep.TerrainError):
                    self.prepare(mock.Mock(side_effect=AssertionError('must not download')))
                self.assertEqual(self.snapshot(), before)

    def test_forged_huge_npy_header_rejected_before_np_load(self):
        self.prepare()
        header = repr({'descr': '<u2', 'fortran_order': False, 'shape': (10 ** 6, 10 ** 6)})
        header = header.encode('latin1')
        header += b' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + b'\n'
        forged = b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header + b'\x00' * 18
        (self.out / (NAME + '.npy')).write_bytes(forged)
        before = self.snapshot()
        boom = mock.Mock(side_effect=AssertionError('np.load must not run on unvalidated cache'))
        with mock.patch.object(prep.np, 'load', boom):
            with self.assertRaises(prep.TerrainError):
                self.prepare(mock.Mock(side_effect=AssertionError('must not download')))
        boom.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_oversize_cache_files_rejected_before_parsing(self):
        for key, limit_name in (('.json', 'MAX_JSON_BYTES'), ('.hgt.gz', 'MAX_GZIP_BYTES')):
            with self.subTest(file=key):
                self.setUp()
                self.prepare()
                before = self.snapshot()
                with mock.patch.object(prep, limit_name, 8):
                    with self.assertRaises(prep.TerrainError):
                        self.prepare(mock.Mock(side_effect=AssertionError('must not download')))
                self.assertEqual(self.snapshot(), before)
        self.setUp()
        self.prepare()
        (self.out / (NAME + '.npy')).write_bytes(b'\x00' * (3 * 3 * 2 + prep.NPY_HEADER_ALLOWANCE + 1))
        before = self.snapshot()
        with mock.patch.object(prep.np, 'load', side_effect=AssertionError('np.load called')):
            with self.assertRaises(prep.TerrainError):
                self.prepare(mock.Mock(side_effect=AssertionError('must not download')))
        self.assertEqual(self.snapshot(), before)

    def test_bad_payload_leaves_no_files(self):
        for payload in (RAW + b'\x00', RAW[:-2], b''):
            with self.subTest(size=len(payload)):
                self.setUp()
                with self.assertRaises(prep.TerrainError):
                    self.prepare(make_fetch(payload))
                self.assertEqual(self.snapshot(), {})

    def test_failure_cleans_only_own_files(self):
        json_path = self.out / (NAME + '.json')
        inner = make_fetch()

        def racing(url):
            self.out.mkdir(parents=True, exist_ok=True)
            json_path.write_bytes(b'keep')
            return inner(url)

        with self.assertRaises((OSError, prep.TerrainError)):
            self.prepare(racing)
        self.assertEqual(self.snapshot(), {NAME + '.json': b'keep'})


class FakeCodec:
    METHODS = ('raw', 'rev')

    def __init__(self, tag=b'F', writable=True, fortran=False, log=None, label=''):
        self.tag = tag
        self.writable = writable
        self.fortran = fortran
        self.log = log
        self.label = label

    def encode(self, arr, method):
        body = np.ascontiguousarray(arr, dtype='<u2').tobytes()
        if method == 'rev':
            body = body[::-1]
        shape = np.asarray(arr.shape, dtype='<u4').tobytes()
        return self.tag + bytes([self.METHODS.index(method)]) + shape + body

    def decode(self, data):
        if self.log is not None:
            self.log.append(self.label)
        h, w = (int(v) for v in np.frombuffer(data[2:10], dtype='<u4'))
        body = data[10:]
        if data[1] == 1:
            body = body[::-1]
        out = np.frombuffer(body, dtype='<u2').reshape(h, w)
        if self.writable:
            out = out.copy()
        if self.fortran:
            out = np.asfortranarray(out)
        return out


SMALL = np.arange(120, dtype=np.uint16).reshape(12, 10).astype('<u2')


class CompareDecodeTests(unittest.TestCase):
    def test_central_crop_is_deterministic(self):
        arr = np.arange(10 * 12, dtype=np.uint16).reshape(10, 12)
        sub, row, col = cd.central_crop(arr, 4)
        self.assertEqual((row, col), (3, 4))
        self.assertTrue(np.array_equal(sub, arr[3:7, 4:8]))
        self.assertTrue(sub.flags.c_contiguous)
        self.assertIsNone(cd.central_crop(arr, 11))

    def test_build_cases_only_uses_fixed_sizes_that_fit(self):
        def fixtures(size):
            return [('flat', 'synthetic', np.zeros((size, size), dtype='<u2'))]

        big = np.zeros((1100, 1200), dtype='<u2')
        cases = cd.build_cases(['x/tile.npy'], lambda p: big, fixtures)
        got = [(c['kind'], tuple(c['shape']), c['row_origin'], c['col_origin']) for c in cases]
        self.assertEqual(got, [
            ('synthetic', (256, 256), 0, 0),
            ('real-crop', (256, 256), 422, 472),
            ('real-crop', (1024, 1024), 38, 88),
            ('real-full', (1100, 1200), 0, 0),
        ])
        self.assertTrue(all('tile' in c['label'] for c in cases[1:]))
        small = np.zeros((100, 100), dtype='<u2')
        cases = cd.build_cases(['t.npy'], lambda p: small, fixtures)
        self.assertEqual([c['kind'] for c in cases], ['synthetic', 'real-full'])

    def test_percentiles_and_relative(self):
        s = cd.summarize_times([1_000_000, 2_000_000, 3_000_000, 4_000_000])
        self.assertAlmostEqual(s['median_ms'], 2.5)
        self.assertAlmostEqual(s['p95_ms'], 3.85)
        self.assertEqual(s['samples_ns'], [1_000_000, 2_000_000, 3_000_000, 4_000_000])
        r = cd.relative(4.0, 2.0)
        self.assertAlmostEqual(r['new_over_old_percent'], 50.0)
        self.assertAlmostEqual(r['speedup'], 2.0)
        self.assertIsNone(cd.relative(0.0, 1.0)['speedup'])

    def test_check_decoded_rejects_bad_results(self):
        good = SMALL.copy()
        cd.check_decoded(good, SMALL, 'good')
        readonly = SMALL.copy()
        readonly.setflags(write=False)
        bad = {
            'dtype': SMALL.astype('>u2'),
            'shape': SMALL[:6].copy(),
            'readonly': readonly,
            'fortran': np.asfortranarray(SMALL),
            'values': SMALL + 1,
            'type': SMALL.tobytes(),
        }
        for name, value in bad.items():
            with self.subTest(name):
                with self.assertRaises(cd.ValidationError):
                    cd.check_decoded(value, SMALL, name)

    def test_validate_case_method(self):
        old, new = FakeCodec(), FakeCodec()
        for method in FakeCodec.METHODS:
            enc = cd.validate_case_method(old, new, SMALL, method)
            self.assertEqual(enc, old.encode(SMALL, method))
        for broken in (FakeCodec(tag=b'X'), FakeCodec(writable=False), FakeCodec(fortran=True)):
            with self.assertRaises(cd.ValidationError):
                cd.validate_case_method(old, broken, SMALL, 'raw')

    def test_measure_method_order_and_samples(self):
        log = []
        old = FakeCodec(log=log, label='old')
        new = FakeCodec(log=log, label='new')
        enc = old.encode(SMALL, 'rev')
        res = cd.measure_method(old.decode, new.decode, enc, SMALL, repeats=4, warmup=2)
        self.assertEqual(sorted(log[:4]), ['new', 'new', 'old', 'old'])
        self.assertEqual(log[4:], ['old', 'new', 'new', 'old', 'old', 'new', 'new', 'old'])
        for side in ('baseline', 'current'):
            self.assertEqual(len(res[side]['samples_ns']), 4)
            self.assertEqual(res[side]['repeats'], 4)
        self.assertIn('speedup', res)
        self.assertIn('new_over_old_percent', res)

    def test_measure_method_validates_results(self):
        old, new = FakeCodec(), FakeCodec(writable=False)
        enc = old.encode(SMALL, 'raw')
        with self.assertRaises(cd.ValidationError):
            cd.measure_method(old.decode, new.decode, enc, SMALL, repeats=2, warmup=0)

    def test_run_benchmark_records_are_serializable(self):
        case = cd.make_case('tiny/full', 'real-full', 'mem', 0, 0, SMALL)
        logs = []
        results = cd.run_benchmark(FakeCodec(), FakeCodec(), [case], FakeCodec.METHODS,
                                   repeats=3, warmup=1, log=logs.append)
        self.assertTrue(logs)
        rec = results[0]
        self.assertNotIn('array', rec)
        self.assertEqual(rec['array_sha256'], hashlib.sha256(SMALL.tobytes()).hexdigest())
        self.assertEqual(rec['raw_bytes'], SMALL.nbytes)
        self.assertEqual(set(rec['methods']), set(FakeCodec.METHODS))
        for entry in rec['methods'].values():
            self.assertGreater(entry['container_bytes'], 0)
            self.assertEqual(len(entry['container_sha256']), 64)
        json.dumps(results, allow_nan=False)

    def test_methods_mismatch_detected(self):
        other = FakeCodec()
        other.METHODS = ('raw',)
        with self.assertRaises(cd.ValidationError):
            cd.resolve_methods(other, FakeCodec())

    def test_report_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sub' / 'report.json'
            cd.write_report(path, {'a': 1})
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                cd.write_report(path, {'a': 2})
            self.assertEqual(path.read_bytes(), first)

    def test_cli_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'existing.json'
            out.write_bytes(b'keep')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertNotEqual(cd.main(['--output', str(out), '--repeats', '30']), 0)
                with self.assertRaises(SystemExit):
                    cd.parse_args(['--output', str(out), '--repeats', '29'])
                with self.assertRaises(SystemExit):
                    cd.parse_args([])
            self.assertEqual(out.read_bytes(), b'keep')
        args = cd.parse_args(['--output', 'x.json'])
        self.assertEqual(args.repeats, 50)
        self.assertEqual(args.baseline, 'benchmarks/reference/codec_v1.py')


if __name__ == '__main__':
    unittest.main()

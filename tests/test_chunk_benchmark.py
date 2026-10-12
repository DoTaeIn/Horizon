import unittest
from unittest import mock

import numpy as np

from horizon import chunk_benchmark
from horizon.chunk_benchmark import run_chunk_benchmark
from horizon.codec import METHODS, decode, encode

_DTYPE = np.dtype('<u2')

RESULT_KEYS = {
    'method',
    'chunk_size',
    'grid',
    'chunk_count',
    'whole_container_bytes',
    'chunk_index_bytes',
    'chunk_payload_bytes',
    'chunked_container_bytes',
    'size_ratio_chunked_to_whole',
    'region_schedule',
    'whole_encode',
    'chunked_encode',
    'whole_decode',
    'chunked_reconstruct',
    'region_chunk_decode',
    'region_full_decode_crop',
    'exact_match',
}

TIMING_KEYS = {'raw_ms', 'median_ms', 'p95_ms'}
REGION_DESC_KEYS = {'repeat', 'row', 'col', 'height', 'width', 'chunk_index', 'verified'}


def _make_arr(height, width, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=(height, width), dtype=np.uint16).astype(_DTYPE)


class ChunkGridTests(unittest.TestCase):
    def test_divisible_grid(self):
        entries = chunk_benchmark._chunk_entries(8, 8, 4)
        self.assertEqual(chunk_benchmark._grid_dims(8, 8, 4), (2, 2))
        self.assertEqual(len(entries), 4)
        for _row, _col, h, w in entries:
            self.assertEqual((h, w), (4, 4))

    def test_non_divisible_edges_shrink_to_fit(self):
        height, width, chunk_size = 10, 7, 4
        entries = chunk_benchmark._chunk_entries(height, width, chunk_size)
        rows, cols = chunk_benchmark._grid_dims(height, width, chunk_size)
        self.assertEqual((rows, cols), (3, 2))
        self.assertEqual(len(entries), rows * cols)

        covered = np.zeros((height, width), dtype=bool)
        for row, col, h, w in entries:
            self.assertGreaterEqual(h, 1)
            self.assertGreaterEqual(w, 1)
            self.assertLessEqual(row + h, height)
            self.assertLessEqual(col + w, width)
            self.assertFalse(covered[row:row + h, col:col + w].any(), 'chunks overlap')
            covered[row:row + h, col:col + w] = True
        self.assertTrue(covered.all(), 'chunks do not cover the whole map')

    def test_chunk_size_larger_than_map_yields_single_chunk(self):
        entries = chunk_benchmark._chunk_entries(5, 6, 999)
        self.assertEqual(entries, [(0, 0, 5, 6)])
        self.assertEqual(chunk_benchmark._grid_dims(5, 6, 999), (1, 1))


class RegionScheduleTests(unittest.TestCase):
    def test_deterministic_and_diverse(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = chunk_benchmark._region_schedule(rng1, chunk_count=6, repeats=10)
        s2 = chunk_benchmark._region_schedule(rng2, chunk_count=6, repeats=10)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 10)
        # First full cycle (one per chunk) is a permutation: every chunk visited once.
        self.assertEqual(sorted(s1[:6]), list(range(6)))
        # Over more than one cycle every chunk is covered, not just a single one.
        self.assertEqual(set(s1), set(range(6)))

    def test_single_chunk_repeats_trivially(self):
        rng = np.random.default_rng(1)
        s = chunk_benchmark._region_schedule(rng, chunk_count=1, repeats=5)
        self.assertEqual(s, [0, 0, 0, 0, 0])


class LookupAndCropTests(unittest.TestCase):
    def test_lookup_entry_is_o1_and_matches_linear_scan(self):
        arr = _make_arr(9, 5, seed=2)
        entries = chunk_benchmark._chunk_entries(9, 5, 3)
        grid_rows, grid_cols = chunk_benchmark._grid_dims(9, 5, 3)
        blobs = chunk_benchmark._encode_chunks(arr, entries, 'zstd')
        container, _index_bytes = chunk_benchmark._build_container(
            'zstd', 3, grid_rows, grid_cols, entries, blobs
        )
        chunk_count = len(entries)
        _method, _cs, _gr, _gc, unpacked, base = chunk_benchmark._unpack_index(container)

        for i, (row, col, h, w) in enumerate(entries):
            lrow, lcol, lh, lw, abs_offset, length = chunk_benchmark._lookup_entry(
                container, chunk_count, i
            )
            self.assertEqual((lrow, lcol, lh, lw), (row, col, h, w))
            _urow, _ucol, _uh, _uw, uoff, ulen = unpacked[i]
            self.assertEqual(abs_offset, base + uoff)
            self.assertEqual(length, ulen)
            blob = container[abs_offset:abs_offset + length]
            np.testing.assert_array_equal(decode(blob), arr[row:row + h, col:col + w])

    def test_crop_copy_is_independently_owned(self):
        arr = _make_arr(8, 8)
        decoded = arr.copy()
        cropped = chunk_benchmark._crop_copy(decoded, 2, 2, 3, 3)
        self.assertTrue(cropped.flags['OWNDATA'])
        self.assertFalse(np.shares_memory(cropped, decoded))
        np.testing.assert_array_equal(cropped, decoded[2:5, 2:5])


class OrderPolicyTests(unittest.TestCase):
    def test_alternates_by_repeat_parity(self):
        arr = _make_arr(8, 8)
        height, width = arr.shape
        chunk_size = 4
        entries = chunk_benchmark._chunk_entries(height, width, chunk_size)
        grid_rows, grid_cols = chunk_benchmark._grid_dims(height, width, chunk_size)
        schedule = [0, 1, 2, 3, 0]
        expected_whole_bytes = arr.tobytes()

        order_log = []
        real_whole = chunk_benchmark._time_whole_encode
        real_chunk = chunk_benchmark._time_chunked_encode

        def whole_wrapper(*a, **kw):
            order_log.append('whole')
            return real_whole(*a, **kw)

        def chunk_wrapper(*a, **kw):
            order_log.append('chunk')
            return real_chunk(*a, **kw)

        with mock.patch.object(chunk_benchmark, '_time_whole_encode', side_effect=whole_wrapper), \
                mock.patch.object(chunk_benchmark, '_time_chunked_encode', side_effect=chunk_wrapper):
            chunk_benchmark._bench_chunk_config(
                arr, height, width, chunk_size, grid_rows, grid_cols, entries,
                schedule, expected_whole_bytes, 'zstd', 5,
            )

        # The first pair is the untimed warmup call (always whole-then-chunk).
        self.assertEqual(order_log[:2], ['whole', 'chunk'])
        loop_pairs = [order_log[j:j + 2] for j in range(2, len(order_log), 2)]
        self.assertEqual(len(loop_pairs), 5)
        for i, pair in enumerate(loop_pairs):
            expected = ['whole', 'chunk'] if i % 2 == 0 else ['chunk', 'whole']
            self.assertEqual(pair, expected, f'repeat {i}')


class RunChunkBenchmarkTests(unittest.TestCase):
    def test_schema_and_result_count(self):
        arr = _make_arr(10, 9)
        report = run_chunk_benchmark(arr, chunk_sizes=(4, 5), repeats=5)
        self.assertEqual(report['schema_version'], 2)
        for key in ('python', 'platform', 'numpy', 'zstandard', 'lz4'):
            self.assertIn(key, report['environment'])
        self.assertEqual(report['input']['shape'], [10, 9])
        self.assertEqual(report['input']['raw_bytes'], arr.nbytes)
        self.assertEqual(len(report['input']['sha256']), 64)
        self.assertEqual(report['settings']['chunk_sizes'], [4, 5])
        # Truthful warmup metadata: exactly one untimed warmup call per timed path.
        self.assertEqual(report['settings']['warmup'], 1)
        self.assertEqual(report['settings']['warmup'], chunk_benchmark.WARMUP)
        self.assertIn('order_policy', report['settings'])
        self.assertIn('region_schedule_policy', report['settings'])
        self.assertIn('container_format', report['settings'])
        self.assertIn('private', report['settings']['container_format'])
        self.assertIn('no adversarial', report['settings']['container_format'])

        results = report['results']
        self.assertEqual(len(results), 2 * len(METHODS))
        for row in results:
            self.assertEqual(set(row), RESULT_KEYS)
            self.assertTrue(row['exact_match'])
            self.assertIn(row['method'], METHODS)
            self.assertEqual(len(row['region_schedule']), 5)
            for desc in row['region_schedule']:
                self.assertEqual(set(desc), REGION_DESC_KEYS)
                self.assertTrue(desc['verified'])
                self.assertLessEqual(desc['row'] + desc['height'], 10)
                self.assertLessEqual(desc['col'] + desc['width'], 9)
            for timing_key in (
                'whole_encode', 'chunked_encode', 'whole_decode',
                'chunked_reconstruct', 'region_chunk_decode', 'region_full_decode_crop',
            ):
                stats = row[timing_key]
                self.assertEqual(set(stats), TIMING_KEYS)
                self.assertEqual(len(stats['raw_ms']), 5)
                self.assertGreaterEqual(stats['median_ms'], 0)
                self.assertGreaterEqual(stats['p95_ms'], 0)

    def test_region_schedule_diverse_within_one_config(self):
        arr = _make_arr(12, 12)
        report = run_chunk_benchmark(arr, chunk_sizes=(4,), repeats=9)
        row = report['results'][0]
        chunk_indices = {d['chunk_index'] for d in row['region_schedule']}
        self.assertGreater(len(chunk_indices), 1, 'region schedule should visit more than one chunk')

    def test_schedule_shared_across_methods(self):
        arr = _make_arr(10, 9)
        report = run_chunk_benchmark(arr, chunk_sizes=(4,), repeats=6)
        rows = [r for r in report['results'] if r['chunk_size'] == 4]
        self.assertEqual(len(rows), len(METHODS))
        sequences = [[d['chunk_index'] for d in r['region_schedule']] for r in rows]
        self.assertTrue(all(seq == sequences[0] for seq in sequences))

    def test_index_byte_accounting(self):
        arr = _make_arr(10, 9)
        report = run_chunk_benchmark(arr, chunk_sizes=(4,), repeats=5)
        header_bytes = report['settings']['chunk_index_header_bytes']
        entry_bytes = report['settings']['chunk_index_entry_bytes']
        for row in report['results']:
            self.assertEqual(
                row['chunk_index_bytes'], header_bytes + entry_bytes * row['chunk_count']
            )
            self.assertEqual(
                row['chunked_container_bytes'],
                row['chunk_index_bytes'] + row['chunk_payload_bytes'],
            )
            self.assertEqual(row['whole_container_bytes'], len(encode(arr, method=row['method'])))

    def test_same_seed_is_reproducible(self):
        arr = _make_arr(12, 11)
        a = run_chunk_benchmark(arr, chunk_sizes=(4, 5), repeats=5, seed=7)
        b = run_chunk_benchmark(arr, chunk_sizes=(4, 5), repeats=5, seed=7)
        schedules_a = [[d['chunk_index'] for d in r['region_schedule']] for r in a['results']]
        schedules_b = [[d['chunk_index'] for d in r['region_schedule']] for r in b['results']]
        self.assertEqual(schedules_a, schedules_b)

    def test_non_divisible_map_round_trips(self):
        arr = _make_arr(13, 17, seed=3)
        report = run_chunk_benchmark(arr, chunk_sizes=(5,), repeats=5)
        for row in report['results']:
            self.assertEqual(row['grid'], [3, 4])
            self.assertEqual(row['chunk_count'], 12)
            self.assertTrue(row['exact_match'])

    def test_warmup_plus_repeats_call_counts(self):
        # decode() fires once per (whole_decode | region_chunk | region_crop) call, and
        # once per chunk inside each chunked_reconstruct call: (warmup=1 + repeats) * (3 + chunk_count).
        # repeats must stay >= MIN_REPEATS (5) or run_chunk_benchmark rejects it before any call happens.
        arr = _make_arr(6, 6)
        chunk_size = 3
        repeats = 5
        entries = chunk_benchmark._chunk_entries(6, 6, chunk_size)
        chunk_count = len(entries)
        expected_decode_calls_per_method = (1 + repeats) * (3 + chunk_count)

        real_decode = chunk_benchmark.decode
        with mock.patch.object(chunk_benchmark, 'decode', side_effect=real_decode) as mock_decode:
            run_chunk_benchmark(arr, chunk_sizes=(chunk_size,), repeats=repeats)

        self.assertEqual(mock_decode.call_count, len(METHODS) * expected_decode_calls_per_method)

    def test_warmup_plus_repeats_encode_call_counts(self):
        # encode() fires once per whole_encode call, and once per chunk inside each
        # chunked_encode call. Each of those two paths runs once as bootstrap (not through
        # the timed helper), once as untimed warmup (through the timed helper), and once per
        # timed repeat: (bootstrap=1 + warmup=1 + repeats) * (1 whole-call + chunk_count chunk-calls).
        arr = _make_arr(6, 6)
        chunk_size = 3
        repeats = 5
        entries = chunk_benchmark._chunk_entries(6, 6, chunk_size)
        chunk_count = len(entries)
        expected_encode_calls_per_method = (2 + repeats) * (1 + chunk_count)

        real_encode = chunk_benchmark.encode
        with mock.patch.object(chunk_benchmark, 'encode', side_effect=real_encode) as mock_encode:
            run_chunk_benchmark(arr, chunk_sizes=(chunk_size,), repeats=repeats)

        self.assertEqual(mock_encode.call_count, len(METHODS) * expected_encode_calls_per_method)


class ValidationTests(unittest.TestCase):
    def test_bad_arr_rejected_before_encode(self):
        bad_arrays = [
            np.zeros((4, 4), dtype=np.float32),
            np.zeros(4, dtype=np.uint16),
            np.zeros((0, 3), dtype=np.uint16),
            [[1, 2], [3, 4]],
        ]
        with mock.patch.object(chunk_benchmark, 'encode', side_effect=AssertionError):
            for arr in bad_arrays:
                with self.assertRaises(ValueError):
                    run_chunk_benchmark(arr, chunk_sizes=(2,), repeats=5)

    def test_bad_chunk_sizes(self):
        arr = _make_arr(8, 8)
        for chunk_sizes in ((), (0,), (-1,), (1.5,), ('x',), (True,), None):
            with self.assertRaises((ValueError, TypeError), msg=repr(chunk_sizes)):
                run_chunk_benchmark(arr, chunk_sizes=chunk_sizes, repeats=5)

    def test_duplicate_chunk_sizes_rejected(self):
        arr = _make_arr(8, 8)
        with self.assertRaises(ValueError):
            run_chunk_benchmark(arr, chunk_sizes=(4, 4), repeats=5)

    def test_out_of_u32_chunk_size_rejected(self):
        arr = _make_arr(4, 4)
        with self.assertRaises(ValueError):
            run_chunk_benchmark(arr, chunk_sizes=(0x100000000,), repeats=5)

    def test_u32_boundary_chunk_size_accepted(self):
        arr = _make_arr(4, 4)
        report = run_chunk_benchmark(arr, chunk_sizes=(0xFFFFFFFF,), repeats=5)
        self.assertEqual(report['settings']['chunk_sizes'], [0xFFFFFFFF])
        self.assertEqual(report['results'][0]['grid'], [1, 1])

    def test_chunk_sizes_accepts_single_int(self):
        arr = _make_arr(8, 8)
        report = run_chunk_benchmark(arr, chunk_sizes=4, repeats=5)
        self.assertEqual(report['settings']['chunk_sizes'], [4])

    def test_bad_repeats(self):
        arr = _make_arr(8, 8)
        for repeats in (4, 0, False, 'x', 1.5):
            with self.assertRaises(ValueError, msg=repr(repeats)):
                run_chunk_benchmark(arr, chunk_sizes=(4,), repeats=repeats)

    def test_bad_seed(self):
        arr = _make_arr(8, 8)
        for seed in (-1, True, 1.5, 'x'):
            with self.assertRaises(ValueError, msg=repr(seed)):
                run_chunk_benchmark(arr, chunk_sizes=(4,), repeats=5, seed=seed)


class ContainerRoundTripTests(unittest.TestCase):
    def test_pack_unpack_index_round_trip(self):
        arr = _make_arr(9, 5, seed=2)
        entries = chunk_benchmark._chunk_entries(9, 5, 3)
        grid_rows, grid_cols = chunk_benchmark._grid_dims(9, 5, 3)
        blobs = chunk_benchmark._encode_chunks(arr, entries, 'zstd')
        container, index_bytes = chunk_benchmark._build_container(
            'zstd', 3, grid_rows, grid_cols, entries, blobs
        )
        self.assertEqual(len(container), len(index_bytes) + sum(len(b) for b in blobs))

        method, chunk_size, rows, cols, unpacked_entries, base = chunk_benchmark._unpack_index(container)
        self.assertEqual(method, 'zstd')
        self.assertEqual(chunk_size, 3)
        self.assertEqual((rows, cols), (grid_rows, grid_cols))
        self.assertEqual(len(unpacked_entries), len(entries))

        reconstructed = chunk_benchmark._reconstruct_container(container, 9, 5)
        np.testing.assert_array_equal(reconstructed, arr)

        for (row, col, h, w, offset, length), blob in zip(unpacked_entries, blobs):
            self.assertEqual(length, len(blob))
            chunk_blob = container[base + offset: base + offset + length]
            np.testing.assert_array_equal(decode(chunk_blob), arr[row:row + h, col:col + w])


if __name__ == '__main__':
    unittest.main()

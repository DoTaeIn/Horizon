'''Independent tests for horizon.town_region.load_region.

load_region(path, region, mode='warm_buffered') -> (array, metrics) is expected
to read the full HCIX header/index on every call (no cross-call cache, no
reused file handle), then read ONLY the payload bytes of chunks that
intersect an arbitrary, positive, in-bounds region -- unlike
horizon.town_disk.load_chunked's region path, which only accepts a region
that is the exact extent of a single grid chunk. These tests build fixtures
with horizon.town_disk.prepare_case (the existing, already-tested writer) and
exercise horizon.town_region.load_region against the resulting chunked.hcix
container, independent of its implementation.

horizon.town_region is a required import: if it is missing or broken, this
module must fail loudly (an ImportError/collection error), not swallow it as
a skip, since that can hide a broken implementation import.
'''

from __future__ import annotations

import platform
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from horizon import town_disk, town_region
from horizon.chunk_benchmark import _INDEX_ENTRY, _INDEX_HEADER
from horizon.codec import HEADER_SIZE, METHODS

IS_DARWIN = platform.system() == 'Darwin'
_DTYPE = np.dtype('<u2')


def _make_arr(height, width, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=(height, width), dtype=np.uint16).astype(_DTYPE)


def _read_index(path):
    '''Parse the HCIX header/entries directly from disk, independent of any
    module under test, to compute expected chunk layouts/byte counts.'''
    data = Path(path).read_bytes()
    _magic, _version, _method_id, _reserved, chunk_size, rows, cols, count = (
        _INDEX_HEADER.unpack_from(data)
    )
    base = _INDEX_HEADER.size
    entries = []
    for i in range(count):
        row, col, h, w, offset, length = _INDEX_ENTRY.unpack_from(data, base + i * _INDEX_ENTRY.size)
        entries.append((int(row), int(col), int(h), int(w), int(offset), int(length)))
    payload_base = base + count * _INDEX_ENTRY.size
    return {
        'chunk_size': int(chunk_size),
        'rows': int(rows),
        'cols': int(cols),
        'count': int(count),
        'entries': entries,
        'header_bytes': base,
        'payload_base': payload_base,
    }


def _expected_chunk_coords(row, col, h, w, chunk_size, rows, cols):
    r0, r1 = row // chunk_size, (row + h - 1) // chunk_size
    c0, c1 = col // chunk_size, (col + w - 1) // chunk_size
    coords = []
    for gr in range(min(r0, rows - 1), min(r1, rows - 1) + 1):
        for gc in range(min(c0, cols - 1), min(c1, cols - 1) + 1):
            coords.append([gr, gc])
    return coords


def _expected_selected_bytes(index, coords):
    '''header + full index entries (always read) + only the selected chunks'
    payload bytes (the only part load_region should read selectively).'''
    total = index['payload_base']  # header + every index entry
    for gr, gc in coords:
        i = gr * index['cols'] + gc
        total += index['entries'][i][5]  # length
    return total


class TownRegionTestCase(unittest.TestCase):
    def setUp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def case_dir(self, name='case'):
        return self.dir / name

    def _prepare(self, height, width, chunk_size, method='zstd', seed=1, name='case'):
        arr = _make_arr(height, width, seed=seed)
        info = town_disk.prepare_case(arr, str(self.case_dir(name)), method, chunk_size=chunk_size)
        return arr, info

    def _assert_owned(self, out):
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(out.flags['C_CONTIGUOUS'])
        self.assertTrue(out.flags['OWNDATA'] or out.base is None)

    def _assert_metrics_shape(self, metrics, expected_coords):
        for key in (
            'total_ns', 'read_ns', 'decode_ns', 'bytes_read', 'bytes_requested',
            'read_calls', 'seek_calls', 'chunks_loaded', 'chunk_coordinates',
        ):
            self.assertIn(key, metrics, f'metrics missing {key!r}: {metrics!r}')
        self.assertEqual(metrics['total_ns'], metrics['read_ns'] + metrics['decode_ns'])
        self.assertEqual(metrics['chunks_loaded'], len(expected_coords))
        self.assertEqual(
            sorted(tuple(c) for c in metrics['chunk_coordinates']),
            sorted(tuple(c) for c in expected_coords),
        )


# --------------------------------------------------------------------------- #
# correctness across region geometries
# --------------------------------------------------------------------------- #

class RegionGeometryTests(TownRegionTestCase):
    def _check(self, height, width, chunk_size, region, name, seed=1):
        arr, info = self._prepare(height, width, chunk_size, seed=seed, name=name)
        index = _read_index(info['chunked_path'])
        row, col, h, w = region
        expected = arr[row:row + h, col:col + w]
        expected_coords = _expected_chunk_coords(row, col, h, w, chunk_size, index['rows'], index['cols'])

        out, metrics = town_region.load_region(info['chunked_path'], region)

        np.testing.assert_array_equal(out, expected)
        self._assert_owned(out)
        self._assert_metrics_shape(metrics, expected_coords)
        return out, metrics, index, expected_coords

    def test_single_chunk_region(self):
        self._check(64, 64, 16, (2, 2, 5, 5), 'single')

    def test_horizontal_boundary_region(self):
        # chunk_size=16: columns 0-15 and 16-31 are distinct grid cells in the
        # same grid row; this region straddles that column boundary only.
        self._check(64, 64, 16, (2, 12, 4, 8), 'horizontal')

    def test_vertical_boundary_region(self):
        self._check(64, 64, 16, (12, 2, 8, 4), 'vertical')

    def test_four_way_boundary_region(self):
        # Straddles both the row and column boundary at 16: touches all four
        # grid cells (0,0), (0,1), (1,0), (1,1).
        self._check(64, 64, 16, (12, 12, 8, 8), 'fourway')

    def test_region_larger_than_chunk_size(self):
        height, width, chunk_size = 600, 600, 256
        # Height span of 300 exceeds the 256 chunk size and crosses two grid
        # rows while staying in a single grid column.
        self._check(height, width, chunk_size, (0, 0, 300, 10), 'roi-gt-256')

    def test_full_map_region(self):
        height, width, chunk_size = 96, 80, 16
        self._check(height, width, chunk_size, (0, 0, height, width), 'fullmap')

    def test_odd_rectangular_map_edges(self):
        height, width, chunk_size = 513, 769, 128
        arr, info = self._prepare(height, width, chunk_size, name='odd')
        index = _read_index(info['chunked_path'])
        regions = [
            (0, 0, 10, 10),                              # top-left corner
            (height - 5, width - 7, 5, 7),               # bottom-right edge chunk
            (height - 1, 0, 1, width),                   # last row strip
            (0, width - 1, height, 1),                   # last column strip
            (height // 2 - 64, width // 2 - 64, 200, 300),  # spans several interior chunks
        ]
        for region in regions:
            with self.subTest(region=region):
                row, col, h, w = region
                expected = arr[row:row + h, col:col + w]
                expected_coords = _expected_chunk_coords(
                    row, col, h, w, chunk_size, index['rows'], index['cols']
                )
                out, metrics = town_region.load_region(info['chunked_path'], region)
                np.testing.assert_array_equal(out, expected)
                self._assert_owned(out)
                self._assert_metrics_shape(metrics, expected_coords)

    def test_single_pixel_region(self):
        height, width, chunk_size = 40, 40, 16
        self._check(height, width, chunk_size, (0, 0, 1, 1), 'pixel-origin')
        self._check(height, width, chunk_size, (height - 1, width - 1, 1, 1), 'pixel-corner')
        self._check(height, width, chunk_size, (16, 16, 1, 1), 'pixel-boundary')


# --------------------------------------------------------------------------- #
# all codecs, 256 chunk size: 4-way straddle and 513x769 odd-rectangular edges
# --------------------------------------------------------------------------- #

class AllCodecsTests(TownRegionTestCase):
    def test_four_way_roi_all_methods_256_chunk_exact_bytes_and_coords(self):
        height, width, chunk_size = 600, 600, 256
        region = (240, 240, 40, 40)  # straddles all four grid cells at the 256 boundary
        row, col, h, w = region
        for method in METHODS:
            with self.subTest(method=method):
                arr, info = self._prepare(
                    height, width, chunk_size, method=method, name=f'allcodec-fourway-{method}'
                )
                index = _read_index(info['chunked_path'])
                expected_coords = _expected_chunk_coords(row, col, h, w, chunk_size, index['rows'], index['cols'])
                self.assertEqual(sorted(tuple(c) for c in expected_coords), [(0, 0), (0, 1), (1, 0), (1, 1)])

                out, metrics = town_region.load_region(info['chunked_path'], region)

                np.testing.assert_array_equal(out, arr[row:row + h, col:col + w])
                self._assert_metrics_shape(metrics, expected_coords)
                self.assertEqual(metrics['bytes_read'], _expected_selected_bytes(index, expected_coords))

    def test_odd_rectangular_edges_all_methods_256_chunk_exact_bytes_and_coords(self):
        height, width, chunk_size = 513, 769, 256
        regions = [
            (0, 0, 20, 20),                     # top-left corner chunk
            (height - 3, width - 5, 3, 5),       # bottom-right shrunken edge chunk
            (height - 1, 0, 1, width),           # last-row strip across every column chunk
            (0, width - 1, height, 1),           # last-column strip across every row chunk
        ]
        for method in METHODS:
            arr, info = self._prepare(
                height, width, chunk_size, method=method, name=f'allcodec-odd-{method}'
            )
            index = _read_index(info['chunked_path'])
            for region in regions:
                with self.subTest(method=method, region=region):
                    row, col, h, w = region
                    expected_coords = _expected_chunk_coords(
                        row, col, h, w, chunk_size, index['rows'], index['cols']
                    )
                    out, metrics = town_region.load_region(info['chunked_path'], region)
                    np.testing.assert_array_equal(out, arr[row:row + h, col:col + w])
                    self._assert_metrics_shape(metrics, expected_coords)
                    self.assertEqual(
                        metrics['bytes_read'], _expected_selected_bytes(index, expected_coords)
                    )


# --------------------------------------------------------------------------- #
# selected-only I/O: exact byte counts, no over-reading
# --------------------------------------------------------------------------- #

class SelectedOnlyReadTests(TownRegionTestCase):
    def test_bytes_read_equals_header_plus_index_plus_selected_payloads_only(self):
        height, width, chunk_size = 600, 600, 256
        arr, info = self._prepare(height, width, chunk_size, name='bytes')
        index = _read_index(info['chunked_path'])
        region = (240, 240, 40, 40)  # four-way straddle region
        row, col, h, w = region
        expected_coords = _expected_chunk_coords(row, col, h, w, chunk_size, index['rows'], index['cols'])
        self.assertEqual(len(expected_coords), 4)

        _out, metrics = town_region.load_region(info['chunked_path'], region)

        expected_bytes = _expected_selected_bytes(index, expected_coords)
        self.assertEqual(metrics['bytes_read'], expected_bytes)
        # Only 4 of the 9 grid chunks' payloads are read, never all of them.
        self.assertLess(metrics['bytes_read'], info['chunked_bytes'])

    def test_single_chunk_region_reads_only_that_payload(self):
        height, width, chunk_size = 128, 128, 32
        arr, info = self._prepare(height, width, chunk_size, name='single-bytes')
        index = _read_index(info['chunked_path'])
        region = (0, 0, 10, 10)
        expected_coords = _expected_chunk_coords(0, 0, 10, 10, chunk_size, index['rows'], index['cols'])
        self.assertEqual(len(expected_coords), 1)

        _out, metrics = town_region.load_region(info['chunked_path'], region)

        expected_bytes = _expected_selected_bytes(index, expected_coords)
        self.assertEqual(metrics['bytes_read'], expected_bytes)


# --------------------------------------------------------------------------- #
# invalid arguments / out-of-bounds
# --------------------------------------------------------------------------- #

class InvalidArgumentTests(TownRegionTestCase):
    def setUp(self):
        super().setUp()
        self.arr, self.info = self._prepare(64, 64, 16, name='invalid')

    def test_negative_row_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (-1, 0, 4, 4))

    def test_negative_col_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (0, -1, 4, 4))

    def test_zero_height_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (0, 0, 0, 4))

    def test_zero_width_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (0, 0, 4, 0))

    def test_wrong_length_tuple_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (0, 0, 4))
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (0, 0, 4, 4, 4))

    def test_non_integer_values_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (0.5, 0, 4, 4))

    def test_region_exceeds_height_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (60, 0, 10, 4))

    def test_region_exceeds_width_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (0, 60, 4, 10))

    def test_region_entirely_outside_map_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (1000, 1000, 4, 4))

    def test_unsupported_mode_rejected(self):
        with self.assertRaises(ValueError):
            town_region.load_region(self.info['chunked_path'], (0, 0, 4, 4), mode='bogus')


# --------------------------------------------------------------------------- #
# corrupt index / payload for the selected chunk
# --------------------------------------------------------------------------- #

class CorruptSelectedChunkTests(TownRegionTestCase):
    def test_corrupt_index_entry_for_selected_chunk_rejected(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='corrupt-index')
        region = (2, 2, 5, 5)  # entirely inside grid chunk (0, 0)
        index = _read_index(info['chunked_path'])

        data = bytearray(Path(info['chunked_path']).read_bytes())
        row, col, h, w, offset, length = _INDEX_ENTRY.unpack_from(data, index['header_bytes'])
        bad_offset = offset + length + 10_000  # well past the real payload
        patched = _INDEX_ENTRY.pack(row, col, h, w, bad_offset, length)
        data[index['header_bytes']:index['header_bytes'] + _INDEX_ENTRY.size] = patched
        corrupt_path = self.dir / 'corrupt_index.hcix'
        corrupt_path.write_bytes(bytes(data))

        with self.assertRaises(ValueError):
            town_region.load_region(str(corrupt_path), region)

    def test_corrupt_payload_for_selected_chunk_rejected(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='corrupt-payload')
        region = (2, 2, 5, 5)  # entirely inside grid chunk (0, 0)
        index = _read_index(info['chunked_path'])

        data = bytearray(Path(info['chunked_path']).read_bytes())
        _row, _col, _h, _w, offset, length = index['entries'][0]
        payload_start = index['payload_base'] + offset
        # Mangle the chunk container's magic bytes so decode() must reject it.
        data[payload_start:payload_start + 4] = b'\x00\x00\x00\x00'
        corrupt_path = self.dir / 'corrupt_payload.hcix'
        corrupt_path.write_bytes(bytes(data))

        with self.assertRaises(ValueError):
            town_region.load_region(str(corrupt_path), region)

    def test_corrupt_payload_body_bytes_rejected_via_checksum(self):
        '''Flip a byte inside the compressed body (well past the container
        header), not the magic -- decode()'s own sha256-of-samples check,
        not a header-field check, must be what catches this.'''
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='corrupt-body')
        region = (2, 2, 5, 5)  # entirely inside grid chunk (0, 0)
        index = _read_index(info['chunked_path'])

        data = bytearray(Path(info['chunked_path']).read_bytes())
        _row, _col, _h, _w, offset, length = index['entries'][0]
        payload_start = index['payload_base'] + offset
        self.assertGreater(length, HEADER_SIZE + 8, 'chunk payload too small to corrupt its body')
        body_byte = payload_start + HEADER_SIZE + 4  # inside the compressed body, after the header
        data[body_byte] ^= 0xFF
        corrupt_path = self.dir / 'corrupt_body.hcix'
        corrupt_path.write_bytes(bytes(data))

        with self.assertRaises(ValueError):
            town_region.load_region(str(corrupt_path), region)

    def test_truncated_selected_payload_rejected_without_leaking_handle(self):
        '''A file truncated mid-way through the selected chunk's payload must
        be rejected as a short read, and must not leak the open descriptor.'''
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='short-read')
        region = (2, 2, 5, 5)  # entirely inside grid chunk (0, 0)
        index = _read_index(info['chunked_path'])

        data = Path(info['chunked_path']).read_bytes()
        _row, _col, _h, _w, offset, length = index['entries'][0]
        payload_start = index['payload_base'] + offset
        truncated = data[:payload_start + length - 5]  # cut 5 bytes off the selected payload
        truncated_path = self.dir / 'truncated.hcix'
        truncated_path.write_bytes(truncated)

        real_open = open
        opened = []

        class _TrackingFile:
            def __init__(self, fh):
                self._fh = fh
                self.close_calls = 0

            def __getattr__(self, name):
                return getattr(self._fh, name)

            def close(self):
                self.close_calls += 1
                return self._fh.close()

        def tracking_open(*a, **kw):
            fh = _TrackingFile(real_open(*a, **kw))
            opened.append(fh)
            return fh

        with mock.patch('builtins.open', tracking_open):
            with self.assertRaises(ValueError):
                town_region.load_region(str(truncated_path), region)

        self.assertTrue(opened, 'load_region must open the file before failing on a short read')
        for fh in opened:
            self.assertEqual(fh.close_calls, 1, 'a short read must not leak the open file handle')


# --------------------------------------------------------------------------- #
# repeat requests: fresh open every call, no aliasing between results
# --------------------------------------------------------------------------- #

class RepeatRequestTests(TownRegionTestCase):
    def test_repeated_calls_open_fresh_and_return_independent_arrays(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='repeat')
        region = (12, 12, 8, 8)

        real_open = open
        open_calls = []

        def spy_open(*a, **kw):
            fh = real_open(*a, **kw)
            open_calls.append(a[0] if a else kw.get('file'))
            return fh

        with mock.patch('builtins.open', spy_open):
            out1, metrics1 = town_region.load_region(info['chunked_path'], region)
            first_open_count = len(open_calls)
            self.assertGreater(first_open_count, 0)
            out2, metrics2 = town_region.load_region(info['chunked_path'], region)
            second_open_count = len(open_calls) - first_open_count

        # Each call must independently open the file; the second call is not
        # served from any handle or cache left over from the first.
        self.assertEqual(second_open_count, first_open_count)

        np.testing.assert_array_equal(out1, out2)
        self.assertFalse(np.shares_memory(out1, out2))
        self.assertIsNot(out1, out2)
        self.assertEqual(metrics1['bytes_read'], metrics2['bytes_read'])
        self.assertEqual(metrics1['chunks_loaded'], metrics2['chunks_loaded'])

    @unittest.skipUnless(IS_DARWIN, 'macos_nocache is Darwin-only')
    def test_macos_nocache_mode_matches_warm_buffered(self):
        height, width, chunk_size = 48, 48, 16
        arr, info = self._prepare(height, width, chunk_size, name='nocache')
        region = (4, 4, 10, 10)
        expected = arr[4:14, 4:14]

        out_warm, _m1 = town_region.load_region(info['chunked_path'], region, mode='warm_buffered')
        out_cold, _m2 = town_region.load_region(info['chunked_path'], region, mode='macos_nocache')

        np.testing.assert_array_equal(out_warm, expected)
        np.testing.assert_array_equal(out_cold, expected)


if __name__ == '__main__':
    unittest.main()

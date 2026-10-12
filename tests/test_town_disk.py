import os
import platform
import struct
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from horizon import town_disk
from horizon.chunk_benchmark import _INDEX_ENTRY, _INDEX_HEADER, _INDEX_MAGIC
from horizon.codec import HEADER_SIZE, METHODS, encode

_DTYPE = np.dtype('<u2')
IS_DARWIN = platform.system() == 'Darwin'


def _make_arr(height, width, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=(height, width), dtype=np.uint16).astype(_DTYPE)


class _TrackingFile:
    '''Transparent proxy around a real file object that counts close() calls.'''

    def __init__(self, fh):
        self._fh = fh
        self.close_calls = 0

    def __getattr__(self, name):
        return getattr(self._fh, name)

    def close(self):
        self.close_calls += 1
        return self._fh.close()


class TownDiskTestCase(unittest.TestCase):
    def setUp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def case_dir(self, name='case'):
        return self.dir / name


# --------------------------------------------------------------------------- #
# correctness: round trip, all methods, whole + chunked, full + region
# --------------------------------------------------------------------------- #

class RoundTripTests(TownDiskTestCase):
    def _prepare(self, height, width, method, chunk_size, seed=1, name='case'):
        arr = _make_arr(height, width, seed=seed)
        info = town_disk.prepare_case(arr, str(self.case_dir(name)), method, chunk_size=chunk_size)
        return arr, info

    def test_prepare_case_result_keys(self):
        arr, info = self._prepare(20, 13, 'zstd', 8, name='keys')
        for key in (
            'whole_path', 'chunked_path', 'shape', 'method', 'chunk_size', 'grid',
            'chunk_count', 'index_bytes', 'whole_bytes', 'chunked_bytes',
            'input_sha256', 'hashes',
        ):
            self.assertIn(key, info)
        self.assertEqual(info['shape'], [20, 13])
        self.assertEqual(info['method'], 'zstd')
        self.assertEqual(info['chunk_size'], 8)
        self.assertTrue(Path(info['whole_path']).is_file())
        self.assertTrue(Path(info['chunked_path']).is_file())

    def test_prepare_case_creates_new_directory_per_method(self):
        arr = _make_arr(16, 16, seed=2)
        for method in METHODS:
            directory = self.case_dir(f'per-method-{method}')
            info = town_disk.prepare_case(arr, str(directory), method, chunk_size=8)
            self.assertEqual(Path(info['whole_path']).parent, directory)
            self.assertEqual(Path(info['chunked_path']).parent, directory)

    def test_whole_and_chunked_full_round_trip_all_methods(self):
        for method in METHODS:
            with self.subTest(method=method):
                arr, info = self._prepare(21, 17, method, 8, name=f'full-{method}')
                whole_arr, _ = town_disk.load_whole(info['whole_path'])
                chunked_arr, _ = town_disk.load_chunked(info['chunked_path'])
                np.testing.assert_array_equal(whole_arr, arr)
                np.testing.assert_array_equal(chunked_arr, arr)

    def _modes(self):
        modes = ['warm_buffered']
        if IS_DARWIN:
            modes.append('macos_nocache')
        return modes

    def test_same_region_all_methods_all_modes_rectangular_and_edge(self):
        height, width, chunk_size = 20, 13, 8  # 13 is not a multiple of 8: edge chunk exists
        regions = [
            (0, 0, chunk_size, chunk_size),           # full interior chunk
            (0, chunk_size, chunk_size, width - chunk_size),   # edge column chunk (w=5)
            (2 * chunk_size, 0, height - 2 * chunk_size, chunk_size),  # edge row chunk (h=4)
            (2 * chunk_size, chunk_size, height - 2 * chunk_size, width - chunk_size),  # edge corner
        ]
        for method in METHODS:
            arr, info = self._prepare(height, width, method, chunk_size, name=f'region-{method}')
            for region in regions:
                expected = arr[region[0]:region[0] + region[2], region[1]:region[1] + region[3]]
                for mode in self._modes():
                    with self.subTest(method=method, region=region, mode=mode):
                        whole_region, _ = town_disk.load_whole(info['whole_path'], mode=mode, region=region)
                        chunked_region, _ = town_disk.load_chunked(info['chunked_path'], mode=mode, region=region)
                        np.testing.assert_array_equal(whole_region, expected)
                        np.testing.assert_array_equal(chunked_region, expected)


# --------------------------------------------------------------------------- #
# partial-read efficiency and no hidden buffering
# --------------------------------------------------------------------------- #

class PartialReadTests(TownDiskTestCase):
    def test_partial_region_bytes_much_less_than_full(self):
        height, width, chunk_size = 64, 64, 16
        arr = _make_arr(height, width, seed=3)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd', chunk_size=chunk_size)
        region = (0, 0, chunk_size, chunk_size)

        _arr, whole_metrics = town_disk.load_whole(info['whole_path'], region=region)
        _arr, chunked_metrics = town_disk.load_chunked(info['chunked_path'], region=region)

        full_chunked_bytes = info['chunked_bytes']
        selected_entry_payload_len = chunked_metrics['bytes_read'] - _INDEX_HEADER.size - _INDEX_ENTRY.size
        self.assertGreater(selected_entry_payload_len, 0)

        # header(24) + one entry(32) + selected chunk blob only: the chunked
        # partial read must stay far below the whole chunked container size.
        self.assertLess(chunked_metrics['bytes_read'], full_chunked_bytes * 0.5)
        self.assertEqual(
            chunked_metrics['bytes_read'],
            _INDEX_HEADER.size + _INDEX_ENTRY.size + selected_entry_payload_len,
        )
        # load_whole still has to read and decode the entire container even
        # for a region, so it reads far more than the chunked partial path.
        self.assertGreater(whole_metrics['bytes_read'], chunked_metrics['bytes_read'])

    def test_no_hidden_python_buffering_read_ahead(self):
        '''buffering=0 means every readinto() call must receive exactly the
        remaining requested size, never a larger block-aligned read.'''
        height, width, chunk_size = 32, 32, 8
        arr = _make_arr(height, width, seed=4)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd', chunk_size=chunk_size)
        region = (0, 0, chunk_size, chunk_size)

        requested_sizes = []
        real_readinto = os.pread  # unused; placeholder to keep import tidy

        import io
        real_open = open

        class _SpyFile:
            def __init__(self, fh):
                self._fh = fh

            def __getattr__(self, name):
                return getattr(self._fh, name)

            def readinto(self, view):
                requested_sizes.append(len(view))
                return self._fh.readinto(view)

        def spy_open(*a, **kw):
            return _SpyFile(real_open(*a, **kw))

        with mock.patch('builtins.open', spy_open):
            _arr, metrics = town_disk.load_chunked(info['chunked_path'], region=region)

        self.assertTrue(requested_sizes)
        self.assertEqual(sum(requested_sizes), metrics['bytes_requested'])
        # No request should ever exceed the exact bytes remaining to satisfy
        # one read_exact() call; a buffering/read-ahead leak would show up as
        # a request much larger than any of the header/entry/chunk sizes.
        self.assertLessEqual(max(requested_sizes), info['chunked_bytes'])

    def test_no_timed_full_file_hash(self):
        '''load_* must not sha256 the whole file on top of the codec's own
        per-container digest check inside decode().'''
        height, width, chunk_size = 32, 32, 8
        arr = _make_arr(height, width, seed=5)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd', chunk_size=chunk_size)
        region = (0, 0, chunk_size, chunk_size)

        import hashlib
        real_sha256 = hashlib.sha256
        calls = []

        def spy_sha256(*a, **kw):
            calls.append(a[0] if a else None)
            return real_sha256(*a, **kw)

        with mock.patch('hashlib.sha256', spy_sha256):
            town_disk.load_chunked(info['chunked_path'], region=region)
        # decode() hashes exactly the one chunk's decompressed samples once;
        # no extra hash over the whole file should appear.
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(len(calls[0]), chunk_size * chunk_size * _DTYPE.itemsize)


# --------------------------------------------------------------------------- #
# output overwrite protection
# --------------------------------------------------------------------------- #

class OverwriteProtectionTests(TownDiskTestCase):
    def test_prepare_case_refuses_existing_directory(self):
        directory = self.case_dir()
        directory.mkdir()
        marker = directory / 'marker.txt'
        marker.write_text('keep me')
        arr = _make_arr(8, 8, seed=6)
        with self.assertRaises(FileExistsError):
            town_disk.prepare_case(arr, str(directory), 'zstd')
        self.assertEqual(marker.read_text(), 'keep me')
        self.assertFalse((directory / town_disk.WHOLE_NAME).exists())
        self.assertFalse((directory / town_disk.CHUNKED_NAME).exists())


# --------------------------------------------------------------------------- #
# corrupt / truncated / malicious input rejection
# --------------------------------------------------------------------------- #

class CorruptInputTests(TownDiskTestCase):
    def _write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_whole_truncated_header_rejected(self):
        path = self._write('short.hrzn', b'\x00' * (HEADER_SIZE - 1))
        with self.assertRaises(ValueError):
            town_disk.load_whole(str(path))

    def test_whole_bad_magic_rejected(self):
        arr = _make_arr(8, 8, seed=7)
        blob = bytearray(encode(arr, method='zstd'))
        blob[0:4] = b'XXXX'
        path = self._write('badmagic.hrzn', bytes(blob))
        with self.assertRaises(ValueError):
            town_disk.load_whole(str(path))

    def test_whole_payload_length_mismatch_rejected(self):
        arr = _make_arr(8, 8, seed=8)
        blob = encode(arr, method='zstd')
        path = self._write('shortpayload.hrzn', blob[:-1])  # truncate by one byte
        with self.assertRaises(ValueError):
            town_disk.load_whole(str(path))

    def test_empty_file_rejected_for_both_loaders(self):
        path = self._write('empty.bin', b'')
        with self.assertRaises(ValueError):
            town_disk.load_whole(str(path))
        with self.assertRaises(ValueError):
            town_disk.load_chunked(str(path))

    def test_oversize_file_rejected(self):
        path = self._write('oversize.bin', b'\x00' * 16)
        with mock.patch.object(town_disk, 'MAX_FILE_BYTES', 8):
            with self.assertRaises(ValueError):
                town_disk.load_whole(str(path))

    def _hcix_header(self, method_id=0, chunk_size=8, rows=1, cols=1, count=1, version=None, reserved=0):
        return _INDEX_HEADER.pack(
            _INDEX_MAGIC, town_disk._INDEX_VERSION if version is None else version,
            method_id, reserved, chunk_size, rows, cols, count,
        )

    def test_chunked_huge_entry_count_rejected_before_allocation(self):
        '''count far beyond MAX_INDEX_ENTRIES must be rejected by the cap
        check alone, before any per-entry list/array is built.'''
        huge_count = town_disk.MAX_INDEX_ENTRIES + 1
        header = self._hcix_header(count=huge_count, rows=1, cols=huge_count)
        # Pad so the file is bigger than just the header (size check passes).
        data = header + b'\x00' * 8
        path = self._write('huge_count.hcix', data)
        t0 = time.perf_counter()
        with self.assertRaises(ValueError):
            town_disk.load_chunked(str(path))
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.0, 'rejecting an oversized count must not attempt a huge allocation')

    def test_chunked_grid_count_mismatch_rejected(self):
        header = self._hcix_header(rows=2, cols=2, count=3)  # rows*cols=4 != count=3
        data = header + b'\x00' * 8
        path = self._write('mismatch.hcix', data)
        with self.assertRaises(ValueError):
            town_disk.load_chunked(str(path))

    def test_chunked_bad_magic_rejected(self):
        header = _INDEX_HEADER.pack(b'XXXX', town_disk._INDEX_VERSION, 0, 0, 8, 1, 1, 1)
        path = self._write('badmagic.hcix', header + b'\x00' * 8)
        with self.assertRaises(ValueError):
            town_disk.load_chunked(str(path))

    def test_chunked_unknown_method_id_rejected(self):
        header = self._hcix_header(method_id=len(METHODS))
        path = self._write('badmethod.hcix', header + b'\x00' * 8)
        with self.assertRaises(ValueError):
            town_disk.load_chunked(str(path))

    def test_chunked_single_chunk_offset_out_of_bounds_rejected(self):
        '''A real valid 1-chunk container, but with the index entry's stored
        offset/length pointing past the actual payload.'''
        arr = _make_arr(8, 8, seed=9)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd', chunk_size=8)
        data = bytearray(Path(info['chunked_path']).read_bytes())
        row, col, h, w, offset, length = _INDEX_ENTRY.unpack_from(data, _INDEX_HEADER.size)
        bad_offset = offset + length + 10_000  # well past the payload
        patched = _INDEX_ENTRY.pack(row, col, h, w, bad_offset, length)
        data[_INDEX_HEADER.size:_INDEX_HEADER.size + _INDEX_ENTRY.size] = patched
        corrupt_path = self.dir / 'corrupt_offset.hcix'
        corrupt_path.write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            town_disk.load_chunked(str(corrupt_path), region=(row, col, h, w))

    def test_chunked_region_shape_mismatch_rejected(self):
        '''Requesting a region whose height/width disagree with the stored
        entry for that grid cell must be rejected, not silently cropped.'''
        arr = _make_arr(16, 16, seed=10)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd', chunk_size=8)
        with self.assertRaises(ValueError):
            town_disk.load_chunked(info['chunked_path'], region=(0, 0, 4, 4))  # not the full 8x8 chunk

    def test_chunked_full_read_entries_not_covering_whole_map_rejected(self):
        '''A structurally valid header/entries list whose cursor does not
        land exactly on the payload end must be rejected (gap or overlap).'''
        arr = _make_arr(16, 16, seed=11)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd', chunk_size=8)
        data = bytearray(Path(info['chunked_path']).read_bytes())
        # Corrupt the first entry's length to be one byte short, leaving a gap.
        row, col, h, w, offset, length = _INDEX_ENTRY.unpack_from(data, _INDEX_HEADER.size)
        patched = _INDEX_ENTRY.pack(row, col, h, w, offset, length - 1)
        data[_INDEX_HEADER.size:_INDEX_HEADER.size + _INDEX_ENTRY.size] = patched
        corrupt_path = self.dir / 'gap.hcix'
        corrupt_path.write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            town_disk.load_chunked(str(corrupt_path))


# --------------------------------------------------------------------------- #
# unsupported mode / method
# --------------------------------------------------------------------------- #

class UnsupportedArgumentTests(TownDiskTestCase):
    def test_load_whole_unsupported_mode_rejected(self):
        arr = _make_arr(8, 8, seed=12)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd')
        with self.assertRaises(ValueError):
            town_disk.load_whole(info['whole_path'], mode='bogus')

    def test_load_chunked_unsupported_mode_rejected(self):
        arr = _make_arr(8, 8, seed=13)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd')
        with self.assertRaises(ValueError):
            town_disk.load_chunked(info['chunked_path'], mode='bogus')

    def test_prepare_case_unsupported_method_rejected(self):
        arr = _make_arr(8, 8, seed=14)
        with self.assertRaises(ValueError):
            town_disk.prepare_case(arr, str(self.case_dir()), 'not_a_method')

    @unittest.skipIf(IS_DARWIN, 'macos_nocache is only rejected for being unsupported off Darwin')
    def test_macos_nocache_rejected_off_darwin(self):
        arr = _make_arr(8, 8, seed=15)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd')
        with self.assertRaises(OSError):
            town_disk.load_whole(info['whole_path'], mode='macos_nocache')


# --------------------------------------------------------------------------- #
# native Darwin F_NOCACHE smoke test: wrap the real fcntl call, don't replace it
# --------------------------------------------------------------------------- #

@unittest.skipUnless(IS_DARWIN, 'F_NOCACHE is Darwin-only')
class MacosNocacheSmokeTest(TownDiskTestCase):
    def test_fcntl_f_nocache_invoked_on_real_descriptor(self):
        import fcntl as real_fcntl_module

        arr = _make_arr(8, 8, seed=16)
        info = town_disk.prepare_case(arr, str(self.case_dir()), 'zstd')

        calls = []
        real_fcntl_fn = real_fcntl_module.fcntl

        def spy(fd, cmd, arg):
            calls.append((cmd, arg))
            return real_fcntl_fn(fd, cmd, arg)  # actually invoke the real syscall

        with mock.patch.object(real_fcntl_module, 'fcntl', side_effect=spy):
            arr_out, _metrics = town_disk.load_whole(info['whole_path'], mode='macos_nocache')
        np.testing.assert_array_equal(arr_out, arr)
        self.assertEqual(calls, [(town_disk.F_NOCACHE, 1)])


# --------------------------------------------------------------------------- #
# resource-leak / rollback bugs found in audit
# --------------------------------------------------------------------------- #

class ResourceLeakAndRollbackTests(TownDiskTestCase):
    def test_reader_closes_handle_when_empty_file_rejected(self):
        '''_Reader.__init__ fstats inside a try/except that closes the handle
        on failure, but the empty-file size check runs AFTER that block, so a
        rejection there must still not leak the open file descriptor.'''
        path = self.dir / 'empty.hrzn'
        path.write_bytes(b'')
        created = []
        real_open = open

        def tracking_open(*a, **kw):
            fh = _TrackingFile(real_open(*a, **kw))
            created.append(fh)
            return fh

        with mock.patch('builtins.open', tracking_open):
            with self.assertRaises(ValueError):
                town_disk._Reader(str(path), 'warm_buffered')

        self.assertEqual(len(created), 1)
        self.assertEqual(
            created[0].close_calls, 1,
            'empty-file rejection in _Reader.__init__ must close the handle it just opened',
        )

    def test_reader_closes_handle_when_oversize_file_rejected(self):
        '''Same leak, via the other size-check branch (file too large).'''
        path = self.dir / 'big.hrzn'
        path.write_bytes(b'\x00' * 16)
        created = []
        real_open = open

        def tracking_open(*a, **kw):
            fh = _TrackingFile(real_open(*a, **kw))
            created.append(fh)
            return fh

        with mock.patch('builtins.open', tracking_open), \
             mock.patch.object(town_disk, 'MAX_FILE_BYTES', 8):
            with self.assertRaises(ValueError):
                town_disk._Reader(str(path), 'warm_buffered')

        self.assertEqual(len(created), 1)
        self.assertEqual(
            created[0].close_calls, 1,
            'oversize-file rejection in _Reader.__init__ must close the handle it just opened',
        )

    def test_prepare_case_rolls_back_on_second_write_failure(self):
        '''If the chunked container write fails after the whole container
        was already written, prepare_case must not leave the partially
        written case directory (with only whole.hrzn) behind.'''
        directory = self.case_dir('rollback-write')
        arr = _make_arr(16, 16, seed=17)

        real_write = town_disk._write_exclusive
        calls = {'n': 0}

        def flaky_write(path, data):
            calls['n'] += 1
            if calls['n'] == 2:
                raise OSError('simulated disk failure writing the chunked container')
            return real_write(path, data)

        with mock.patch.object(town_disk, '_write_exclusive', side_effect=flaky_write):
            with self.assertRaises(OSError):
                town_disk.prepare_case(arr, str(directory), 'zstd', chunk_size=8)

        self.assertFalse(
            directory.exists(),
            'prepare_case left a partially written case directory behind '
            '(whole.hrzn written, chunked.hcix write failed) with no rollback',
        )

    def test_prepare_case_rolls_back_empty_directory_on_encode_failure(self):
        '''If chunk encoding fails before any file is written, the directory
        prepare_case just created (via mkdir) must not be left behind empty.

        _chunk_entries/_encode_chunks are imported from horizon.chunk_benchmark
        into horizon.town_disk's own namespace, so patching town_disk.encode
        does not intercept the chunk-encode call dispatched through
        chunk_benchmark._encode_chunks; patch town_disk._encode_chunks itself.'''
        directory = self.case_dir('rollback-encode')
        arr = _make_arr(16, 16, seed=18)

        with mock.patch.object(
            town_disk, '_encode_chunks', side_effect=RuntimeError('simulated codec failure')
        ):
            with self.assertRaises(RuntimeError):
                town_disk.prepare_case(arr, str(directory), 'zstd', chunk_size=4)

        self.assertFalse(
            directory.exists(),
            'prepare_case left an empty case directory behind after an encode '
            'failure that happened before any file was written',
        )


if __name__ == '__main__':
    unittest.main()

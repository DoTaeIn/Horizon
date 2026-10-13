'''Tests for the upcoming horizon.town_cache module: ChunkRegionCache (an LRU
chunk cache over a chunked HCIX container) and WholeMapResident (a fully
resident whole-map container), per the agreed contract below.

horizon.town_cache is a required import: if it is missing or broken, this
module must fail loudly (an ImportError/collection error), not swallow it as
a skip -- that would hide a broken or absent implementation.

Assumed contract (implementation does not exist yet; these tests pin it down):

  ChunkRegionCache(path, max_chunks=8, mode='warm_buffered')
  WholeMapResident(path, mode)

  Both are context managers (`with cache: ...`) whose close() is idempotent
  (calling it more than once, with or without the `with` block, never raises).

  Both expose:
    .load(region) -> (array, metrics)
        array: an owned (OWNDATA or base is None), C-contiguous <u2 array of
               exactly the requested (row, col, height, width) ROI.
        metrics: dict with the standard town_disk-style keys (total_ns,
               read_ns, decode_ns, bytes_read, bytes_requested, read_calls,
               seek_calls) scoped to this call, plus 'resident_bytes' -- the
               object's .resident_bytes snapshot taken after this load.
    .startup_metrics
        dict with the same standard total/read/decode_ns + byte/read-counter
        keys, populated once during __init__ and never mutated afterwards.
    .resident_bytes
        the exact total nbytes of the decoded numpy arrays currently retained
        in memory (cached chunks, or the one resident whole-map array) --
        excludes any output ROI array returned by load() and any Python/dict
        bookkeeping overhead.

  ChunkRegionCache.load()'s metrics dict is scoped to that one call only (the
  object itself keeps no cumulative counters beyond .resident_bytes and
  .resident_chunks); each call's metrics additionally carries:
    .chunks_requested    grid cells this call's ROI touched (hits + misses)
    .chunks_loaded        cells whose blob had to be read this call (== cache_misses)
    .cache_hits           cells this call served from the in-memory cache
    .cache_misses         cells this call had to read from disk
    .evictions             chunks this call dropped to stay within max_chunks
    .chunk_coordinates    [[grid_row, grid_col], ...] every cell this call
                         selected, in visit order
    .hit_coordinates      the subsequence of chunk_coordinates that were hits
    .miss_coordinates     the subsequence of chunk_coordinates that were misses
    .evicted_coordinates  [[grid_row, grid_col], ...] chunks this call
                         evicted, in eviction order

  ChunkRegionCache reads the HCIX header and the full index exactly once, in
  __init__, and keeps one unbuffered (buffering=0) file descriptor open for
  its whole lifetime -- load() never reopens the file. A cache hit for a
  chunk costs 0 payload bytes read; a cache miss reads exactly that chunk's
  stored payload length, nothing more. Eviction is strict LRU.

  Within one load() call, a straddled ROI's cells that are *already* cached
  are processed and stitched first, in grid row-major order, each becoming
  most-recently-used immediately -- only then are the cells that were
  initially missing processed, in grid row-major order, each one evicting the
  current LRU chunk (if at capacity) right before it is decoded. This two-pass
  "hits first" order exists to avoid an avoidable thrash: with pure row-major
  processing (hits and misses interleaved in one pass), a miss visited before
  a same-call hit could evict that hit's chunk before the hit's own turn in
  the scan, turning it into a miss too -- a chunk the cache already had got
  evicted and immediately re-read inside the very call that would have been a
  hit for it. Marking every already-cached cell MRU before any eviction from
  this call happens removes that interaction entirely.

  total_ns is the actual measured wall-clock elapsed of load()'s core work
  (not a sum of a hand-picked subset of spans): it covers argument-independent
  work done after bounds-checking, including LRU bookkeeping (cache
  hit/evict dict operations), output-buffer allocation and stitching, not just
  I/O and decode(). read_ns is the sum of this call's I/O spans only (the
  per-load snapshot fstat plus each miss's payload read); decode_ns is defined
  as total_ns - read_ns, so it is the catch-all for everything else the call
  did -- LRU dict bookkeeping, eviction, np.empty allocation, decode() and
  stitching -- and total_ns == read_ns + decode_ns holds by construction, not
  by measuring each independently.

  Each load() re-fstats the cache's own pinned descriptor and fails closed
  (raises) if the file has been modified in place (size, mtime or ctime
  differs from what __init__ observed). Replacing the path with a brand new
  file (os.replace) does not change the already-open descriptor's inode, so
  the cache keeps serving the original snapshot -- that is acceptable, not a
  failure; in practice the unlink inside os.replace usually also changes the
  original inode's ctime, so a fail-closed raise instead is an equally
  acceptable, documented outcome.

  WholeMapResident reads and decodes the whole-map container exactly once, in
  __init__; every load() after that only crops an owned copy out of the
  resident decoded array and performs no file I/O at all. Because of this, the
  underlying file may even be deleted after construction and load() still
  works -- it is reading from memory, not disk.

  max_chunks must be a positive int (bool is explicitly rejected even though
  it is an int subclass).
'''

from __future__ import annotations

import os
import platform
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from horizon import town_cache, town_disk
from horizon.chunk_benchmark import _INDEX_ENTRY, _INDEX_HEADER
from horizon.codec import METHODS

IS_DARWIN = platform.system() == 'Darwin'
_DTYPE = np.dtype('<u2')

_STANDARD_METRIC_KEYS = (
    'total_ns', 'read_ns', 'decode_ns',
    'bytes_read', 'bytes_requested', 'read_calls', 'seek_calls',
)

# A deterministic, injected cost (sleep) used to prove that non-I/O work
# (allocation, LRU dict bookkeeping) is actually counted into decode_ns,
# instead of asserting a noisy wall-clock ratio. sleep() only guarantees a
# *lower* bound on elapsed time, so the assertion threshold is comfortably
# below the injected amount to absorb scheduler jitter without being able to
# pass by coincidence on an un-instrumented, effectively-instant code path.
_INJECTED_SLEEP_S = 0.05
_INJECTED_SLEEP_NS = int(_INJECTED_SLEEP_S * 1e9)
_MIN_OBSERVED_NS = int(_INJECTED_SLEEP_NS * 0.5)


def _make_arr(height, width, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=(height, width), dtype=np.uint16).astype(_DTYPE)


def _read_index(path):
    '''Parse the HCIX header/entries directly from disk, independent of
    town_cache, to compute expected chunk layouts/byte counts/shapes.'''
    data = Path(path).read_bytes()
    _magic, _version, _method_id, _reserved, chunk_size, rows, cols, count = (
        _INDEX_HEADER.unpack_from(data)
    )
    base = _INDEX_HEADER.size
    entries = {}
    for i in range(count):
        row, col, h, w, offset, length = _INDEX_ENTRY.unpack_from(data, base + i * _INDEX_ENTRY.size)
        entries[(i // cols, i % cols)] = (int(row), int(col), int(h), int(w), int(offset), int(length))
    payload_base = base + count * _INDEX_ENTRY.size
    return {
        'chunk_size': int(chunk_size),
        'rows': int(rows),
        'cols': int(cols),
        'count': int(count),
        'entries': entries,
        'header_bytes': base,
        'payload_base': payload_base,
        'index_bytes': base + count * _INDEX_ENTRY.size,
    }


def _chunk_bytes(index, grid_row, grid_col):
    _row, _col, h, w, _off, _len = index['entries'][(grid_row, grid_col)]
    return h * w * _DTYPE.itemsize


def _grid_cells(row, col, h, w, chunk_size, rows, cols):
    r0, r1 = row // chunk_size, (row + h - 1) // chunk_size
    c0, c1 = col // chunk_size, (col + w - 1) // chunk_size
    cells = []
    for gr in range(min(r0, rows - 1), min(r1, rows - 1) + 1):
        for gc in range(min(c0, cols - 1), min(c1, cols - 1) + 1):
            cells.append((gr, gc))
    return cells


class TownCacheTestCase(unittest.TestCase):
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

    def _assert_standard_metrics(self, metrics):
        for key in _STANDARD_METRIC_KEYS:
            self.assertIn(key, metrics, f'metrics missing {key!r}: {metrics!r}')
        self.assertEqual(metrics['total_ns'], metrics['read_ns'] + metrics['decode_ns'])
        self.assertIn('resident_bytes', metrics)


# --------------------------------------------------------------------------- #
# ChunkRegionCache: correctness across region geometries, all codecs
# --------------------------------------------------------------------------- #

class ChunkCacheCorrectnessTests(TownCacheTestCase):
    def _check(self, height, width, chunk_size, region, name, seed=1, max_chunks=8):
        arr, info = self._prepare(height, width, chunk_size, seed=seed, name=name)
        row, col, h, w = region
        expected = arr[row:row + h, col:col + w]
        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=max_chunks) as cache:
            out, metrics = cache.load(region)
        np.testing.assert_array_equal(out, expected)
        self._assert_owned(out)
        self._assert_standard_metrics(metrics)
        return out, metrics, info

    def test_single_chunk_region(self):
        self._check(64, 64, 16, (2, 2, 5, 5), 'single')

    def test_four_way_boundary_region(self):
        self._check(64, 64, 16, (12, 12, 8, 8), 'fourway')

    def test_odd_rectangular_map_edges(self):
        height, width, chunk_size = 513, 769, 128
        arr, info = self._prepare(height, width, chunk_size, name='odd')
        regions = [
            (0, 0, 10, 10),                                  # top-left corner
            (height - 5, width - 7, 5, 7),                   # bottom-right edge chunk
            (height - 1, 0, 1, width),                       # last row strip
            (0, width - 1, height, 1),                       # last column strip
            (height // 2 - 64, width // 2 - 64, 200, 300),   # spans several interior chunks
        ]
        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=32) as cache:
            for region in regions:
                with self.subTest(region=region):
                    row, col, h, w = region
                    expected = arr[row:row + h, col:col + w]
                    out, metrics = cache.load(region)
                    np.testing.assert_array_equal(out, expected)
                    self._assert_owned(out)
                    self._assert_standard_metrics(metrics)

    def test_all_codecs_four_way_and_odd_edges(self):
        for method in METHODS:
            with self.subTest(method=method):
                height, width, chunk_size = 513, 769, 256
                arr, info = self._prepare(
                    height, width, chunk_size, method=method, name=f'allcodec-{method}'
                )
                regions = [
                    (240, 240, 40, 40),               # straddles all four grid cells
                    (0, 0, 20, 20),                    # top-left corner chunk
                    (height - 3, width - 5, 3, 5),      # bottom-right shrunken edge chunk
                ]
                with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=32) as cache:
                    for region in regions:
                        row, col, h, w = region
                        out, _metrics = cache.load(region)
                        np.testing.assert_array_equal(out, arr[row:row + h, col:col + w])


# --------------------------------------------------------------------------- #
# ChunkRegionCache: hit/miss byte accounting
# --------------------------------------------------------------------------- #

class ChunkCacheByteAccountingTests(TownCacheTestCase):
    def test_miss_reads_exact_index_parsed_payload_length(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='missbytes')
        index = _read_index(info['chunked_path'])
        region = (2, 2, 5, 5)  # inside grid chunk (0, 0) only
        expected_payload = _chunk_bytes(index, 0, 0)
        # _chunk_bytes gives decoded size; the on-disk payload length is what
        # must be read, so pull it directly from the parsed entry instead.
        payload_len = index['entries'][(0, 0)][5]

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=4) as cache:
            _out, metrics = cache.load(region)

        self.assertEqual(metrics['bytes_read'], payload_len)
        self.assertEqual(metrics['cache_misses'], 1)
        self.assertEqual(metrics['cache_hits'], 0)
        self.assertEqual(metrics['chunks_loaded'], 1)
        self.assertEqual(metrics['miss_coordinates'], [[0, 0]])

    def test_repeated_identical_load_is_a_pure_hit_with_zero_bytes(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='hitbytes')
        region = (2, 2, 5, 5)

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=4) as cache:
            out1, metrics1 = cache.load(region)
            out2, metrics2 = cache.load(region)

        np.testing.assert_array_equal(out1, out2)
        self.assertFalse(np.shares_memory(out1, out2))
        self.assertGreater(metrics1['bytes_read'], 0)
        self.assertEqual(metrics1['cache_misses'], 1)
        self.assertEqual(metrics2['bytes_read'], 0)
        self.assertEqual(metrics2['cache_misses'], 0)
        self.assertEqual(metrics2['cache_hits'], 1)
        self.assertEqual(metrics2['hit_coordinates'], [[0, 0]])

    def test_four_way_straddle_misses_bytes_match_index_sum(self):
        height, width, chunk_size = 600, 600, 256
        arr, info = self._prepare(height, width, chunk_size, name='fourwaybytes')
        index = _read_index(info['chunked_path'])
        region = (240, 240, 40, 40)
        cells = _grid_cells(240, 240, 40, 40, chunk_size, index['rows'], index['cols'])
        self.assertEqual(len(cells), 4)
        expected_bytes = sum(index['entries'][c][5] for c in cells)

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=8) as cache:
            _out, metrics = cache.load(region)

        self.assertEqual(metrics['bytes_read'], expected_bytes)
        self.assertEqual(metrics['chunks_requested'], 4)
        self.assertEqual(metrics['cache_misses'], 4)
        self.assertEqual(sorted(map(tuple, metrics['chunk_coordinates'])), sorted(cells))


# --------------------------------------------------------------------------- #
# ChunkRegionCache: deterministic LRU eviction, backtracking, capacity 1
# --------------------------------------------------------------------------- #

class LRUEvictionTests(TownCacheTestCase):
    def _single_chunk_region(self, grid_col, chunk_size):
        return (0, grid_col * chunk_size, chunk_size, chunk_size)

    def test_sequential_row_major_lru_eviction_order(self):
        # A 1 x 4 grid of chunks; max_chunks=2 forces eviction after the 3rd
        # distinct chunk is touched. Each load() touches exactly one chunk,
        # so the eviction order is unambiguous regardless of any in-call
        # chunk-iteration order town_cache might use internally.
        chunk_size = 16
        height, width = chunk_size, chunk_size * 4
        arr, info = self._prepare(height, width, chunk_size, name='lru-sequential')

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=2) as cache:
            _o, m = cache.load(self._single_chunk_region(0, chunk_size))  # miss: cache={0}
            self.assertEqual((m['cache_misses'], m['evictions']), (1, 0))
            self.assertEqual(cache.resident_chunks, [[0, 0]])

            _o, m = cache.load(self._single_chunk_region(1, chunk_size))  # miss: cache={0,1}
            self.assertEqual((m['cache_misses'], m['evictions']), (1, 0))
            self.assertEqual(cache.resident_chunks, [[0, 0], [0, 1]])

            _o, m = cache.load(self._single_chunk_region(2, chunk_size))  # miss, evicts 0 (LRU)
            self.assertEqual(m['evicted_coordinates'], [[0, 0]])
            self.assertEqual(cache.resident_chunks, [[0, 1], [0, 2]])

            _o, m = cache.load(self._single_chunk_region(0, chunk_size))  # miss again (was evicted)
            self.assertEqual(m['cache_misses'], 1)
            self.assertEqual(m['evicted_coordinates'], [[0, 1]])  # 1 was now LRU
            self.assertEqual(cache.resident_chunks, [[0, 2], [0, 0]])

            _o, m = cache.load(self._single_chunk_region(0, chunk_size))  # hit: just reloaded
            self.assertEqual(m['cache_hits'], 1)
            self.assertEqual(m['evicted_coordinates'], [])  # a hit evicts nothing
            self.assertEqual(cache.resident_chunks, [[0, 2], [0, 0]])  # 0 is now most-recent

    def test_backtracking_to_still_cached_chunk_is_a_hit(self):
        chunk_size = 16
        height, width = chunk_size, chunk_size * 3
        arr, info = self._prepare(height, width, chunk_size, name='lru-backtrack')

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=3) as cache:
            for i in range(3):
                _o, m = cache.load(self._single_chunk_region(i, chunk_size))
                self.assertEqual(m['evictions'], 0)
            # All 3 chunks fit (capacity 3): revisiting any of them is a hit.
            _out, metrics = cache.load(self._single_chunk_region(0, chunk_size))
            self.assertEqual(metrics['bytes_read'], 0)
            self.assertEqual(metrics['cache_hits'], 1)
            self.assertEqual(metrics['evictions'], 0)

    def test_capacity_one_roi_larger_than_cache_still_correct(self):
        # A single load() whose ROI spans a 2x2 grid of chunks, but the cache
        # can hold only 1 chunk at a time: every chunk after the first evicts
        # the previously resident one, yet the stitched ROI must still be
        # byte-exact. This call is cold (nothing cached beforehand), so every
        # cell is an initially-missing cell and the hits-first-then-misses
        # ordering has nothing to reorder here -- all 4 cells are still
        # visited in plain grid row-major order, so the expected outcome is
        # unchanged by that policy.
        chunk_size = 16
        height, width = chunk_size * 2, chunk_size * 2
        arr, info = self._prepare(height, width, chunk_size, name='capacity-one')
        region = (0, 0, height, width)

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=1) as cache:
            out, metrics = cache.load(region)

        np.testing.assert_array_equal(out, arr)
        self.assertEqual(metrics['chunks_requested'], 4)
        self.assertEqual(metrics['cache_misses'], 4)
        self.assertEqual(metrics['cache_hits'], 0)
        self.assertEqual(metrics['evictions'], 3)
        # Exactly one chunk's worth of decoded bytes remains resident.
        index = _read_index(info['chunked_path'])
        self.assertEqual(metrics['resident_bytes'], _chunk_bytes(index, 1, 1))


# --------------------------------------------------------------------------- #
# ChunkRegionCache: overlapping-ROI regression -- "hits first, then misses"
# must avoid a same-call chunk being evicted (by an earlier-processed miss)
# before its own turn as a hit, which would otherwise make repeated
# alternating loads over two overlapping ROIs thrash instead of settle.
# --------------------------------------------------------------------------- #

class OverlapThrashRegressionTests(TownCacheTestCase):
    '''A=(240,240,256,256), B=(240,260,256,256) on a 1024x1024 map at
    chunk_size=256 (grid 4x4). A's grid cells are {(0,0),(0,1),(1,0),(1,1)};
    B's are {(0,1),(0,2),(1,1),(1,2)}; they share exactly {(0,1),(1,1)} and
    each has 2 cells the other does not.

    With a naive single-pass row-major walk that evicts-before-decoding each
    miss as it is met, loading A then B then A then B... does not settle: A's
    row-major order visits its non-shared cell (0,0) *before* its shared
    cells, so a miss on (0,0) can evict a shared cell before that shared
    cell's own turn in the very same call -- turning a chunk the cache
    actually already had into a miss too. Processing every already-cached
    cell first (and marking it MRU) before any eviction removes that
    interaction, and the two ROIs should settle into a stable, independently
    verifiable hit/miss pattern instead.
    '''

    _HEIGHT = _WIDTH = 1024
    _CHUNK_SIZE = 256
    _REGION_A = (240, 240, 256, 256)
    _REGION_B = (240, 260, 256, 256)
    _CELLS_A = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    _CELLS_B = frozenset({(0, 1), (0, 2), (1, 1), (1, 2)})
    _SHARED = _CELLS_A & _CELLS_B
    _A_ONLY = _CELLS_A - _SHARED
    _B_ONLY = _CELLS_B - _SHARED

    def setUp(self):
        super().setUp()
        self.assertEqual(self._SHARED, {(0, 1), (1, 1)})
        self.assertEqual(self._A_ONLY, {(0, 0), (1, 0)})
        self.assertEqual(self._B_ONLY, {(0, 2), (1, 2)})

    def _run(self, max_chunks, expected_hit_sets, expected_miss_sets, name):
        arr, info = self._prepare(
            self._HEIGHT, self._WIDTH, self._CHUNK_SIZE, name=name
        )
        index = _read_index(info['chunked_path'])
        regions = [self._REGION_A, self._REGION_B] * ((len(expected_hit_sets) + 1) // 2)
        regions = regions[:len(expected_hit_sets)]

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=max_chunks) as cache:
            for i, (region, exp_hits, exp_misses) in enumerate(
                zip(regions, expected_hit_sets, expected_miss_sets)
            ):
                row, col, h, w = region
                out, metrics = cache.load(region)

                # Correctness and ownership must hold on every single
                # iteration, steady-state or not.
                np.testing.assert_array_equal(out, arr[row:row + h, col:col + w])
                self._assert_owned(out)

                got_hits = {tuple(c) for c in metrics['hit_coordinates']}
                got_misses = {tuple(c) for c in metrics['miss_coordinates']}
                self.assertEqual(
                    got_hits, exp_hits, f'iteration {i} ({region}): unexpected hit set'
                )
                self.assertEqual(
                    got_misses, exp_misses, f'iteration {i} ({region}): unexpected miss set'
                )
                self.assertEqual(metrics['cache_hits'], len(exp_hits))
                self.assertEqual(metrics['cache_misses'], len(exp_misses))
                self.assertEqual(metrics['chunks_requested'], len(exp_hits) + len(exp_misses))

                # Independently verify the exact miss payload bytes from the
                # index, parsed without any help from town_cache.
                expected_bytes = sum(index['entries'][c][5] for c in exp_misses)
                self.assertEqual(
                    metrics['bytes_read'], expected_bytes,
                    f'iteration {i} ({region}): bytes_read does not match the index-parsed '
                    f'payload lengths of this call\'s own miss set',
                )

    def test_capacity_four_settles_to_two_hit_two_miss_forever(self):
        # Cold start: all 4 of A's cells miss. From then on, *every* load
        # (whether A or B) must hit exactly the 2 shared cells and miss
        # exactly its own 2 unique cells -- not degrade back to 4 misses on
        # alternation, which is the thrash this ordering fixes.
        hits = [set(), self._SHARED, self._SHARED, self._SHARED, self._SHARED, self._SHARED]
        misses = [
            self._CELLS_A, self._B_ONLY, self._A_ONLY, self._B_ONLY, self._A_ONLY, self._B_ONLY,
        ]
        self._run(4, hits, misses, name='thrash-cap4')

    def test_capacity_eight_settles_to_all_hits_once_everything_fits(self):
        # Capacity 8 can hold all 6 distinct cells {A ∪ B}, so once both ROIs
        # have been loaded once each, every further load of either is a pure
        # hit, forever.
        hits = [set(), self._SHARED, self._CELLS_A, self._CELLS_B, self._CELLS_A, self._CELLS_B]
        misses = [self._CELLS_A, self._B_ONLY, set(), set(), set(), set()]
        self._run(8, hits, misses, name='thrash-cap8')


# --------------------------------------------------------------------------- #
# ChunkRegionCache: total_ns is the real elapsed core time, not a sum of a
# hand-picked subset of spans -- allocation and LRU bookkeeping must show up
# in decode_ns too. Verified with an injected, deterministic delay rather
# than a noisy wall-clock ratio assertion.
# --------------------------------------------------------------------------- #

class TimingSemanticsTests(TownCacheTestCase):
    def test_output_allocation_cost_is_counted_in_decode_ns(self):
        '''np.empty() builds the per-call output buffer before any chunk is
        touched; it is not I/O, so a deterministic delay injected into it
        must surface in decode_ns (and therefore total_ns), not read_ns.'''
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='timing-alloc')
        real_empty = np.empty

        def slow_empty(*a, **kw):
            time.sleep(_INJECTED_SLEEP_S)
            return real_empty(*a, **kw)

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=4) as cache:
            with mock.patch('numpy.empty', side_effect=slow_empty):
                _out, metrics = cache.load((0, 0, 8, 8))

        self.assertEqual(metrics['total_ns'], metrics['read_ns'] + metrics['decode_ns'])
        self.assertGreaterEqual(metrics['decode_ns'], _MIN_OBSERVED_NS)
        self.assertGreaterEqual(metrics['total_ns'], _MIN_OBSERVED_NS)

    def test_cache_hit_lru_bookkeeping_cost_is_counted_in_decode_ns(self):
        '''A pure cache hit does zero I/O, but it still has to run
        OrderedDict.move_to_end on the real cache dict. A deterministic delay
        injected into that call on this cache's own dict must surface in
        decode_ns even though bytes_read for this call is 0.'''
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='timing-hit')
        region = (2, 2, 5, 5)  # single chunk (0, 0)

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=4) as cache:
            cache.load(region)  # prime: first load is a miss

            real_cache_dict = cache._cache
            real_move_to_end = real_cache_dict.move_to_end

            def slow_move_to_end(*a, **kw):
                time.sleep(_INJECTED_SLEEP_S)
                return real_move_to_end(*a, **kw)

            with mock.patch.object(real_cache_dict, 'move_to_end', side_effect=slow_move_to_end):
                _out, metrics = cache.load(region)  # now a pure hit

        self.assertEqual(metrics['cache_hits'], 1)
        self.assertEqual(metrics['bytes_read'], 0)
        self.assertEqual(metrics['total_ns'], metrics['read_ns'] + metrics['decode_ns'])
        self.assertGreaterEqual(metrics['decode_ns'], _MIN_OBSERVED_NS)

    def test_eviction_bookkeeping_cost_is_counted_in_decode_ns(self):
        '''Eviction (OrderedDict.popitem) runs on a miss, interleaved with but
        distinct from the payload read and the decode call. A deterministic
        delay injected into popitem on this cache's own dict must surface in
        decode_ns, not read_ns, and bytes_read must stay exactly the one
        miss's payload length regardless of how long eviction itself took.'''
        height, width, chunk_size = 32, 32, 16  # 2x2 grid
        arr, info = self._prepare(height, width, chunk_size, name='timing-evict')
        index = _read_index(info['chunked_path'])

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=1) as cache:
            cache.load((0, 0, 8, 8))  # prime: cache now holds chunk (0, 0)

            real_cache_dict = cache._cache
            real_popitem = real_cache_dict.popitem

            def slow_popitem(*a, **kw):
                time.sleep(_INJECTED_SLEEP_S)
                return real_popitem(*a, **kw)

            with mock.patch.object(real_cache_dict, 'popitem', side_effect=slow_popitem):
                # Capacity 1: loading the other chunk forces an eviction.
                _out, metrics = cache.load((chunk_size, chunk_size, 8, 8))

        self.assertEqual(metrics['evictions'], 1)
        self.assertEqual(metrics['cache_misses'], 1)
        self.assertEqual(metrics['bytes_read'], index['entries'][(1, 1)][5])
        self.assertEqual(metrics['total_ns'], metrics['read_ns'] + metrics['decode_ns'])
        self.assertGreaterEqual(metrics['decode_ns'], _MIN_OBSERVED_NS)


# --------------------------------------------------------------------------- #
# ChunkRegionCache: resident_bytes accounting
# --------------------------------------------------------------------------- #

class ChunkCacheResidentBytesTests(TownCacheTestCase):
    def test_resident_bytes_grows_with_distinct_cached_chunks(self):
        chunk_size = 16
        height, width = chunk_size, chunk_size * 3
        arr, info = self._prepare(height, width, chunk_size, name='resident-grow')
        index = _read_index(info['chunked_path'])

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=3) as cache:
            self.assertEqual(cache.resident_bytes, 0)
            cache.load((0, 0, chunk_size, chunk_size))
            self.assertEqual(cache.resident_bytes, _chunk_bytes(index, 0, 0))
            cache.load((0, chunk_size, chunk_size, chunk_size))
            self.assertEqual(
                cache.resident_bytes, _chunk_bytes(index, 0, 0) + _chunk_bytes(index, 0, 1)
            )

    def test_resident_bytes_shrinks_on_eviction(self):
        chunk_size = 16
        height, width = chunk_size, chunk_size * 3
        arr, info = self._prepare(height, width, chunk_size, name='resident-shrink')
        index = _read_index(info['chunked_path'])

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=1) as cache:
            cache.load((0, 0, chunk_size, chunk_size))
            self.assertEqual(cache.resident_bytes, _chunk_bytes(index, 0, 0))
            cache.load((0, chunk_size, chunk_size, chunk_size))
            self.assertEqual(cache.resident_bytes, _chunk_bytes(index, 0, 1))

    def test_load_metrics_resident_bytes_matches_object_attribute(self):
        chunk_size = 16
        height, width = chunk_size, chunk_size * 2
        arr, info = self._prepare(height, width, chunk_size, name='resident-metrics')

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=2) as cache:
            _out, metrics = cache.load((0, 0, chunk_size, chunk_size))
            self.assertEqual(metrics['resident_bytes'], cache.resident_bytes)


# --------------------------------------------------------------------------- #
# ChunkRegionCache: constructor startup accounting
# --------------------------------------------------------------------------- #

class ChunkCacheStartupTests(TownCacheTestCase):
    def test_startup_metrics_read_header_and_index_only_no_payload(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='startup')
        index = _read_index(info['chunked_path'])

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=4) as cache:
            metrics = cache.startup_metrics
            for key in _STANDARD_METRIC_KEYS:
                self.assertIn(key, metrics)
            self.assertEqual(metrics['bytes_read'], index['index_bytes'])
            self.assertEqual(cache.resident_bytes, 0)
            self.assertEqual(cache.resident_chunks, [])

    def test_startup_metrics_frozen_after_loads(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='startup-frozen')

        with town_cache.ChunkRegionCache(info['chunked_path'], max_chunks=4) as cache:
            before = dict(cache.startup_metrics)
            cache.load((0, 0, 8, 8))
            cache.load((16, 16, 8, 8))
            self.assertEqual(cache.startup_metrics, before)


# --------------------------------------------------------------------------- #
# ChunkRegionCache: max_chunks validation
# --------------------------------------------------------------------------- #

class MaxChunksValidationTests(TownCacheTestCase):
    def setUp(self):
        super().setUp()
        self.arr, self.info = self._prepare(32, 32, 16, name='maxchunks')

    def test_zero_max_chunks_rejected(self):
        with self.assertRaises(ValueError):
            town_cache.ChunkRegionCache(self.info['chunked_path'], max_chunks=0)

    def test_negative_max_chunks_rejected(self):
        with self.assertRaises(ValueError):
            town_cache.ChunkRegionCache(self.info['chunked_path'], max_chunks=-1)

    def test_non_integer_max_chunks_rejected(self):
        with self.assertRaises(ValueError):
            town_cache.ChunkRegionCache(self.info['chunked_path'], max_chunks=1.5)

    def test_bool_max_chunks_rejected(self):
        with self.assertRaises(ValueError):
            town_cache.ChunkRegionCache(self.info['chunked_path'], max_chunks=True)
        with self.assertRaises(ValueError):
            town_cache.ChunkRegionCache(self.info['chunked_path'], max_chunks=False)

    def test_positive_integer_max_chunks_accepted(self):
        with town_cache.ChunkRegionCache(self.info['chunked_path'], max_chunks=3) as cache:
            self.assertEqual(cache.max_chunks, 3)


# --------------------------------------------------------------------------- #
# ChunkRegionCache: invalid region / mode
# --------------------------------------------------------------------------- #

class ChunkCacheInvalidArgumentTests(TownCacheTestCase):
    def setUp(self):
        super().setUp()
        self.arr, self.info = self._prepare(64, 64, 16, name='invalid')

    def test_negative_row_rejected(self):
        with town_cache.ChunkRegionCache(self.info['chunked_path']) as cache:
            with self.assertRaises(ValueError):
                cache.load((-1, 0, 4, 4))

    def test_zero_height_rejected(self):
        with town_cache.ChunkRegionCache(self.info['chunked_path']) as cache:
            with self.assertRaises(ValueError):
                cache.load((0, 0, 0, 4))

    def test_wrong_length_tuple_rejected(self):
        with town_cache.ChunkRegionCache(self.info['chunked_path']) as cache:
            with self.assertRaises(ValueError):
                cache.load((0, 0, 4))

    def test_region_exceeds_bounds_rejected(self):
        with town_cache.ChunkRegionCache(self.info['chunked_path']) as cache:
            with self.assertRaises(ValueError):
                cache.load((60, 60, 10, 10))

    def test_unsupported_mode_rejected(self):
        with self.assertRaises(ValueError):
            town_cache.ChunkRegionCache(self.info['chunked_path'], mode='bogus')


# --------------------------------------------------------------------------- #
# ChunkRegionCache: in-place modification detection (fail closed)
# --------------------------------------------------------------------------- #

class InPlaceModificationTests(TownCacheTestCase):
    def test_in_place_size_change_rejected_on_next_load(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='inplace-size')

        with town_cache.ChunkRegionCache(info['chunked_path']) as cache:
            cache.load((0, 0, 8, 8))
            with open(info['chunked_path'], 'ab') as fh:
                fh.write(b'\x00' * 16)  # append bytes: size changes in place
            with self.assertRaises(Exception):
                cache.load((16, 16, 8, 8))

    def test_in_place_mtime_change_rejected_on_next_load(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='inplace-mtime')

        with town_cache.ChunkRegionCache(info['chunked_path']) as cache:
            cache.load((0, 0, 8, 8))
            st = os.stat(info['chunked_path'])
            new_mtime = st.st_mtime + 120
            os.utime(info['chunked_path'], (new_mtime, new_mtime))
            with self.assertRaises(Exception):
                cache.load((16, 16, 8, 8))

    def test_replacing_path_with_new_file_either_keeps_snapshot_or_fails_closed(self):
        '''os.replace swaps the directory entry to a new inode; the cache's
        already-open descriptor still refers to the original file's bytes, so
        if the snapshot check passes it must keep returning the *original*
        map, never a mix with the replacement. In practice the unlink that
        replace() performs changes the original inode's ctime, so the
        documented, equally-acceptable outcome is that the next load fails
        closed instead -- both outcomes are tested for here, never a silent
        read of the replacement file's bytes.'''
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='replace-src')
        other_arr, other_info = self._prepare(32, 32, chunk_size, seed=99, name='replace-dst')

        with town_cache.ChunkRegionCache(info['chunked_path']) as cache:
            first, _m = cache.load((0, 0, 8, 8))
            np.testing.assert_array_equal(first, arr[0:8, 0:8])

            os.replace(other_info['chunked_path'], info['chunked_path'])

            try:
                second, _m = cache.load((16, 16, 8, 8))
            except ValueError:
                pass  # fail-closed outcome: documented and acceptable
            else:
                np.testing.assert_array_equal(second, arr[16:24, 16:24])


# --------------------------------------------------------------------------- #
# ChunkRegionCache: corruption of a selected chunk's body/decode
# --------------------------------------------------------------------------- #

class ChunkCacheCorruptionTests(TownCacheTestCase):
    def test_corrupt_selected_chunk_payload_rejected(self):
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='corrupt-payload')
        index = _read_index(info['chunked_path'])
        _row, _col, _h, _w, offset, _length = index['entries'][(0, 0)]
        payload_start = index['payload_base'] + offset

        data = bytearray(Path(info['chunked_path']).read_bytes())
        data[payload_start:payload_start + 4] = b'\x00\x00\x00\x00'  # mangle container magic
        corrupt_path = self.dir / 'corrupt.hcix'
        corrupt_path.write_bytes(bytes(data))

        with town_cache.ChunkRegionCache(str(corrupt_path)) as cache:
            with self.assertRaises(ValueError):
                cache.load((2, 2, 5, 5))

    def test_corrupt_selected_chunk_body_bytes_rejected_via_checksum(self):
        from horizon.codec import HEADER_SIZE
        height, width, chunk_size = 64, 64, 16
        arr, info = self._prepare(height, width, chunk_size, name='corrupt-body')
        index = _read_index(info['chunked_path'])
        _row, _col, _h, _w, offset, length = index['entries'][(0, 0)]
        payload_start = index['payload_base'] + offset
        self.assertGreater(length, HEADER_SIZE + 8)

        data = bytearray(Path(info['chunked_path']).read_bytes())
        data[payload_start + HEADER_SIZE + 4] ^= 0xFF  # flip a byte in the compressed body
        corrupt_path = self.dir / 'corrupt_body.hcix'
        corrupt_path.write_bytes(bytes(data))

        with town_cache.ChunkRegionCache(str(corrupt_path)) as cache:
            with self.assertRaises(ValueError):
                cache.load((2, 2, 5, 5))


# --------------------------------------------------------------------------- #
# ChunkRegionCache: closing, idempotency, descriptor lifecycle, leak safety
# --------------------------------------------------------------------------- #

class ChunkCacheCloseTests(TownCacheTestCase):
    def test_close_is_idempotent(self):
        arr, info = self._prepare(32, 32, 16, name='close-idempotent')
        cache = town_cache.ChunkRegionCache(info['chunked_path'])
        cache.load((0, 0, 8, 8))
        cache.close()
        cache.close()  # must not raise
        cache.close()

    def test_context_manager_close_is_idempotent_with_explicit_close(self):
        arr, info = self._prepare(32, 32, 16, name='close-ctx')
        with town_cache.ChunkRegionCache(info['chunked_path']) as cache:
            cache.load((0, 0, 8, 8))
            cache.close()  # closing early inside the `with` must not raise
        # __exit__'s own close() call must also be a no-op here, not an error.

    def test_descriptor_opened_once_not_reopened_per_load(self):
        arr, info = self._prepare(64, 64, 16, name='fd-once')
        real_open = open
        open_calls = []

        def spy_open(*a, **kw):
            open_calls.append(a[0] if a else kw.get('file'))
            return real_open(*a, **kw)

        with mock.patch('builtins.open', spy_open):
            with town_cache.ChunkRegionCache(info['chunked_path']) as cache:
                after_init = len(open_calls)
                cache.load((0, 0, 8, 8))
                cache.load((16, 16, 8, 8))
                cache.load((32, 32, 8, 8))
        self.assertEqual(len(open_calls), after_init, 'load() must not open the file again')

    def test_construction_failure_does_not_leak_descriptor(self):
        corrupt_path = self.dir / 'bad.hcix'
        corrupt_path.write_bytes(b'\x00' * 8)  # too small to hold a valid HCIX header

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
                town_cache.ChunkRegionCache(str(corrupt_path))

        self.assertTrue(opened)
        for fh in opened:
            self.assertEqual(fh.close_calls, 1, 'a failed construction must not leak its descriptor')


# --------------------------------------------------------------------------- #
# ChunkRegionCache: output non-aliasing
# --------------------------------------------------------------------------- #

class ChunkCacheNonAliasingTests(TownCacheTestCase):
    def test_independent_loads_never_share_memory(self):
        arr, info = self._prepare(64, 64, 16, name='nonalias')
        region = (12, 12, 8, 8)
        with town_cache.ChunkRegionCache(info['chunked_path']) as cache:
            out1, _m1 = cache.load(region)
            out2, _m2 = cache.load(region)
            out1[0, 0] = (int(out1[0, 0]) + 1) % 65536
            self.assertFalse(np.shares_memory(out1, out2))
            np.testing.assert_array_equal(out2, arr[12:20, 12:20])


# --------------------------------------------------------------------------- #
# WholeMapResident: single read/decode at construction, pure-memory loads
# --------------------------------------------------------------------------- #

class WholeMapResidentTests(TownCacheTestCase):
    def test_startup_reads_and_decodes_full_container_once(self):
        height, width = 64, 48
        arr, info = self._prepare(height, width, 16, name='whole-startup')

        with town_cache.WholeMapResident(info['whole_path'], 'warm_buffered') as resident:
            metrics = resident.startup_metrics
            for key in _STANDARD_METRIC_KEYS:
                self.assertIn(key, metrics)
            self.assertEqual(metrics['bytes_read'], os.path.getsize(info['whole_path']))
            self.assertGreater(metrics['decode_ns'], 0)
            self.assertEqual(resident.resident_bytes, height * width * _DTYPE.itemsize)

    def test_subsequent_loads_do_no_file_io(self):
        height, width = 64, 48
        arr, info = self._prepare(height, width, 16, name='whole-noio')

        with town_cache.WholeMapResident(info['whole_path'], 'warm_buffered') as resident:
            real_open = open
            open_calls = []

            def spy_open(*a, **kw):
                open_calls.append(a[0] if a else kw.get('file'))
                return real_open(*a, **kw)

            with mock.patch('builtins.open', spy_open):
                out, metrics = resident.load((10, 10, 8, 8))

            self.assertEqual(open_calls, [])
            self.assertEqual(metrics['bytes_read'], 0)
            self.assertEqual(metrics['read_calls'], 0)
            np.testing.assert_array_equal(out, arr[10:18, 10:18])

    def test_roi_correctness_boundary_and_odd_edges(self):
        height, width = 513, 769
        arr, info = self._prepare(height, width, 128, name='whole-odd')
        regions = [
            (0, 0, 10, 10),
            (height - 5, width - 7, 5, 7),
            (height - 1, 0, 1, width),
            (0, width - 1, height, 1),
            (12, 12, 8, 8),
        ]
        with town_cache.WholeMapResident(info['whole_path'], 'warm_buffered') as resident:
            for region in regions:
                with self.subTest(region=region):
                    row, col, h, w = region
                    out, metrics = resident.load(region)
                    np.testing.assert_array_equal(out, arr[row:row + h, col:col + w])
                    self.assertTrue(out.flags['C_CONTIGUOUS'])
                    self.assertTrue(out.flags['OWNDATA'] or out.base is None)
                    self.assertEqual(metrics['resident_bytes'], resident.resident_bytes)

    def test_all_codecs_round_trip(self):
        for method in METHODS:
            with self.subTest(method=method):
                arr, info = self._prepare(96, 80, 16, method=method, name=f'whole-{method}')
                with town_cache.WholeMapResident(info['whole_path'], 'warm_buffered') as resident:
                    out, _metrics = resident.load((0, 0, 96, 80))
                    np.testing.assert_array_equal(out, arr)

    def test_snapshot_still_works_after_underlying_file_deleted(self):
        arr, info = self._prepare(48, 48, 16, name='whole-deleted')
        with town_cache.WholeMapResident(info['whole_path'], 'warm_buffered') as resident:
            os.remove(info['whole_path'])
            out, metrics = resident.load((4, 4, 10, 10))
            np.testing.assert_array_equal(out, arr[4:14, 4:14])
            self.assertEqual(metrics['bytes_read'], 0)

    def test_independent_loads_never_share_memory(self):
        arr, info = self._prepare(48, 48, 16, name='whole-nonalias')
        with town_cache.WholeMapResident(info['whole_path'], 'warm_buffered') as resident:
            out1, _m1 = resident.load((4, 4, 10, 10))
            out2, _m2 = resident.load((4, 4, 10, 10))
            self.assertFalse(np.shares_memory(out1, out2))
            self.assertIsNot(out1, out2)

    def test_close_is_idempotent(self):
        arr, info = self._prepare(32, 32, 16, name='whole-close-idempotent')
        resident = town_cache.WholeMapResident(info['whole_path'], 'warm_buffered')
        resident.load((0, 0, 8, 8))
        resident.close()
        resident.close()

    def test_unsupported_mode_rejected(self):
        arr, info = self._prepare(32, 32, 16, name='whole-badmode')
        with self.assertRaises(ValueError):
            town_cache.WholeMapResident(info['whole_path'], 'bogus')

    def test_invalid_region_rejected(self):
        arr, info = self._prepare(32, 32, 16, name='whole-badregion')
        with town_cache.WholeMapResident(info['whole_path'], 'warm_buffered') as resident:
            with self.assertRaises(ValueError):
                resident.load((0, 0, 0, 4))
            with self.assertRaises(ValueError):
                resident.load((30, 30, 10, 10))

    @unittest.skipUnless(IS_DARWIN, 'macos_nocache is Darwin-only')
    def test_macos_nocache_mode_matches_warm_buffered(self):
        height, width = 48, 48
        arr, info = self._prepare(height, width, 16, name='whole-nocache')
        region = (4, 4, 10, 10)
        expected = arr[4:14, 4:14]

        with town_cache.WholeMapResident(info['whole_path'], 'warm_buffered') as r_warm:
            out_warm, _m1 = r_warm.load(region)
        with town_cache.WholeMapResident(info['whole_path'], 'macos_nocache') as r_cold:
            out_cold, _m2 = r_cold.load(region)

        np.testing.assert_array_equal(out_warm, expected)
        np.testing.assert_array_equal(out_cold, expected)


if __name__ == '__main__':
    unittest.main()

'''Resident-state loaders: a bounded chunk LRU vs. a whole-map resident.

horizon.town_disk and horizon.town_region are deliberately stateless: every
call opens the file, reads, closes, decodes, and keeps nothing. That answers
"what does one ROI fetch cost when nothing may be remembered", but not "what
does it cost to keep moving around a town when the process is allowed to
remember things". This module holds the two honest answers to that question, as
two objects with the same shape so a benchmark can swap one for the other:

  ChunkRegionCache(path, max_chunks=8)  pays the HCIX header+index read once in
                                        its constructor, keeps the unbuffered
                                        descriptor open, and keeps up to
                                        max_chunks *decoded* chunks in an LRU.
                                        Each load reads only the blobs of the
                                        chunks it is missing.
  WholeMapResident(path)                pays one full read+decode in its
                                        constructor and keeps the entire
                                        decoded map. Each load is a crop, with
                                        no file I/O at all.

What is and is not measured
---------------------------
Setup is never free and is never hidden: both constructors do their real work
(open, fstat, read, validate, and for the resident the full decode and
checksum) and report it in `startup_metrics`, measured across the whole setup
core -- up to and including installing the state the object keeps, stopping
only where the metrics dict itself is built. A per-load `metrics` then
describes only that load, and the per-load I/O counters are deltas taken since
startup, so they count the payload reads that request actually issued and never
the constructor's metadata reads.

total_ns is measured elapsed time, not a sum of selected spans. One clock
reading is taken before anything the request does -- before the snapshot check,
before the counter baseline, before the output array is allocated -- and one
after the last chunk has been stitched and the LRU updated, immediately before
the metrics dict is assembled. Everything the load really does is inside that
interval: output allocation, cache lookups, eviction, LRU bookkeeping, reads,
decode, stitching. A cache total_ns is therefore directly comparable with
town_region.load_region's total_ns for the same ROI, which is the whole point
of having both.

Within that interval read_ns is the sum of the I/O spans only: the per-load
fstat snapshot check (which a 100%-hit load still pays -- that is why an all-hit
load reports read_ns > 0 with zero read_calls) plus, for each miss, the
exact-size payload read and that blob's container-header checks. decode_ns is
defined as total_ns - read_ns, so it carries *all* the non-I/O work --
allocation, cache lookups, eviction, LRU updates, decode, checksum, stitching.
total_ns == read_ns + decode_ns therefore holds exactly, by construction rather
than by rounding, while total_ns stays real elapsed time. The same shape holds
for the whole-map resident, where read_ns is 0 because no file is touched and
decode_ns is the measured crop including its allocation.

Argument validation sits outside the measured interval in both classes, which
matches the stateless loaders: town_disk and town_region validate mode and
region before their own t0 too, so a rejected call costs no measured time in
any arm.

Snapshot contract
-----------------
Both classes pin the file: the constructor opens it once and every later load
is served through that same descriptor (the cache) or from the array decoded
from it (the resident). That makes an instance a snapshot of one inode, and it
is only sound if the bytes behind it do not move. So every ChunkRegionCache
load fstats the pinned descriptor and compares (st_size, st_mtime_ns,
st_ctime_ns) against the values pinned at construction; any difference fails
closed -- the instance is closed and invalidated, and the load raises rather
than stitching a mix of old index and new payload. The index is validated once,
so a silent in-place rewrite is exactly the case that cheap check must catch.

Replacing the path (rename a new file over it) is a different thing and is not
a mutation of the pinned bytes: the descriptor keeps serving the original
inode, so loads keep returning the original map and a caller who expects to see
the new file must build a new instance. Note that such a replacement unlinks
the old directory entry, which changes the pinned inode's link count and hence
its st_ctime -- so in practice the snapshot check usually *does* trip on a
replace and fails closed. Both outcomes are intended: this module never
silently follows a file it did not open.

Neither class is thread-safe; one instance belongs to one caller. The container
is the same private experimental HCIX benchmark container (see
town_disk.CONTAINER_NOTE) and is treated as untrusted input: this module adds
no format and no validator of its own, it reuses town_disk's.
'''

from __future__ import annotations

import os
import time
from collections import OrderedDict

import numpy as np

from horizon.chunk_benchmark import _INDEX_ENTRY, _INDEX_HEADER
from horizon.codec import HEADER_SIZE, decode
from horizon.town_disk import (
    _DTYPE,
    _check_decoded,
    _check_mode,
    _crop_copy,
    _entry,
    _int_arg,
    _parse_codec_header,
    _parse_index_header,
    _Reader,
    _validate_entries,
    load_whole,
)
from horizon.town_region import _check_roi

DEFAULT_MAX_CHUNKS = 8

SNAPSHOT_NOTE = (
    'an instance pins the descriptor it opened; every load fstats it and fails '
    'closed on any change to (size, mtime_ns, ctime_ns). Replacing the path '
    'leaves the pinned descriptor on the original inode, but the unlink that '
    'replacement performs changes that inode ctime, so a replace usually trips '
    'the check too'
)


def _stat_snapshot(fd):
    '''The identity of the pinned file, as the three fields a rewrite moves.'''
    st = os.fstat(fd)
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _elapsed_metrics(total_ns, read_ns, counters):
    '''Assemble the standard metric keys from measured elapsed time.

    town_disk's (t0, t_read, t_end) split assumes all I/O precedes all decoding;
    a cache load interleaves them, so the phases are described the other way
    round: total_ns is the measured elapsed time of the whole core load, read_ns
    is the summed I/O spans inside it, and decode_ns is the remainder -- every
    non-I/O cost, nothing dropped. The identity total_ns == read_ns + decode_ns
    is then exact and total_ns is still real elapsed time.
    '''
    if not 0 <= read_ns <= total_ns:
        raise ValueError(f'read_ns {read_ns} is not within the {total_ns} ns measured load')
    metrics = {
        'total_ns': total_ns,
        'read_ns': read_ns,
        'decode_ns': total_ns - read_ns,
    }
    metrics.update(counters)
    return metrics


def _zero_counters():
    return {'bytes_read': 0, 'bytes_requested': 0, 'read_calls': 0, 'seek_calls': 0}


def _chunk_counters(coordinates=(), hits=(), misses=(), evicted=()):
    '''The chunk/cache half of a metrics dict; all-zero and empty by default.

    Both classes report these keys always -- empty at startup, and permanently
    empty for the whole-map resident -- so a cache load, a resident load and
    either startup dict line up key-for-key in a benchmark table. The lists are
    rebuilt per call, never shared.
    '''
    return {
        'chunks_requested': len(coordinates),
        'chunks_loaded': len(misses),
        'cache_hits': len(hits),
        'cache_misses': len(misses),
        'evictions': len(evicted),
        'chunk_coordinates': list(coordinates),
        'hit_coordinates': list(hits),
        'miss_coordinates': list(misses),
        'evicted_coordinates': list(evicted),
    }


def _validate_chunk_blob(payload, length, method, c_h, c_w, label):
    '''Re-run town_region's per-chunk container checks on a freshly read blob.

    A chunk's own codec header must agree with the index on shape and method and
    must account for exactly the bytes the index allotted it, before decode() is
    handed anything.
    '''
    b_method, b_h, b_w, b_plen = _parse_codec_header(payload, label)
    if (b_h, b_w) != (c_h, c_w):
        raise ValueError(f'{label} container declares {b_h}x{b_w}, index says {c_h}x{c_w}')
    if b_method != method:
        raise ValueError(f'{label} is encoded with {b_method}, index header says {method}')
    if HEADER_SIZE + b_plen != length:
        raise ValueError(f'{label} payload length {b_plen} disagrees with {length}')


def _stitch(out, block, entry, row, col):
    '''Copy this chunk's intersection with the ROI into the output.

    The intersection is computed in map coordinates, then expressed in
    chunk-local and ROI-local coordinates.
    '''
    c_row, c_col, c_h, c_w = entry[0], entry[1], entry[2], entry[3]
    h, w = out.shape
    top, left = max(c_row, row), max(c_col, col)
    bottom, right = min(c_row + c_h, row + h), min(c_col + c_w, col + w)
    out[top - row:bottom - row, left - col:right - col] = (
        block[top - c_row:bottom - c_row, left - c_col:right - c_col]
    )


def _freeze(block):
    '''Make a retained array unwritable.

    Nothing this module returns aliases a retained array -- every load stitches
    or crops into a fresh output -- and flipping the flag makes that structural
    rather than a thing to remember: an accidental in-place write to resident
    state raises instead of corrupting later loads.
    '''
    block.flags.writeable = False
    return block


class ChunkRegionCache:
    '''Arbitrary-ROI loads over a persistent LRU of decoded HCIX chunks.

    The constructor does the metadata work once: it opens the file unbuffered,
    pins its stat snapshot, reads the index header and the full index, and
    validates that the index is a contiguous, gap-free, non-overlapping cover of
    the map (town_disk's own checkers, unchanged). No payload is read and no
    chunk is decoded at startup, so `resident_bytes` is 0 until the first load
    and `startup_metrics`' chunk counters are zero with empty coordinate lists.
    The elapsed cost of all of it is in `startup_metrics`.

    Each load takes an arbitrary in-bounds ROI, which in general straddles a
    rectangular span of grid cells. The ROI selects that span in grid row-major
    order, and the selected cells are then served in exactly two phases. Which
    cell lands in which phase depends only on the ROI and on the LRU state at
    the instant the request starts, so an audit can simulate the whole policy
    without reading this code:

      phase 1 -- every selected cell that is already resident *when the request
        starts*, in grid row-major order. Each is stitched into the output and
        moved to most-recently-used. No seek, no read, no decode, and above all
        no eviction happens in this phase.
      phase 2 -- every remaining selected cell, in grid row-major order. Each
        one seeks and reads only its own blob, validates that blob's container
        header against the index, evicts if the cache is full, decodes (which
        verifies the codec's sha256), inserts itself as most-recently-used, and
        is stitched immediately.

    Doing all the hits first is what makes the cache behave sanely on the motion
    pattern it exists for: overlapping ROIs. A single sequential pass can read a
    miss early, evict the least-recently-used entry to make room, and so destroy
    a chunk the *same request* was about to hit -- panning back and forth across
    a chunk boundary then turns reusable chunks into repeat misses even when the
    working set fits in the cache. Phase 1 refreshes every reusable chunk to
    most-recently-used before phase 2 is allowed to evict anything, so an ROI
    needing exactly `max_chunks` chunks never evicts a chunk it still needs. Two
    ROIs overlapping in 2 of 4 chunks on a 4-chunk cache settle at a steady
    2 hits / 2 misses per step after the first load, instead of degrading; on an
    8-chunk cache the same alternation becomes all hits once both ROIs have been
    seen.

    Eviction happens just before a miss is decoded, never after inserting: the
    decoded set is therefore never even transiently over capacity, so retained
    bytes stay bounded by max_chunks * chunk_size**2 * 2 at every instant --
    phase 1 only reorders entries and never grows the cache, so it cannot breach
    that bound either. An ROI spanning more chunks than the cache can hold is
    still correct: its hits are all stitched in phase 1, before any eviction can
    occur, and each phase-2 chunk is stitched immediately after being decoded,
    so a chunk evicted later in the same request has already contributed its
    pixels. Such an ROI simply evicts its own earlier chunks as it goes and will
    miss on them again next time -- a real cost of a bounded cache, reported
    (`evictions`, `evicted_coordinates`) rather than smoothed over.

    Any failure while reading, decoding or updating state closes the instance
    and clears the cache: a half-applied mutation, or a descriptor whose file
    moved under it, is not something to retry against. A rejected *argument* (a
    malformed or out-of-bounds ROI) changes nothing and leaves the instance
    usable.
    '''

    def __init__(self, path, max_chunks=DEFAULT_MAX_CHUNKS, mode='warm_buffered'):
        self._closed = False
        self._reader = None
        self._cache = OrderedDict()
        self.path = path
        self.mode = _check_mode(mode)
        self.max_chunks = _int_arg('max_chunks', max_chunks, 1)

        t0 = time.perf_counter_ns()
        reader = _Reader(path, self.mode)
        # From here on the descriptor is ours; anything that rejects the file
        # must close it rather than leak it into a dead instance.
        try:
            self._snapshot = _stat_snapshot(reader._fh.fileno())
            if reader.size <= _INDEX_HEADER.size:
                raise ValueError(f'{path}: {reader.size} bytes is too small for a chunk index')
            head = reader.read_exact(_INDEX_HEADER.size, 0, 'chunk index header')
            method, chunk_size, grid_rows, grid_cols, count = _parse_index_header(
                head, reader.size
            )
            payload_base = _INDEX_HEADER.size + count * _INDEX_ENTRY.size
            payload_bytes = reader.size - payload_base
            index_bytes = reader.read_exact(
                count * _INDEX_ENTRY.size, _INDEX_HEADER.size, 'chunk index entries'
            )
            entries = [_entry(index_bytes, i * _INDEX_ENTRY.size) for i in range(count)]
            height, width = _validate_entries(
                entries, chunk_size, grid_rows, grid_cols, payload_bytes
            )
            # Everything above is I/O plus header/index validation, so it is all
            # read_ns. Installing the state the object keeps is setup too, so it
            # stays inside the measured interval rather than after it.
            io_ns = time.perf_counter_ns() - t0

            self._reader = reader
            self._entries = entries
            self._payload_base = payload_base
            self.method = method
            self.chunk_size = chunk_size
            self.grid_rows = grid_rows
            self.grid_cols = grid_cols
            self.height = height
            self.width = width
        except BaseException:
            reader.close()
            self._reader = None
            self._closed = True
            raise
        total_ns = time.perf_counter_ns() - t0

        # No payload was read and nothing was decoded, so decode_ns here is only
        # the non-I/O setup bookkeeping and the counters are exactly the
        # constructor's own metadata reads.
        startup = _elapsed_metrics(total_ns, io_ns, self._counters())
        startup['resident_bytes'] = 0
        startup['chunk_count'] = count
        startup.update(_chunk_counters())
        self._startup_metrics = startup

    # ----------------------------------------------------------------- state

    def _counters(self):
        reader = self._reader
        return {
            'bytes_read': reader.bytes_read,
            'bytes_requested': reader.bytes_requested,
            'read_calls': reader.read_calls,
            'seek_calls': reader.seek_calls,
        }

    def _counter_delta(self, base):
        now = self._counters()
        return {key: now[key] - base[key] for key in base}

    @property
    def closed(self):
        return self._closed

    @property
    def startup_metrics(self):
        return dict(self._startup_metrics)

    @property
    def resident_bytes(self):
        '''Exact bytes of decoded chunk data retained right now.

        Sums the retained arrays' own .nbytes: the index, the stat snapshot and
        any metrics dict are excluded, and so is every array already handed to a
        caller. Bounded by max_chunks * chunk_size**2 * 2, and 0 once closed.
        '''
        return sum(block.nbytes for block in self._cache.values())

    @property
    def resident_chunks(self):
        '''Retained chunk coordinates, least- to most-recently-used.'''
        return [[row, col] for row, col in self._cache]

    def close(self):
        '''Close the pinned descriptor and drop every decoded chunk.

        Idempotent, and safe to call on a half-built or already-invalidated
        instance. The cache is cleared even if the close itself fails, so no
        decoded data outlives the object.
        '''
        self._closed = True
        reader, self._reader = self._reader, None
        try:
            if reader is not None:
                reader.close()
        finally:
            self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _check_open(self):
        if self._closed or self._reader is None:
            raise ValueError('cache is closed')

    def _check_snapshot(self):
        '''Fail closed if the pinned file changed in place since construction.'''
        current = _stat_snapshot(self._reader._fh.fileno())
        if current != self._snapshot:
            raise ValueError(
                f'{self.path}: file changed under the cache '
                f'(size,mtime_ns,ctime_ns {self._snapshot} -> {current}); '
                f'the validated index no longer describes it'
            )

    # ------------------------------------------------------------------ load

    def load(self, region):
        '''Load an arbitrary in-bounds ROI, reusing cached chunks.

        `region` is the exact extent to return -- (row, col, height, width) in
        map samples, with no alignment requirement -- and must lie fully inside
        the map; a partially out-of-bounds ROI is rejected, not clipped.

        Returns (array, metrics). The array is an owned, writable, C-contiguous
        <u2 array of exactly (height, width); it is freshly allocated and shares
        no memory with any cached chunk, so the caller may mutate it freely
        without touching cache state (the cached arrays are themselves held
        read-only). metrics has the standard keys -- total_ns (== read_ns +
        decode_ns), read_ns, decode_ns, bytes_read, bytes_requested, read_calls,
        seek_calls, where the four counters are this load's payload I/O only --
        plus:

          resident_bytes        decoded bytes retained after this load
          chunks_requested      grid cells the ROI selected
          chunks_loaded         cells whose blob had to be read (== cache_misses)
          cache_hits            cells served from the LRU
          cache_misses          cells that were not
          evictions             chunks dropped to stay within max_chunks
          chunk_coordinates     every selected [grid_row, grid_col], row-major
          hit_coordinates       the phase-1 cells, in grid row-major order
          miss_coordinates      the phase-2 cells, in grid row-major order
          evicted_coordinates   what was actually evicted, in eviction order

        total_ns is the measured elapsed time of the whole core load (see the
        module docstring): the clock starts before the snapshot check, the
        counter baseline and the output allocation, and stops after the last
        stitch and LRU update, immediately before this dict is built. read_ns is
        the I/O inside that interval; decode_ns is everything else in it.
        '''
        row, col, h, w = _check_roi(region)
        self._check_open()
        # Bounds-check against the map validated at construction before anything
        # is sized from the ROI. Argument validation is deliberately outside the
        # measured interval, exactly as in the stateless loaders.
        if row + h > self.height or col + w > self.width:
            raise ValueError(
                f'region ({row},{col},{h},{w}) does not fit in the '
                f'{self.height}x{self.width} map'
            )

        try:
            return self._load(row, col, h, w)
        except BaseException:
            # A failed read, decode or cache mutation leaves no trustworthy
            # state: invalidate rather than let the next load build on it.
            self.close()
            raise

    def _load(self, row, col, h, w):
        t0 = time.perf_counter_ns()
        read_ns = 0
        base = self._counters()

        # The snapshot check is real work on the critical path of *every* load,
        # hits included, so it is timed into read_ns even though it reads no
        # payload bytes and moves no counter.
        t = time.perf_counter_ns()
        self._check_snapshot()
        read_ns += time.perf_counter_ns() - t

        chunk_size = self.chunk_size
        first_grid_row, last_grid_row = row // chunk_size, (row + h - 1) // chunk_size
        first_grid_col, last_grid_col = col // chunk_size, (col + w - 1) // chunk_size

        out = np.empty((h, w), dtype=_DTYPE)
        coordinates = []
        hits = []
        misses = []
        evicted = []
        pending = []

        # Phase 1: the cells already resident when the request started, in grid
        # row-major order. Each is stitched and refreshed to most-recently-used
        # before any eviction is possible, so a miss later in this same request
        # can never evict a chunk this request still needs.
        for grid_row in range(first_grid_row, last_grid_row + 1):
            for grid_col in range(first_grid_col, last_grid_col + 1):
                key = (grid_row, grid_col)
                coordinates.append([grid_row, grid_col])
                entry = self._entries[grid_row * self.grid_cols + grid_col]
                block = self._cache.get(key)
                if block is None:
                    pending.append((key, entry))
                    continue
                hits.append([grid_row, grid_col])
                self._cache.move_to_end(key)
                _stitch(out, block, entry, row, col)

        # Phase 2: the rest, still in grid row-major order. Only now may the
        # cache shrink, and only to make room for the chunk being decoded.
        for key, entry in pending:
            grid_row, grid_col = key
            c_row, c_col, c_h, c_w, offset, length = entry
            label = f'chunk ({grid_row},{grid_col})'
            misses.append([grid_row, grid_col])

            t = time.perf_counter_ns()
            payload = self._reader.read_exact(
                length, self._payload_base + offset, f'{label} payload'
            )
            _validate_chunk_blob(payload, length, self.method, c_h, c_w, label)
            read_ns += time.perf_counter_ns() - t

            # Evict first, then decode: the retained set is never transiently
            # over capacity, so resident_bytes stays bounded even mid-load.
            while len(self._cache) >= self.max_chunks:
                old_key, _old = self._cache.popitem(last=False)
                evicted.append([old_key[0], old_key[1]])
            block = _freeze(_check_decoded(decode(payload), c_h, c_w, label))
            self._cache[key] = block
            _stitch(out, block, entry, row, col)
            block = None

        total_ns = time.perf_counter_ns() - t0

        metrics = _elapsed_metrics(total_ns, read_ns, self._counter_delta(base))
        metrics['resident_bytes'] = self.resident_bytes
        metrics.update(_chunk_counters(coordinates, hits, misses, evicted))
        return out, metrics


class WholeMapResident:
    '''The whole decoded map kept in memory; loads are pure crops.

    The constructor performs a real whole-map load -- town_disk.load_whole with
    no region, so the container is opened, read in full, decoded and
    sha256-verified -- and retains the resulting array. That is the entire cost
    of this arm and it is all in `startup_metrics`; the array is decode()'s own
    owned output, so the resident state is never a view of a caller's buffer or
    of any other array.

    Each load then copies the exact ROI out of the resident array with no file
    I/O whatsoever: the file is already closed by the time the constructor
    returns. Per-load I/O counters are therefore structurally zero, not
    coincidentally zero, and the chunk/cache counters are reported as zeros and
    empty lists so a per-load metrics dict lines up key-for-key with
    ChunkRegionCache's. total_ns is the measured elapsed crop, allocation
    included, with read_ns 0 and decode_ns equal to total_ns.
    '''

    def __init__(self, path, mode='warm_buffered'):
        self._closed = False
        self._data = None
        self.path = path
        self.mode = _check_mode(mode)

        # load_whole opens, reads, closes and decodes, and reports its own real
        # elapsed split with total_ns == read_ns + decode_ns by construction.
        data, startup = load_whole(path, mode=self.mode, region=None)
        self._data = _freeze(data)
        self.height, self.width = data.shape
        startup = dict(startup)
        startup['resident_bytes'] = data.nbytes
        startup.update(_chunk_counters())
        self._startup_metrics = startup

    @property
    def closed(self):
        return self._closed

    @property
    def startup_metrics(self):
        return dict(self._startup_metrics)

    @property
    def resident_bytes(self):
        '''Exact bytes of decoded map data retained right now (0 once closed).'''
        return 0 if self._data is None else self._data.nbytes

    def close(self):
        '''Drop the resident map. Idempotent.'''
        self._closed = True
        self._data = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def load(self, region):
        '''Copy an arbitrary in-bounds ROI out of the resident map.

        `region` is required and is the exact extent returned: (row, col,
        height, width), fully inside the map. Returns (array, metrics); the
        array is an owned, writable copy, never a view of the resident array, so
        mutating it cannot corrupt resident state.

        The clock starts before the output is allocated and stops after the copy
        completes, immediately before the metrics dict is built, so total_ns is
        the elapsed cost of the whole crop path and is measured on the same
        footing as a cache load. read_ns and every I/O counter are zero -- no
        file is touched -- so decode_ns == total_ns, and the chunk/cache keys are
        present but zero/empty for direct comparison with a cache load.
        '''
        row, col, h, w = _check_roi(region)
        if self._closed or self._data is None:
            raise ValueError('resident map is closed')

        t0 = time.perf_counter_ns()
        out = _crop_copy(self._data, row, col, h, w)
        total_ns = time.perf_counter_ns() - t0

        metrics = _elapsed_metrics(total_ns, 0, _zero_counters())
        metrics['resident_bytes'] = self.resident_bytes
        metrics.update(_chunk_counters())
        return out, metrics


__all__ = [
    'DEFAULT_MAX_CHUNKS',
    'SNAPSHOT_NOTE',
    'ChunkRegionCache',
    'WholeMapResident',
]

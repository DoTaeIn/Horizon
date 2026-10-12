'''Real disk-load comparison: whole-map container vs. chunked HCIX container.

Both loaders do actual file I/O every call: they open the file anew, fstat it,
seek, read exactly the bytes they need, close, and only then decode. Nothing is
cached between calls inside this module and no handle is reused, so a timing is
always "open -> read -> close -> decode", never "decode a blob somebody else
already read".

Two arms are measured:

  load_whole(path, region=...)    read the entire whole-map container, decode
                                  the full map, copy the region out of it.
  load_chunked(path, region=...)  read the HCIX header, one 32-byte index
                                  entry, and that chunk's payload (three
                                  seek+read pairs, nothing else), decode it.

With region=None both read and decode everything: load_whole decodes one
container, load_chunked reads the whole index plus the payload in one
contiguous read and assembles the grid.

Modes:

  warm_buffered   plain reads; the OS page cache is expected to be warm. This
                  module never primes it -- priming is the caller's/CLI's job,
                  so that no untimed read is hidden inside a loader.
  macos_nocache   fcntl(fd, F_NOCACHE, 1) on the freshly opened descriptor
                  (Darwin only, no fallback: it raises if unavailable). That
                  call *requests* that the OS not cache this descriptor's
                  data; it is not a verified cache bypass. This module does
                  not check whether a read actually missed the page cache, it
                  evicts nothing that is already cached, and it does not touch
                  the SSD/controller/drive caches. A macos_nocache timing can
                  therefore still be served from cached data and is never a
                  cold-hardware read.

Every file handle is opened with buffering=0 and every read asks for an exact
byte count, so Python never reads ahead past a partial read's selected range
and `read_calls`/`bytes_read` reflect what was really asked of the kernel.

read_ns covers the whole I/O phase (open, F_NOCACHE, fstat, seeks, header and
index parsing, reads, close); decode_ns covers decoding plus materializing the
owned region copy. There is no whole-file hashing inside a timed load; the only
digest work is the per-container sha256 that horizon.codec.decode always does
on its own output, which both arms pay identically.

The HCIX container written here is byte-for-byte the one
horizon.chunk_benchmark builds (its private, experimental benchmark format, not
a production format). Unlike that module's parser, the reader here treats the
bytes on disk as untrusted: every count, coordinate, shape, offset and length
is bounds-checked, and the full-read path additionally verifies the entries are
contiguous, non-overlapping and cover the whole map exactly.
'''

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import struct
import time
from pathlib import Path

try:  # POSIX only; macos_nocache refuses to run without it (see _check_mode).
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX hosts
    fcntl = None

import numpy as np

from horizon.benchmark import validate_heightmap
from horizon.chunk_benchmark import (
    _INDEX_ENTRY,
    _INDEX_HEADER,
    _INDEX_MAGIC,
    _INDEX_VERSION,
    _build_container,
    _chunk_entries,
    _encode_chunks,
    _grid_dims,
)
from horizon.codec import HEADER_SIZE, MAX_RAW_BYTES, METHODS, VERSION, decode, encode

MODES = ('warm_buffered', 'macos_nocache')
DEFAULT_CHUNK_SIZE = 256

# Caps every allocation this module makes from on-disk numbers. MAX_RAW_BYTES is
# the codec's own 64 MiB raw-samples limit; a container can only be larger than
# its raw map by the headers/index, and incompressible payloads stay near raw.
MAX_FILE_BYTES = MAX_RAW_BYTES + (1 << 20)
MAX_INDEX_ENTRIES = (MAX_FILE_BYTES - _INDEX_HEADER.size) // _INDEX_ENTRY.size

# /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/fcntl.h:
#     #define F_NOCACHE       48      /* turn data caching off/on for this fd */
F_NOCACHE = 48

WHOLE_NAME = 'whole.hrzn'
CHUNKED_NAME = 'chunked.hcix'

_DTYPE = np.dtype('<u2')
_U32_MAX = 0xFFFFFFFF
# Mirrors the documented horizon.codec container header (magic, version,
# method, predictor, reserved, height, width, payload length, sha256). Parsed
# here only to learn how many payload bytes to read; decode() re-validates the
# header in full, including its CRC.
_CODEC_HEADER = struct.Struct('<4sBBBBIIQ32s')
_CODEC_MAGIC = b'HRZN'

CACHE_NOTE = (
    'macos_nocache requests per-descriptor caching be disabled via '
    'F_NOCACHE (48); this is a request to the kernel, not a verified bypass. '
    'Nothing here confirms the reads actually missed the page cache, no '
    'already-cached pages are evicted, and the SSD/controller caches are '
    'untouched, so a macos_nocache timing is not a cold-hardware read and may '
    'still be served from cached data'
)
CONTAINER_NOTE = (
    'private experimental HCIX benchmark container (same bytes as '
    'horizon.chunk_benchmark), not a production format; the reader here '
    'validates it as untrusted input'
)


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #

def _int_arg(name, value, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return value


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}; expected one of {MODES}')
    if mode == 'macos_nocache':
        if platform.system() != 'Darwin':
            raise OSError(
                f'macos_nocache requires Darwin (F_NOCACHE); this is {platform.system()!r}'
            )
        if fcntl is None:
            raise OSError('macos_nocache requires the fcntl module')
    return mode


def _check_method(method):
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f'unknown method {method!r}; expected one of {METHODS}')
    return method


def _check_region(region):
    '''Normalize region=(row, col, height, width); None means "everything".'''
    if region is None:
        return None
    if isinstance(region, (str, bytes)) or not hasattr(region, '__len__'):
        raise ValueError('region must be a (row, col, height, width) sequence')
    if len(region) != 4:
        raise ValueError('region must have exactly 4 elements (row, col, height, width)')
    row = _int_arg('region row', region[0], 0, _U32_MAX)
    col = _int_arg('region col', region[1], 0, _U32_MAX)
    h = _int_arg('region height', region[2], 1, _U32_MAX)
    w = _int_arg('region width', region[3], 1, _U32_MAX)
    return row, col, h, w


# --------------------------------------------------------------------------- #
# counted, exact-size reader
# --------------------------------------------------------------------------- #

class _Reader:
    '''Unbuffered handle that only ever reads exact, pre-validated sizes.

    buffering=0 means no Python-level read-ahead: a 32-byte index-entry read
    asks the kernel for 32 bytes, not for a block. Short reads are retried and
    every attempt is counted, so read_calls/bytes_requested/bytes_read describe
    the real I/O instead of an idealized one.
    '''

    def __init__(self, path, mode):
        self.read_calls = 0
        self.bytes_requested = 0
        self.bytes_read = 0
        self.seek_calls = 0
        self._fh = open(path, 'rb', buffering=0)
        # Everything that can fail after the open lives in this try, size
        # validation included: a rejected file must not leak its descriptor.
        try:
            if mode == 'macos_nocache':
                # No fallback on purpose: a silently-cached run would be a
                # measurement of the wrong thing.
                fcntl.fcntl(self._fh.fileno(), F_NOCACHE, 1)
            self.size = os.fstat(self._fh.fileno()).st_size
            if self.size <= 0:
                raise ValueError(f'{path}: file is empty')
            if self.size > MAX_FILE_BYTES:
                raise ValueError(
                    f'{path}: {self.size} bytes exceeds the {MAX_FILE_BYTES} byte limit'
                )
        except BaseException:
            self._fh.close()
            raise

    def seek(self, offset):
        if not 0 <= offset <= self.size:
            raise ValueError(f'seek offset {offset} outside the {self.size}-byte file')
        self._fh.seek(offset)
        self.seek_calls += 1

    def read_exact(self, size, offset, label):
        '''Read exactly `size` bytes starting at `offset`, or raise.

        The range is checked against the fstat'd size *before* a buffer is
        allocated, so a corrupt length field cannot provoke a giant allocation.
        '''
        size = _int_arg(f'{label} size', size, 1)
        if size > MAX_FILE_BYTES or offset < 0 or offset + size > self.size:
            raise ValueError(
                f'{label}: range [{offset}, {offset + size}) outside the {self.size}-byte file'
            )
        self.seek(offset)
        buf = bytearray(size)
        view = memoryview(buf)
        got = 0
        while got < size:
            self.read_calls += 1
            self.bytes_requested += size - got
            n = self._fh.readinto(view[got:])
            if not n:
                raise ValueError(f'{label}: file truncated at {offset + got} ({size} wanted)')
            self.bytes_read += n
            got += n
        return bytes(buf)

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _metrics(t0, t_read, t_end, reader):
    return {
        'total_ns': t_end - t0,
        'read_ns': t_read - t0,
        'decode_ns': t_end - t_read,
        'bytes_read': reader.bytes_read,
        'bytes_requested': reader.bytes_requested,
        'read_calls': reader.read_calls,
        'seek_calls': reader.seek_calls,
    }


# --------------------------------------------------------------------------- #
# header parsing
# --------------------------------------------------------------------------- #

def _parse_codec_header(blob, label):
    '''Return (method, height, width, payload_len) from a codec container header.'''
    magic, version, method_id, _pred, _res, height, width, plen, _digest = (
        _CODEC_HEADER.unpack_from(blob)
    )
    if magic != _CODEC_MAGIC:
        raise ValueError(f'{label}: bad container magic')
    if version != VERSION:
        raise ValueError(f'{label}: unsupported container version {version}')
    if method_id >= len(METHODS):
        raise ValueError(f'{label}: unknown container method id {method_id}')
    if height < 1 or width < 1:
        raise ValueError(f'{label}: non-positive container dimensions')
    if height * width * _DTYPE.itemsize > MAX_RAW_BYTES:
        raise ValueError(f'{label}: container dimensions exceed the {MAX_RAW_BYTES} byte raw limit')
    if plen < 1 or plen > MAX_FILE_BYTES:
        raise ValueError(f'{label}: implausible payload length {plen}')
    return METHODS[method_id], int(height), int(width), int(plen)


def _parse_index_header(blob, file_size):
    '''Validate the HCIX header and return (method, chunk_size, rows, cols, count).

    The header carries no map dimensions, so full dims are derived later from
    the entries; a partial read never needs them.
    '''
    magic, version, method_id, reserved, chunk_size, rows, cols, count = (
        _INDEX_HEADER.unpack_from(blob)
    )
    if magic != _INDEX_MAGIC:
        raise ValueError('bad chunk index magic')
    if version != _INDEX_VERSION:
        raise ValueError(f'unsupported chunk index version {version}')
    if method_id >= len(METHODS):
        raise ValueError(f'unknown chunk index method id {method_id}')
    if reserved != 0:
        raise ValueError('chunk index reserved field must be 0')
    if chunk_size < 1 or rows < 1 or cols < 1 or count < 1:
        raise ValueError('chunk index grid must be positive')
    if count > MAX_INDEX_ENTRIES or rows * cols != count:
        raise ValueError(f'chunk index count {count} inconsistent with grid {rows}x{cols}')
    # The smallest map a rows x cols grid at this chunk_size can describe has
    # full chunks everywhere except a 1-sample-wide last row/column. Rejecting
    # that lower bound against the codec's raw limit kills an absurd grid
    # before any index buffer is sized from `count`.
    min_height = (rows - 1) * chunk_size + 1
    min_width = (cols - 1) * chunk_size + 1
    if min_height * min_width * _DTYPE.itemsize > MAX_RAW_BYTES:
        raise ValueError(
            f'chunk index grid {rows}x{cols} at chunk_size={chunk_size} needs at least '
            f'{min_height}x{min_width} samples, over the {MAX_RAW_BYTES} byte raw limit'
        )
    # Every chunk costs an index entry plus at least a bare codec header, so a
    # file too small to hold that much cannot have this many chunks.
    required = _INDEX_HEADER.size + count * (_INDEX_ENTRY.size + HEADER_SIZE)
    if required > file_size:
        raise ValueError(
            f'chunk index of {count} entries needs at least {required} bytes, '
            f'file has {file_size}'
        )
    return METHODS[method_id], int(chunk_size), int(rows), int(cols), int(count)


def _entry(blob, offset=0):
    row, col, h, w, chunk_offset, length = _INDEX_ENTRY.unpack_from(blob, offset)
    return int(row), int(col), int(h), int(w), int(chunk_offset), int(length)


def _validate_entries(entries, chunk_size, rows, cols, payload_bytes):
    '''Check the index describes a contiguous, gap-free, non-overlapping cover.

    The on-disk entries are in grid row-major order, so rebuilding the expected
    grid from (chunk_size, rows, cols) and the dimensions implied by the last
    entry validates coordinates, shapes, completeness and non-overlap in one
    pass. Offsets must then start at 0, be strictly consecutive, and end exactly
    at the last payload byte.
    '''
    last_row, last_col, last_h, last_w, _off, _len = entries[-1]
    height, width = last_row + last_h, last_col + last_w
    if height * width * _DTYPE.itemsize > MAX_RAW_BYTES:
        raise ValueError(f'chunk index implies {height}x{width}, over the raw size limit')
    if _grid_dims(height, width, chunk_size) != (rows, cols):
        raise ValueError(
            f'chunk index grid {rows}x{cols} does not match the {height}x{width} map '
            f'it describes at chunk_size={chunk_size}'
        )
    expected = _chunk_entries(height, width, chunk_size)
    if len(expected) != len(entries):
        raise ValueError('chunk index entry count does not match its grid')
    cursor = 0
    for (want_row, want_col, want_h, want_w), got in zip(expected, entries):
        row, col, h, w, offset, length = got
        if (row, col, h, w) != (want_row, want_col, want_h, want_w):
            raise ValueError(
                f'chunk index entry ({row},{col},{h},{w}) is not the expected grid cell '
                f'({want_row},{want_col},{want_h},{want_w})'
            )
        if offset != cursor:
            raise ValueError(f'chunk at ({row},{col}) starts at {offset}, expected {cursor}')
        if length < HEADER_SIZE or offset + length > payload_bytes:
            raise ValueError(f'chunk at ({row},{col}) has out-of-range length {length}')
        cursor = offset + length
    if cursor != payload_bytes:
        raise ValueError(f'chunk payloads cover {cursor} of {payload_bytes} bytes')
    return height, width


def _check_decoded(block, h, w, label):
    if block.shape != (h, w) or block.dtype != _DTYPE:
        raise ValueError(f'{label}: decoded {block.shape}/{block.dtype}, expected {(h, w)}/<u2')
    return block


def _crop_copy(decoded, row, col, h, w):
    '''Own the region outright (a copy, not a view) so both arms return arrays
    owned exactly the way decode() always owns its output.'''
    height, width = decoded.shape
    if row + h > height or col + w > width:
        raise ValueError(
            f'region ({row},{col},{h},{w}) does not fit in the {height}x{width} map'
        )
    return decoded[row:row + h, col:col + w].copy()


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #

def _write_exclusive(path, data):
    '''Create path (never overwrite), write it fully, fsync it.

    The O_EXCL open happens outside the cleanup block on purpose: if it fails
    because something already lives at `path`, that something is not ours and
    must survive. Only a file this call actually created is unlinked.
    '''
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise
    return len(data)


def _fsync_dir(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare_case(arr, directory, method, chunk_size=DEFAULT_CHUNK_SIZE):
    '''Write the whole-map and chunked containers for `arr` into a new directory.

    `directory` must not already exist: its final component is created
    exclusively, so a run can never overwrite or silently reuse another run's
    containers. Missing parents are created if needed and are left alone
    afterwards, whether or not they already existed.

    Both containers are encoded and size-checked *before* anything is created
    on disk, so an encoding failure leaves the filesystem untouched. Each file
    is then written with O_EXCL and fsynced, and the directory entry is fsynced
    too. If any write or fsync fails, only the files this call created and the
    directory it created are removed; a pre-existing path is never deleted.

    Returns a JSON-ready dict (paths are str) describing what was written,
    including the sha256 of each container file and of the source samples.
    '''
    validate_heightmap(arr, label='arr')
    method = _check_method(method)
    chunk_size = _int_arg('chunk_size', chunk_size, 1, _U32_MAX)

    arr = np.ascontiguousarray(arr)
    height, width = arr.shape
    raw = arr.tobytes()

    whole_blob = encode(arr, method=method)
    entries = _chunk_entries(height, width, chunk_size)
    grid_rows, grid_cols = _grid_dims(height, width, chunk_size)
    blobs = _encode_chunks(arr, entries, method)
    chunked_blob, index_bytes = _build_container(
        method, chunk_size, grid_rows, grid_cols, entries, blobs
    )
    for name, blob in (('whole', whole_blob), ('chunked', chunked_blob)):
        if len(blob) > MAX_FILE_BYTES:
            raise ValueError(f'{name} container is {len(blob)} bytes, over the file limit')

    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # exclusive: raises FileExistsError if it already exists
    whole_path = directory / WHOLE_NAME
    chunked_path = directory / CHUNKED_NAME
    written = []
    try:
        _fsync_dir(directory)
        whole_bytes = _write_exclusive(whole_path, whole_blob)
        written.append(whole_path)
        chunked_bytes = _write_exclusive(chunked_path, chunked_blob)
        written.append(chunked_path)
        _fsync_dir(directory)
    except BaseException:
        # Undo exactly what this call made: the files it wrote, then the
        # directory it created (rmdir, so a non-empty dir is left for
        # inspection rather than recursively deleted).
        for path in written:
            with contextlib.suppress(OSError):
                os.unlink(path)
        with contextlib.suppress(OSError):
            os.rmdir(directory)
        raise

    return {
        'whole_path': str(whole_path),
        'chunked_path': str(chunked_path),
        'shape': [int(height), int(width)],
        'method': method,
        'chunk_size': chunk_size,
        'grid': [int(grid_rows), int(grid_cols)],
        'chunk_count': len(entries),
        'index_bytes': len(index_bytes),
        'whole_bytes': whole_bytes,
        'chunked_bytes': chunked_bytes,
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'hashes': {
            'whole_sha256': hashlib.sha256(whole_blob).hexdigest(),
            'chunked_sha256': hashlib.sha256(chunked_blob).hexdigest(),
        },
        'container_note': CONTAINER_NOTE,
    }


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def load_whole(path, mode='warm_buffered', region=None):
    '''Load the whole-map container from disk; crop `region` out after decoding.

    Reads the header, then the payload, then closes the file: the whole
    container has to come off disk and be decoded even when only a region is
    wanted. region=None returns the full map. Returns (array, metrics).
    '''
    mode = _check_mode(mode)
    region = _check_region(region)

    t0 = time.perf_counter_ns()
    with _Reader(path, mode) as reader:
        if reader.size <= HEADER_SIZE:
            raise ValueError(f'{path}: {reader.size} bytes is too small for a container')
        header = reader.read_exact(HEADER_SIZE, 0, 'container header')
        _c_method, height, width, plen = _parse_codec_header(header, 'container header')
        if HEADER_SIZE + plen != reader.size:
            raise ValueError(
                f'{path}: header claims {HEADER_SIZE + plen} bytes, file has {reader.size}'
            )
        payload = reader.read_exact(plen, HEADER_SIZE, 'container payload')
    t_read = time.perf_counter_ns()

    decoded = _check_decoded(decode(header + payload), height, width, 'whole container')
    if region is None:
        result = decoded
    else:
        result = _crop_copy(decoded, *region)
    t_end = time.perf_counter_ns()
    return result, _metrics(t0, t_read, t_end, reader)


def load_chunked(path, mode='warm_buffered', region=None):
    '''Load from the chunked HCIX container on disk.

    region=None: read the index, validate it covers the map exactly, read all
    payload bytes in one contiguous read, and assemble the full map.

    region=(row, col, height, width): read only the fixed-size header, only the
    one 32-byte index entry the region selects (located by grid arithmetic from
    the header -- the other entries are never read or scanned), and only that
    chunk's payload. The region must be the exact extent of a single grid chunk:
    chunk-size aligned, exactly chunk_size in any interior grid row/column, and
    equal to the stored entry's height/width -- which for a last-row/last-column
    chunk is the shrunken edge size. The selected chunk's own container header
    must also agree with the index on both dimensions and on the codec method.

    Returns (array, metrics).
    '''
    mode = _check_mode(mode)
    region = _check_region(region)

    t0 = time.perf_counter_ns()
    with _Reader(path, mode) as reader:
        if reader.size <= _INDEX_HEADER.size:
            raise ValueError(f'{path}: {reader.size} bytes is too small for a chunk index')
        head = reader.read_exact(_INDEX_HEADER.size, 0, 'chunk index header')
        method, chunk_size, grid_rows, grid_cols, count = _parse_index_header(head, reader.size)
        payload_base = _INDEX_HEADER.size + count * _INDEX_ENTRY.size
        payload_bytes = reader.size - payload_base

        if region is None:
            index_bytes = reader.read_exact(
                count * _INDEX_ENTRY.size, _INDEX_HEADER.size, 'chunk index entries'
            )
            entries = [_entry(index_bytes, i * _INDEX_ENTRY.size) for i in range(count)]
            height, width = _validate_entries(
                entries, chunk_size, grid_rows, grid_cols, payload_bytes
            )
            # Entries are validated contiguous and gap-free, so the payload is
            # one contiguous span: a single read, not count reads.
            payload = reader.read_exact(payload_bytes, payload_base, 'chunk payloads')
            plan = None
        else:
            row, col, h, w = region
            if row % chunk_size or col % chunk_size:
                raise ValueError(
                    f'region ({row},{col}) is not aligned to chunk_size={chunk_size}'
                )
            grid_row, grid_col = row // chunk_size, col // chunk_size
            if grid_row >= grid_rows or grid_col >= grid_cols:
                raise ValueError(
                    f'region chunk ({grid_row},{grid_col}) outside the {grid_rows}x{grid_cols} grid'
                )
            if h > chunk_size or w > chunk_size:
                raise ValueError(f'region {h}x{w} is larger than chunk_size={chunk_size}')
            # Only the last grid row/column may hold a shrunken edge chunk;
            # every interior cell is exactly chunk_size. Checking that against
            # the grid from the header needs no other entry.
            if grid_row < grid_rows - 1 and h != chunk_size:
                raise ValueError(
                    f'region height {h} must be chunk_size={chunk_size} for interior grid '
                    f'row {grid_row} of {grid_rows}'
                )
            if grid_col < grid_cols - 1 and w != chunk_size:
                raise ValueError(
                    f'region width {w} must be chunk_size={chunk_size} for interior grid '
                    f'column {grid_col} of {grid_cols}'
                )
            index = grid_row * grid_cols + grid_col
            entry_bytes = reader.read_exact(
                _INDEX_ENTRY.size,
                _INDEX_HEADER.size + index * _INDEX_ENTRY.size,
                'chunk index entry',
            )
            e_row, e_col, e_h, e_w, offset, length = _entry(entry_bytes)
            if (e_row, e_col, e_h, e_w) != (row, col, h, w):
                raise ValueError(
                    f'index entry {index} describes ({e_row},{e_col},{e_h},{e_w}), '
                    f'not the requested region ({row},{col},{h},{w})'
                )
            if length < HEADER_SIZE or offset < 0 or offset + length > payload_bytes:
                raise ValueError(
                    f'chunk {index} range [{offset}, {offset + length}) outside the '
                    f'{payload_bytes}-byte payload'
                )
            payload = reader.read_exact(length, payload_base + offset, 'chunk payload')
            c_method, c_h, c_w, c_plen = _parse_codec_header(payload, f'chunk {index}')
            if (c_h, c_w) != (h, w):
                raise ValueError(
                    f'chunk {index} container declares {c_h}x{c_w}, index says {h}x{w}'
                )
            if c_method != method:
                raise ValueError(
                    f'chunk {index} is encoded with {c_method}, index header says {method}'
                )
            if HEADER_SIZE + c_plen != length:
                raise ValueError(f'chunk {index} payload length {c_plen} disagrees with {length}')
            plan = (h, w)
    t_read = time.perf_counter_ns()

    if plan is None:
        out = np.empty((height, width), dtype=_DTYPE)
        for row, col, h, w, offset, length in entries:
            block = decode(payload[offset:offset + length])
            out[row:row + h, col:col + w] = _check_decoded(block, h, w, f'chunk ({row},{col})')
        result = out
    else:
        sel_h, sel_w = plan
        result = _check_decoded(decode(payload), sel_h, sel_w, 'selected chunk')
    t_end = time.perf_counter_ns()
    return result, _metrics(t0, t_read, t_end, reader)


__all__ = [
    'CACHE_NOTE',
    'CONTAINER_NOTE',
    'DEFAULT_CHUNK_SIZE',
    'F_NOCACHE',
    'MAX_FILE_BYTES',
    'MODES',
    'load_chunked',
    'load_whole',
    'prepare_case',
]

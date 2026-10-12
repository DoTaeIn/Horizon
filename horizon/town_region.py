'''Arbitrary-ROI loading from the chunked HCIX container on disk.

horizon.town_disk.load_chunked's region path is deliberately narrow: the region
must be exactly one grid chunk, so it can locate a single index entry by
arithmetic and never look at the rest of the index. This module answers the
other question -- an arbitrary rectangle, which in general straddles a 2-D
block of chunks -- while keeping the same discipline:

  * every call does real I/O: open, fstat, seek, read exact sizes, close, and
    only then decode. Nothing is cached here, no handle outlives a call, and
    the metadata (header + index) is read inside the timed call too, so a
    timing is never "decode blobs somebody else already read".
  * the file is untrusted input. The header, the full index and every selected
    chunk's own container header are validated with town_disk's existing
    checkers before a single output byte is allocated.
  * only the chunk blobs the ROI actually intersects are read. The index is
    read in full because it is small, fixed-size metadata (32 bytes per chunk)
    and validating the cover needs all of it; the payload is not.

read_ns covers the whole I/O phase (open, F_NOCACHE, fstat, header and index
parsing and validation, the per-chunk seeks and reads, close). decode_ns covers
decoding the chunks and stitching their cropped intersections into the exact
ROI. Metrics are town_disk.load_whole's keys plus chunks_loaded and
chunk_coordinates, so an ROI load is directly comparable with a whole-map load.

The container format is the same private experimental HCIX benchmark container
horizon.chunk_benchmark writes (see town_disk.CONTAINER_NOTE); this module adds
no format of its own.
'''

from __future__ import annotations

import time

import numpy as np

from horizon.chunk_benchmark import _INDEX_ENTRY, _INDEX_HEADER
from horizon.codec import HEADER_SIZE, decode
from horizon.town_disk import (
    _DTYPE,
    _check_decoded,
    _check_mode,
    _check_region,
    _entry,
    _metrics,
    _parse_codec_header,
    _parse_index_header,
    _Reader,
    _validate_entries,
)


def _check_roi(region):
    '''Normalize a required (row, col, height, width) ROI.

    Unlike town_disk's loaders there is no region=None spelling here: an ROI
    load with no ROI is a whole-map load, which load_whole/load_chunked already
    do, so None is an error rather than a silent full read.
    '''
    if region is None:
        raise ValueError('region is required; use load_whole/load_chunked for the full map')
    return _check_region(region)


def load_region(path, region, mode='warm_buffered'):
    '''Load an arbitrary rectangular ROI from the chunked HCIX container.

    `region` is required and is the exact extent to return: (row, col, height,
    width), in map samples, with no alignment requirement. It must lie fully
    inside the map the container describes -- a partially out-of-bounds ROI is
    rejected rather than clipped.

    The call reads the index header and the full (small) index, validates that
    the index describes a contiguous, gap-free, non-overlapping cover of the
    map, then reads only the payload blobs of the chunks the ROI intersects --
    one exact-size read per chunk, nothing else. The file is closed before any
    decoding happens. Each chunk is then decoded, checked against the shape the
    index claims for it, and its intersection with the ROI is cropped and
    copied into the output.

    Returns (array, metrics). The array is an owned, C-contiguous <u2 array of
    exactly (height, width); metrics has load_whole's keys plus:

      chunks_loaded      how many chunk blobs were read
      chunk_coordinates  [[grid_row, grid_col], ...] of those chunks, in grid
                         row-major order
    '''
    mode = _check_mode(mode)
    row, col, h, w = _check_roi(region)

    t0 = time.perf_counter_ns()
    with _Reader(path, mode) as reader:
        if reader.size <= _INDEX_HEADER.size:
            raise ValueError(f'{path}: {reader.size} bytes is too small for a chunk index')
        head = reader.read_exact(_INDEX_HEADER.size, 0, 'chunk index header')
        method, chunk_size, grid_rows, grid_cols, count = _parse_index_header(head, reader.size)
        payload_base = _INDEX_HEADER.size + count * _INDEX_ENTRY.size
        payload_bytes = reader.size - payload_base

        # The index is fixed-size metadata (32 bytes per chunk) and validating
        # the cover needs all of it; reading it whole is what lets the ROI be
        # arbitrary instead of chunk-aligned.
        index_bytes = reader.read_exact(
            count * _INDEX_ENTRY.size, _INDEX_HEADER.size, 'chunk index entries'
        )
        entries = [_entry(index_bytes, i * _INDEX_ENTRY.size) for i in range(count)]
        height, width = _validate_entries(
            entries, chunk_size, grid_rows, grid_cols, payload_bytes
        )

        # Bounds-check the ROI against the validated map before anything is
        # sized from it.
        if row + h > height or col + w > width:
            raise ValueError(
                f'region ({row},{col},{h},{w}) does not fit in the {height}x{width} map'
            )

        # The validated index is the grid in row-major order, so the chunks the
        # ROI touches are a rectangular span of grid cells.
        first_grid_row, last_grid_row = row // chunk_size, (row + h - 1) // chunk_size
        first_grid_col, last_grid_col = col // chunk_size, (col + w - 1) // chunk_size

        selected = []
        for grid_row in range(first_grid_row, last_grid_row + 1):
            for grid_col in range(first_grid_col, last_grid_col + 1):
                index = grid_row * grid_cols + grid_col
                c_row, c_col, c_h, c_w, offset, length = entries[index]
                payload = reader.read_exact(
                    length, payload_base + offset, f'chunk ({grid_row},{grid_col}) payload'
                )
                # The chunk's own container header must agree with the index on
                # shape and method, exactly as load_chunked requires.
                b_method, b_h, b_w, b_plen = _parse_codec_header(
                    payload, f'chunk ({grid_row},{grid_col})'
                )
                if (b_h, b_w) != (c_h, c_w):
                    raise ValueError(
                        f'chunk ({grid_row},{grid_col}) container declares {b_h}x{b_w}, '
                        f'index says {c_h}x{c_w}'
                    )
                if b_method != method:
                    raise ValueError(
                        f'chunk ({grid_row},{grid_col}) is encoded with {b_method}, '
                        f'index header says {method}'
                    )
                if HEADER_SIZE + b_plen != length:
                    raise ValueError(
                        f'chunk ({grid_row},{grid_col}) payload length {b_plen} disagrees '
                        f'with {length}'
                    )
                selected.append((grid_row, grid_col, c_row, c_col, c_h, c_w, payload))
    t_read = time.perf_counter_ns()

    out = np.empty((h, w), dtype=_DTYPE)
    for grid_row, grid_col, c_row, c_col, c_h, c_w, payload in selected:
        block = _check_decoded(
            decode(payload), c_h, c_w, f'chunk ({grid_row},{grid_col})'
        )
        # Intersection of this chunk with the ROI, in map coordinates, then the
        # same window expressed in chunk-local and ROI-local coordinates.
        top, left = max(c_row, row), max(c_col, col)
        bottom, right = min(c_row + c_h, row + h), min(c_col + c_w, col + w)
        out[top - row:bottom - row, left - col:right - col] = (
            block[top - c_row:bottom - c_row, left - c_col:right - c_col]
        )
    t_end = time.perf_counter_ns()

    metrics = _metrics(t0, t_read, t_end, reader)
    metrics['chunks_loaded'] = len(selected)
    metrics['chunk_coordinates'] = [[gr, gc] for gr, gc, *_rest in selected]
    return out, metrics


__all__ = ['load_region']

# Town-map real-disk benchmark method

`scripts/compare_town_disk.py` measures real disk I/O for loading the same
synthetic town heightmap used by `scripts/compare_town.py`, comparing a
whole-map container against a chunked (`chunk_size=256`) container, on both
a full-map workload and a single-chunk partial workload. It is a thin CLI
around two library modules it does not implement:

- `horizon.town_map.make_town_map(size, seed)` — builds the map (see
  `docs/town-benchmark-method.md` for provenance notes; unchanged here).
- `horizon.town_disk` — `prepare_case(arr, directory, method, chunk_size=256)`
  writes the on-disk containers; `load_whole(path, mode, region=None)` and
  `load_chunked(path, mode, region=None)` each do real file I/O (open, seek,
  read, close, decode) and return `(array, metrics)`.

This document describes what the CLI measures and its boundaries. It does
not re-document `horizon.town_disk`'s internals (see that module's own
docstrings), and it does not claim to improve or extend them.

## What is measured

For one `--size`/`--seed`, and for every method in `horizon.codec.METHODS`
(`zstd`, `lz4`, `delta_zstd`):

1. `prepare_case` writes one whole-map container and one `chunk_size=256`
   chunked (HCIX) container for that method into its own exclusive
   subdirectory under `--containers-dir`.
2. For every requested `--modes` entry, two workloads are timed:
   - **full** — `load_whole(region=None)` vs `load_chunked(region=None)`:
     both reconstruct the entire map from disk.
   - **partial** — `load_whole(region=chunk)` (decodes the *whole* container,
     then crops an owned copy out of it) vs `load_chunked(region=chunk)` (a
     true single-chunk disk read: header, one index entry, that chunk's
     payload only — see `horizon.town_disk.load_chunked`).

Every measured call's output is validated (exact `ndarray` type, shape,
dtype, byte content) **after** its timer has already stopped, so validation
never adds to a measured path. Container files are sha256'd before and after
the whole run; any mismatch aborts with an error, since a changing container
would invalidate every timing already taken against it.

## Workload region selection

`region` is always one exact, grid-aligned `chunk_size=256` chunk — row/col
multiples of 256, with edge chunks shrunk to fit the map exactly the way
`horizon.chunk_benchmark._chunk_entries` does (and `horizon.town_disk`
re-validates on read). There is no arbitrary-rectangle or multi-chunk ROI
engine: a region spanning more than one grid chunk, or not aligned to the
grid, is not something this benchmark or `horizon.town_disk` supports or
measures.

Per repeat, a pixel coordinate `(row_px, col_px)` is drawn **uniformly** from
the run's seeded RNG via `Generator.integers(0, size)`, independently per
axis. `integers`' upper bound is exclusive, so the draw's range is exactly
`[0, size-1]` — every pixel, including the last row/column, is reachable
with equal probability. (An earlier version of this script drew from
`Generator.beta(0.5, 0.5)` scaled by `size - 1`; since `beta(0.5, 0.5)` has
support `(0, 1)` and essentially never returns exactly `1.0`, `int(x * (size
- 1))` could not land on pixel `size - 1`, so the last row/column was
unreachable. That bias is why this draw is now uniform.) The pixel is mapped
to the grid chunk containing it. One schedule entry is drawn per repeat
(length always equals `--repeats`), and the same per-repeat schedule of
pixels/chunks is reused for every method and mode, so the same disk offsets
are compared across them. Both the drawn pixel and the resulting chunk are
recorded.

## Priming and cache modes

- `warm_buffered`: before **every** measured `load_whole`/`load_chunked`
  call — whichever workload and whichever arm (whole or chunked) is about to
  run — this script does one untimed, ordinary buffered read of **both**
  the whole-map container file and the chunked container file for that
  method, discarding the bytes. Priming both files every time (not just the
  one about to be measured) keeps both arms symmetrically warmed, so a
  comparison never advantages one arm because its file happened to be read
  more recently.
- `macos_nocache`: this script does **no** priming. That is a deliberate
  choice to keep the arm meaningful (nothing this script does would un-cache
  the page cache anyway), not a claim of a cold read. `horizon.town_disk`
  only *requests* `fcntl(fd, F_NOCACHE, 1)` on the reading descriptor — that
  request has not been verified by this script or that module to evict any
  page, and has not been verified to reach the SSD, drive, or
  storage-controller caches. It is also **not** a claim that the OS page
  cache never serves this read from pages already resident from an earlier
  warmup, repeat, or process: `F_NOCACHE` is documented (on Darwin) to stop
  this descriptor from retaining newly read data in the cache, not to force
  every read on this descriptor to bypass pages the cache already holds.
  Available only on Darwin; the CLI fails loudly (no fallback) if requested
  elsewhere.

Every path gets `--warmups` (2) full untimed passes of both workloads before
any timed repeat, independent of priming, so the first *measured* repeat is
never also the first time a given code path has run.

## Ordering

Execution order of the measured pair (whole vs. chunked) within a workload
alternates by repeat parity — even repeats run whole/whole-crop first, odd
repeats run chunked/chunk-read first — independently for the full and
partial workloads, the same alternation strategy `horizon.chunk_benchmark`
uses for its in-memory comparisons. Method and mode ordering is **not**
alternated: the outer loops are method, then mode, in the fixed order
`horizon.codec.METHODS` and `--modes` were given in. This is recorded as a
static-order caveat rather than implemented as a further alternation, since
the per-repeat whole/chunk swap already addresses the dominant source of
ordering bias within a single (method, mode) cell, and alternating method or
mode order would multiply the number of container files kept warm/cold at
once without a clear benefit.

## What is recorded

Per `(method, mode)`, for both the full and partial workloads: raw
nanosecond samples for `total_ns`/`read_ns`/`decode_ns`, plus `bytes_read`,
`bytes_requested`, `read_calls` and `seek_calls` per repeat, and
`median_ms`/`p95_ms` (`numpy.percentile`, linear interpolation) computed
from the raw samples. `bytes_read`/`bytes_requested`/`read_calls`/
`seek_calls` are the application-level `read()`/`seek()` call counts
`horizon.town_disk`'s reader itself tallies — they describe what Python
asked the kernel for, not a measurement of physical SSD or storage
controller transfer size. The partial workload also records, per repeat,
the drawn `(row_px, col_px)` and the resulting exact `(row, col, height,
width)` chunk.

The report additionally records: the environment (Python, platform,
library versions); sha256 of every `horizon/*.py` source file and of this
script, so a report can be tied to the exact code that produced it; the
generated map's shape/dtype/sha256; and each container file's sha256 before
and after the run.

## I/O boundaries and what this does *not* measure

- **Chunk-index overhead is included, not subtracted.** The chunked
  container's header and index bytes are read as part of every chunked
  timing (`load_chunked`'s `read_ns` and `bytes_read` reflect them); this
  benchmark does not try to isolate "pure payload" cost from index cost.
- **No arbitrary-ROI engine.** As above: `region` only ever addresses one
  exact grid chunk. A real "read any rectangular window" engine, spanning
  multiple chunks or arbitrary offsets, is out of scope for both
  `horizon.town_disk` and this script.
- **`macos_nocache` is not a verified cold read.** See "Priming and cache
  modes" above — the `F_NOCACHE` request is not verified to evict anything,
  does not reach device/controller/drive caches, and does not guarantee the
  OS page cache skips pages it already holds.
- **Byte/call counts are application syscall counts, not physical transfer
  measurements.** See "What is recorded" above.
- **Single machine, single process.** No claim is made about behavior under
  concurrent readers, network filesystems, or different storage hardware.
- **This script does not implement `horizon.town_disk`.** If that module is
  missing, the CLI fails with a clear import error rather than falling back
  to a stub or an in-memory substitute.

## CLI

```
scripts/compare_town_disk.py
  --size N                 default 1024 (bounds: horizon.town_map.MIN_SIZE/MAX_SIZE)
  --seed N                 default 20261012; exactly one seed per run
  --repeats N               default 50, minimum 30
  --output PATH             required; must not already exist
  --containers-dir PATH     required; must not already exist
  --modes a,b               default warm_buffered,macos_nocache on Darwin,
                            otherwise warm_buffered; values restricted to
                            warm_buffered, macos_nocache
```

`--containers-dir` is left on disk after a successful run for audit (one
subdirectory per method, each holding `whole.hrzn` and `chunked.hcix`, as
written by `horizon.town_disk.prepare_case`). On failure, the script removes
only the container files and directories it itself created this run, then
exits non-zero; nothing pre-existing is ever touched, and there is no
silent fallback path.

This script only runs the comparison; it does not interpret the results as
pass/fail or draw performance conclusions. That analysis is left to whoever
audits the resulting report.

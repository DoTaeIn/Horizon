# Town-map simulated-movement real-disk benchmark method

`scripts/compare_town_movement.py` measures real disk I/O for loading a
moving 256x256 character viewport over the same synthetic town heightmap
used by `scripts/compare_town.py` and `scripts/compare_town_disk.py`,
comparing a whole-container ROI crop against a multi-chunk region read, on
deterministic simulated-character trajectories. It is a thin CLI around
library modules it does not implement:

- `horizon.town_map.make_town_map(size, seed)` — builds the map (see
  `docs/town-benchmark-method.md`; unchanged here).
- `horizon.town_disk` — `prepare_case(arr, directory, method,
  chunk_size=256)` writes the on-disk containers (one whole, one
  `chunk_size=256` chunked); `load_whole(path, mode, region=None)` does real
  file I/O and returns `(array, metrics)`.
- `horizon.town_region` — `load_region(path, region, mode='warm_buffered')`
  reads only the `chunk_size=256` grid chunks a region overlaps from a
  chunked container (the region need not be chunk-aligned or a single exact
  chunk) and returns `(array, metrics)`, where `metrics` carries the same
  keys `horizon.town_disk`'s readers use (`total_ns`, `read_ns`,
  `decode_ns`, `bytes_read`, `bytes_requested`, `read_calls`, `seek_calls`)
  plus `chunks_loaded` (int) and `chunk_coordinates` (the grid cells it
  touched). This script does not implement it; if it is not importable, the
  CLI fails with a clear error instead of falling back to a stub (see
  "Import boundary" below).

This document describes what the CLI measures and its boundaries. It does
not re-document `horizon.town_disk` or `horizon.town_region`'s internals,
and does not claim to improve or extend them.

## What is measured

For one `--size`/`--seed`, and for every method in `horizon.codec.METHODS`
(`zstd`, `lz4`, `delta_zstd`):

1. `horizon.town_disk.prepare_case` writes one whole-map container and one
   `chunk_size=256` chunked container for that method into its own
   exclusive subdirectory under `--containers-dir` — exactly as
   `scripts/compare_town_disk.py` does; no separate preparation path exists
   for `horizon.town_region`, which reads the same chunked container.
2. For every requested `--modes` entry, and for each of three deterministic
   trajectory kinds (`smooth_walk`, `boundary_backtrack`, `teleport`; see
   "Trajectories" below), `--repeats` continuous runs of `--steps` frames
   each are timed. Every frame measures two arms over the frame's 256x256
   viewport region:
   - **whole** — `horizon.town_disk.load_whole(whole_path, mode=mode,
     region=ROI)`: decodes the *entire* whole container, then crops an
     owned copy out of it.
   - **region** — `horizon.town_region.load_region(chunked_path, ROI,
     mode=mode)`: reads and decodes only the `chunk_size=256` grid chunks
     the ROI overlaps. Because the viewport is centered on a trajectory
     point and only clamped (not grid-aligned), a frame's ROI commonly
     spans more than one chunk — unlike
     `horizon.town_disk.load_chunked`, which only ever addresses one exact
     grid chunk, `load_region` is the multi-chunk arbitrary-ROI reader.

Every loaded array, from both arms, on every frame, is validated (exact
`ndarray` type, shape, dtype, byte content against a crop of the
in-memory map) immediately after that call returns — after its own
metrics/timing have already been captured by the library function, so
validation is never inside a timed span. Container files are sha256'd
before and after the whole run; any mismatch aborts with an error.

## Trajectories

Three kinds are generated once per run, from the seeded RNG (one
independent `numpy.random.default_rng([seed, kind_index])` stream per
kind), over `--steps` steps each, and then reused unchanged across every
method and mode — the same schedule-sharing strategy
`scripts/compare_town_disk.py` uses for its region schedule. Positions are
never regenerated between repeats; each repeat replays the identical
center-point sequence to isolate timing variance from path variance.

- **smooth_walk** — a momentum random walk: acceleration drawn from
  `Normal(0, 8.0)` each step, velocity clipped to `[-24, 24]` px/step,
  position clipped to the map. This wanders continuously and, over
  `--steps` (default 120) steps across a `--size >= 513` map, crosses many
  `chunk_size=256` grid boundaries.
- **boundary_backtrack** — one axis (row or column) and one *interior
  viewport-origin* boundary are chosen once: a `chunk_size=256` grid line
  `boundary` strictly between `0` and `max_origin = size - 256` (never the
  last grid line, which would have no room above it and would clamp to the
  same origin every step). The oscillation amplitude base is capped to
  `min(room, 12)` — the room available on the tighter side of `boundary`,
  further capped at 12px so a wide-open boundary (e.g. one near the center
  of a large map) still produces a tight backtrack rather than a long-range
  hop — and the center on that axis is set to `boundary + 128 + sign *
  delta` (alternating `sign` every step, `delta = min(room, amplitude_base
  + step % 3)` growing slightly with a 3-step sawtooth, both capped so the
  origin never clamps and the largest single-step jump is `2 *
  min(room, 14) <= 28px`). Because `center - 128` is exactly `boundary +
  sign * delta`, the viewport **origin** itself — not just the center —
  lands on alternating sides of the exact grid line every step, so the
  selected chunk set changes on essentially every frame. The other axis is
  held at a fixed, interior (unclamped) origin, drawn once uniformly from
  `[0, max_origin]`.
- **teleport** — `--steps` independent uniform draws over the full map, no
  correlation between consecutive positions: the opposite of a smooth walk,
  one discontinuous jump per frame.

A trajectory center `(center_row, center_col)` becomes a region via
clamping: `row = clamp(center_row - 128, 0, size - 256)`, `col =
clamp(center_col - 128, 0, size - 256)`, region `= (row, col, 256, 256)`.
With `--size >= 513`, `size - 256 >= 257 > 0` always holds, so the viewport
is always exactly 256x256 — only its offset is clamped at map edges, it is
never shrunk.

## Priming and cache modes

- `warm_buffered`: once at the start of each `(trajectory_kind, repeat)`
  continuous run — **not** once per frame — this script does one untimed,
  ordinary buffered read of both the whole container file and the chunked
  container file for that method, discarding the bytes. No further priming
  read happens for the rest of that run; this is a deliberate departure
  from `scripts/compare_town_disk.py` (which primes every measured call),
  because this benchmark is specifically about a *continuous* run of
  frames, and re-priming every frame would hide the real cache behavior a
  moving viewport produces.
- `macos_nocache`: no priming is performed by this script. Available only
  on Darwin; the CLI fails loudly (no fallback) if requested elsewhere.

### No application cache (either arm)

Neither arm is backed by any application-level cache, in this script,
`horizon.town_disk`, or `horizon.town_region`: every frame, for both arms,
opens its container file fresh, reads exactly the bytes it needs, closes
it, and decodes — nothing from one frame, repeat, or trajectory is reused
by the next. `warm_buffered`/`macos_nocache` describe a request about the
*OS page cache*, never this script's own state, and neither mode is a
verified physically-cold read: `warm_buffered` only means this script
primed the OS page cache once per run before timing; `macos_nocache` only
means the reader asked the kernel (`fcntl(fd, F_NOCACHE, 1)`) not to retain
newly read pages for that descriptor — that request is not verified to
evict anything already cached, and does not reach
SSD/controller/drive caches (see `horizon.town_disk.CACHE_NOTE`).

## Ordering

Within each `(trajectory_kind, repeat)` continuous run, a starting parity
is tracked per `(method, mode)` cell: frame 0 of that run uses the current
parity to decide which arm (whole or region) goes first, frame 1 the
opposite, and so on — the measured pair alternates every frame. The
starting parity itself flips once per `(trajectory_kind, repeat)` run
(carried across repeats and across trajectory kinds within a `(method,
mode)` cell), so which arm leads at frame 0 also alternates run to run.
Method and mode ordering is not alternated — the outer loops are method,
then mode, in the fixed order `horizon.codec.METHODS` and `--modes` were
given in, the same static-order caveat `scripts/compare_town_disk.py`
records.

## What is recorded

Per `(method, mode, trajectory_kind)`: every repeat's raw per-frame
metrics (both arms, in full, including `chunks_loaded`/`chunk_coordinates`
for the region arm), plus an aggregate block computed over every frame of
every repeat:

- `median_ms`/`p95_ms` (`numpy.percentile`, linear interpolation)/`max_ms`
  for `total`, `read`, and `decode`, per arm.
- Per-frame `bytes_read`, `bytes_requested`, `read_calls`, `seek_calls`
  lists, per arm.
- For the region arm only: the per-frame `chunks_loaded` list, the
  per-frame `chunk_coordinates` list, and a `chunk_count_distribution`
  (how many frames loaded 1 chunk, 2 chunks, etc.).
- `trajectory_totals`: per-repeat sums of `total_ns` and `bytes_read`
  across all frames in that repeat, then `median`/`p95`/`max` of those
  per-repeat sums, per arm, under `time_ms` (milliseconds — the sums are
  taken in nanoseconds but reported in milliseconds, like every other
  timing in this report) and `bytes` — the trajectory-level time/bytes
  totals, as opposed to the per-frame aggregate above.

The report also records: the environment (Python, platform, library
versions); sha256 of every `horizon/*.py` source file and of this script;
the generated map's shape/dtype/sha256; each container file's sha256
before and after the run; the full trajectory position sequences (so a
report is independently replayable); and the policy text blocks quoted
above (`order_policy`, `trajectory_policy`, `viewport_policy`,
`priming_policy`, `validation_policy`, `no_appcache_note`) verbatim, for
audit without cross-referencing this file.

## Import boundary

This script implements neither `horizon.town_disk` nor
`horizon.town_region`. Both are imported lazily inside `main()`; if either
is missing, the CLI raises a clear `ValidationError` (no stub, no in-memory
substitute) and exits non-zero after cleaning up only what it created.

## I/O boundaries and what this does *not* measure

- **No arbitrary multi-chunk engine is implemented here.** `load_region`'s
  behavior, correctness and performance are entirely
  `horizon.town_region`'s; this script only calls it and validates its
  output.
- **`macos_nocache` is not a verified cold read**, and **`warm_buffered` is
  not a verified warm-cache guarantee for the OS** — both are requests, not
  guarantees (see "No application cache" above).
- **Byte/call counts are application syscall counts**, not physical
  transfer measurements.
- **Single machine, single process.** No claim is made about concurrent
  readers, network filesystems, or different storage hardware.
- **Chunk-index overhead is included, not subtracted**, in the region arm's
  `read_ns`/`bytes_read`, the same convention
  `scripts/compare_town_disk.py` uses for its chunked arm.

## CLI

```
scripts/compare_town_movement.py
  --size N                 default 1024 (bounds: 513 .. horizon.town_map.MAX_SIZE)
  --seed N                 default 20261012; exactly one seed per run
  --steps N                 default 120, minimum 30
  --repeats N               default 3, minimum 1
  --output PATH             required; must not already exist
  --containers-dir PATH     required; must not already exist
  --modes a,b               default warm_buffered,macos_nocache on Darwin,
                            otherwise warm_buffered; values restricted to
                            warm_buffered, macos_nocache
```

`--size` has a higher floor (513) here than
`scripts/compare_town_disk.py`'s (`horizon.town_map.MIN_SIZE`, 64): a
256-wide viewport clamped into `[0, size - 256]` needs room on both sides
of the first interior `chunk_size=256` grid line for the trajectory shapes
above (especially `boundary_backtrack`) to be meaningful.

`--containers-dir` is left on disk after a successful run for audit (one
subdirectory per method, each holding `whole.hrzn` and `chunked.hcix`, as
written by `horizon.town_disk.prepare_case`). On failure, the script
removes only the container files and directories it itself created this
run, then exits non-zero; nothing pre-existing is ever touched, and there
is no silent fallback path.

This script only runs the comparison; it does not interpret the results as
pass/fail or draw performance conclusions. That analysis is left to
whoever audits the resulting report.

# Town-map application-level cache comparison method

`scripts/compare_town_cache.py` measures what it costs to keep moving a
viewport around the same synthetic town heightmap used by
`scripts/compare_town.py`/`scripts/compare_town_disk.py`/
`scripts/compare_town_movement.py`, when the *application* is allowed to
remember state between frames, instead of the always-stateless
`horizon.town_disk`/`horizon.town_region` readers those other scripts
exercise. It never generates a new map or new trajectories of its own: it
replays an existing `scripts/compare_town_movement.py` report's recorded
`trajectories[kind]` positions verbatim, via `--schedule`. It is a thin CLI
around library modules it does not implement:

- `horizon.town_map.make_town_map(size, seed)` — re-derives the map the
  schedule was built from, so its sha256 can be checked against the
  schedule's recorded `map.sha256` (see "Schedule verification" below).
- `horizon.town_disk.prepare_case(arr, directory, method, chunk_size)` —
  writes the on-disk containers this run replays against; unchanged here.
- `horizon.town_cache.ChunkRegionCache(path, max_chunks, mode)` and
  `horizon.town_cache.WholeMapResident(path, mode)` — the two stateful
  loaders under comparison. This script implements neither their cache
  policy nor their I/O; see `horizon/town_cache.py`'s own docstrings for
  what `startup_metrics`/`load()` metrics mean. Independent tests for the
  module itself live in `tests/test_town_cache.py`.
- `horizon.town_region.load_region(path, roi, mode)` — the stateless
  baseline, reused unchanged from `scripts/compare_town_movement.py`.

This document describes what the CLI measures and its boundaries. It does
not re-document `horizon.town_cache`'s internals, and reports no benchmark
numbers of its own — those live in `benchmarks/town-cache-audit.md` and the
raw report files it names, produced and independently checked separately
from this document.

## Arms compared

Four arms are run per `(method, mode, trajectory_kind, repeat)`, all driven
by the same replayed positions:

- **stateless** — `horizon.town_region.load_region(chunked_path, roi,
  mode)`: no cache, real file I/O on every call, exactly as
  `scripts/compare_town_movement.py`'s `region` arm. Its metrics lack cache
  concepts (no `cache_hits`, no `resident_bytes`); this script fills those
  keys in honestly rather than omitting them, so every arm's metrics dict
  has the same keys: every chunk touched counts as a miss, nothing is ever
  retained (`cache_hits=0`, `cache_misses=chunks_loaded`, `evictions=0`,
  `resident_bytes=0`).
- **lru{N}** — one `horizon.town_cache.ChunkRegionCache(chunked_path,
  max_chunks=N, mode)` per `--cache-sizes` entry (default `4,8`, so `lru4`
  and `lru8` by default). A bounded LRU of decoded chunks; see
  `horizon/town_cache.py` for the exact two-phase hits-then-misses load
  order and eviction policy.
- **whole_resident** — `horizon.town_cache.WholeMapResident(whole_path,
  mode)`. The entire map is read, decoded and retained once at
  construction; every `load()` afterward is a pure in-memory crop with zero
  file I/O. This arm's near-zero per-load cost is real for *this*
  benchmark's single small map and is not evidence that whole-map residency
  is universally the right choice — it ignores memory cost and any scenario
  with more than one resident map (see "What this does not measure" below).

Every arm gets a **freshly constructed session** once per `(method, mode,
trajectory_kind, repeat)` — never reused across repeats. `stateless` has no
session by definition; `lru{N}` starts from an empty cache; `whole_resident`
starts from a freshly read-and-decoded map. All four sessions for a given
`(method, mode, trajectory_kind, repeat)` are constructed up front, before
any frame of that run is timed, and all are closed again (`session.close()`)
once that run's frames are done — including if construction or a frame
raises, so a failed repeat never leaks an open descriptor or a resident
array.

Nothing here is hard-coded to a particular grid: `DOCUMENTED_CHUNK_SIZE` and
`DOCUMENTED_VIEWPORT` (both `256`, recorded into every report's `settings`
alongside the schedule's actual `chunk_size`/`viewport`) describe what the
published runs in `benchmarks/town-cache-audit.md` used, not a constraint
this script enforces — any positive `chunk_size`/`viewport` that fits the
recorded map is accepted and replayed, which is what makes small fixture
schedules usable in tests (see `tests/test_town_cache_cli.py`).

## Schedule verification

`--schedule` must be an existing `scripts/compare_town_movement.py` report
(e.g. `benchmarks/town-movement-1024-v1.json`). Before anything else runs,
and before any container is written:

1. The schedule's `settings` block (size, seed, steps, chunk_size, viewport,
   methods, modes, trajectory_kinds) is validated for shape and type —
   wrong type, missing key, out-of-range value or an unsupported mode name
   all reject the run with a specific message, not a generic crash.
2. `horizon.town_map.make_town_map(size=settings.size, seed=settings.seed)`
   re-derives the map and its sha256 is asserted against the schedule's
   recorded `map.sha256`.
3. `trajectories[kind]` is read and validated (right length, in-bounds
   integer `[row, col]` pairs) but **never regenerated, clipped or
   reordered** — those exact recorded positions are what gets replayed.
4. **Every** file recorded in `source_files_sha256` is re-hashed and must
   still be present and byte-identical — **including
   `scripts/compare_town_movement.py` itself**, the script that produced
   the schedule. This is strict by design: a recorded source file that is
   missing, or whose bytes changed, rejects the run outright. Nothing is
   treated as "out of scope" or silently skipped; a replay can never
   confirm that a file it does not re-hash still matches what was recorded.
   (A pre-audit draft of this script carved out an exception for the
   movement script and for any absent file — that exception is gone.)

Any failure in any of the above aborts with a `ValidationError` before any
container is written, and nothing this run would have created is left
behind.

## What is measured, per frame

For every requested `--modes` entry and every trajectory kind the schedule
recorded, `--repeats` continuous runs of the schedule's recorded positions
are replayed. Every frame computes the exact ROI from the recorded center
point (clamped into bounds the same way as
`scripts/compare_town_movement.py`), then calls every arm's loader on that
identical ROI and validates the returned array (`ndarray` type, exact
`(height, width)` shape, dtype, byte content against a crop of the in-memory
original map) **strictly after** the timed call returns, so validation is
never inside a timed span. Every metrics dict, from every arm, is also
checked for the full required key set before being recorded — a library
function silently dropping a key is a loud test failure here, not a missing
column in a report.

- `total_ns` is each arm's own measured wall-clock elapsed time for that
  `load()`/`load_region()` call — not a sum of hand-picked spans. For the
  cache arms this includes LRU bookkeeping, eviction and output allocation,
  not just I/O and decode; see `horizon/town_cache.py`'s own docstring for
  the exact accounting (`total_ns == read_ns + decode_ns` by construction).
- Every arm reports the full `horizon.town_disk`-style metric keys
  (`total_ns`, `read_ns`, `decode_ns`, `bytes_read`, `bytes_requested`,
  `read_calls`, `seek_calls`), plus the chunk/cache keys (`resident_bytes`,
  `chunks_requested`, `chunks_loaded`, `cache_hits`, `cache_misses`,
  `evictions`, `chunk_coordinates`, `hit_coordinates`, `miss_coordinates`,
  `evicted_coordinates`) — identical shape across all four arms, `stateless`
  included, so a benchmark table never has to special-case a missing key.

## Timing units: raw nanoseconds vs. aggregated milliseconds

Every **raw** value recorded per frame or per repeat is an integer
nanosecond count and is named with an `_ns` suffix:
`repeats[].frames[].arms[arm].total_ns`,
`repeats[].startup_metrics[arm].total_ns`,
`repeats[].trajectory_ns[arm].*`. Every **aggregated** statistic is
computed in milliseconds and named `*_ms` inside its stat block
(`median_ms`/`p95_ms`/`max_ms`, `numpy.percentile` with linear
interpolation) — no aggregate is ever labelled with a unit it is not
actually in, and no raw value is ever reported pre-rounded to milliseconds.

## Raw per-repeat records

Per `(method, mode, trajectory_kind)`, the report keeps, per repeat:

- `construct_order` — the rotated arm order sessions were actually built in
  for that repeat (see "Rotation" below).
- `startup_metrics[arm]` — that repeat's session's own raw constructor
  metrics (`total_ns`, `read_ns`, `decode_ns`, byte/call counters),
  recorded **verbatim**, once per repeat (not once per frame, since it
  belongs to session construction). `stateless` has no session, so its
  `startup_metrics` is the genuine all-zero dict, not an approximation.
- `trajectory_ns[arm]` — per repeat, that arm's raw
  `startup_total_ns` (== that repeat's `startup_metrics[arm].total_ns`),
  `frames_total_ns` (the exact sum of that repeat's own frame `total_ns`
  values for that arm) and `total_ns_including_startup` (`= startup_total_ns
  + frames_total_ns`). These are loader time only — not wall-clock frames;
  they exclude rendering, physics, validation and report writing.
- `frames[]` — every frame's `roi`, the realized `arm_order` for that frame,
  and each arm's full metrics dict for that frame: the raw per-frame
  record, not just a derived statistic.

## Aggregation

The `aggregate` block per arm, computed over every frame and every repeat in
that `(method, mode, trajectory_kind)`, is a set of millisecond stat blocks
plus a few raw integer totals:

- `startup_total` — stats over each repeat's one `startup_metrics[arm].total_ns`.
- `frame_total` / `frame_total_hit_only` / `frame_total_miss_involving` —
  stats over every frame's `total_ns`; over just the frames with zero
  misses; and over just the frames with at least one miss. The hit/miss
  split is only meaningful for `lru{N}` — `whole_resident` never reports a
  miss (vacuously all "hit-only"), and `stateless` never reports a hit
  (every chunk it touches counts as a miss, so it is vacuously all
  "miss-involving").
- `first_viewport` — stats over, per repeat, that repeat's own
  `trajectory_ns[arm].startup_total_ns` plus **that same repeat's own frame
  0** `total_ns` (never a different repeat's frame, and never averaged
  across repeats before combining). This is "time to first rendered
  viewport for a freshly constructed session".
- `trajectory_total_including_startup` — stats over each repeat's
  `trajectory_ns[arm].total_ns_including_startup`.
- `trajectory_total_movement_only` — stats over each repeat's
  `trajectory_ns[arm].frames_total_ns` (loader time only, startup excluded).
- `bytes_read_total_including_startup` — every frame's `bytes_read` plus
  every repeat's startup `bytes_read`, summed (raw integer, not an
  aggregate stat block).
- `hit_rate` — total `cache_hits` / total `chunks_requested` across all
  frames (`None` if no chunks were ever requested, e.g. a zero-step run).
- `evictions_total`, `resident_bytes_max`, `resident_bytes_end` — the last
  two taken across every frame's `resident_bytes`, so they describe the
  per-call snapshot, not a continuously sampled high-water mark between
  calls.
- `hit_only_frame_count` / `miss_involving_frame_count` — how many frames
  fell in each bucket above.

Every one of these is independently recomputable from the raw
`repeats[].startup_metrics`/`repeats[].trajectory_ns`/`repeats[].frames`
records in the same report — `tests/test_town_cache_cli.py` does exactly
that recomputation rather than trusting the aggregate block at face value.

The report also records: the environment (Python, platform, library
versions), sha256 of every `horizon/*.py` source file, of this script
itself, and of the memory worker script when present, the regenerated map's
shape/dtype/sha256, the schedule's own path and sha256 and the verification
outcome, each container file's sha256 before and after the whole run
(re-checked *after* the memory sub-measurement runs too, since it reads the
same files), and the policy text blocks quoted in this script's module
docstring (`rotation_policy`, `priming_policy`, `session_policy`,
`validation_policy`, `timing_unit_policy`, `trajectory_total_policy`,
`memory_policy`, `whole_resident_note`) verbatim.

## Rotation

Within a frame, arm execution order rotates by `(trajectory_index + repeat +
frame_index) % num_arms`; session construction order within a repeat
rotates by `(trajectory_index + repeat) % num_arms`. Both realized orders
are recorded per frame/repeat (`arm_order`, `construct_order`) specifically
so no arm is systematically measured first or last across the whole run.

## Priming and cache modes

- `warm_buffered`: once per `(trajectory_kind, repeat)` run, **before any
  arm's session is constructed or timed**, this script does one untimed,
  ordinary buffered read of both the whole and chunked container files for
  that method, discarding the bytes. No further priming happens mid-run.
- `macos_nocache`: no priming is performed by this script at all. Available
  only on Darwin.

Requesting a mode explicitly via `--modes` on a platform that does not
support it (`macos_nocache` off Darwin) is a hard, loud rejection — never a
silent rewrite to a different mode than what was asked for. Only the
schedule's own *default* mode list (used when `--modes` is omitted) may be
filtered down to what the current platform supports, and only as long as at
least one supported mode remains; if filtering would leave zero modes, that
is also a rejection, not a silent fallback.

As with `scripts/compare_town_movement.py`, neither mode is a verified
physically-cold read: `warm_buffered` only means this script primed the OS
page cache once per run before timing; `macos_nocache` only means the
reader asked the kernel (`fcntl(fd, F_NOCACHE, 1)`) not to retain newly read
pages for that descriptor. Neither reaches SSD/controller/drive caches, and
neither claim is about this script's own application-level state — that is
exactly the thing the cache arms vs. `stateless` are contrasting, not
something either mode flag controls.

## Memory sub-measurement

Timing and memory are measured separately, in separate processes, so that
Python's own allocator/GC state and whatever large reports or arrays a long
timing run has accumulated can never leak into a memory measurement.
`--memory-repeats N` (default 3; `0` disables it for fast smoke runs) shells
out to `scripts/measure_town_cache_memory.py` as a subprocess after the
timing run finishes, passing it `--schedule`, `--containers-dir`,
`--cache-sizes`, `--repeats N` and a fresh `--output` sidecar path (the main
report's own `--output` path with `.memory.json` appended, which must not
already exist — checked once before any container is written, and
re-checked immediately before the subprocess launches, to narrow the race).
Its resulting report is embedded verbatim under this script's own
`report["memory"]["report"]`, alongside the memory report's own path and
sha256 and the memory worker script's own sha256. If that subprocess exits
non-zero, or exits zero but writes no file, or writes invalid JSON, this
script raises rather than silently continuing without a memory section; any
partial sidecar the worker left behind on failure is deliberately not
deleted, so it remains available for inspection. On success, this run's own
container-file hashes are re-verified **after** the memory subprocess runs,
since that subprocess reads the same container files this run wrote.

**`scripts/measure_town_cache_memory.py` is a separate CLI** with its own
schema and its own module docstring — this document does not redefine its
contract. In short: one fresh Python subprocess per `(method,
trajectory_kind, arm, repeat)` sample, so no sample can inherit the parent's
already-decoded map, previous large reports, or another arm's resident
state (each sample's `pid` is recorded so two samples can be confirmed never
to collide); and its own schedule verification is just as strict as this
script's — nothing recorded in `source_files_sha256`, including any of the
three driver scripts' own hashes, is treated as out of its scope.

### What the memory measurement distinguishes

Three different numbers answer three different questions, and the memory
sub-measurement keeps them distinct rather than reporting one and calling it
"memory" (field names as actually reported by
`scripts/measure_town_cache_memory.py`):

- **`resident_decoded_bytes_peak` / `resident_decoded_bytes_end`** — each
  cache object's own `.resident_bytes`: the sum of `.nbytes` of the decoded
  numpy arrays it is actually holding (cached chunks for `lru{N}`, the one
  resident array for `whole_resident`, always `0` for `stateless`). This is
  an exact, deterministic count with no measurement noise, bounded by
  construction (`max_chunks * chunk_size**2 * itemsize` for `lru{N}`); it
  excludes the per-call output ROI array handed back to the caller, the
  chunk index, and any Python/dict bookkeeping overhead.
- **`tracemalloc_startup_{current,peak}_bytes` / `tracemalloc_movement_{current,peak}_bytes`
  / `tracemalloc_overall_peak_bytes`** — Python-level allocations
  `tracemalloc` can see, measured after imports and a `gc.collect()` so
  neither is included, split into the constructor phase (`startup_*`, peak
  reset to zero immediately after) and the replay-loop phase (`movement_*`).
  This excludes any allocation `tracemalloc` cannot instrument (native codec
  library buffers that bypass Python's allocator, for instance), so it is
  not a complete memory account, but it is useful for seeing *when*
  (relative to setup vs. steady-state movement) Python-level memory grows.
- **`process_rss_baseline_hwm_bytes` / `process_rss_startup_hwm_bytes` /
  `process_rss_overall_hwm_bytes`** — the OS-reported resident-set-size
  high-water mark for the whole fresh child process
  (`resource.getrusage(RUSAGE_SELF).ru_maxrss`; Darwin reports bytes
  directly, Linux reports KiB and is converted to bytes here, any other
  platform or a missing `resource` module reports the value as unavailable
  rather than guessing a unit). This includes the Python interpreter,
  NumPy, every imported codec library, and everything else in that
  process's address space, not just this benchmark's own state. It is
  monotonically non-decreasing within one process — even the "baseline"
  reading is already a high-water mark, not a current value — and it is the
  only one of the three that can be dominated by import/runtime overhead
  unrelated to the cache under test. A smaller "exact retained bytes" number
  for one arm does not guarantee a smaller RSS peak for that arm — the two
  are measuring different things and are reported separately for exactly
  that reason.

No numeric result from any of these three is asserted or invented in this
document; see `benchmarks/town-cache-audit.md` for the actual measured
figures and their own caveats.

## Import boundary

This script implements none of `horizon.town_disk`, `horizon.town_cache`,
or `horizon.town_region`. All three are imported lazily inside `main()`; if
any is missing, the CLI raises a clear `ValidationError` (no stub, no
in-memory substitute) and exits non-zero after cleaning up only what it
created this run.

## What this does not measure

- **No arbitrary multi-chunk engine or cache policy is implemented here.**
  `ChunkRegionCache`'s and `WholeMapResident`'s behavior, correctness and
  performance are entirely `horizon.town_cache`'s; this script only
  constructs them, calls their documented API, and validates their output.
- **`whole_resident`'s near-zero per-load cost reflects this benchmark's one
  small map, resident for the whole run.** It says nothing about memory
  pressure from holding multiple whole maps resident at once, about a map
  too large to fit in memory, or about any workload other than "one map,
  one process, read repeatedly".
- **`macos_nocache` is not a verified cold read**, and **`warm_buffered` is
  not a verified warm-cache guarantee for the OS** — see "Priming and cache
  modes" above.
- **Byte/call counts are application syscall counts**, not physical
  transfer measurements.
- **Single machine, single process** (per timing run; the memory
  sub-measurement is single machine, multiple short-lived processes). No
  claim is made about concurrent readers, network filesystems, or different
  storage hardware.
- **This is a heightmap loader benchmark.** It says nothing about a game
  engine's mesh, texture, GPU upload, or collision-geometry costs.

## CLI

```
scripts/compare_town_cache.py
  --schedule PATH           required; an existing scripts/compare_town_movement.py
                            report to replay trajectories from
  --repeats N                default 3, minimum 1
  --cache-sizes N,N,...      default 4,8; positive, unique integers; one lru{N}
                            arm per entry
  --modes a,b                default: the schedule's own recorded modes
                            (filtered to what this platform supports); an
                            explicit value restricted to warm_buffered,
                            macos_nocache and rejected outright (not
                            filtered) if unsupported here
  --memory-repeats N         default 3; 0 disables the memory sub-measurement;
                            any other value requires
                            scripts/measure_town_cache_memory.py to exist
  --output PATH              required; must not already exist
  --containers-dir PATH      required; must not already exist
```

`--containers-dir` is left on disk after a successful run for audit (one
subdirectory per method, each holding the whole and chunked containers
`horizon.town_disk.prepare_case` writes). On failure, the script removes
only the container files and directories it itself created this run (and,
if it owns one, the memory sidecar it created — determined by matching the
exact `(dev, inode)` and sha256 it recorded right after writing it, so a
file that was replaced or is no longer provably the same one is left
alone), then exits non-zero; nothing pre-existing is ever touched, and
there is no silent fallback path. The memory sub-measurement's own
`--output` sidecar follows the same must-not-already-exist rule as the main
`--output`.

This script only runs the comparison; it does not interpret the results as
pass/fail or draw performance conclusions. That analysis is left to
whoever audits the resulting report — see `benchmarks/town-cache-audit.md`.

# Town-map chunk benchmark method

`scripts/compare_town.py` measures the warm, in-memory cost of processing one
synthetic heightmap as a whole versus processing it in fixed-size chunks, at
several chunk sizes. It is a thin CLI around two library functions:

- `horizon.town_map.make_town_map(size=1024, seed=20261012)` — builds the map.
- `horizon.chunk_benchmark.run_chunk_benchmark(arr, chunk_sizes=(64, 128, 256, 512), repeats=30, seed=20261012)` —
  times whole-array and chunked processing of that map.
- `horizon.town_map.town_plan(size, seed)` — optional; its returned geometry
  (river, routes, flats, crossings) feeds the semantic `_layout.svg` preview
  described below. Used read-only and only if present.

This script and this document do not implement or specify those functions'
internals; they only rely on the signatures above. Everything here describes
the comparison method and the preview rendering, not the generators
themselves.

## Map provenance

`make_town_map` produces an **original, procedurally generated** heightmap.
Its layout is loosely inspired by the "towns connected by routes" structure
seen in ORAS (*Pokémon Omega Ruby / Alpha Sapphire*) — i.e. a few dense hub
regions joined by corridor-like paths — purely as a shape reference for
generating varied synthetic terrain. The map contains **no assets, tile data,
or content extracted or derived from any actual game**. It is deterministic
synthetic data for benchmarking only, in the same spirit as the `flat`,
`plane`, `smooth_terrain`, and `random` fixtures in `horizon/benchmark.py`.

The generated array is expected to follow this repository's existing
heightmap contract: a nonempty 2D little-endian `uint16` array (checked here
by reusing `horizon.benchmark.validate_heightmap`, not by re-implementing
validation).

## What is measured

For a given `--size`/`--seed`, `compare_town.py`:

1. Calls `make_town_map(size, seed)` once and validates the result.
2. Calls `run_chunk_benchmark(arr, chunk_sizes, repeats, seed)` once on that
   same in-memory array.
3. Wraps the returned result, plus settings/environment/provenance metadata,
   in one JSON report.

Both calls happen in the same process, back to back, on an array that is
already fully resident in memory. There is no filesystem I/O anywhere inside
`run_chunk_benchmark`'s timed sections: map generation (`make_town_map`) and
this script's own report/preview writes happen entirely outside those timed
calls. The timed encode/decode calls do include in-memory (de)serialization —
encoding the chunked form packs a binary index (`_pack_index`) alongside
compressing each chunk, and decoding it parses that index (`_unpack_index`)
before slicing out one chunk's bytes — so that packing/parsing cost is part
of what `chunked_encode`/`chunked_reconstruct`/`region_chunk_decode` measure,
not excluded from it. There is no engine or GPU involvement in the timed
path — only whatever `run_chunk_benchmark` itself measures internally
(whole-map vs. chunked/partial traversal or processing, per the chunk sizes
given).

This is a **warm in-memory, full-map-vs-partial comparison**: "full" means
operating on the entire array at once; "partial" means operating on it through
fixed-size chunks, repeated at each requested chunk size. The benchmark does
not model cold caches, disk layout, streaming from an engine, or multi-process
contention.

## Reproducibility

- `size`, `seed`, `repeats`, and `chunk_sizes` are recorded verbatim in the
  report's `settings` block.
- The generated map's shape, dtype, and SHA-256 are recorded in the report's
  `map` block. Re-running with the same `size`/`seed` should reproduce the same
  hash, since `make_town_map` is documented as deterministic given a seed; this
  script does not itself add any additional randomness.
- The report also records the wall-clock cost of map generation
  (`map.generate_wall_s`) separately from the chunk benchmark itself
  (`chunk_benchmark_wall_s`), so one-time setup cost is not confused with the
  per-chunk-size timings inside `results`.
- `environment` records Python/OS/CPU/NumPy details for the invoking machine,
  matching the pattern used by the other `scripts/compare_*.py` tools in this
  repository.

## Chunk index format and overhead (as currently implemented)

`horizon/chunk_benchmark.py` cuts the map into a grid of independent chunks,
encodes each chunk on its own with the existing container codec
(`horizon.codec`, 60-byte header per container, `horizon.codec.HEADER_SIZE`),
and packs a small binary index in front of the concatenated chunk blobs so the
chunked form can be measured and round-tripped. As of this writing that index
is:

- a 24-byte fixed header (`magic, version, method_id, reserved, chunk_size,
  grid_rows, grid_cols, chunk_count`, struct format `<4sBBHIIII`), plus
- 32 bytes per chunk entry (`row, col, height, width, offset, length`, struct
  format `<IIIIQQ`).

So for `N` chunks the index alone costs `24 + 32*N` bytes, and each chunk's
own blob additionally carries its own 60-byte codec header on top of its
compressed payload: **total** index+header bookkeeping for the chunked form
is `24 + 32*N + 60*N` bytes. Relative to the single whole-map container,
which pays that 60-byte header only once, the chunked form's **additional**
bookkeeping is `24 + 32*N + 60*N - 60` bytes.

That bookkeeping term grows as `chunk_size` shrinks (more, smaller chunks ⇒
more 60-byte headers and a bigger index), but it is not the only thing
`size_ratio_chunked_to_whole` reflects: that ratio also carries however the
sum of the per-chunk compressed payloads compares to the single whole-map
compressed payload, and that comparison is data-dependent — splitting into
chunks can lose cross-chunk redundancy a codec would otherwise exploit, so
it most often costs size, but it is not guaranteed to. Don't assume
`size_ratio_chunked_to_whole` necessarily grows monotonically as `chunk_size`
shrinks; the bookkeeping term does, the compression term can move either
way, and the report's ratio is their combined, measured result.

This binary chunk index (`HCIX`, version 1) is a **private, in-process
benchmark artifact**, scoped to one call of `run_chunk_benchmark`. It exists
only so the benchmark can round-trip and verify the chunked form against the
whole-map form within the same run. `horizon/chunk_benchmark.py` is explicit
about this (`CONTAINER_FORMAT_NOTE`, and the comment above `_unpack_index`):
its parser trusts its own just-built output and performs **no adversarial
validation** — no bounds/overflow checks on offset or length, no checksum —
because it only ever parses data this module produced in the same call. It is
not a hardened, versioned, or public storage format: nothing in this
repository persists it to disk, reads it back across runs, or commits to its
layout as a stable API.

**Edge/remainder handling is supported, not unknown:** when `size` is not an
exact multiple of a given `chunk_size`, the grid's last row and last column of
chunks are shrunk to fit the remaining rows/columns (see
`_chunk_entries`/`_grid_dims` in `horizon/chunk_benchmark.py`). No chunk is
padded, dropped, or wrapped; every chunk's `(row, col, height, width)` is
recorded in the index exactly as encoded.

## Random-region retrieval method (as currently implemented)

`run_chunk_benchmark` compares two ways of getting one region's pixels back:

- `region_chunk_decode`: an O(1) index-entry lookup (`_lookup_entry`, direct
  offset arithmetic, no scan) + a container slice + decoding just that one
  chunk's blob.
- `region_full_decode_crop`: decoding the whole-map blob, then copying the
  same region out of the decoded array (`_crop_copy`, an explicit `.copy()`,
  not a view — so this arm's result is "owned" the same way `decode()`'s
  output always is, which keeps the comparison from quietly favoring one arm
  through aliasing).

To avoid the two biases a single fixed measurement would have:

- **Region coverage** (`REGION_SCHEDULE_POLICY`): for each `chunk_size`, a
  `repeats`-long schedule of chunk indices is built once, before the per-method
  loop, from cyclic random permutations of the shared seeded RNG — each full
  cycle visits every grid chunk exactly once, so a `repeats` run exercises
  many distinct chunks (not one hammered chunk) and the same schedule is
  reused unchanged across all `horizon.codec.METHODS`, making the methods
  comparable on identical regions. Every repeat's region is still verified
  against the expected bytes, same as the whole-map/chunked checks.
- **Measurement order** (`ORDER_POLICY`): per repeat, which half of each
  measured pair runs first alternates by repeat parity — even repeats run
  one order, odd repeats run it swapped — applied independently to the
  encode pair, the decode/reconstruct pair, and the region-retrieval pair.
  This is the same ordering-bias control `scripts/compare_decode.py` already
  uses for its baseline/current comparison.

The report's `settings.order_policy` and `settings.region_schedule_policy`
strings (and `results[].region_schedule`, a per-repeat list of
`{repeat, row, col, height, width, chunk_index, verified}`) are copied
straight from `run_chunk_benchmark`'s own output; this script does not
recompute or re-describe them, so they stay accurate even if the policy
wording changes upstream.

### Caution: chunk-uniform sampling can oversample tiny edge chunks

The region schedule picks grid **chunks** uniformly at random, not pixels or
area. When `size` is not an exact multiple of `chunk_size`, the edge row/
column of chunks is shrunk to fit (see "Edge/remainder handling" above) and
can be very small — down to 1 pixel wide or tall. A chunk-uniform schedule
gives those tiny edge chunks the same selection probability as a full-size
interior chunk, so on a map with a large edge remainder, a
`region_chunk_decode`/`region_full_decode_crop` comparison can end up
dominated by near-trivial edge requests instead of typical ones.

This showed up in practice, not just in theory: on a 1025×1025 map at
`chunk_size=512`, 17 of 30 scheduled chunks were 1-pixel-wide/tall edge
chunks, and `region_chunk_decode`'s median fell to about 0.05 ms. That is
**not** a performance improvement over the 1024 case — it reflects doing
roughly 1/512th of the pixel work per request, not a faster retrieval path.
Don't compare `region_chunk_decode`/`region_full_decode_crop` medians across
different `size`/`chunk_size` combinations without first checking
`results[].region_schedule` for how many scheduled chunks were edge-shrunk.

A fixed-area or pixel/coordinate-based access distribution, instead of
chunk-uniform, would avoid this skew and is a natural next step for a
dedicated partial-read experiment; `run_chunk_benchmark` does not currently
offer that mode.

## General overhead caveats

- `chunk_sizes` smaller than `size` imply more, smaller chunks; Python-level
  looping and function-call overhead per chunk is expected to grow as chunk
  size shrinks, on top of the concrete index/header overhead described above.
- A single invocation reflects one process, on one machine, at one point in
  time. Treat absolute numbers as illustrative; compare relative differences
  between chunk sizes from the same report, not across machines or across
  reports generated at different times.
- `repeats` controls how many timed repetitions `run_chunk_benchmark` performs
  internally; this script does not add its own repetition loop around that
  call.

## Resolved issue: `make_town_map` briefly returned a 3D array

During development, `horizon.town_map.make_town_map(size, seed)` briefly
returned an array of shape `(size, size, size)` instead of the documented
`(size, size)`, for every size tested. The cause was in `_carve_river`: it
called `_river_weight(..., v=rows[:, None], ...)`, so the `bed` array
`_river_weight` returned already had shape `(size, 1)`; `_carve_river` then
did `bed[:, None]`, adding a spurious third axis, which broadcast against the
`(size, size)` surface into `(size, size, size)`. This was fixed upstream in
`horizon/town_map.py` (not by this script or this document).

While that bug was present, `build_report`'s call to
`horizon.benchmark.validate_heightmap` correctly rejected the malformed 3D
output with a clear `ValueError` ("expected a 2D array, got 3D") instead of
silently misbehaving or writing a corrupt report — `compare_town.py` never
worked around it. `tests/test_compare_town.py::RealTownMapSmokeTest` caught
this automatically (it failed while the bug was present, and passes now that
`make_town_map` returns a proper 2D array).

## CLI

```
python scripts/compare_town.py \
  --size 1024 --seed 20261012 --repeats 30 --chunk-sizes 64,128,256,512 \
  --output benchmarks/town-run.json \
  [--preview-dir benchmarks/preview]
```

- `--size`, `--seed`, `--repeats`, `--chunk-sizes` default to the values shown
  above (`--size`/`--repeats` default to 1024/30; `--seed` matches the library
  defaults; `--chunk-sizes` is a comma-separated list).
- `--size` must be in `[horizon.town_map.MIN_SIZE, horizon.town_map.MAX_SIZE]`
  (currently `[64, 4096]`), imported directly from `horizon.town_map` rather
  than restated, so this script's bound tracks the real contract exactly —
  checked before `make_town_map` is called, so an invalid `--size` fails at
  argument-parsing time instead of after generating a map.
- `--seed` must be in `[0, 2**64)`, matching `horizon.town_map`'s own seed
  contract.
- `--output` is required and must not already exist; the script refuses to
  overwrite an existing report.
- `--preview-dir` is optional. When given, the script writes a `.svg` and a
  `.npy`, named `town_<size>_<seed>.svg`/`.npy`, plus a second
  `town_<size>_<seed>_layout.svg` whenever `horizon.town_map.town_plan` is
  available (see "Preview artifacts" below); none of these are overwritten if
  already present.
- All outputs are written all-or-nothing: if any output path already exists,
  or if writing any one of them fails partway through, nothing from that run
  is left on disk (previously created files in the same invocation are
  removed before the script exits with a nonzero status).

## Preview artifacts

The optional preview is a debugging aid, not a measurement:

- The `.npy` file is the exact generated map (`uint16`, little-endian),
  suitable for reloading with `numpy.load` or `horizon.benchmark.load_heightmap`.
- The `.svg` file is a **downsampled** (strided, deterministic) view of the
  same map — at most 128 cells per side regardless of `--size` — rendered with
  a small fixed terrain-like color gradient and a legend bar showing the
  map's actual min/max height values. It is for a quick visual sanity check
  only; it is not pixel-accurate at large `--size` and should not be used to
  infer exact per-cell values. No image/plotting dependency is used — the SVG
  is built with plain string formatting.

### `_layout.svg`: a second, semantic preview

Height alone does not reveal the town/route structure: a height-only gradient
looks flat or ambiguous wherever a terrace, a road cut, and a riverbank land
on a similar color. When `horizon.town_map.town_plan(size, seed)` is
available, `compare_town.py` also renders `town_<size>_<seed>_layout.svg`:
the same sampled height raster and legend as the plain `.svg`, with a vector
overlay drawn from `town_plan()`'s returned geometry —

- the river centerline, in blue;
- every route's path, in a pale road stroke, labeled with its name (e.g.
  "North Route");
- each flattened building pad and the plaza, as an outlined/filled ellipse,
  labeled with its name;
- a marker at each bridge and each ford crossing (distinct shape/color per
  kind), with both explained in the on-image legend.

The legend includes an on-image disclaimer ("Schematic metadata; no game
assets.", kept short so it isn't clipped at small `--size`/preview widths)
and this document repeats the fuller version: **these overlay labels are
schematic feature metadata read from `town_plan()`'s dict (names, normalized
positions, radii) — not game assets, not production geometry, and not
pixel-level ground truth.** All text drawn into either SVG (titles, route
and pad names, the legend lines, the disclaimer) is XML-escaped before being
inserted, so unexpected characters in any of that text cannot produce
malformed or injected markup.

`town_plan` is treated as optional: if the imported `horizon.town_map` module
does not expose it (e.g. an older module, or a test stub), `compare_town.py`
skips `_layout.svg` and still writes the plain `.svg` and `.npy` as before —
this is not an error.

# Decode speed and real terrain audit — 2026-10-12

What this change does: It restores large heightmaps faster while keeping the v1
file bytes and validation rules. It also prepares two public terrain tiles and
compares the old and new decoders on the same inputs.

Looks good. Ship for this local Python prototype.

## Work and scope

The requested Claude CLI workers ran in parallel: one Opus for the codec and one
Sonnet for terrain preparation and comparison tools. The parent reviewed their
code, rejected regressions, ran all checks, and wrote this report. The final small
map guard came from Opus with repository access and tools disabled. The parent
integrated that returned function. A final Sonnet test-path correction was denied
by automatic approval review because repository test contents might be sent to an
external API. The parent corrected that test locally; the denied action was not
retried. Both workers had already completed their implementation work.

Reviewed callers: `horizon/benchmark.py`, pack/unpack in `horizon/__main__.py`,
the new preparation/comparison scripts, codec tests, terrain tests, data outputs,
and the saved baseline. No dependencies, public codec signatures, compression
settings, or container versions changed.

Assumed load: one local process and one user, nonempty 2D `<u2` inputs up to the
existing 64 MiB raw limit. Concurrent edits to input/cache files are outside this
prototype's contract. This review is about the changed paths, not a whole-repo
security assessment.

## Issues found and resolved

1. **Full-array uint32 accumulation regressed large maps.** The first Opus
   candidate helped a small microbenchmark but made 3601-square accumulation
   about 40.9ms versus 22.0ms. Its traced synthetic decode peak was about
   129.7MB versus 77.8MB. Rejected. The final code uses complete-row batches with
   at most 65,536 uint32 samples, or 256 KiB of explicit scratch storage.
   See `rejected-candidate-memory.json` for the rejected memory result.
2. **Batch scratch overhead regressed small maps.** A second candidate was faster
   on full terrain but slowed some 256-square maps; plane median was 0.4475ms
   versus 0.3453ms in that paired run. Rejected. The final code retains the
   original uint16 accumulation when the whole map fits the batch bound or one
   row exceeds it. `rejected-small-map-timing.json` and its memory companion are
   intermediate evidence, not the final result.
3. **Cached NPY headers were loaded before validation.** A small file claiming a
   million-by-million array could trigger a large allocation. Sonnet changed the
   cache path to bounded reads and reused `horizon.benchmark.load_heightmap`,
   which checks dimensions, dtype, raw size, and file length before `np.load`.
   A forged-header test verifies that `np.load` is never reached. Oversized gzip,
   metadata, and NPY caches also have regression checks. Files remain untouched
   when validation fails.
4. **A corruption test kept a stale temporary path.** Three subcases failed before
   reaching the code under test. The parent moved path resolution inside each
   closure so it follows that subcase's current temporary directory. All three
   now exercise the intended rejection.

No remaining must-fix finding was found in the reviewed paths.

## Correctness and data evidence

- `python -m unittest discover -s tests -q`: **94 tests passed** after the final
  codec change. Coverage includes batch boundaries, partial batches, the small
  path, very wide rows, uint32 overflow, independent v1 container construction,
  malformed frames, no aliasing, CLI behavior, and cache safety.
- Both real tiles were downloaded, then reused through verified cache loading.
  Each is 3601×3601 with 25,934,402 raw sample bytes. Korea `N37E127` has valid
  heights -63…1460m; Alps `N46E007` has 398…4469m. Both have zero missing samples.
- The source is [Mapzen/Tilezen Skadi on AWS Open Data](https://registry.opendata.aws/terrain-tiles/),
  a composited real-world elevation model. It is not claimed to be exclusively
  original NASA SRTM. Source and attribution links are preserved in each tile's
  metadata; absent source-imagery headers are not guessed.
- Signed big-endian int16 meters are mapped to little-endian uint16 by adding
  32768. Subtracting that offset reproduces the HGT bytes exactly. No smoothing,
  resizing, quantization, clipping, or missing-sample removal was performed.
- The parent independently verified the inverse HGT bytes and all four hashes
  per tile: downloaded gzip, raw HGT, normalized sample bytes, and NPY file.
- On all ten cases and all three methods, old and current encoders produced
  identical containers. Both decoders read both encoders' outputs correctly.
  Every measured result was checked for exact values, shape, dtype, writable
  storage, and C order. CRC and SHA-256 remain enabled during timed decoding.

## Final paired measurements

Windows, Python 3.10.6, NumPy 2.2.6, zstandard 0.25.0, lz4 4.4.5.
Five warmups per decoder, then 50 repetitions, alternating old/current call order.
Workers had stopped before timing. Time covers the whole in-memory decode call,
including internal checks; external result comparison is outside the timer.
Percentiles use linear interpolation. Raw samples and environment details are
in [decode-real-v2.json](decode-real-v2.json).

| Input, delta+Zstd | Old median ms | New median ms | Old p95 ms | New p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Korea central 256² | 0.4003 | 0.3956 | 0.4648 | 0.4256 |
| Korea central 1024² | 7.8566 | 6.9986 | 8.8185 | 7.5057 |
| Korea full 3601² | 98.8238 | 85.5461 | 106.8574 | 97.5503 |
| Alps central 256² | 0.4129 | 0.4053 | 0.5756 | 0.5537 |
| Alps central 1024² | 8.0931 | 7.2201 | 8.6343 | 8.6038 |
| Alps full 3601² | 100.2365 | 86.4612 | 109.0717 | 94.9466 |

Full-tile median decode time fell **13.44% and 13.74%**; p95 fell **8.71% and
12.95%**. Central 1024-square medians fell about 11%. Small synthetic maps are
essentially unchanged: smooth median 0.3245→0.3234ms, p95 0.3723→0.3748ms;
random median 0.2730→0.2742ms. These small differences are not a proven win or
regression. This is one run on one device, not a statistical or hardware-wide claim.

The final delta containers are 7,996,675B and 8,742,429B, respectively 33.50% and
40.77% smaller than direct Zstd. They are byte-identical to the old version.
Current direct Zstd full-tile medians are 81.9383ms and 81.8254ms. Delta+Zstd
therefore still costs about 4.4% and 5.7% more decode time in exchange for size.
The larger relative delta cost on small maps also remains.

Traced peak allocation for a full delta decode fell from about **85.80→60.39MB**
on Korea and **86.55→61.13MB** on Alps (decimal MB, five observations each).
Removing the checksum's full-array byte copy more than offsets the bounded
scratch buffer. [decode-memory-v2.json](decode-memory-v2.json) contains the peaks.
These are Python/NumPy allocations seen by `tracemalloc`, not process RSS; input
and container allocations precede tracing and untraced native allocations are
excluded. Peak memory of the full preparation/benchmark process is not measured.

Baseline source SHA-256:
`9fa7a91167868235661113086f3dc5185387427ccd311ea3fa76147d4af63eec`.
Final codec source SHA-256:
`4e69e6748af02fbfd443537749c3eb16cb08d3b059f994db8448f41d47a2a3c2`.

## Reproduce

Run from the repository root. Output reports must have new filenames.

```powershell
uv run python -m unittest discover -s tests -q
uv run python scripts/prepare_terrain.py
uv run python scripts/compare_decode.py --baseline benchmarks/reference/codec_v1.py --input data/terrain/N37E127.npy --input data/terrain/N46E007.npy --repeats 50 --output benchmarks/new-comparison.json
```

On a fresh checkout containing metadata only, use an empty `--output-dir` and
update comparison input paths, as described in [terrain instructions](../data/terrain/README.md).
Compare new source/data hashes to the saved report before treating runs as the
same experiment. A source baseline is executed as Python: use the supplied
trusted snapshot, not a downloaded unknown script.

The small [memory probe](reference/measure_memory.py) reproduces the allocation
method with the cached tiles at the default paths. Change its output filename
before running it again; it refuses to overwrite the existing report.

Verdict: Ship for local prototype testing.

Not checked: game-engine integration, disk/cold-cache loading, GPU/collision
readiness, other hardware, process RSS, concurrent cache writers, handcrafted
game maps, dictionary Zstd, or world generation. Faster codec decoding alone does
not establish faster gameplay loading.

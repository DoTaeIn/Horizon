# Cache measurement CLI final audit (2026-10-13)

Claude Opus implemented the timing CLI and updated both READMEs; Claude Sonnet
implemented the memory CLI, CLI tests and method document. Codex reviewed the
changes, ran the final tests and measurements, and independently recomputed the
raw evidence. The cache library itself was implemented by Claude in the preceding
experiment. No Claude background agents were used in this completion pass.

The earlier [draft audit](town-cache-audit.md) and v1 measurements are historical
records. Its missing-memory-worker and incomplete-CLI findings are resolved in
this final version. The exact draft source is preserved in
[reference/compare_town_cache_draft_v1.py](reference/compare_town_cache_draft_v1.py)
so its source hashes remain auditable after the production CLI changes.

## Accepted implementation and corrected findings

- Timing CLI records every repeat's raw constructor metrics and the sum of
  startup plus all load calls. Aggregate names now distinguish millisecond
  statistics from raw nanoseconds. Partial constructor failures close sessions.
- Every recorded schedule source hash is verified, including scripts and the
  memory worker itself when recorded. Missing files, mismatches, malformed
  schedules and explicit unsupported modes fail before measurement.
- Memory defaults now work end to end. Each arm/path/codec/repeat gets a fresh
  interpreter with only a small ROI request and expected output-sequence hash.
  The source map and full timing report stay in the parent process.
- The reviewer rejected an intermediate memory verifier that prohibited repeated
  coordinates, rejected seed zero and exempted its own source hash. These were
  fixed: standing still and revisiting locations are valid trajectory operations.
- The reviewer rejected tests that checked only frame structure while claiming
  to verify raw startup/trajectory values. The accepted tests check the actual
  startup records, timing identities and sums. An independent audit separately
  recomputes statistics without calling the CLI's aggregation helper.
- Existing outputs are never overwritten. Memory sidecars are preflighted;
  cleanup after subsequent failure checks the owned sidecar's identity and hash.
  Container hashes are checked after both timing and memory measurements.

## Verification

`python -m unittest discover -s tests -q`: **381 tests, 376 passed, 5 skipped**
(platform-specific cases). Final suite completed in 11.920 seconds on this host.
The cache library's earlier randomized audit covered 2,700 ROI requests and 1,350
LRU transitions on a rectangular 513×769 map with capacities 1, 4 and 8.

Final measurements replay the exact positions in both previously published
movement schedules (seeds 20261012 and 20261013): 1024×1024 maps, 256 chunks,
256-square viewports, 3 codecs, 2 I/O modes, 3 paths, 120 steps, 3 repeats and
4 loader arms. **51,840 load records**, with startup records inside each report.
Memory runs are separate, warm-buffered only: **216 fresh processes**.

[Independent evidence](town-cache-final-evidence.json) recomputes the realized arm
orders, ROIs, two-phase LRU transitions, hits/misses/evictions, exact returned read
bytes/calls, retained decoded bytes, timing identities, raw trajectory totals,
all published timing aggregates and report/source/container hashes. Memory
sequence hashes are independently regenerated from the source map; retained bytes
and total read bytes are cross-checked against the timing runs.

- [Default-seed final timing and embedded memory](town-cache-1024-v2.json)
- [Default-seed memory sidecar](town-cache-1024-v2.memory.json)
- [Second-seed final timing and embedded memory](town-cache-1024-seed2-v2.json)
- [Second-seed memory sidecar](town-cache-1024-seed2-v2.memory.json)
- [Independent final audit script](audit_town_cache_final.py)
- [Method and commands](../docs/town-cache-method.md)

Both timing JSON files use compact formatting (about 9.8 MiB each). Generated
containers stay in ignored `data/terrain-local/` directories. The independent
file-byte audit expects those locally generated files at the report's recorded
paths; on a fresh checkout they must first be regenerated with the recorded map,
codec and chunk settings. Raw reports, hashes and evidence remain committed.

## Findings and limits

For default-seed warm-buffered delta+Zstd, the 8-chunk cache's median request
cost was 0.007708 ms walking, 0.008208 ms at boundaries and 0.264479 ms teleporting.
The resident whole map was faster per request (0.003708 / 0.003333 / 0.007480 ms).
Including startup, the 120-step median totals were 4.576 / 2.582 / 41.418 ms for
8 chunks versus 4.667 / 4.664 / 4.992 ms for the whole map. Walking totals are
close; boundary reuse helps the 8-chunk cache; teleports favor whole residency.
The second seed has the same qualitative pattern. No universal winner follows.

Retained decoded arrays are exactly 0.5 MiB at capacity 4 and at most 1 MiB at
capacity 8, versus 2 MiB for the whole map. The boundary path uses only six
unique chunks, so its capacity-8 cache actually retains 0.75 MiB. A returned
256-square viewport adds 0.125 MiB outside these retained-array counts.

Default-seed walking median overall tracemalloc peaks were 0.629 / 0.942 /
1.443 / 5.578 MiB for stateless / 4 chunks / 8 chunks / whole map; the whole
map's movement-only traced peak was 2.135 MiB. Corresponding absolute process
RSS high-water marks were 38.375 / 37.469 / 38.516 / 40.875 MiB. Tracemalloc
excludes untracked native allocations. RSS includes runtime/imports, and its
baseline is itself a high-water mark, not current RSS. These are separate
measurements and do not establish a universal RSS saving.

The raw default-seed 4-chunk teleport maximum was 31.677208 ms; it is retained
in the report. Its cause was not established. No outliers were discarded.
F_NOCACHE is a per-handle request, not proof of page eviction or physical cold
SSD reads. These loader-only measurements exclude engine objects, GPU uploads,
collision preparation and frame pacing, and do not predict game FPS.

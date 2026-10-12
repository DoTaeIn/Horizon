# Real terrain disk loading audit — 2026-10-12

What this change does: It compares current direct Zstd with current delta+Zstd
from opening a real container file through completed decoding. It measures both
primed ordinary reads and Windows file data cache bypass reads.

Looks good. Ship as a local measurement tool. The results do not establish a
loading-speed advantage for delta+Zstd on this device.

## Work and review scope

One Claude CLI Opus worker and one Claude CLI Sonnet worker ran in parallel.
Opus wrote the measurement script. Sonnet wrote independent tests from the same
interface contract. They received generic specifications with tools disabled;
repository files and real terrain samples were not sent to the workers. The
parent reviewed the code, checked native calls, ran tests and real measurements,
and rejected the first measurement implementation. Opus supplied the reader fix
and its regression checks; the parent integrated and verified them.

Reviewed paths: `scripts/compare_disk.py`, `tests/test_disk.py`, the existing
`horizon.codec.encode/decode` and validated NPY loader, generated containers,
reports, and reproduction instructions. The codec itself is unchanged. No new
dependencies or cache-purging utilities were installed.

Assumed load: one local benchmark process, two full uint16 tiles, and no other
writer modifying benchmark inputs. This is not an engine or concurrent-server
benchmark. Both workers had stopped before timing began.

## Resolved before acceptance

1. **The initial buffered reader requested the allocation cap, not file size.**
   - **What this is:** The reader must limit allocations while reading complete
     container files of about 8–15MB.
   - **Problem:** `read(MAX_FILE_BYTES + 1)` requested about 65MiB even for those
     smaller files. On this Windows/Python build it added substantial allocation
     cost. An alternating 30-sample check found about 12.5–13.3ms for that reader
     versus 2.2–4.2ms when requesting actual size. The first warm-mode results
     were unsuitable for accepting a representative loading comparison.
   - **Fix:** Opus changed the reader to get size from the open handle with
     `fstat`, validate the cap before allocation, read only `size + 1`, and reject
     changed lengths. Two regression tests check the requested size and rejection
     before reading oversized files. All measurements were rerun.
   - **If we skip it:** Artificial allocation cost obscures the value of smaller
     files and distorts the cached versus bypass comparison. This finding is
     resolved; no outstanding must-fix finding remains in the reviewed paths.

The rejected result is `rejected-disk-load-cap-allocation.json`; the diagnostic
check is `rejected-disk-buffer-allocation.json`. Neither is used for final results.

## Conditions and timing boundaries

- **Warm file data cache:** Before every measured call, the complete same file
  is read with the ordinary reader outside the timer. The measured call then
  opens, reads, and closes the file again, and decodes the newly returned bytes.
  Cache residency is encouraged by immediate priming, not inspected with page
  residency counters.
- **OS file data cache bypass:** Every measured read opens its own handle with
  `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` (`0x28000000`). The aligned
  synchronous read bypasses the Windows file data cache. No global purge occurs.
- **This is not full physical cold loading.** Device/controller caches and file
  metadata can remain cached. Runtime code and libraries are warmed in both
  conditions. These semantics follow Microsoft's
  [file buffering documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering).
- Each method and condition gets two complete unmeasured warmups and 50 measured
  calls per tile. Method order reverses every repetition. Condition order
  reverses in two-repetition blocks. Both tiles use all samples, without choosing
  the best run. The report contains 400 timed calls and their raw measurements.
- The total timer starts before the reader and ends after `decode`. It includes
  open/read/close and allocation/copy/release costs, plus codec CRC and SHA-256
  checks. The bypass path also includes its volume query and aligned allocation.
  External exact-result validation runs after timing. Total/read/decode median
  and p95 are calculated independently; component percentiles are not added.
- The ordinary Python reader and native bypass reader use different API paths.
  Comparing codecs within either condition is the intended comparison. The
  difference between conditions cannot be attributed only to cache misses;
  native setup and buffer-copy costs also differ.
- All containers are prepared before timing, written exclusively to a new
  directory, flushed and `fsync`ed. They are never padded or changed for alignment.
  Every timed call reads a file again; no compressed bytes or decoded arrays are
  retained as an application cache.

## Data and native API evidence

Inputs are the same real, composited Skadi DEMs from the previous audit: Korea
`N37E127` and Alps `N46E007`, each 3601×3601 and 25,934,402 raw bytes. Source NPY
and normalized sample hashes match that run. There is no resizing or new data
transformation. See [terrain provenance](../data/terrain/README.md).

| Tile | Direct Zstd container | Delta+Zstd container | Size reduction |
| --- | ---: | ---: | ---: |
| Korea | 12,024,378B | 7,996,675B | 33.50% |
| Alps | 14,761,014B | 8,742,429B | 40.77% |

The parent independently checked all four generated file hashes against the
previous in-memory audit. They match exactly. Each timed decode matched the
source values, shape and dtype, and returned writable C-order storage. All files
also retained the same length and SHA-256 after timing.

Actual native-read evidence is in
[disk-load-native-evidence.json](disk-load-native-evidence.json). Auditor wrappers
called the real configured Win32 APIs for all four full containers and checked:

- `CreateFileW` received the bypass flags, read-only access and `OPEN_EXISTING`.
- Read request lengths and allocated buffer addresses were 4096-byte aligned.
  Returned byte counts matched each unpadded file's exact size, including EOF.
- Returned bytes matched ordinary reads and saved file hashes.
- Buffers and handles were released. A forced `ReadFile` error 87 propagated as
  an error, with both resources still released and no buffered fallback.

The volume is **C:, NTFS**, with 512-byte logical sectors and 4096-byte physical
sectors. Physical alignment was independently queried with
`GetFileInformationByHandleEx(FileStorageInfo)`. The measurement tool also records
its per-volume logical geometry. No physical disk I/O counters were collected.
The disk model could not be inspected with the sandbox's unavailable storage CIM
query, and was not guessed.

## Final results

Windows 10.0.26300, Python 3.10.6, NumPy 2.2.6, zstandard 0.25.0, lz4 4.4.5.
CPU identifier and all source hashes are in
[disk-load-real-v2.json](disk-load-real-v2.json).

| Tile and condition | Zstd total median / p95 ms | Delta+Zstd total median / p95 ms | Delta median change |
| --- | ---: | ---: | ---: |
| Korea, warm | 82.8357 / 98.7448 | 86.5765 / 101.8089 | +4.52% |
| Alps, warm | 84.5971 / 90.8549 | 86.9619 / 99.5722 | +2.80% |
| Korea, bypass | 92.7157 / 110.2883 | 92.8759 / 105.6630 | +0.17% |
| Alps, bypass | 93.5403 / 106.7221 | 94.7094 / 108.2406 | +1.25% |

Positive change means delta+Zstd takes longer. Direct Zstd is faster by median
under primed-cache reads. Bypass medians are close, with no clear delta loading
win. Bypass p95 favors delta on Korea and direct Zstd on Alps, so a general tail
latency advantage is not established. These are one paired run on one machine;
small differences are not a statistical or hardware-wide claim.

| Tile and condition | Zstd read / decode median ms | Delta read / decode median ms |
| --- | ---: | ---: |
| Korea, warm | 3.1747 / 79.7865 | 2.1507 / 84.4295 |
| Alps, warm | 3.8669 / 80.7094 | 2.3251 / 84.5164 |
| Korea, bypass | 10.2652 / 82.8026 | 7.5630 / 85.4021 |
| Alps, bypass | 11.9294 / 82.0471 | 8.0582 / 86.7559 |

Smaller delta files reduce read time. Additional reconstruction work consumes
that saving. Decoding remains the larger component. These component medians
are separate summaries, not numbers to add to reconstruct the total median.
The established advantage here is storage size, not faster end-to-end loading.

## Verification and reproduction

The complete suite ran **130 tests: 127 passed, 3 skipped**. The skips are two
symlink-creation checks unavailable with current Windows privileges, and the
non-Windows-only rejection check. Actual Windows bypass reads were not skipped:
files of 1, 4095, 4096, 4097, 8193, and 3MiB+123 bytes matched exactly and remained
unchanged. Mocked timer tests confirm priming, read/decode boundaries, result
validation outside timing, and no fallback on bypass failure.

Additional auditor checks confirmed rejection of existing reports and workdirs,
preservation of their contents, and cleanup of only newly created files when
preparation fails after the first input. All 400 raw samples have positive
durations and `total_ns == read_ns + decode_ns`.

Run from the repository root, after other workers stop:

```powershell
uv run python -m unittest discover -s tests -q
uv run python scripts/compare_disk.py --input data/terrain/N37E127.npy --input data/terrain/N46E007.npy --containers-dir data/disk-load-new --repeats 50 --output benchmarks/disk-load-new.json
```

Use new paths for both outputs. The report parent must exist. For a fresh
checkout, prepare the data as described in the terrain instructions and adjust
input paths. Match input/container hashes before comparing runs. Bypass is
mandatory: unsupported platforms or native failures stop the run without a
successful report. Failure after preparation can leave the valid prepared files
for inspection; existing files are never recursively cleaned up.

Final script SHA-256:
`a4f2deb970b393dee6629c21e19b28dd79bd1c04290eb4dda84ac864e7981b5e`.
Unchanged codec SHA-256:
`4e69e6748af02fbfd443537749c3eb16cb08d3b059f994db8448f41d47a2a3c2`.

Verdict: Ship as a local comparison tool. Keep the size benefit separate from a
loading-speed claim.

Not checked: physical device/controller cache eviction, cold boot, hardware I/O
telemetry, other drives, other hardware, process RSS, concurrent writers,
handcrafted game maps, dictionary Zstd, engine/GPU/collision readiness, or world
generation. This test does not measure complete gameplay loading.

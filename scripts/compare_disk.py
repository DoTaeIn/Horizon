#!/usr/bin/env python3
'''Local heightmap disk-load benchmark: zstd vs delta_zstd containers on Windows.

Measures reading a prepared container file plus decoding it, either with the
OS file cache warm (warm_buffered) or with the Windows file data cache bypassed
for the reading handle (os_cache_bypass). Cache bypass is NOT a physically cold
device read: SSD/controller/drive caches are not evicted.
'''
from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from horizon import codec  # noqa: E402
from horizon.benchmark import load_heightmap  # noqa: E402
from horizon.codec import decode, encode  # noqa: E402

MAX_FILE_BYTES = 65 * 1024 * 1024
METHODS = ('zstd', 'delta_zstd')
MODES = ('warm_buffered', 'os_cache_bypass')
WARMUPS = 2
DEFAULT_REPEATS = 50
MIN_REPEATS = 30
MODE_BLOCK = 2

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
BYPASS_FLAGS = FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN
MEM_COMMIT = 0x00001000
MEM_RESERVE = 0x00002000
MEM_RELEASE = 0x00008000
PAGE_READWRITE = 0x04
MIN_ALIGNMENT = 4096
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_K32 = None


def _log(message):
    print(message, file=sys.stderr, flush=True)


def _kernel32():
    '''Return (kernel32, wintypes), configured lazily; Windows only.'''
    global _K32
    if sys.platform != 'win32':
        raise OSError('os_cache_bypass requires Windows')
    if _K32 is None:
        from ctypes import wintypes as wt
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p,
                                  wt.DWORD, wt.DWORD, wt.HANDLE]
        k.CreateFileW.restype = wt.HANDLE
        k.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD,
                               ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
        k.ReadFile.restype = wt.BOOL
        k.CloseHandle.argtypes = [wt.HANDLE]
        k.CloseHandle.restype = wt.BOOL
        k.GetFileSizeEx.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
        k.GetFileSizeEx.restype = wt.BOOL
        k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
        k.VirtualAlloc.restype = ctypes.c_void_p
        k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
        k.VirtualFree.restype = wt.BOOL
        k.GetVolumePathNameW.argtypes = [wt.LPCWSTR, wt.LPWSTR, wt.DWORD]
        k.GetVolumePathNameW.restype = wt.BOOL
        k.GetDiskFreeSpaceW.argtypes = [wt.LPCWSTR] + [ctypes.POINTER(wt.DWORD)] * 4
        k.GetDiskFreeSpaceW.restype = wt.BOOL
        _K32 = (k, wt)
    return _K32


def _win_error(name):
    code = ctypes.get_last_error()
    return ctypes.WinError(code, f'{name} failed (error {code})')


def _check_size(size):
    if size <= 0:
        raise ValueError('file is empty')
    if size > MAX_FILE_BYTES:
        raise ValueError(f'file exceeds {MAX_FILE_BYTES} bytes')


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def volume_geometry(path):
    '''Query the volume root and sector size for path (Windows only).'''
    k32, wt = _kernel32()
    root = ctypes.create_unicode_buffer(32768)
    if not k32.GetVolumePathNameW(os.path.abspath(path), root, len(root)):
        raise _win_error('GetVolumePathNameW')
    spc, bps, free, total = wt.DWORD(), wt.DWORD(), wt.DWORD(), wt.DWORD()
    if not k32.GetDiskFreeSpaceW(root.value, ctypes.byref(spc), ctypes.byref(bps),
                                 ctypes.byref(free), ctypes.byref(total)):
        raise _win_error('GetDiskFreeSpaceW')
    sector = bps.value
    if sector <= 0 or sector & (sector - 1):
        raise OSError(f'unexpected bytes per sector: {sector}')
    return {
        'volume_root': root.value,
        'bytes_per_sector': sector,
        'sectors_per_cluster': spc.value,
        'alignment_bytes': max(MIN_ALIGNMENT, sector),
    }


def read_buffered(path):
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        _check_size(size)
        data = f.read(size + 1)
    if len(data) != size:
        raise ValueError(
            "file size changed during read: expected %d bytes, got %d" % (size, len(data))
        )
    return data

def read_unbuffered(path):
    k32, wt = _kernel32()
    path = os.path.abspath(path)
    _check_size(os.stat(path).st_size)
    align = volume_geometry(path)['alignment_bytes']
    handle = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None,
                             OPEN_EXISTING, BYPASS_FLAGS, None)
    if handle is None or handle == INVALID_HANDLE_VALUE:
        raise _win_error('CreateFileW')
    buf = None
    try:
        size = ctypes.c_longlong()
        if not k32.GetFileSizeEx(handle, ctypes.byref(size)):
            raise _win_error('GetFileSizeEx')
        _check_size(size.value)
        request = -(-size.value // align) * align
        buf = k32.VirtualAlloc(None, request, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not buf:
            raise _win_error('VirtualAlloc')
        if buf % align:
            raise OSError(f'VirtualAlloc buffer not aligned to {align} bytes')
        got = wt.DWORD()
        if not k32.ReadFile(handle, buf, request, ctypes.byref(got), None):
            raise _win_error('ReadFile')
        if got.value != size.value:
            raise OSError(f'ReadFile returned {got.value} bytes, expected {size.value}')
        data = ctypes.string_at(buf, size.value)
    finally:
        errors = []
        if buf and not k32.VirtualFree(buf, 0, MEM_RELEASE):
            errors.append(_win_error('VirtualFree'))
        if not k32.CloseHandle(handle):
            errors.append(_win_error('CloseHandle'))
    if errors:
        raise errors[0]
    return data


def validate_result(actual, expected):
    if type(actual) is not np.ndarray:
        raise ValueError(f'decoded result is {type(actual).__name__}, not ndarray')
    if actual.shape != expected.shape:
        raise ValueError(f'shape mismatch: {actual.shape} != {expected.shape}')
    if actual.dtype.str != expected.dtype.str:
        raise ValueError(f'dtype mismatch: {actual.dtype.str} != {expected.dtype.str}')
    if not actual.flags.c_contiguous or not actual.flags.writeable:
        raise ValueError('decoded array is not C-contiguous and writable')
    if not np.array_equal(actual, expected):
        raise ValueError('decoded values differ from source')


def measure_once(path, expected, mode):
    if mode == 'warm_buffered':
        read_buffered(path)  # prime the OS file cache outside the timer
        reader = read_buffered
    elif mode == 'os_cache_bypass':
        reader = read_unbuffered
    else:
        raise ValueError(f'unknown mode: {mode!r}')
    t0 = time.perf_counter_ns()
    blob = reader(path)
    t1 = time.perf_counter_ns()
    result = decode(blob)
    t2 = time.perf_counter_ns()
    del blob
    validate_result(result, expected)
    return {'total_ns': t2 - t0, 'read_ns': t1 - t0, 'decode_ns': t2 - t1}


def summarize(samples):
    if not samples:
        raise ValueError('no samples')
    out = {'repeats': len(samples)}
    for name in ('total', 'read', 'decode'):
        values = np.array([s[f'{name}_ns'] for s in samples], dtype=np.float64)
        out[name] = {
            'median_ms': float(np.median(values)) / 1e6,
            'p95_ms': float(np.percentile(values, 95, method='linear')) / 1e6,
        }
    out['samples_ns'] = list(samples)
    return out


def write_report(path, report):
    path = Path(path)
    text = json.dumps(report, indent=2, allow_nan=False)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _repeats(text):
    value = int(text)
    if value < MIN_REPEATS:
        raise argparse.ArgumentTypeError(f'--repeats must be >= {MIN_REPEATS}')
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--input', action='append', required=True, type=Path,
                   help='full-tile heightmap .npy (repeatable)')
    p.add_argument('--containers-dir', required=True, type=Path,
                   help='new directory for container files (must not exist)')
    p.add_argument('--output', required=True, type=Path,
                   help='JSON report path (must not exist)')
    p.add_argument('--repeats', type=_repeats, default=DEFAULT_REPEATS)
    return p.parse_args(argv)


def _prepare(inputs, workdir):
    workdir.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir()
    created, cases, arrays = [], [], []
    try:
        for idx, src in enumerate(inputs):
            _log(f'preparing case {idx}: {src}')
            source_sha = _sha256_file(src)
            arr = load_heightmap(src)
            if (type(arr) is not np.ndarray or arr.ndim != 2 or arr.size == 0
                    or arr.dtype.str != '<u2' or not arr.flags.c_contiguous):
                raise ValueError(f'unexpected array from load_heightmap: {src}')
            if _sha256_file(src) != source_sha:
                raise ValueError(f'source changed while loading: {src}')
            arr.flags.writeable = False
            files = {}
            for method in METHODS:
                blob = bytes(encode(arr, method))
                validate_result(decode(blob), arr)
                _check_size(len(blob))
                path = workdir / f'{idx:03d}_{method}.bin'
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0))
                created.append(path)
                with os.fdopen(fd, 'wb') as f:
                    f.write(blob)
                    f.flush()
                    os.fsync(f.fileno())
                files[method] = {'path': str(path), 'container_bytes': len(blob),
                                 'sha256': hashlib.sha256(blob).hexdigest()}
                del blob
            cases.append({
                'name': f'{idx:03d}_{src.stem}',
                'source': str(src),
                'source_npy_sha256': source_sha,
                'shape': list(arr.shape),
                'dtype': arr.dtype.str,
                'raw_bytes': int(arr.nbytes),
                'array_sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
                'files': files,
            })
            arrays.append(arr)
    except BaseException:
        for path in reversed(created):
            with contextlib.suppress(OSError):
                path.unlink()
        with contextlib.suppress(OSError):
            workdir.rmdir()  # only succeeds if empty
        raise
    return cases, arrays


def _order(rep):
    modes = MODES if (rep // MODE_BLOCK) % 2 == 0 else MODES[::-1]
    methods = METHODS if rep % 2 == 0 else METHODS[::-1]
    return modes, methods


def _benchmark(case, expected, repeats):
    paths = {m: Path(case['files'][m]['path']) for m in METHODS}
    for mode in MODES:
        for method in METHODS:
            for _ in range(WARMUPS):
                measure_once(paths[method], expected, mode)
    samples = {mode: {m: [] for m in METHODS} for mode in MODES}
    for rep in range(repeats):
        modes, methods = _order(rep)
        for mode in modes:
            for method in methods:
                samples[mode][method].append(measure_once(paths[method], expected, mode))
        if (rep + 1) % 10 == 0 or rep + 1 == repeats:
            _log(f'  repetition {rep + 1}/{repeats}')
    return samples


def _compare(summaries, files):
    z, d = summaries['zstd']['total'], summaries['delta_zstd']['total']
    return {
        'delta_zstd_over_zstd_total_median': d['median_ms'] / z['median_ms'],
        'delta_zstd_over_zstd_total_p95': d['p95_ms'] / z['p95_ms'],
        'delta_zstd_over_zstd_container_bytes':
            files['delta_zstd']['container_bytes'] / files['zstd']['container_bytes'],
    }


def _verify_containers(cases):
    for case in cases:
        for method, info in case['files'].items():
            data = read_buffered(info['path'])
            info['post_benchmark_bytes'] = len(data)
            info['post_benchmark_sha256'] = hashlib.sha256(data).hexdigest()
            info['post_benchmark_match'] = (
                info['post_benchmark_bytes'] == info['container_bytes']
                and info['post_benchmark_sha256'] == info['sha256'])
            if not info['post_benchmark_match']:
                raise ValueError(f'container changed during benchmark: {info[chr(112) + chr(97) + chr(116) + chr(104)]}')


def _source_hashes():
    pkg = Path(codec.__file__).resolve().parent
    hashes = {'scripts/compare_disk.py': _sha256_file(Path(__file__).resolve())}
    for p in sorted(pkg.rglob('*.py')):
        hashes[p.relative_to(pkg.parent).as_posix()] = _sha256_file(p)
    return hashes


def _environment():
    return {
        'python': sys.version,
        'implementation': platform.python_implementation(),
        'os': platform.platform(),
        'os_version': platform.version(),
        'machine': platform.machine(),
        'cpu': platform.processor(),
        'cpu_count': os.cpu_count(),
        'numpy': np.__version__,
        'zstandard': metadata.version('zstandard'),
        'lz4': metadata.version('lz4'),
    }


SCOPE = (
    'End-to-end load of prepared container files from local disk followed by '
    'horizon.codec.decode, per method. Timed read includes open, read, close and, '
    'for os_cache_bypass, the volume geometry query, aligned buffer allocation, '
    'copy to bytes and buffer release. warm_buffered: each measured read is preceded '
    'by an unmeasured complete ordinary read of the same file, so the OS file cache '
    'is expected to be present. os_cache_bypass: the reading handle is opened with '
    'FILE_FLAG_NO_BUFFERING, bypassing the Windows file data cache for that read; '
    'this is NOT a physically cold device read (SSD/controller/drive caches are not '
    'evicted, no global cache purge is performed, and physical read counters were '
    'not verified).'
)


def _run(args):
    script_sha = _sha256_file(Path(__file__).resolve())
    cases, arrays = _prepare(args.input, args.containers_dir)
    geometry = volume_geometry(args.containers_dir)
    _log(f'volume geometry: {geometry}')
    for case, expected in zip(cases, arrays):
        name = case['name']
        _log(f'benchmarking {name}')
        samples = _benchmark(case, expected, args.repeats)
        case['modes'] = {mode: {m: summarize(samples[mode][m]) for m in METHODS}
                         for mode in MODES}
        case['comparisons'] = {mode: _compare(case['modes'][mode], case['files'])
                               for mode in MODES}
    del arrays
    _verify_containers(cases)
    return {
        'schema_version': 1,
        'scope': SCOPE,
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_sha256': script_sha,
        'codec_source_sha256': _sha256_file(Path(codec.__file__).resolve()),
        'source_files_sha256': _source_hashes(),
        'environment': _environment(),
        'settings': {
            'repeats': args.repeats,
            'warmups': WARMUPS,
            'methods': list(METHODS),
            'modes': list(MODES),
            'codec_version': codec.VERSION,
            'codec_header_size': codec.HEADER_SIZE,
            'codec_max_raw_bytes': codec.MAX_RAW_BYTES,
            'compression_level': codec.ZSTD_LEVEL,
            'max_file_bytes': MAX_FILE_BYTES,
            'timer': 'time.perf_counter_ns: before reader, after reader, after decode; '
                     'total_ns = read_ns + decode_ns; validation after final timer',
            'p95_policy': 'numpy.percentile(method=linear) and numpy.median over all raw '
                          'samples; total/read/decode summarized independently, never summed; '
                          'no min-of-runs',
            'order': {
                'cases': 'sequential in --input order; each case fully benchmarked before the next',
                'warmups': f'{WARMUPS} unmeasured complete measure_once calls per mode+method '
                           'per case, in MODES x METHODS order, before timed repetitions',
                'method_order': 'per repetition r: zstd,delta_zstd when r is even, reversed when odd',
                'mode_order': f'blocks of {MODE_BLOCK} repetitions: warm_buffered,os_cache_bypass '
                              f'when (r // {MODE_BLOCK}) is even, reversed when odd',
                'nesting': 'within a repetition: for each mode in mode order, for each method '
                           'in method order, one measured call',
            },
            'caching_policy': 'containers written once to a new --containers-dir with fsync, '
                              'kept for audit; no in-process data cache; every measured call '
                              'reads the file anew and discards bytes and decoded array after '
                              'validation; only source arrays are retained for validation',
        },
        'cache_mode_evidence': {
            'warm_buffered': {
                'api': 'Python open(rb), fstat size validation, read of actual file size+1 bytes',
                'priming': 'complete unmeasured ordinary read immediately before each measured read',
            },
            'os_cache_bypass': {
                'api': 'CreateFileW, GetFileSizeEx, VirtualAlloc, one synchronous ReadFile, '
                       'VirtualFree(MEM_RELEASE), CloseHandle',
                'desired_access': GENERIC_READ,
                'share_mode': FILE_SHARE_READ,
                'creation_disposition': OPEN_EXISTING,
                'flags_and_attributes': BYPASS_FLAGS,
                'flags_and_attributes_hex': hex(BYPASS_FLAGS),
                'flag_values': {
                    'FILE_FLAG_NO_BUFFERING': hex(FILE_FLAG_NO_BUFFERING),
                    'FILE_FLAG_SEQUENTIAL_SCAN': hex(FILE_FLAG_SEQUENTIAL_SCAN),
                },
                'alignment_policy': f'max({MIN_ALIGNMENT}, bytes_per_sector) from GetDiskFreeSpaceW '
                                    'on the GetVolumePathNameW root, queried per call inside the '
                                    'read timing; request rounded up to alignment; buffer from '
                                    'VirtualAlloc (allocation-granularity aligned, verified per call)',
                'eof_policy': 'ReadFile byte count must equal GetFileSizeEx size from the same '
                              'handle; file on disk is never padded or modified',
                'volume_geometry': geometry,
                'limitations': 'bypasses only the Windows file data cache for the reading handle; '
                               'device and controller caches are not evicted; physical read '
                               'counters and disk model were not queried or verified',
            },
        },
        'cases': cases,
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        if os.path.lexists(args.output):
            raise FileExistsError(f'refusing to overwrite existing report: {args.output}')
        if not args.output.parent.is_dir():
            raise FileNotFoundError(f'report directory does not exist: {args.output.parent}')
        if os.path.lexists(args.containers_dir):
            raise FileExistsError(f'containers dir already exists: {args.containers_dir}')
        _kernel32()  # os_cache_bypass is mandatory; fail before any preparation
        report = _run(args)
        write_report(args.output, report)
    except Exception as exc:
        _log(f'error: {type(exc).__name__}: {exc}')
        return 1
    _log(f'report written: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

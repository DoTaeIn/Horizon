#!/usr/bin/env python3
'''Compare warm in-memory decode time of a baseline codec and horizon.codec.

This measures decode() of an already-encoded bytes object held in memory. It is
not a disk or engine streaming benchmark. Run it only after all other workers
have finished so that timings are not disturbed.
'''
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_BASELINE = 'benchmarks/reference/codec_v1.py'
MIN_REPEATS = 30
DEFAULT_REPEATS = 50
WARMUP = 5
SYNTHETIC_SIZE = 256
CROP_SIZES = (256, 1024)
ORDER_POLICY = ('even repetition index: baseline then current; '
                'odd repetition index: current then baseline')
LIBRARIES = ('numpy', 'zstandard', 'lz4', 'zstd', 'pyzstd', 'horizon')


class ValidationError(RuntimeError):
    pass


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError('baseline codec not found: ' + str(p))
    spec = importlib.util.spec_from_file_location('horizon_baseline_codec', p)
    if spec is None or spec.loader is None:
        raise ImportError('cannot load baseline codec from ' + str(p))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def load_current():
    try:
        return importlib.import_module('horizon.codec')
    except ModuleNotFoundError:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module('horizon.codec')


def load_benchmark_helpers():
    benchmark = importlib.import_module('horizon.benchmark')
    return benchmark.load_heightmap, benchmark.make_fixtures


def resolve_methods(baseline, current):
    methods = tuple(current.METHODS)
    base = getattr(baseline, 'METHODS', None)
    if base is not None and tuple(base) != methods:
        raise ValidationError(f'METHODS differ: baseline={tuple(base)} current={methods}')
    return methods


def make_case(label, kind, source, row, col, arr):
    if arr.ndim != 2 or arr.size == 0 or arr.dtype != np.dtype('<u2'):
        raise ValueError(f'{label}: expected a non-empty 2D little-endian uint16 array, '
                         f'got shape={arr.shape} dtype={arr.dtype}')
    return {
        'label': label,
        'kind': kind,
        'source': source,
        'row_origin': int(row),
        'col_origin': int(col),
        'shape': [int(arr.shape[0]), int(arr.shape[1])],
        'array': np.ascontiguousarray(arr),
    }


def central_crop(arr, size):
    '''Deterministic central crop, or None when the map is too small.'''
    h, w = arr.shape
    if h < size or w < size:
        return None
    row, col = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(arr[row:row + size, col:col + size]), row, col


def build_cases(inputs, load_heightmap, make_fixtures, synthetic_size=SYNTHETIC_SIZE):
    cases = []
    for name, kind, arr in make_fixtures(synthetic_size):
        source = f'make_fixtures({synthetic_size})'
        cases.append(make_case('synthetic/' + str(name), str(kind), source, 0, 0, arr))
    for path in inputs:
        arr = load_heightmap(Path(path))
        tile = Path(path).stem
        source = str(path)
        for size in CROP_SIZES:
            crop = central_crop(arr, size)
            if crop is None or crop[0].shape == arr.shape:
                continue
            sub, row, col = crop
            cases.append(make_case(f'{tile}/crop{size}@r{row}c{col}', 'real-crop', source, row, col, sub))
        cases.append(make_case(f'{tile}/full', 'real-full', source, 0, 0, arr))
    return cases


def check_decoded(result, expected, label):
    if not isinstance(result, np.ndarray):
        raise ValidationError(f'{label}: result is {type(result).__name__}, not ndarray')
    if result.dtype.str != expected.dtype.str:
        raise ValidationError(f'{label}: dtype {result.dtype.str} != {expected.dtype.str}')
    if result.shape != expected.shape:
        raise ValidationError(f'{label}: shape {result.shape} != {expected.shape}')
    if not result.flags.writeable:
        raise ValidationError(label + ': result is not writable')
    if not result.flags.c_contiguous:
        raise ValidationError(label + ': result is not C contiguous')
    if not np.array_equal(result, expected):
        raise ValidationError(label + ': decoded values differ from the input')


def validate_case_method(baseline, current, arr, method):
    '''Byte-identical containers and cross-decode compatibility; returns the container.'''
    enc_old = bytes(baseline.encode(arr, method))
    enc_new = bytes(current.encode(arr, method))
    if enc_old != enc_new:
        raise ValidationError(f'{method}: baseline and current containers differ '
                              f'({len(enc_old)} vs {len(enc_new)} bytes)')
    for dec_label, dec in (('baseline', baseline.decode), ('current', current.decode)):
        for enc_label, enc in (('baseline', enc_old), ('current', enc_new)):
            check_decoded(dec(enc), arr, f'{method}: {dec_label}.decode({enc_label} container)')
    return enc_old


def summarize_times(ns):
    ms = np.asarray(ns, dtype=np.float64) / 1e6
    return {
        'repeats': len(ns),
        'median_ms': float(np.percentile(ms, 50, method='linear')),
        'p95_ms': float(np.percentile(ms, 95, method='linear')),
        'samples_ns': [int(x) for x in ns],
    }


def relative(old_ms, new_ms):
    if old_ms <= 0 or new_ms <= 0:
        return {'new_over_old_percent': None, 'speedup': None}
    return {'new_over_old_percent': 100.0 * new_ms / old_ms, 'speedup': old_ms / new_ms}


def measure_method(baseline_decode, current_decode, encoded, expected, repeats, warmup=WARMUP):
    '''Time only the decode calls; validate every result after timing.'''
    for _ in range(warmup):
        baseline_decode(encoded)
        current_decode(encoded)
    clock = time.perf_counter_ns
    old_ns, new_ns = [], []
    for rep in range(repeats):
        order = ('old', 'new') if rep % 2 == 0 else ('new', 'old')
        results = {}
        for which in order:
            fn = baseline_decode if which == 'old' else current_decode
            t0 = clock()
            out = fn(encoded)
            t1 = clock()
            (old_ns if which == 'old' else new_ns).append(t1 - t0)
            results[which] = out
        check_decoded(results['old'], expected, f'timed baseline decode, repetition {rep}')
        check_decoded(results['new'], expected, f'timed current decode, repetition {rep}')
        del results
    old = summarize_times(old_ns)
    new = summarize_times(new_ns)
    out = {'baseline': old, 'current': new}
    out.update(relative(old['median_ms'], new['median_ms']))
    return out


def run_benchmark(baseline, current, cases, methods, repeats, warmup=WARMUP, log=None):
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    results = []
    for index, case in enumerate(cases, 1):
        arr = case['array']
        label = case['label']
        log(f'[{index}/{len(cases)}] {label} shape={case["shape"]}')
        record = {k: v for k, v in case.items() if k != 'array'}
        record['dtype'] = arr.dtype.str
        record['array_sha256'] = sha256_hex(arr.tobytes())
        record['raw_bytes'] = int(arr.nbytes)
        record['methods'] = {}
        for method in methods:
            encoded = validate_case_method(baseline, current, arr, method)
            timing = measure_method(baseline.decode, current.decode, encoded, arr, repeats, warmup)
            entry = {
                'container_bytes': len(encoded),
                'container_sha256': sha256_hex(encoded),
                'compression_ratio': arr.nbytes / len(encoded),
                'container_over_raw': len(encoded) / arr.nbytes,
            }
            entry.update(timing)
            record['methods'][method] = entry
            log('    {}: baseline {:.4f} ms, current {:.4f} ms, speedup {}'.format(
                method, timing['baseline']['median_ms'], timing['current']['median_ms'], timing['speedup']))
        results.append(record)
    return results


def cpu_name():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def library_versions():
    out = {}
    for dist in LIBRARIES:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


def environment():
    return {
        'cpu': cpu_name(),
        'machine': platform.machine(),
        'logical_cpus': os.cpu_count(),
        'os': platform.platform(),
        'python': sys.version,
        'python_implementation': platform.python_implementation(),
        'libraries': library_versions(),
    }


def write_report(path, report):
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with open(target, 'x', encoding='utf-8', newline='\n') as fh:
            created = True
            fh.write(text)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


def _repeats(text):
    value = int(text)
    if value < MIN_REPEATS:
        raise argparse.ArgumentTypeError(f'repeats must be at least {MIN_REPEATS}')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Warm in-memory decode comparison: baseline vs horizon.codec.')
    parser.add_argument('--baseline', default=DEFAULT_BASELINE)
    parser.add_argument('--input', action='append', default=None, help='NPY heightmap (repeatable)')
    parser.add_argument('--repeats', type=_repeats, default=DEFAULT_REPEATS)
    parser.add_argument('--output', required=True, help='new JSON report path; never overwritten')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        print('error: output already exists, refusing to overwrite: ' + str(output), file=sys.stderr)
        return 2
    inputs = list(args.input or [])
    try:
        baseline = load_baseline(args.baseline)
        current = load_current()
        load_heightmap, make_fixtures = load_benchmark_helpers()
        methods = resolve_methods(baseline, current)
        cases = build_cases(inputs, load_heightmap, make_fixtures)
        results = run_benchmark(baseline, current, cases, methods, args.repeats)
        current_file = getattr(current, '__file__', None)
        report = {
            'schema_version': 1,
            'scope': 'warm in-memory decode of an already-encoded bytes object; '
                     'not disk or engine streaming',
            'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'repeats': args.repeats,
            'warmup_decodes_per_decoder': WARMUP,
            'order_policy': ORDER_POLICY,
            'timer': 'time.perf_counter_ns around decode calls only',
            'percentile_method': 'numpy.percentile(method=linear); median is the 50th percentile',
            'checksums': {
                'array_sha256': True,
                'container_sha256': True,
                'post_timing_result_validation': True,
                'note': 'only external result verification and provenance hashing happen outside '
                        'the timed regions; the codec internal CRC/SHA256 checks remain enabled '
                        'and are included in the timed decode calls',
            },
            'environment': environment(),
            'baseline': {'path': str(args.baseline), 'sha256': file_sha256(args.baseline)},
            'current': {
                'module': current.__name__,
                'path': current_file,
                'sha256': file_sha256(current_file) if current_file else None,
            },
            'methods': list(methods),
            'inputs': [{'path': str(p), 'sha256': file_sha256(p)} for p in inputs],
            'cases': results,
        }
        write_report(output, report)
    except (ValidationError, OSError, ImportError, ValueError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 1
    print('wrote ' + str(output), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())

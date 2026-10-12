#!/usr/bin/env python3
'''Fetch public Skadi HGT tiles and store them as lossless uint16 heightmaps.

Mapzen/Tilezen Skadi is a real-world composited DEM, not necessarily original
NASA SRTM exclusively. Nothing here claims ground-truth accuracy.
'''
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import http.client
import importlib
import io
import json
import re
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np

SIDE = 3601
MAX_GZIP_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024
NPY_HEADER_ALLOWANCE = 4096
TIMEOUT_S = 60
NODATA_M = -32768
OFFSET = 32768
DEFAULT_OUTPUT_DIR = 'data/terrain'

TILE_URLS = {
    'N37E127': 'https://elevation-tiles-prod.s3.amazonaws.com/skadi/N37/N37E127.hgt.gz',
    'N46E007': 'https://elevation-tiles-prod.s3.amazonaws.com/skadi/N46/N46E007.hgt.gz',
}
REGISTRY_URL = 'https://registry.opendata.aws/terrain-tiles/'
SOURCE_DOCS_URL = 'https://github.com/tilezen/joerd/blob/master/docs/data-sources.md'
ATTRIBUTION_URL = 'https://github.com/tilezen/joerd/blob/master/docs/attribution.md'
ATTRIBUTION = [
    'Mapzen / Tilezen',
    'Terrain data courtesy of the U.S. Geological Survey.',
    'Produced using Copernicus data and information funded by the European Union - EU-DEM layers.',
    'NOAA / ETOPO1',
]
EQUATION = 'u = h + 32768 (h: signed int16 meters, u: uint16 little-endian)'
INVERSE = 'h = u - 32768, stored as big-endian int16, reproduces the original HGT bytes exactly'
KEPT_HEADERS = ('content-length', 'content-type', 'last-modified', 'etag')
TILE_RE = re.compile('^([NS])([0-9]{2})([EW])([0-9]{3})$')
VOLATILE_KEYS = ('fetched_utc', 'http_headers', 'source_imagery_headers', 'source_imagery_note')


class TerrainError(ValueError):
    pass


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def tile_extent(name):
    m = TILE_RE.match(name)
    if m is None:
        raise ValueError('invalid tile name: ' + repr(name))
    lat = int(m.group(2)) * (1 if m.group(1) == 'N' else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == 'E' else -1)
    return {
        'south_lat': lat,
        'north_lat': lat + 1,
        'west_lon': lon,
        'east_lon': lon + 1,
        'crs': 'WGS84 geographic degrees',
        'row_order': 'north-to-south',
        'col_order': 'west-to-east',
        'note': 'samples sit on the integer-degree grid; both edges are included',
    }


def file_names(name):
    return {'gzip': name + '.hgt.gz', 'npy': name + '.npy', 'json': name + '.json'}


def tile_paths(name, out):
    return {key: Path(out) / value for key, value in file_names(name).items()}


def download_limited(url, max_bytes=MAX_GZIP_BYTES, timeout=TIMEOUT_S):
    '''Fetch one allowlisted HTTPS URL without auth; return (bytes, kept headers).'''
    if url not in TILE_URLS.values():
        raise TerrainError('refusing to fetch non-allowlisted URL: ' + url)
    request = urllib.request.Request(url, headers={'User-Agent': 'horizon-terrain-prep/1'})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        final = resp.geturl()
        if final != url:
            raise TerrainError('unexpected redirect to ' + str(final))
        headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        declared = headers.get('content-length', '')
        if declared.isdigit() and int(declared) > max_bytes:
            raise TerrainError(f'download declares {declared} bytes, limit is {max_bytes}')
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise TerrainError(f'download exceeds the {max_bytes} byte limit')
    kept = {k: v for k, v in headers.items() if k in KEPT_HEADERS or k.startswith('x-amz-meta-')}
    return data, kept


def gunzip_hgt(data, side=SIDE):
    '''Decompress with a hard output cap; require exactly side*side*2 bytes.'''
    expected = side * side * 2
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as gz:
            raw = gz.read(expected + 1)
    except (OSError, EOFError, zlib.error) as exc:
        raise TerrainError(f'invalid gzip stream: {exc}') from exc
    if len(raw) != expected:
        got = 'more than ' + str(expected) if len(raw) > expected else str(len(raw))
        raise TerrainError(f'HGT payload is {got} bytes, expected exactly {expected}')
    return raw


def denormalize_hgt(u):
    return (u.astype(np.int32) - OFFSET).astype('>i2').tobytes()


def normalize_hgt(raw, side=SIDE):
    '''Big-endian int16 HGT bytes -> C-contiguous little-endian uint16 (lossless).'''
    raw = bytes(raw)
    if len(raw) != side * side * 2:
        raise TerrainError(f'HGT is {len(raw)} bytes, expected {side * side * 2}')
    h = np.frombuffer(raw, dtype='>i2').reshape(side, side)
    u = (h.astype(np.int32) + OFFSET).astype('<u2')
    if denormalize_hgt(u) != raw:
        raise TerrainError('uint16 normalization is not invertible to the original HGT bytes')
    return u


def summarize_values(u):
    valid = u != NODATA_M + OFFSET
    count = int(valid.sum())
    out = {
        'nodata_count': int(u.size - count),
        'valid_count': count,
        'valid_min_m': None,
        'valid_max_m': None,
    }
    if count:
        values = u[valid]
        out['valid_min_m'] = int(values.min()) - OFFSET
        out['valid_max_m'] = int(values.max()) - OFFSET
    return out


def npy_bytes(u):
    buf = io.BytesIO()
    np.save(buf, u, allow_pickle=False)
    return buf.getvalue()


def build_metadata(name, side, headers, gz, raw, u, npy, fetched_utc):
    imagery = {k: v for k, v in sorted(headers.items()) if k.startswith('x-amz-meta-')}
    if imagery:
        note = 'Copied verbatim from the HTTP response; not independently verified.'
    else:
        note = 'No source imagery header was present in the response; sources are not identified.'
    meta = {
        'schema_version': 1,
        'tile': name,
        'source_url': TILE_URLS[name],
        'dataset': 'Mapzen/Tilezen Skadi terrain tiles (HGT, composited real-world DEM)',
        'dataset_note': 'Composited from several public elevation sources; not necessarily original '
                        'NASA SRTM exclusively. No ground-truth accuracy is claimed.',
        'extent': tile_extent(name),
        'fetched_utc': fetched_utc,
        'http_headers': dict(sorted(headers.items())),
        'source_imagery_headers': imagery,
        'source_imagery_note': note,
        'shape': [side, side],
        'raw_format': 'signed int16 big-endian meters, rows north-to-south, nodata -32768',
        'normalized_dtype': '<u2',
        'normalization': {
            'equation': EQUATION,
            'inverse': INVERSE,
            'offset': OFFSET,
            'nodata_m': NODATA_M,
            'nodata_u': NODATA_M + OFFSET,
        },
        'processing': 'none: no smoothing, clipping, interpolation, resizing or region selection',
        'sha256': {
            'gzip': sha256_hex(gz),
            'raw_hgt': sha256_hex(raw),
            'normalized_uint16_le': sha256_hex(u.tobytes()),
            'npy_file': sha256_hex(npy),
        },
        'sizes_bytes': {'gzip': len(gz), 'raw_hgt': len(raw), 'npy_file': len(npy)},
        'files': file_names(name),
        'registry_url': REGISTRY_URL,
        'source_docs_url': SOURCE_DOCS_URL,
        'attribution_url': ATTRIBUTION_URL,
        'attribution': list(ATTRIBUTION),
    }
    meta.update(summarize_values(u))
    return meta


def read_bounded(path, limit):
    '''Read a whole file, refusing anything larger than limit bytes.'''
    with open(path, 'rb') as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise TerrainError(f'{Path(path).name} exceeds the {limit} byte limit')
    return data


def load_cached_npy(path):
    '''Load via horizon.benchmark.load_heightmap, which validates before allocating.'''
    try:
        benchmark = importlib.import_module('horizon.benchmark')
    except ModuleNotFoundError:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        benchmark = importlib.import_module('horizon.benchmark')
    return benchmark.load_heightmap(Path(path))


def verify_cached(name, out, side=SIDE):
    '''Re-derive everything from the cached .hgt.gz and compare with .json/.npy.'''
    paths = tile_paths(name, out)
    try:
        meta = json.loads(read_bounded(paths['json'], MAX_JSON_BYTES).decode('utf-8'))
        gz = read_bounded(paths['gzip'], MAX_GZIP_BYTES)
        npy = read_bounded(paths['npy'], side * side * 2 + NPY_HEADER_ALLOWANCE)
    except (OSError, ValueError) as exc:
        raise TerrainError(f'{name}: cannot read cached files: {exc}') from exc
    if not isinstance(meta, dict):
        raise TerrainError(f'{name}: metadata is not a JSON object')
    try:
        raw = gunzip_hgt(gz, side)
        u = normalize_hgt(raw, side)
        cached = load_cached_npy(paths['npy'])
    except (TerrainError, OSError, ValueError, EOFError) as exc:
        raise TerrainError(f'{name}: cached data is invalid: {exc}') from exc
    problems = []
    expected = build_metadata(name, side, {}, gz, raw, u, npy, '')
    for key, value in expected.items():
        if key in VOLATILE_KEYS:
            if key not in meta:
                problems.append(key)
        elif meta.get(key) != value:
            problems.append(key)
    if cached.dtype != np.dtype('<u2') or not np.array_equal(cached, u):
        problems.append('npy array contents')
    if problems:
        joined = ', '.join(problems)
        raise TerrainError(
            f'{name}: existing outputs in {Path(out)} are corrupt or inconsistent ({joined}); '
            'nothing was modified, inspect or move them manually'
        )
    return meta


def write_exclusive(path, data, created):
    with open(path, 'xb') as fh:
        created.append(path)
        fh.write(data)


def remove_files(paths):
    for path in reversed(list(paths)):
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass


def prepare_tile(name, output_dir=DEFAULT_OUTPUT_DIR, fetch=None, now=None, side=SIDE, created=None):
    '''Return (metadata, reused). Never overwrites; cleans only files it created.'''
    if name not in TILE_URLS:
        raise ValueError('unsupported tile: ' + repr(name))
    out = Path(output_dir)
    paths = tile_paths(name, out)
    present = [k for k, p in paths.items() if p.exists() or p.is_symlink()]
    if len(present) == len(paths):
        return verify_cached(name, out, side), True
    if present:
        missing = [k for k in paths if k not in present]
        raise TerrainError(
            f'{name}: partial existing outputs in {out} (present: {present}, missing: {missing}); '
            'refusing to overwrite or clean up, resolve manually'
        )
    out.mkdir(parents=True, exist_ok=True)
    fetch = fetch or download_limited
    gz, headers = fetch(TILE_URLS[name])
    fetched = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    if len(gz) > MAX_GZIP_BYTES:
        raise TerrainError(f'{name}: compressed download exceeds {MAX_GZIP_BYTES} bytes')
    raw = gunzip_hgt(gz, side)
    u = normalize_hgt(raw, side)
    npy = npy_bytes(u)
    meta = build_metadata(name, side, headers, gz, raw, u, npy, fetched.isoformat(timespec='seconds'))
    meta_bytes = (json.dumps(meta, indent=2, sort_keys=True) + '\n').encode('utf-8')
    mine = []
    try:
        write_exclusive(paths['gzip'], gz, mine)
        write_exclusive(paths['npy'], npy, mine)
        write_exclusive(paths['json'], meta_bytes, mine)
    except BaseException:
        remove_files(mine)
        raise
    if created is not None:
        created.extend(mine)
    return meta, False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Prepare public Skadi terrain tiles as uint16 .npy heightmaps.')
    parser.add_argument('--tile', action='append', choices=sorted(TILE_URLS),
                        help='tile to prepare (repeatable); default: all')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    names = list(dict.fromkeys(args.tile or sorted(TILE_URLS)))
    created = []
    try:
        for name in names:
            meta, reused = prepare_tile(name, args.output_dir, created=created)
            status = 'reused (verified)' if reused else 'downloaded'
            print(f'{name}: {status}, shape={meta["shape"]}, nodata={meta["nodata_count"]}, '
                  f'valid_min_m={meta["valid_min_m"]}, valid_max_m={meta["valid_max_m"]}')
    except BaseException as exc:
        remove_files(created)
        if isinstance(exc, (TerrainError, OSError, ValueError, http.client.HTTPException)):
            print('error: ' + str(exc), file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())

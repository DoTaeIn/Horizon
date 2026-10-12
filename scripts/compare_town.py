#!/usr/bin/env python3
'''Warm in-memory benchmark: whole-map vs chunked traversal of a synthetic town map.

Generates one deterministic heightmap via horizon.town_map.make_town_map and
times horizon.chunk_benchmark.run_chunk_benchmark across chunk sizes on that
same in-memory array. This measures comparative chunking overhead only: no
disk I/O, no engine, no real game assets. See docs/town-benchmark-method.md
for scope, reproducibility notes, and caveats.
'''
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from horizon.town_map import MAX_SIZE, MIN_SIZE  # noqa: E402

DEFAULT_SIZE = 1024
DEFAULT_SEED = 20261012
DEFAULT_REPEATS = 30
MIN_REPEATS = 5
DEFAULT_CHUNK_SIZES = (64, 128, 256, 512)
SEED_EXCLUSIVE_MAX = 2 ** 64  # matches horizon.town_map's own seed contract

PREVIEW_MAX_EDGE = 128  # sampled preview grid edge, independent of --size
PREVIEW_CELL_PX = 3
PREVIEW_LEGEND_H = 24
PREVIEW_LEGEND_STEPS = 256
# Low-to-high terrain-ish stops: (position in [0, 1], (r, g, b)).
PREVIEW_COLOR_STOPS = (
    (0.00, (28, 46, 90)),
    (0.18, (43, 100, 150)),
    (0.32, (62, 140, 90)),
    (0.55, (140, 168, 76)),
    (0.72, (158, 126, 74)),
    (0.88, (120, 92, 70)),
    (1.00, (245, 245, 245)),
)

_LAYOUT_LEGEND_ROW_H = 14
_RIVER_COLOR = '#3f78c8'
_ROAD_COLOR = '#d9c38a'
_PAD_COLOR = '#f2f2f2'
_BRIDGE_COLOR = '#ffd54a'
_FORD_COLOR = '#58d6c9'
_LAYOUT_DISCLAIMER = 'Schematic metadata; no game assets.'
_LAYOUT_LEGEND_ENTRIES = (
    (lambda y: f'<line x1="2" y1="{y}" x2="14" y2="{y}" stroke="{_RIVER_COLOR}" '
               f'stroke-width="2.5"/>', 'River centerline'),
    (lambda y: f'<line x1="2" y1="{y}" x2="14" y2="{y}" stroke="{_ROAD_COLOR}" '
               f'stroke-width="1.5"/>', 'Routes (labeled)'),
    (lambda y: f'<ellipse cx="8" cy="{y}" rx="6" ry="4" fill="{_PAD_COLOR}" '
               f'fill-opacity="0.18" stroke="{_PAD_COLOR}"/>', 'Building pads / plaza'),
    (lambda y: f'<rect x="4" y="{y - 4}" width="8" height="8" fill="{_BRIDGE_COLOR}"/>',
     'Bridge crossing'),
    (lambda y: f'<polygon points="8,{y - 5} 13,{y} 8,{y + 5} 3,{y}" fill="{_FORD_COLOR}"/>',
     'Ford crossing'),
)

LIBRARIES = ('numpy',)


def _escape(text):
    return _xml_escape(str(text), {'"': '&quot;', "'": '&apos;'})


class ValidationError(RuntimeError):
    pass


def _positive_int(name, text, minimum, maximum=None):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{name} must be an integer') from None
    if value < minimum:
        raise argparse.ArgumentTypeError(f'{name} must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise argparse.ArgumentTypeError(f'{name} must be <= {maximum}')
    return value


def _size_type(text):
    return _positive_int('size', text, MIN_SIZE, MAX_SIZE)


def _seed_type(text):
    return _positive_int('seed', text, 0, SEED_EXCLUSIVE_MAX - 1)


def _repeats_type(text):
    return _positive_int('repeats', text, MIN_REPEATS)


def _chunk_sizes_type(text):
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('chunk-sizes must list at least one positive integer')
    sizes = []
    seen = set()
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f'chunk-sizes: {part!r} is not an integer') from None
        if value <= 0:
            raise argparse.ArgumentTypeError(f'chunk-sizes: {value} must be positive')
        if value in seen:
            raise argparse.ArgumentTypeError(f'chunk-sizes: duplicate value {value}')
        seen.add(value)
        sizes.append(value)
    return tuple(sizes)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Warm in-memory comparison of whole-map vs chunked processing on a '
                    'synthetic town map.')
    parser.add_argument('--size', type=_size_type, default=DEFAULT_SIZE)
    parser.add_argument('--seed', type=_seed_type, default=DEFAULT_SEED)
    parser.add_argument('--repeats', type=_repeats_type, default=DEFAULT_REPEATS)
    parser.add_argument('--chunk-sizes', type=_chunk_sizes_type, default=DEFAULT_CHUNK_SIZES)
    parser.add_argument('--output', required=True, help='new JSON report path; never overwritten')
    parser.add_argument('--preview-dir', default=None,
                         help='optional dir for an SVG preview + .npy dump; files inside are '
                              'never overwritten')
    return parser.parse_args(argv)


def _import_horizon(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module(name)


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
        'cpu': platform.processor() or None,
        'machine': platform.machine(),
        'logical_cpus': os.cpu_count(),
        'os': platform.platform(),
        'python': sys.version,
        'python_implementation': platform.python_implementation(),
        'libraries': library_versions(),
    }


def _colormap(t):
    '''Linearly interpolate an RGB tuple for t in [0, 1] across PREVIEW_COLOR_STOPS.'''
    t = min(1.0, max(0.0, t))
    stops = PREVIEW_COLOR_STOPS
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            span = t1 - t0
            frac = (t - t0) / span if span > 0 else 0.0
            return tuple(round(c0[i] + (c1[i] - c0[i]) * frac) for i in range(3))
    return stops[-1][1]


def sample_heightmap(arr, max_edge=PREVIEW_MAX_EDGE):
    '''Deterministic strided downsample to at most max_edge cells per side.'''
    h, w = arr.shape
    row_step = -(-h // max_edge)  # ceil division
    col_step = -(-w // max_edge)
    return arr[::row_step, ::col_step]


def render_svg_preview(arr, title):
    '''Render a sampled, colored SVG heightmap with a legend. Pure stdlib/numpy, no raster deps.'''
    sample = sample_heightmap(arr)
    rows, cols = sample.shape
    lo = int(arr.min())
    hi = int(arr.max())
    span = hi - lo if hi > lo else 1

    cell = PREVIEW_CELL_PX
    width = cols * cell
    height = rows * cell
    title_h = 18
    legend_gap = 10
    label_h = 16
    total_h = title_h + height + legend_gap + PREVIEW_LEGEND_H + label_h

    norm = (sample.astype(np.float64) - lo) / span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" '
        f'viewBox="0 0 {width} {total_h}" font-family="monospace" font-size="11">',
        f'<rect x="0" y="0" width="{width}" height="{total_h}" fill="#101010"/>',
        f'<text x="2" y="{title_h - 5}" fill="#eeeeee">{_escape(title)}</text>',
    ]

    for r in range(rows):
        row_norm = norm[r]
        for c in range(cols):
            color = _colormap(float(row_norm[c]))
            parts.append(
                f'<rect x="{c * cell}" y="{title_h + r * cell}" width="{cell}" height="{cell}" '
                f'fill="rgb({color[0]},{color[1]},{color[2]})"/>'
            )

    legend_y = title_h + height + legend_gap
    step_w = width / PREVIEW_LEGEND_STEPS
    for i in range(PREVIEW_LEGEND_STEPS):
        color = _colormap(i / (PREVIEW_LEGEND_STEPS - 1))
        parts.append(
            f'<rect x="{i * step_w:.2f}" y="{legend_y}" width="{step_w:.2f}" '
            f'height="{PREVIEW_LEGEND_H}" fill="rgb({color[0]},{color[1]},{color[2]})"/>'
        )
    parts.append(
        f'<text x="2" y="{legend_y + PREVIEW_LEGEND_H + label_h - 4}" fill="#eeeeee">min {lo}</text>'
    )
    parts.append(
        f'<text x="{width - 2}" y="{legend_y + PREVIEW_LEGEND_H + label_h - 4}" '
        f'fill="#eeeeee" text-anchor="end">max {hi}</text>'
    )
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def render_layout_svg(arr, plan, title):
    '''Semantic overlay (river, routes, pads/plaza, bridge/ford markers) on the
    same sampled terrain raster as render_svg_preview. plan is the dict returned
    by horizon.town_map.town_plan(size, seed); all labels are schematic feature
    metadata from that dict, not game assets or production geometry.'''
    sample = sample_heightmap(arr)
    rows, cols = sample.shape
    lo = int(arr.min())
    hi = int(arr.max())
    span = hi - lo if hi > lo else 1
    cell = PREVIEW_CELL_PX
    width = cols * cell
    height = rows * cell
    norm = (sample.astype(np.float64) - lo) / span

    title_h = 18
    legend_gap = 10
    label_h = 16
    semantic_rows = len(_LAYOUT_LEGEND_ENTRIES) + 1  # + one disclaimer line
    semantic_legend_h = semantic_rows * _LAYOUT_LEGEND_ROW_H
    total_h = (title_h + height + legend_gap + PREVIEW_LEGEND_H + label_h
               + legend_gap + semantic_legend_h)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" '
        f'viewBox="0 0 {width} {total_h}" font-family="monospace" font-size="11">',
        f'<rect x="0" y="0" width="{width}" height="{total_h}" fill="#101010"/>',
        f'<text x="2" y="{title_h - 5}" fill="#eeeeee">{_escape(title)}</text>',
    ]

    for r in range(rows):
        row_norm = norm[r]
        for c in range(cols):
            color = _colormap(float(row_norm[c]))
            parts.append(
                f'<rect x="{c * cell}" y="{title_h + r * cell}" width="{cell}" height="{cell}" '
                f'fill="rgb({color[0]},{color[1]},{color[2]})"/>'
            )

    def px(u, v):
        return u * width, title_h + v * height

    river = plan['river']
    river_pts = ' '.join(
        f'{x:.1f},{y:.1f}' for x, y in (px(u, v) for u, v in zip(river['x'], river['t']))
    )
    parts.append(
        f'<polyline points="{river_pts}" fill="none" stroke="{_RIVER_COLOR}" '
        f'stroke-width="2.5" stroke-opacity="0.9"/>'
    )

    for route in plan['routes']:
        pts = route['points']
        poly = ' '.join(f'{x:.1f},{y:.1f}' for x, y in (px(u, v) for u, v in pts))
        parts.append(
            f'<polyline points="{poly}" fill="none" stroke="{_ROAD_COLOR}" '
            f'stroke-width="1.5" stroke-opacity="0.85"/>'
        )
        mid_u, mid_v = pts[len(pts) // 2]
        lx, ly = px(mid_u, mid_v)
        label = _escape(route['name'].replace('_', ' ').title())
        parts.append(f'<text x="{lx:.1f}" y="{ly - 4:.1f}" fill="{_ROAD_COLOR}">{label}</text>')

    for flat in plan['flats']:
        cu, cv = flat['center']
        ru, rv = flat['radii']
        cx, cy = px(cu, cv)
        parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{ru * width:.1f}" ry="{rv * height:.1f}" '
            f'fill="{_PAD_COLOR}" fill-opacity="0.18" stroke="{_PAD_COLOR}" stroke-width="1.5"/>'
        )
        label = _escape(flat['name'].replace('_', ' ').title())
        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" fill="{_PAD_COLOR}" text-anchor="middle">{label}</text>'
        )

    for crossing in plan['crossings']:
        cx, cy = px(crossing['u'], crossing['v'])
        if crossing['kind'] == 'bridge':
            parts.append(
                f'<rect x="{cx - 4:.1f}" y="{cy - 4:.1f}" width="8" height="8" '
                f'fill="{_BRIDGE_COLOR}" stroke="#000000" stroke-width="0.5"/>'
            )
        else:
            parts.append(
                f'<polygon points="{cx:.1f},{cy - 5:.1f} {cx + 5:.1f},{cy:.1f} '
                f'{cx:.1f},{cy + 5:.1f} {cx - 5:.1f},{cy:.1f}" '
                f'fill="{_FORD_COLOR}" stroke="#000000" stroke-width="0.5"/>'
            )

    legend_y = title_h + height + legend_gap
    step_w = width / PREVIEW_LEGEND_STEPS
    for i in range(PREVIEW_LEGEND_STEPS):
        color = _colormap(i / (PREVIEW_LEGEND_STEPS - 1))
        parts.append(
            f'<rect x="{i * step_w:.2f}" y="{legend_y}" width="{step_w:.2f}" '
            f'height="{PREVIEW_LEGEND_H}" fill="rgb({color[0]},{color[1]},{color[2]})"/>'
        )
    parts.append(
        f'<text x="2" y="{legend_y + PREVIEW_LEGEND_H + label_h - 4}" fill="#eeeeee">min {lo}</text>'
    )
    parts.append(
        f'<text x="{width - 2}" y="{legend_y + PREVIEW_LEGEND_H + label_h - 4}" '
        f'fill="#eeeeee" text-anchor="end">max {hi}</text>'
    )

    sem_y = legend_y + PREVIEW_LEGEND_H + label_h + legend_gap
    for i, (swatch, label) in enumerate(_LAYOUT_LEGEND_ENTRIES):
        row_y = sem_y + i * _LAYOUT_LEGEND_ROW_H
        parts.append(swatch(row_y))
        parts.append(f'<text x="18" y="{row_y + 9:.1f}" fill="#eeeeee">{_escape(label)}</text>')
    disclaimer_y = sem_y + len(_LAYOUT_LEGEND_ENTRIES) * _LAYOUT_LEGEND_ROW_H
    parts.append(
        f'<text x="2" y="{disclaimer_y + 9:.1f}" fill="#999999">{_escape(_LAYOUT_DISCLAIMER)}</text>'
    )

    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def write_all_exclusive(files):
    '''Create every (path, bytes) pair exclusively. All-or-nothing: on any failure,
    every file already created in this call is removed before re-raising.'''
    created = []
    try:
        for path, data in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'xb') as fh:
                created.append(path)
                fh.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def check_free(path, label):
    if path.exists() or path.is_symlink():
        raise ValidationError(f'{label} already exists, refusing to overwrite: {path}')


def build_report(args, town_map_module, chunk_benchmark_module):
    from horizon.benchmark import validate_heightmap

    t_gen0 = time.perf_counter()
    arr = town_map_module.make_town_map(size=args.size, seed=args.seed)
    t_gen1 = time.perf_counter()
    validate_heightmap(arr, 'make_town_map output')
    arr = np.ascontiguousarray(arr)

    t_bench0 = time.perf_counter()
    chunk_results = chunk_benchmark_module.run_chunk_benchmark(
        arr, chunk_sizes=args.chunk_sizes, repeats=args.repeats, seed=args.seed)
    t_bench1 = time.perf_counter()

    report = {
        'schema_version': 1,
        'scope': ('warm in-memory comparison of whole-map processing against chunked/partial '
                  'processing of one synthetic town map already resident in memory; not a '
                  'disk, streaming, or real game-asset benchmark'),
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'settings': {
            'size': args.size,
            'seed': args.seed,
            'repeats': args.repeats,
            'chunk_sizes': list(args.chunk_sizes),
        },
        'map': {
            'source': 'horizon.town_map.make_town_map (original synthetic map, '
                      'ORAS town-route inspired layout, not derived from or containing '
                      'actual game assets)',
            'shape': [int(n) for n in arr.shape],
            'dtype': arr.dtype.str,
            'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
            'generate_wall_s': t_gen1 - t_gen0,
        },
        'environment': environment(),
        'chunk_benchmark_wall_s': t_bench1 - t_bench0,
        'results': chunk_results,
    }
    return arr, report


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    svg_path = npy_path = layout_path = None
    if args.preview_dir is not None:
        preview_dir = Path(args.preview_dir)
        svg_path = preview_dir / f'town_{args.size}_{args.seed}.svg'
        npy_path = preview_dir / f'town_{args.size}_{args.seed}.npy'
        layout_path = preview_dir / f'town_{args.size}_{args.seed}_layout.svg'

    try:
        check_free(output, 'output')
        if svg_path is not None:
            check_free(svg_path, 'preview SVG')
            check_free(npy_path, 'preview NPY')

        town_map_module = _import_horizon('horizon.town_map')
        chunk_benchmark_module = _import_horizon('horizon.chunk_benchmark')

        arr, report = build_report(args, town_map_module, chunk_benchmark_module)

        files = [(output, (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8'))]
        if svg_path is not None:
            title = f'town map size={args.size} seed={args.seed} (sampled)'
            files.append((svg_path, render_svg_preview(arr, title).encode('utf-8')))
            files.append((npy_path, npy_bytes(arr)))

            # town_plan() is optional: older/stub horizon.town_map modules may not
            # have it, in which case the layout overlay is skipped gracefully.
            town_plan = getattr(town_map_module, 'town_plan', None)
            if town_plan is not None:
                plan = town_plan(size=args.size, seed=args.seed)
                layout_title = f'town map size={args.size} seed={args.seed} layout (schematic)'
                files.append((layout_path, render_layout_svg(arr, plan, layout_title).encode('utf-8')))

        # Re-check right before writing to narrow the race with any concurrent run.
        for path, _ in files:
            check_free(path, str(path))
        write_all_exclusive(files)
    except (ValidationError, OSError, ImportError, ValueError, AttributeError, TypeError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 1

    print('wrote ' + str(output), file=sys.stderr)
    if svg_path is not None:
        written_paths = [str(path) for path, _ in files if path != output]
        print('wrote ' + ', '.join(written_paths), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())

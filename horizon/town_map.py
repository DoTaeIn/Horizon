"""Deterministic original uint16 heightmap of a terraced river town.

``make_town_map(size, seed)`` renders a square little-endian ``uint16``
heightmap in the *spirit* of a handheld role-playing-game region: a terraced
town sitting on one bank of a meandering river, with graded routes leaving it
north, west, east and south, a bridge and a stepping-stone ford over the
water, cliffs between terrace levels, stair flights where a route climbs a
cliff, gently rolling plateaus and a fine seeded detail layer.

Nothing here is derived from a published game. No tile data, map coordinates,
art or layout was copied or measured from any title; the inspiration is purely
structural (a town on terraces, routes that leave it, a river that has to be
crossed). Every feature position, meander and plateau comes out of the seed
through the code below, so the generator is also a self-contained benchmark
fixture: it produces terrain-like, strongly compressible maps without needing
a network fetch or a licence.

Height units are centimetres above an arbitrary datum, which keeps the whole
map inside ``uint16`` with room to spare (roughly 5.2 m .. 77 m here, so
nothing is ever clipped). Elevation is built as::

    SEA_LEVEL + terrace_level * TERRACE_STEP     stepped plateaus, sharp cliffs
      + rolling relief                           continuous, +/- ROLL_AMPLITUDE
      -> flattened building pads and the plaza
      -> river channel carved down, strictly descending downstream
      -> routes stamped on top: graded, stepped into stairs, bridges, a ford
      + seeded detail, suppressed on roads, pads and in the channel

The geometry lives in normalized [0, 1) map coordinates and is sampled at the
requested resolution, so maps of different sizes are the same town at
different detail levels (fine features naturally stop resolving on small
maps). ``town_plan()`` returns that geometry without rasterizing, which is
what the tests and any viewer use instead of re-deriving internals.

Determinism: for a given (size, seed) the output is bit-identical within a
build. The fields are float32, so a different numpy build or CPU could in
principle round an individual sample differently; the structure cannot change.
"""

from __future__ import annotations

import numpy as np

_DTYPE = np.dtype("<u2")
_F = np.float32

DEFAULT_SIZE = 1024
DEFAULT_SEED = 20261012
MIN_SIZE = 64
# Peak working set is roughly 40 bytes per pixel; 4096 keeps that near 700 MB.
MAX_SIZE = 4096

SEA_LEVEL = 6000.0        # height of the lowest terrace, in centimetres
TERRACE_STEP = 320.0      # rise from one terrace to the next (3.2 m cliffs)
TERRACE_LEVELS = 6        # terrace indices 0 (south, low) .. 5 (north, high)
TERRACE_WARP = 1.0        # levels of low-frequency wobble in the terrace edges
ROLL_AMPLITUDE = 90.0     # gentle relief inside a terrace
DETAIL_AMPLITUDE = 12.0   # fine seeded grain
# ROLL_AMPLITUDE + DETAIL_AMPLITUDE stays below TERRACE_STEP / 2 so that the
# terrace a pixel belongs to is always recoverable from its height.

RIVER_X0 = 0.34           # mean normalized easting of the centreline
RIVER_HARMONICS = ((0.105, 1.1), (0.045, 2.3), (0.016, 4.1))  # amplitude, cycles
RIVER_SAMPLES = 513       # centreline samples, independent of map size
RIVER_CORE = 0.016        # normalized half width of the flat bed
RIVER_BANK = 0.040        # normalized half width of the carved valley
RIVER_MIN_CORE_PX = 1.2
RIVER_MIN_BANK_PX = 3.0
CHANNEL_DEPTH = 150.0     # bed clearance below the surrounding ground
CHANNEL_DROP = 0.8        # minimum fall per centreline sample (flow direction)

ROAD_STEP = 1.0 / 1024.0  # arclength spacing of route samples, normalized
ROAD_HALF = 0.009         # normalized half width of the carriageway
ROAD_SHOULDER = 0.018     # normalized half width including the shoulder blend
ROAD_MIN_HALF_PX = 1.0
ROAD_MIN_SHOULDER_PX = 2.0
ROAD_SMOOTH_WINDOW = 25   # samples per grading pass along a route
MAX_ROAD_GRADE = 3.5      # height units per ROAD_STEP; routes stay walkable
STAIR_RISE = 20.0         # riser height where a route climbs a cliff
FORD_RISE = 8.0           # stepping stones above the water at the ford

FLAT_MARGIN = 1.3         # pads reach full flatness at r=1, blend out by 1.3

_TERRACE_OCTAVES = (3, 6, 12)
_ROLL_OCTAVES = (5, 10, 20, 40)
_DETAIL_OCTAVES = (80, 160, 320)

_RAMP_LOW = -0.4                      # terrace ramp at the south edge
_RAMP_HIGH = TERRACE_LEVELS - 0.6     # terrace ramp at the north edge


# --------------------------------------------------------------------------
# argument checking


def _check_size(size):
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise ValueError("size must be an integer")
    size = int(size)
    if size < MIN_SIZE or size > MAX_SIZE:
        raise ValueError(f"size must be in [{MIN_SIZE}, {MAX_SIZE}], got {size}")
    return size


def _check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if not 0 <= seed < 2 ** 64:
        raise ValueError("seed must be in [0, 2**64)")
    return seed


# --------------------------------------------------------------------------
# noise fields


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _lattices(seed, tag, resolutions):
    """Independent square value-noise lattices in [-1, 1]."""
    rng = np.random.default_rng([seed, tag])
    return tuple(rng.uniform(-1.0, 1.0, size=(n, n)).astype(_F) for n in resolutions)


def _value_noise(lattice, u, v):
    """Periodic value noise at broadcastable normalized coordinates."""
    n = lattice.shape[0]
    gx = np.asarray(u, dtype=_F) * n
    gy = np.asarray(v, dtype=_F) * n
    fx = np.floor(gx)
    fy = np.floor(gy)
    tx = _smoothstep(gx - fx)
    ty = _smoothstep(gy - fy)
    i0 = fx.astype(np.intp) % n
    j0 = fy.astype(np.intp) % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    top = lattice[j0, i0]
    top = top + (lattice[j0, i1] - top) * tx
    bot = lattice[j1, i0]
    bot = bot + (lattice[j1, i1] - bot) * tx
    return top + (bot - top) * ty


def _fbm(lattices, u, v):
    """Amplitude-halving sum of the lattices, normalized back to [-1, 1]."""
    total = None
    amp = 1.0
    norm = 0.0
    for lattice in lattices:
        octave = _value_noise(lattice, u, v) * _F(amp)
        total = octave if total is None else total + octave
        norm += amp
        amp *= 0.5
    return total / _F(norm)


def _fields(seed):
    phase = np.random.default_rng([seed, 7]).uniform(
        0.0, 2.0 * np.pi, size=len(RIVER_HARMONICS)
    )
    return {
        "terrace": _lattices(seed, 1, _TERRACE_OCTAVES),
        "roll": _lattices(seed, 2, _ROLL_OCTAVES),
        "detail": _lattices(seed, 3, _DETAIL_OCTAVES),
        "river_phase": phase,
    }


# --------------------------------------------------------------------------
# bare ground: terraces, cliffs, rolling relief


def terrace_height(level):
    """Ground height of a terrace index."""
    return SEA_LEVEL + np.asarray(level, dtype=np.float64) * TERRACE_STEP


def _terrace_level(fields, u, v):
    """Integer terrace index; the floor() is what makes the cliffs."""
    ramp = _RAMP_LOW + (1.0 - np.asarray(v, dtype=_F)) * _F(_RAMP_HIGH - _RAMP_LOW)
    warp = _fbm(fields["terrace"], u, v) * _F(TERRACE_WARP)
    return np.clip(np.floor(ramp + warp), 0.0, _F(TERRACE_LEVELS - 1))


def _bare_surface(fields, u, v):
    level = _terrace_level(fields, u, v)
    roll = _fbm(fields["roll"], u, v) * _F(ROLL_AMPLITUDE)
    return _F(SEA_LEVEL) + level * _F(TERRACE_STEP) + roll


def _flat_weight(flat, u, v):
    cu, cv = flat["center"]
    ru, rv = flat["radii"]
    du = (np.asarray(u, dtype=_F) - _F(cu)) / _F(ru)
    dv = (np.asarray(v, dtype=_F) - _F(cv)) / _F(rv)
    r = np.sqrt(du * du + dv * dv)
    return _smoothstep(
        np.clip((_F(FLAT_MARGIN) - r) / _F(FLAT_MARGIN - 1.0), _F(0.0), _F(1.0))
    )


def _flat_mask(flats, u, v):
    mask = None
    for flat in flats:
        w = _flat_weight(flat, u, v)
        mask = w if mask is None else np.maximum(mask, w)
    return mask


def _town_surface(fields, flats, u, v):
    """Bare ground with the town's building pads and plaza flattened in."""
    height = _bare_surface(fields, u, v)
    for flat in flats:
        w = _flat_weight(flat, u, v)
        height = height + (_F(flat["height"]) - height) * w
    return height


# --------------------------------------------------------------------------
# river


def _river_centreline(fields):
    t = np.linspace(0.0, 1.0, RIVER_SAMPLES)
    x = np.full_like(t, RIVER_X0)
    dx = np.zeros_like(t)
    for (amp, cycles), phase in zip(RIVER_HARMONICS, fields["river_phase"]):
        w = 2.0 * np.pi * cycles
        x += amp * np.sin(w * t + phase)
        dx += amp * w * np.cos(w * t + phase)
    return t, x, dx


def _river_bed(fields, flats, t, x):
    """Bed profile: at least CHANNEL_DEPTH down, and strictly descending.

    The running minimum makes the bed monotone in the flow direction (north to
    south) no matter what the ground above it does, so the channel never has an
    uphill stretch and water would always have somewhere to go.
    """
    ground = np.asarray(_town_surface(fields, flats, x, t), dtype=np.float64)
    target = ground - CHANNEL_DEPTH
    ramp = np.arange(t.size, dtype=np.float64) * CHANNEL_DROP
    bed = np.minimum.accumulate(target + ramp) - ramp
    return ground, bed


def _river_weight(river, u, v, size=None):
    """Carve weight (1 on the bed, 0 past the bank) and the bed height."""
    cx = np.interp(v, river["t"], river["x"])
    slope = np.interp(v, river["t"], river["dx"])
    bed = np.interp(v, river["t"], river["bed"])
    # Perpendicular distance from a horizontal offset, since the centreline is
    # a function of v: divide the easting offset by sqrt(1 + (dx/dv)^2).
    perp = 1.0 / np.sqrt(1.0 + slope * slope)
    if size is None:
        core, bank = RIVER_CORE, RIVER_BANK
        offset = np.abs(u - cx)
    else:
        core = max(RIVER_CORE * size, RIVER_MIN_CORE_PX)
        bank = max(RIVER_BANK * size, RIVER_MIN_BANK_PX)
        offset = np.abs(u - cx * size)
    d = offset * perp
    w = _smoothstep(np.clip((bank - d) / (bank - core), 0.0, 1.0))
    return w, bed


def _carve_river(surf, river, size):
    """Cut the channel into the surface; returns the carve weight as well."""
    rows = (np.arange(size, dtype=np.float64) + 0.5) / size
    cols = np.arange(size, dtype=np.float64) + 0.5
    w, bed = _river_weight(river, cols[None, :], rows[:, None], size=size)
    # bed already carries v's trailing axis, so it broadcasts across columns.
    w = w.astype(_F)
    carved = surf - (surf - bed.astype(_F)) * w
    return np.minimum(surf, carved), w


# --------------------------------------------------------------------------
# routes


def _catmull_rom(control, step):
    """Resample control points onto an arclength-uniform Catmull-Rom spline."""
    pts = np.asarray(control, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("a route needs at least two control points")
    ext = np.vstack([2.0 * pts[0] - pts[1], pts, 2.0 * pts[-1] - pts[-2]])
    pieces = []
    for i in range(pts.shape[0] - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        chord = float(np.hypot(*(p2 - p1)))
        m = max(4, int(np.ceil(chord / (0.25 * step))))
        tt = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]
        pieces.append(
            0.5
            * (
                2.0 * p1
                + (p2 - p0) * tt
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tt ** 2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * tt ** 3
            )
        )
    pieces.append(pts[-1][None, :])
    dense = np.vstack(pieces)
    walk = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(dense[:, 0]), np.diff(dense[:, 1])))]
    )
    total = float(walk[-1])
    count = max(2, int(round(total / step)) + 1)
    want = np.linspace(0.0, total, count)
    return np.column_stack(
        [np.interp(want, walk, dense[:, 0]), np.interp(want, walk, dense[:, 1])]
    )


def _smooth(values, window, passes=2):
    """Box smoothing along a profile with clamped ends."""
    out = np.asarray(values, dtype=np.float64)
    half = window // 2
    if half < 1 or out.size < 3:
        return out.copy()
    half = min(half, (out.size - 1) // 2)
    width = 2 * half + 1
    for _ in range(passes):
        padded = np.concatenate(
            [np.full(half, out[0]), out, np.full(half, out[-1])]
        )
        cumulative = np.concatenate([[0.0], np.cumsum(padded)])
        out = (cumulative[width:] - cumulative[:-width]) / width
    return out


def _limit_grade(profile, max_step):
    """Largest profile <= the input whose consecutive steps are <= max_step.

    Two cummin passes give the exact 1D lower envelope of the cones
    ``profile[j] + max_step * |i - j|``, so the route is a cut into the hill
    that never exceeds the grade -- the ramps either side of the ford fall out
    of the same pass.
    """
    ramp = np.arange(profile.size, dtype=np.float64) * max_step
    out = np.minimum.accumulate(profile - ramp) + ramp
    out = np.minimum.accumulate((out + ramp)[::-1])[::-1] - ramp
    return out


def _dilate(mask, radius):
    out = mask.copy()
    for shift in range(1, radius + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def _route_profile(fields, flats, river, points, ford):
    """Walkable height profile along a route, with stairs and an optional ford."""
    u, v = points[:, 0], points[:, 1]
    ground = np.asarray(_town_surface(fields, flats, u, v), dtype=np.float64)
    profile = _smooth(ground, ROAD_SMOOTH_WINDOW, passes=2)

    corridor = np.zeros(u.size, dtype=bool)
    if ford:
        # Dip to stepping-stone height on the water; _limit_grade then cuts the
        # approach ramps down to it.
        w, bed = _river_weight(river, u, v)
        profile = profile + (bed + FORD_RISE - profile) * w
        corridor = w > 0.05

    profile = _limit_grade(profile, MAX_ROAD_GRADE)

    # Where the grade limiter is saturated the route is climbing a cliff, so
    # quantize it into treads and risers: a flight of stairs.
    jump = np.abs(np.diff(profile))
    steep = np.zeros(u.size, dtype=bool)
    steep[:-1] |= jump > 0.6 * MAX_ROAD_GRADE
    steep[1:] |= jump > 0.6 * MAX_ROAD_GRADE
    steep = _dilate(steep, 6) & ~corridor
    stairs = STAIR_RISE * np.rint(profile / STAIR_RISE)
    return np.where(steep, stairs, profile), steep, corridor


def _stamp_routes(routes, size):
    """Rasterize route decks: per pixel, the weight and height of the nearest."""
    best_w = np.zeros((size, size), dtype=_F)
    best_h = np.zeros((size, size), dtype=_F)
    half = max(ROAD_HALF * size, ROAD_MIN_HALF_PX)
    shoulder = max(ROAD_SHOULDER * size, ROAD_MIN_SHOULDER_PX)
    span = _F(shoulder - half)
    for route in routes:
        px = route["points"][:, 0] * size - 0.5
        py = route["points"][:, 1] * size - 0.5
        heights = route["height"]
        for i in range(px.size - 1):
            x0, y0, x1, y1 = px[i], py[i], px[i + 1], py[i + 1]
            dx, dy = x1 - x0, y1 - y0
            length2 = dx * dx + dy * dy
            if length2 <= 0.0:
                continue
            lo_x = max(0, int(np.floor(min(x0, x1) - shoulder)))
            hi_x = min(size - 1, int(np.ceil(max(x0, x1) + shoulder)))
            lo_y = max(0, int(np.floor(min(y0, y1) - shoulder)))
            hi_y = min(size - 1, int(np.ceil(max(y0, y1) + shoulder)))
            if lo_x > hi_x or lo_y > hi_y:
                continue
            gx = np.arange(lo_x, hi_x + 1, dtype=_F)[None, :]
            gy = np.arange(lo_y, hi_y + 1, dtype=_F)[:, None]
            s = np.clip(((gx - x0) * dx + (gy - y0) * dy) / length2, _F(0.0), _F(1.0))
            d = np.hypot(gx - (x0 + s * dx), gy - (y0 + s * dy))
            w = _smoothstep(np.clip((_F(shoulder) - d) / span, _F(0.0), _F(1.0)))
            h = _F(heights[i]) + s * _F(heights[i + 1] - heights[i])
            view_w = best_w[lo_y:hi_y + 1, lo_x:hi_x + 1]
            view_h = best_h[lo_y:hi_y + 1, lo_x:hi_x + 1]
            closer = w > view_w
            view_w[closer] = w[closer]
            view_h[closer] = h[closer]
    return best_w, best_h


# --------------------------------------------------------------------------
# layout


def _crossings(river, points, name, kind):
    """River crossings of a route, found from the sign of easting - centreline.

    The centreline is single-valued in v, so a sign change between consecutive
    route samples is a crossing and nothing else can be one.
    """
    u, v = points[:, 0], points[:, 1]
    side = u - np.interp(v, river["t"], river["x"])
    found = []
    for i in np.nonzero(np.sign(side[:-1]) * np.sign(side[1:]) < 0.0)[0]:
        a, b = side[i], side[i + 1]
        f = a / (a - b)
        found.append(
            {
                "route": name,
                "kind": kind,
                "u": float(u[i] + f * (u[i + 1] - u[i])),
                "v": float(v[i] + f * (v[i + 1] - v[i])),
                "sample": int(i),
            }
        )
    return found


def _north_offset(v):
    """Easting of the north route relative to the river, 0.12 .. 0.20."""
    phase = 2.2 * np.pi * (0.63 - np.asarray(v, dtype=np.float64)) / 0.60
    return 0.16 + 0.04 * np.sin(phase)


def _layout(fields):
    """Control points for the river-relative town, pads and routes."""
    t, x, dx = _river_centreline(fields)

    def east(v):
        """A point `off` east of the river at northing v, kept on the map."""
        return float(np.interp(v, t, x))

    def node(v, off):
        return (float(np.clip(east(v) + off, 0.02, 0.98)), float(v))

    plaza = node(0.63, float(_north_offset(0.63)))
    pu, pv = plaza

    flats = [
        {"name": "plaza", "center": plaza, "radii": (0.075, 0.055)},
        {"name": "market_pad", "center": (max(pu - 0.085, 0.06), 0.685),
         "radii": (0.045, 0.032)},
        {"name": "workshop_pad", "center": (min(pu + 0.075, 0.94), 0.585),
         "radii": (0.050, 0.036)},
        {"name": "hall_pad", "center": (min(pu + 0.020, 0.94), 0.730),
         "radii": (0.055, 0.030)},
        {"name": "lookout_pad",
         "center": (float(np.clip(east(0.17) + float(_north_offset(0.17)) + 0.035,
                                  0.06, 0.94)), 0.17),
         "radii": (0.045, 0.035)},
    ]
    for flat in flats:
        cu, cv = flat["center"]
        level = float(_terrace_level(fields, cu, cv))
        flat["height"] = float(terrace_height(level))
        flat["level"] = int(level)

    north_v = np.linspace(0.63, 0.03, 25)
    north = [node(v, float(_north_offset(v))) for v in north_v]

    west = [
        plaza,
        node(0.625, 0.060),
        node(0.615, -0.060),
        node(0.600, -0.140),
        (0.02, 0.575),
    ]

    ford = [
        node(0.300, float(_north_offset(0.30))),
        node(0.295, 0.070),
        node(0.285, 0.000),
        node(0.275, -0.070),
        node(0.265, -0.130),
    ]

    gate_u = 0.98  # the plaza is always well west of this
    east_route = [
        plaza,
        (float(np.clip(pu + 0.07, 0.02, 0.97)), 0.655),
        (float(0.5 * (pu + gate_u)), 0.680),
        (gate_u, 0.705),
    ]

    south_v = np.linspace(0.63, 0.98, 9)
    south = [node(v, float(_north_offset(0.63)) + 0.03 * np.sin(6.0 * (v - 0.63)))
             for v in south_v]

    routes = [
        ("north_route", north, False),
        ("west_route", west, False),
        ("ford_trail", ford, True),
        ("east_route", east_route, False),
        ("south_lane", south, False),
    ]
    return {"t": t, "x": x, "dx": dx}, flats, routes, plaza


def town_plan(size=DEFAULT_SIZE, seed=DEFAULT_SEED):
    """Geometry and metadata of the town, without rasterizing the heightmap.

    The returned dict holds the river centreline and bed, every route with its
    walkable height profile, the flattened pads, the river crossings and the
    pixel widths used at this ``size``. Everything except those widths is
    independent of ``size``.
    """
    size = _check_size(size)
    seed = _check_seed(seed)
    fields = _fields(seed)
    river, flats, route_specs, plaza = _layout(fields)
    ground, bed = _river_bed(fields, flats, river["t"], river["x"])
    river["bed"] = bed
    river["ground"] = ground

    routes = []
    crossings = []
    for name, control, is_ford in route_specs:
        points = _catmull_rom(control, ROAD_STEP)
        height, stairs, corridor = _route_profile(
            fields, flats, river, points, is_ford
        )
        routes.append(
            {
                "name": name,
                "control": [tuple(p) for p in control],
                "points": points,
                "height": height,
                "stairs": stairs,
                "ford_corridor": corridor,
                "is_ford": is_ford,
            }
        )
        crossings.extend(
            _crossings(river, points, name, "ford" if is_ford else "bridge")
        )

    return {
        "size": size,
        "seed": seed,
        "plaza": plaza,
        "river": river,
        "routes": routes,
        "flats": flats,
        "crossings": crossings,
        "terrace": {
            "base": SEA_LEVEL,
            "step": TERRACE_STEP,
            "levels": TERRACE_LEVELS,
        },
        "widths_px": {
            "river_core": max(RIVER_CORE * size, RIVER_MIN_CORE_PX),
            "river_bank": max(RIVER_BANK * size, RIVER_MIN_BANK_PX),
            "road_half": max(ROAD_HALF * size, ROAD_MIN_HALF_PX),
            "road_shoulder": max(ROAD_SHOULDER * size, ROAD_MIN_SHOULDER_PX),
        },
    }


# --------------------------------------------------------------------------
# rasterization


def make_town_map(size=DEFAULT_SIZE, seed=DEFAULT_SEED):
    """Render the town as a C-contiguous ``(size, size)`` little-endian uint16 map.

    ``size`` must be an integer in [MIN_SIZE, MAX_SIZE]; any value in that
    range works, powers of two are not special. ``seed`` selects the town.
    """
    size = _check_size(size)
    seed = _check_seed(seed)
    fields = _fields(seed)
    plan = town_plan(size=size, seed=seed)
    flats = plan["flats"]

    axis = (np.arange(size, dtype=_F) + _F(0.5)) / _F(size)
    v = axis[:, None]
    u = axis[None, :]

    surf = _town_surface(fields, flats, u, v)
    quiet = _flat_mask(flats, u, v)

    surf, river_w = _carve_river(surf, plan["river"], size)
    np.maximum(quiet, river_w, out=quiet)
    del river_w

    road_w, road_h = _stamp_routes(plan["routes"], size)
    surf += (road_h - surf) * road_w
    np.maximum(quiet, road_w, out=quiet)
    del road_w, road_h

    detail = _fbm(fields["detail"], u, v) * _F(DETAIL_AMPLITUDE)
    surf += detail * (_F(1.0) - quiet)
    del detail, quiet

    np.clip(surf, _F(0.0), _F(65535.0), out=surf)
    return np.rint(surf).astype(_DTYPE)

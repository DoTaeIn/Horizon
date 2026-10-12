"""Tests for the deterministic town-and-routes heightmap generator.

The generator's structure is checked through horizon.town_map.town_plan(), the
published geometry, rather than by re-deriving module internals: the plan says
where the river, routes, crossings and pads are, and the tests assert that the
raster actually honours it.
"""

import unittest

import numpy as np

from horizon import town_map as tm
from horizon.codec import decode, encode
from horizon.town_map import make_town_map, town_plan

SIZE = 1024
TERRACE_TOL = tm.ROLL_AMPLITUDE + tm.DETAIL_AMPLITUDE + 1.0
_CACHE = {}


def cached_map(size=SIZE, seed=tm.DEFAULT_SEED):
    key = (size, seed)
    if key not in _CACHE:
        _CACHE[key] = make_town_map(size, seed)
    return _CACHE[key]


def cached_plan(size=SIZE, seed=tm.DEFAULT_SEED):
    key = ("plan", size, seed)
    if key not in _CACHE:
        _CACHE[key] = town_plan(size, seed)
    return _CACHE[key]


def terrace_levels(heights):
    """Nearest terrace index per pixel and its height error, in height units."""
    residual = heights.astype(np.float64) - tm.SEA_LEVEL
    level = np.rint(residual / tm.TERRACE_STEP)
    error = np.abs(residual - level * tm.TERRACE_STEP)
    return level, error


def river_rows(plan):
    """Per-row channel geometry in pixels, plus a mask of non-crossing rows."""
    size = plan["size"]
    river = plan["river"]
    rows = np.arange(size)
    v = (rows + 0.5) / size
    centre = np.interp(v, river["t"], river["x"]) * size
    bed = np.interp(v, river["t"], river["bed"])
    slope = np.interp(v, river["t"], river["dx"])
    # The carve measures distance perpendicular to the centreline, so a
    # diagonal reach is wider than RIVER_BANK when measured along a row.
    bank = plan["widths_px"]["river_bank"] * np.sqrt(1.0 + slope * slope)
    open_water = np.ones(size, dtype=bool)
    for crossing in plan["crossings"]:
        open_water &= np.abs(v - crossing["v"]) >= 0.06
    return rows, centre, bed, bank, open_water


def pixel_at(size, u, v):
    """Index of the pixel whose centre is nearest a normalized position."""
    col = int(np.clip(round(u * size - 0.5), 0, size - 1))
    row = int(np.clip(round(v * size - 0.5), 0, size - 1))
    return row, col


class TestInterface(unittest.TestCase):
    def test_defaults(self):
        heights = cached_map()
        self.assertEqual(heights.shape, (tm.DEFAULT_SIZE, tm.DEFAULT_SIZE))
        self.assertEqual(heights.dtype.str, "<u2")
        self.assertTrue(heights.flags["C_CONTIGUOUS"])

    def test_arbitrary_sizes(self):
        for size in (tm.MIN_SIZE, 65, 97, 100, 255, 256):
            with self.subTest(size=size):
                heights = make_town_map(size, seed=tm.DEFAULT_SEED)
                self.assertEqual(heights.shape, (size, size))
                self.assertEqual(heights.dtype.str, "<u2")

    def test_no_clipping_against_the_uint16_ends(self):
        for size in (tm.MIN_SIZE, 97, SIZE):
            with self.subTest(size=size):
                heights = cached_map(size)
                self.assertGreater(int(heights.min()), 0)
                self.assertLess(int(heights.max()), 65535)

    def test_numpy_integer_arguments_are_accepted(self):
        expected = make_town_map(128, 7)
        actual = make_town_map(np.int64(128), np.uint32(7))
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_rejected_sizes(self):
        for size in (63, 0, -1, tm.MAX_SIZE + 1, 128.0, "128", True, None):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    make_town_map(size)

    def test_rejected_seeds(self):
        for seed in (-1, 2 ** 64, 1.5, True, None):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    make_town_map(128, seed)


class TestDeterminism(unittest.TestCase):
    def test_same_arguments_give_identical_bytes(self):
        for size in (tm.MIN_SIZE, 97, 256):
            with self.subTest(size=size):
                first = make_town_map(size, seed=4242)
                second = make_town_map(size, seed=4242)
                self.assertEqual(first.tobytes(), second.tobytes())

    def test_seed_changes_the_whole_map(self):
        first = make_town_map(256, seed=tm.DEFAULT_SEED)
        second = make_town_map(256, seed=tm.DEFAULT_SEED + 1)
        differing = np.mean(first != second)
        self.assertGreater(differing, 0.5)

    def test_seed_moves_the_town_and_the_river(self):
        first = town_plan(256, seed=tm.DEFAULT_SEED)
        second = town_plan(256, seed=tm.DEFAULT_SEED + 1)
        self.assertNotEqual(first["plaza"], second["plaza"])
        self.assertFalse(np.allclose(first["river"]["x"], second["river"]["x"]))

    def test_plan_geometry_does_not_depend_on_size(self):
        small = town_plan(tm.MIN_SIZE)
        large = town_plan(SIZE)
        self.assertEqual(small["plaza"], large["plaza"])
        np.testing.assert_array_equal(small["river"]["bed"], large["river"]["bed"])
        self.assertEqual(
            [(c["route"], c["kind"], c["v"]) for c in small["crossings"]],
            [(c["route"], c["kind"], c["v"]) for c in large["crossings"]],
        )
        for a, b in zip(small["routes"], large["routes"]):
            np.testing.assert_array_equal(a["height"], b["height"])


class TestTerracesAndCliffs(unittest.TestCase):
    def test_most_ground_sits_on_a_terrace(self):
        level, error = terrace_levels(cached_map())
        on_terrace = (error <= TERRACE_TOL) & (level >= 0) & (level < tm.TERRACE_LEVELS)
        # The remainder is the river channel and the graded routes cut into it.
        self.assertGreater(np.mean(on_terrace), 0.85)

    def test_every_terrace_level_is_used(self):
        level, error = terrace_levels(cached_map())
        on_terrace = (error <= TERRACE_TOL) & (level >= 0) & (level < tm.TERRACE_LEVELS)
        for index in range(tm.TERRACE_LEVELS):
            with self.subTest(level=index):
                self.assertGreater(np.mean(level[on_terrace] == index), 0.005)

    def test_plateaus_roll_instead_of_being_flat(self):
        heights = cached_map()
        level, error = terrace_levels(heights)
        on_terrace = (error <= TERRACE_TOL) & (level >= 0) & (level < tm.TERRACE_LEVELS)
        spread = float(np.std(error[on_terrace]))
        self.assertGreater(spread, 8.0)
        self.assertLess(spread, TERRACE_TOL)

    def test_terrace_edges_are_cliffs(self):
        heights = cached_map().astype(np.int64)
        steps = np.abs(np.diff(heights, axis=1))
        cliff = steps >= 0.8 * tm.TERRACE_STEP
        self.assertGreaterEqual(int(steps.max()), int(0.9 * tm.TERRACE_STEP))
        # Cliffs are a thin boundary set: present, but not most of the map.
        self.assertGreater(np.mean(cliff), 1e-4)
        self.assertLess(np.mean(cliff), 0.05)

    def test_detail_layer_perturbs_neighbouring_pixels(self):
        heights = cached_map().astype(np.int64)
        changing = np.mean(np.diff(heights, axis=1) != 0)
        self.assertGreater(changing, 0.4)


class TestRiver(unittest.TestCase):
    def test_bed_descends_strictly_downstream(self):
        bed = cached_plan()["river"]["bed"]
        self.assertTrue(np.all(np.diff(bed) < 0.0))

    def test_bed_clears_the_ground_above_it(self):
        river = cached_plan()["river"]
        clearance = river["ground"] - river["bed"]
        self.assertGreaterEqual(float(clearance.min()), tm.CHANNEL_DEPTH - 1e-6)

    def test_channel_follows_the_planned_bed(self):
        plan = cached_plan()
        heights = cached_map().astype(np.float64)
        rows, centre, bed, _, open_water = river_rows(plan)
        columns = np.clip(np.rint(centre - 0.5).astype(int), 0, plan["size"] - 1)
        deviation = np.abs(heights[rows, columns] - bed)[open_water]
        self.assertLess(float(deviation.max()), 0.4 * tm.CHANNEL_DEPTH)

    def test_channel_is_a_gorge_between_its_banks(self):
        plan = cached_plan()
        size = plan["size"]
        heights = cached_map().astype(np.int64)
        rows, centre, _, bank, open_water = river_rows(plan)
        centre_col = np.clip(np.rint(centre - 0.5).astype(int), 0, size - 1)
        left = np.clip(np.rint(centre - 0.5 - 1.6 * bank).astype(int), 0, size - 1)
        right = np.clip(np.rint(centre - 0.5 + 1.6 * bank).astype(int), 0, size - 1)
        # One bank can be the foot of a cliff, so compare against the higher.
        relief = np.maximum(heights[rows, left], heights[rows, right])
        drop = (relief - heights[rows, centre_col])[open_water]
        self.assertGreater(int(drop.min()), int(0.6 * tm.CHANNEL_DEPTH))

    def test_channel_spans_the_map_north_to_south(self):
        plan = cached_plan()
        _, _, _, _, open_water = river_rows(plan)
        self.assertTrue(open_water[0])
        self.assertTrue(open_water[-1])
        self.assertGreater(np.mean(open_water), 0.5)


class TestRoutes(unittest.TestCase):
    def test_routes_are_present_and_on_the_map(self):
        routes = cached_plan()["routes"]
        self.assertEqual(
            [route["name"] for route in routes],
            ["north_route", "west_route", "ford_trail", "east_route", "south_lane"],
        )
        for route in routes:
            with self.subTest(route=route["name"]):
                points = route["points"]
                self.assertGreater(points.shape[0], 2)
                self.assertEqual(points.shape[0], route["height"].size)
                self.assertGreaterEqual(float(points.min()), 0.0)
                self.assertLessEqual(float(points.max()), 1.0)

    def test_routes_stay_walkable_where_the_ground_does_not(self):
        plan = cached_plan()
        limit = tm.MAX_ROAD_GRADE + tm.STAIR_RISE
        for route in plan["routes"]:
            with self.subTest(route=route["name"]):
                step = np.abs(np.diff(route["height"]))
                self.assertLessEqual(float(step.max()), limit + 1e-9)
        # Meanwhile the surrounding ground still has full-height cliffs.
        heights = cached_map().astype(np.int64)
        self.assertGreater(int(np.abs(np.diff(heights, axis=0)).max()), int(limit))

    def test_raster_carries_the_planned_route_profile(self):
        plan = cached_plan()
        size = plan["size"]
        heights = cached_map().astype(np.float64)
        for route in plan["routes"]:
            with self.subTest(route=route["name"]):
                points = route["points"]
                columns = np.clip(np.rint(points[:, 0] * size - 0.5).astype(int), 0, size - 1)
                rows = np.clip(np.rint(points[:, 1] * size - 0.5).astype(int), 0, size - 1)
                deviation = np.abs(heights[rows, columns] - route["height"])
                # A sample's nearest pixel can belong to the neighbouring tread.
                self.assertLess(float(deviation.max()), 3.0 * tm.STAIR_RISE)
                self.assertLess(float(deviation.mean()), tm.STAIR_RISE)

    def test_climbing_routes_become_stair_flights(self):
        plan = cached_plan()
        total_stairs = sum(int(route["stairs"].sum()) for route in plan["routes"])
        self.assertGreater(total_stairs, 100)
        north = next(r for r in plan["routes"] if r["name"] == "north_route")
        self.assertGreater(north["height"].max() - north["height"].min(), 3 * tm.TERRACE_STEP)
        # Treads: runs of equal height, separated by exact STAIR_RISE risers.
        self.assertGreater(int(np.sum(np.diff(north["height"]) == 0.0)), 20)
        risers = north["height"][north["stairs"]] / tm.STAIR_RISE
        np.testing.assert_allclose(risers, np.rint(risers), atol=1e-9)

    def test_routes_are_smoother_than_the_ground_they_cross(self):
        plan = cached_plan()
        north = next(r for r in plan["routes"] if r["name"] == "north_route")
        points = north["points"]
        ground = tm._town_surface(
            tm._fields(plan["seed"]), plan["flats"], points[:, 0], points[:, 1]
        )
        along_route = float(np.abs(np.diff(north["height"])).max())
        along_ground = float(np.abs(np.diff(np.asarray(ground, dtype=np.float64))).max())
        self.assertLess(along_route, along_ground)


class TestCrossings(unittest.TestCase):
    def test_a_bridge_and_a_ford_cross_the_river(self):
        crossings = cached_plan()["crossings"]
        self.assertGreaterEqual(len(crossings), 2)
        kinds = {crossing["kind"] for crossing in crossings}
        self.assertIn("bridge", kinds)
        self.assertIn("ford", kinds)

    def test_bridge_decks_span_the_channel_at_bank_height(self):
        plan = cached_plan()
        heights = cached_map().astype(np.float64)
        river = plan["river"]
        bridges = [c for c in plan["crossings"] if c["kind"] == "bridge"]
        self.assertTrue(bridges)
        for crossing in bridges:
            with self.subTest(route=crossing["route"]):
                row, col = pixel_at(plan["size"], crossing["u"], crossing["v"])
                deck = heights[row, col]
                bed = float(np.interp(crossing["v"], river["t"], river["bed"]))
                ground = float(np.interp(crossing["v"], river["t"], river["ground"]))
                self.assertGreater(deck, bed + 0.5 * tm.CHANNEL_DEPTH)
                self.assertLess(abs(deck - ground), 0.4 * tm.CHANNEL_DEPTH)

    def test_the_ford_dips_to_the_water_without_going_under_it(self):
        plan = cached_plan()
        heights = cached_map().astype(np.float64)
        river = plan["river"]
        fords = [c for c in plan["crossings"] if c["kind"] == "ford"]
        self.assertTrue(fords)
        for crossing in fords:
            with self.subTest(route=crossing["route"]):
                row, col = pixel_at(plan["size"], crossing["u"], crossing["v"])
                stones = heights[row, col]
                bed = float(np.interp(crossing["v"], river["t"], river["bed"]))
                ground = float(np.interp(crossing["v"], river["t"], river["ground"]))
                self.assertGreaterEqual(stones, bed - 1.0)
                self.assertLessEqual(stones, bed + 4.0 * tm.FORD_RISE)
                self.assertLess(stones, ground - 0.4 * tm.CHANNEL_DEPTH)


class TestFlats(unittest.TestCase):
    def test_pads_are_flat_and_sit_on_a_terrace(self):
        plan = cached_plan()
        size = plan["size"]
        heights = cached_map().astype(np.float64)
        self.assertGreaterEqual(len(plan["flats"]), 3)
        for flat in plan["flats"]:
            with self.subTest(pad=flat["name"]):
                offset = (flat["height"] - tm.SEA_LEVEL) / tm.TERRACE_STEP
                self.assertAlmostEqual(offset, round(offset), places=9)
                cu, cv = flat["center"]
                ru, rv = flat["radii"]
                top, left = pixel_at(size, cu - 0.6 * ru, cv - 0.6 * rv)
                bottom, right = pixel_at(size, cu + 0.6 * ru, cv + 0.6 * rv)
                patch = heights[top:bottom + 1, left:right + 1]
                self.assertGreater(patch.size, 4)
                # Most of the pad is exactly the pad height; routes cut the rest.
                self.assertGreater(np.mean(patch == flat["height"]), 0.6)


class TestResolutionCoherence(unittest.TestCase):
    def test_coarse_shape_is_the_same_town_at_every_size(self):
        small = make_town_map(256).astype(np.float64)
        large = cached_map(1024).astype(np.float64)
        blocks = 16
        small_means = small.reshape(blocks, 256 // blocks, blocks, 256 // blocks).mean(
            axis=(1, 3)
        )
        large_means = large.reshape(blocks, 1024 // blocks, blocks, 1024 // blocks).mean(
            axis=(1, 3)
        )
        self.assertLess(float(np.abs(small_means - large_means).max()), tm.TERRACE_STEP / 4)
        correlation = np.corrcoef(small_means.ravel(), large_means.ravel())[0, 1]
        self.assertGreater(correlation, 0.99)


class TestCodecFixture(unittest.TestCase):
    def test_round_trips_through_the_container_and_compresses(self):
        heights = cached_map(256)
        blob = encode(heights)
        np.testing.assert_array_equal(decode(blob), heights)
        ratio = len(blob) / heights.nbytes
        # Terrain-like, so far better than random but not degenerate.
        self.assertLess(ratio, 0.4)
        self.assertGreater(ratio, 0.001)


if __name__ == "__main__":
    unittest.main()

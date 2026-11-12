import pickle
from pathlib import Path
from dataclasses import replace
from functools import reduce

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString

from transitforge.dataclasses import TransitRoute, TransitStop
from transitforge.network import TransitRouteNetwork
from transitforge.utils import _normalize_ref


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

# Two consecutive "lines" along the same parallel. They meet around
# lon 7.10 (A ends at 7.1000, B starts at 7.1005, ~38 m apart), so no
# single route covers A_COORDS + B_COORDS but a chain of two does.
A_COORDS = [(46.0, 7.0), (46.0, 7.05), (46.0, 7.1)]
B_COORDS = [(46.0, 7.1005), (46.0, 7.15), (46.0, 7.2)]

BUS_LEG = {"route": "12", "routeType": "BUS"}


def make_route(relation_id, coords, route_type="bus", ref="12", **kwargs):
    """Route whose stops and geometry both follow `coords` ((lat, lon))."""
    stops = [
        TransitStop(relation_id * 100 + i, lat, lon, f"S{i}")
        for i, (lat, lon) in enumerate(coords)
    ]
    geometry = (
        LineString([(lon, lat) for lat, lon in coords])
        if len(coords) >= 2
        else None
    )

    return TransitRoute(
        relation_id=relation_id,
        route_type=route_type,
        ref=ref,
        stops=stops,
        geometry=geometry,
        **kwargs,
    )


def make_leg(start, end, route="12", route_type="BUS"):
    return {
        "route": route,
        "routeType": route_type,
        "from": {"lat": start[0], "lon": start[1]},
        "to": {"lat": end[0], "lon": end[1]},
    }


def prepared(coords):
    g = np.radians(np.asarray(coords, dtype=float))
    return g, np.cos(g[:, 0])


def assert_coords(actual, expected, atol=1e-9):
    np.testing.assert_allclose(actual, expected, atol=atol)


# ----------------------------------------------------------------------
# Indexes / serialization (existing tests)
# ----------------------------------------------------------------------


def test_network_indexes_routes_by_normalized_ref(route):
    network = TransitRouteNetwork({route.relation_id: route})

    assert route.relation_id in network.routes
    assert network.by_ref["12"] == [route.relation_id]


def test_network_ignores_routes_without_ref(route):
    route.ref = None

    network = TransitRouteNetwork({route.relation_id: route})

    assert network.by_ref == {}


# ----------------------------------------------------------------------
# Indexes / serialization (new)
# ----------------------------------------------------------------------


def test_network_groups_routes_sharing_a_ref():
    r1 = make_route(1, A_COORDS, route_type="bus")
    r2 = make_route(2, B_COORDS, route_type="train")

    network = TransitRouteNetwork({1: r1, 2: r2})

    assert network.by_ref["12"] == [1, 2]


def test_network_builds_bbox_from_stops():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    assert network._bbox[1] == pytest.approx((46.0, 46.0, 7.0, 7.1))


def test_network_has_no_bbox_for_routes_without_stops():
    empty = TransitRoute(relation_id=1, route_type="bus", ref="12")

    network = TransitRouteNetwork({1: empty})

    assert 1 not in network._bbox
    # ... but it is still indexed by ref
    assert network.by_ref["12"] == [1]


def test_network_starts_with_empty_caches():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    assert network._arrays == {}
    assert network._cand_cache == {}
    assert network._type_routes == {}
    assert network._type_bbox == {}


@pytest.mark.parametrize(
    "gtfs_label, osm_type",
    [
        ("TRAM", "tram"),
        ("TRAM", "light_rail"),
        ("CABLE_TRAM", "tram"),
        ("SUBWAY", "subway"),
        ("METRO", "monorail"),
        ("RAIL", "train"),
        ("TRAIN", "train"),
        ("BUS", "bus"),
        ("BUS", "trolleybus"),
        ("TROLLEYBUS", "bus"),
        ("FERRY", "ferry"),
        ("CABLE_CAR", "aerialway"),
        ("GONDOLA", "aerialway"),
        ("FUNICULAR", "funicular"),
    ],
)
def test_gtfs_type_to_osm_mapping(gtfs_label, osm_type):
    network = TransitRouteNetwork({})

    assert osm_type in network.GTFS_TYPE_TO_OSM_ROUTES[gtfs_label]


def test_getstate_drops_derived_indexes(route):
    network = TransitRouteNetwork({route.relation_id: route})

    state = network.__getstate__()

    assert "routes" in state

    for key in (
        "by_ref",
        "_bbox",
        "_arrays",
        "_cand_cache",
        "_type_routes",
        "_type_bbox",
    ):
        assert key not in state


def test_pickle_roundtrip_rebuilds_indexes():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    loaded = pickle.loads(pickle.dumps(network))

    assert loaded.by_ref["12"] == [1]
    assert loaded._bbox[1] == pytest.approx((46.0, 46.0, 7.0, 7.1))
    assert loaded._cand_cache == {}


def test_load_rebuilds_indexes_when_missing(tmp_path, monkeypatch):
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})
    del network.by_ref

    monkeypatch.setattr(pickle, "load", lambda f: network)

    path = tmp_path / "network.pkl"
    path.write_bytes(b"")

    loaded = TransitRouteNetwork.load(path)

    assert loaded.by_ref["12"] == [1]


def test_save_accepts_str_path_and_reports(tmp_path, capsys):
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    path = tmp_path / "sub" / "network.pkl"

    network.save(str(path))

    assert path.exists()
    assert "Saved network to" in capsys.readouterr().out


# ----------------------------------------------------------------------
# Simple helpers (single leg)
# ----------------------------------------------------------------------


def test_nearest_stop(route):
    network = TransitRouteNetwork({route.relation_id: route})

    index, distance = network._nearest_stop(
        route.stops,
        (46.0050, 7.0050),
    )

    assert index == 1
    assert distance == pytest.approx(0.0)


def test_nearest_stop_with_no_stops():
    network = TransitRouteNetwork({})

    index, distance = network._nearest_stop([], (46.0, 7.0))

    assert index == -1
    assert distance == float("inf")


def test_score_route_for_leg_sums_endpoint_distances():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    score = network._score_route_for_leg(
        route,
        (46.0, 7.0),
        (46.0, 7.1),
        750,
    )

    assert score == pytest.approx(0.0, abs=1e-3)


def test_score_route_for_leg_without_stops():
    route = TransitRoute(relation_id=1, route_type="bus", ref="12")
    network = TransitRouteNetwork({1: route})

    assert (
        network._score_route_for_leg(route, (46.0, 7.0), (46.0, 7.1), 750)
        is None
    )


def test_score_route_for_leg_start_too_far():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    assert (
        network._score_route_for_leg(route, (47.0, 8.0), (46.0, 7.1), 750)
        is None
    )


def test_score_route_for_leg_end_too_far():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    assert (
        network._score_route_for_leg(route, (46.0, 7.0), (47.0, 8.0), 750)
        is None
    )


def test_score_route_for_leg_wrong_order():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    assert (
        network._score_route_for_leg(route, (46.0, 7.1), (46.0, 7.0), 750)
        is None
    )


# ----------------------------------------------------------------------
# Stop sequence matching (pure python reference)
# ----------------------------------------------------------------------


def test_match_stop_sequence(route, stop_sequence):
    network = TransitRouteNetwork({route.relation_id: route})

    result = network._match_stop_sequence(
        route,
        stop_sequence,
        max_stop_distance_m=750,
    )

    assert result is not None

    indices, total_distance = result

    assert indices == [0, 1, 2]
    assert total_distance == pytest.approx(0.0)


def test_match_stop_sequence_allows_missing_intermediate_osm_stops():
    stops = [
        TransitStop(1, 46.0000, 7.0000, "A"),
        TransitStop(2, 46.0050, 7.0050, "B"),
        TransitStop(3, 46.0100, 7.0100, "C"),
        TransitStop(4, 46.0150, 7.0150, "D"),
        TransitStop(5, 46.0200, 7.0200, "E"),
    ]

    route = TransitRoute(
        relation_id=1,
        route_type="bus",
        ref="12",
        stops=stops,
        geometry=LineString(
            [
                (7.0000, 46.0000),
                (7.0050, 46.0050),
                (7.0100, 46.0100),
                (7.0150, 46.0150),
                (7.0200, 46.0200),
            ]
        ),
    )

    network = TransitRouteNetwork({1: route})

    result = network._match_stop_sequence(
        route,
        [
            (46.0000, 7.0000),
            (46.0100, 7.0100),
            (46.0200, 7.0200),
        ],
        max_stop_distance_m=750,
    )

    assert result is not None

    indices, distance = result

    assert indices == [0, 2, 4]
    assert distance == pytest.approx(0.0)


def test_match_stop_sequence_rejects_wrong_direction(route):
    network = TransitRouteNetwork({route.relation_id: route})

    result = network._match_stop_sequence(
        route,
        [
            (46.0100, 7.0100),
            (46.0050, 7.0050),
            (46.0000, 7.0000),
        ],
        max_stop_distance_m=750,
    )

    assert result is None


def test_match_stop_sequence_rejects_far_stops(route):
    network = TransitRouteNetwork({route.relation_id: route})

    result = network._match_stop_sequence(
        route,
        [
            (40.0, -70.0),
            (41.0, -71.0),
        ],
        max_stop_distance_m=750,
    )

    assert result is None


def test_match_stop_sequence_route_without_stops():
    route = TransitRoute(relation_id=1, route_type="bus", ref="12")
    network = TransitRouteNetwork({1: route})

    assert (
        network._match_stop_sequence(route, [(46.0, 7.0), (46.1, 7.1)], 750)
        is None
    )


def test_match_stop_sequence_requires_two_coordinates():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    assert network._match_stop_sequence(route, [(46.0, 7.0)], 750) is None
    assert network._match_stop_sequence(route, [], 750) is None


def test_score_route_for_stop_sequence():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    score = network._score_route_for_stop_sequence(route, A_COORDS, 750)

    assert score == pytest.approx(0.0, abs=1e-3)


def test_score_route_for_stop_sequence_no_match():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    assert (
        network._score_route_for_stop_sequence(
            route, [(47.0, 8.0), (47.1, 8.1)], 750
        )
        is None
    )


# ----------------------------------------------------------------------
# Vectorised matching
# ----------------------------------------------------------------------


def test_route_arrays_are_radians_and_cached():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    lat_r, lon_r, cos_r = network._route_arrays(route)

    assert lat_r == pytest.approx(np.radians([46.0, 46.0, 46.0]))
    assert lon_r == pytest.approx(np.radians([7.0, 7.05, 7.1]))
    assert cos_r == pytest.approx(np.cos(lat_r))

    # Second call returns the very same cached tuple.
    assert network._route_arrays(route) is network._arrays[1]
    assert network._route_arrays(route) is network._route_arrays(route)


def test_match_prepared_ok():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})
    g, g_cos = prepared(A_COORDS)

    indices, total = network._match_prepared(route, g, g_cos, 750)

    assert indices == [0, 1, 2]
    assert total == pytest.approx(0.0, abs=1e-3)


def test_match_prepared_ok_with_progress():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})
    g, g_cos = prepared(A_COORDS)

    assert network._match_prepared(
        route, g, g_cos, 750, return_progress=True
    ) == (3, None, "ok")


def test_match_prepared_skips_osm_stops_in_order():
    route = make_route(1, [(46.0, 7.0 + 0.01 * i) for i in range(5)])
    network = TransitRouteNetwork({1: route})
    g, g_cos = prepared([(46.0, 7.0), (46.0, 7.02), (46.0, 7.04)])

    indices, _ = network._match_prepared(route, g, g_cos, 750)

    assert indices == [0, 2, 4]


def test_match_prepared_too_far():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})
    g, g_cos = prepared([(46.0, 7.0), (46.5, 7.5)])

    assert network._match_prepared(route, g, g_cos, 750) is None

    k, d, status = network._match_prepared(
        route, g, g_cos, 750, return_progress=True
    )

    assert (k, status) == (1, "too_far")
    assert d > 750


def test_match_prepared_out_of_osm_stops():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})
    # One more GTFS stop than the OSM route has stops.
    g, g_cos = prepared(A_COORDS + [A_COORDS[-1]])

    assert network._match_prepared(route, g, g_cos, 750) is None

    assert network._match_prepared(
        route, g, g_cos, 750, return_progress=True
    ) == (3, None, "out_of_osm_stops")


# ----------------------------------------------------------------------
# Best route for a full stop sequence
# ----------------------------------------------------------------------


def test_best_stop_sequence_route_matches_forward():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    best_route, indices = network._best_stop_sequence_route(
        [route], A_COORDS, 750
    )

    assert best_route is route
    assert indices == [0, 1, 2]


def test_best_stop_sequence_route_matches_reversed_direction():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    best_route, indices = network._best_stop_sequence_route(
        [route], A_COORDS[::-1], 750
    )

    assert best_route is route
    assert indices == [0, 1, 2]


def test_best_stop_sequence_route_needs_two_stops_and_candidates():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    assert network._best_stop_sequence_route([route], [(46.0, 7.0)], 750) is None
    assert network._best_stop_sequence_route([], A_COORDS, 750) is None


def test_best_stop_sequence_route_rejects_by_bbox():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    # First stop is on the route, last one far outside its bbox.
    assert (
        network._best_stop_sequence_route(
            [route], [(46.0, 7.0), (47.0, 8.0)], 750
        )
        is None
    )
    # Last stop is on the route, first one far outside its bbox.
    assert (
        network._best_stop_sequence_route(
            [route], [(47.0, 8.0), (46.0, 7.1)], 750
        )
        is None
    )


def test_best_stop_sequence_route_skips_routes_without_bbox():
    empty = TransitRoute(relation_id=1, route_type="bus", ref="12")
    network = TransitRouteNetwork({1: empty})

    assert network._best_stop_sequence_route([empty], A_COORDS, 750) is None


def test_best_stop_sequence_route_rejects_when_stops_do_not_fit():
    # Terminals are inside the bbox, but the middle stop is far away.
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    assert (
        network._best_stop_sequence_route(
            [route],
            [(46.0, 7.0), (46.02, 7.05), (46.0, 7.1)],
            750,
        )
        is None
    )


def test_best_stop_sequence_route_prefers_lowest_total_distance():
    exact = make_route(1, A_COORDS)
    shifted = make_route(2, [(lat + 0.001, lon) for lat, lon in A_COORDS])
    network = TransitRouteNetwork({1: exact, 2: shifted})

    best_route, _ = network._best_stop_sequence_route(
        [shifted, exact], A_COORDS, 750
    )

    assert best_route is exact


# ----------------------------------------------------------------------
# Candidate filtering
# ----------------------------------------------------------------------


def test_filter_fallback_candidates_without_ref_returns_all():
    r1 = make_route(1, A_COORDS, ref="1")
    r2 = make_route(2, A_COORDS, ref="2")
    network = TransitRouteNetwork({1: r1, 2: r2})

    assert network._filter_fallback_candidates([r1, r2], "") == [r1, r2]


def test_filter_fallback_candidates_generic_ref_uses_metadata():
    ter = make_route(1, A_COORDS, ref="10", name="TER 10")
    other = make_route(2, A_COORDS, ref="11", name="Bus 11")
    network = TransitRouteNetwork({1: ter, 2: other})

    result = network._filter_fallback_candidates(
        [ter, other], _normalize_ref("TER")
    )

    assert result == [ter]


def test_filter_fallback_candidates_generic_ref_uses_network_operator():
    by_network = make_route(1, A_COORDS, ref="1", network="TGV Lyria")
    by_operator = make_route(2, A_COORDS, ref="2", operator="TGV")
    other = make_route(3, A_COORDS, ref="3", name="Bus 3")
    network = TransitRouteNetwork({1: by_network, 2: by_operator, 3: other})

    result = network._filter_fallback_candidates(
        [by_network, by_operator, other], _normalize_ref("TGV")
    )

    assert result == [by_network, by_operator]


def test_filter_fallback_candidates_generic_without_match_returns_all():
    r1 = make_route(1, A_COORDS, ref="1", name="Bus 1")
    r2 = make_route(2, A_COORDS, ref="2", name="Bus 2")
    network = TransitRouteNetwork({1: r1, 2: r2})

    assert network._filter_fallback_candidates(
        [r1, r2], _normalize_ref("TGV")
    ) == [r1, r2]


def test_filter_fallback_candidates_exact_ref():
    ic5 = make_route(1, A_COORDS, ref="IC5")
    ir35 = make_route(2, A_COORDS, ref="IR35")
    network = TransitRouteNetwork({1: ic5, 2: ir35})

    assert network._filter_fallback_candidates(
        [ic5, ir35], _normalize_ref("IC5")
    ) == [ic5]


def test_filter_fallback_candidates_substring_both_ways():
    short = make_route(1, A_COORDS, ref="IC5")
    long = make_route(2, A_COORDS, ref="IC5A")
    other = make_route(3, A_COORDS, ref="S1")
    network = TransitRouteNetwork({1: short, 2: long, 3: other})

    # candidate ref contains requested ref
    assert network._filter_fallback_candidates(
        [long, other], _normalize_ref("IC5")
    ) == [long]

    # requested ref contains candidate ref
    assert network._filter_fallback_candidates(
        [short, other], _normalize_ref("IC5A")
    ) == [short]


def test_filter_fallback_candidates_uses_nat_ref_and_old_ref():
    by_nat = make_route(1, A_COORDS, ref="x", nat_ref="RE33")
    by_old = make_route(2, A_COORDS, ref="y", old_ref="RE44")
    other = make_route(3, A_COORDS, ref="z")
    network = TransitRouteNetwork({1: by_nat, 2: by_old, 3: other})

    assert network._filter_fallback_candidates(
        [by_nat, by_old, other], _normalize_ref("RE33")
    ) == [by_nat]
    assert network._filter_fallback_candidates(
        [by_nat, by_old, other], _normalize_ref("RE44")
    ) == [by_old]


def test_filter_fallback_candidates_no_match_returns_all():
    r1 = make_route(1, A_COORDS, ref="S1")
    r2 = make_route(2, A_COORDS, ref="S2")
    network = TransitRouteNetwork({1: r1, 2: r2})

    assert network._filter_fallback_candidates(
        [r1, r2], _normalize_ref("RE33")
    ) == [r1, r2]


def test_compute_compatible_candidates_filters_exact_ref_by_type():
    bus = make_route(1, A_COORDS, route_type="bus")
    train = make_route(2, A_COORDS, route_type="train")
    network = TransitRouteNetwork({1: bus, 2: train})

    assert network._compute_compatible_candidates("12", "BUS") == [bus]
    assert network._compute_compatible_candidates("12", "RAIL") == [train]


def test_compute_compatible_candidates_unknown_label_does_not_filter():
    bus = make_route(1, A_COORDS, route_type="bus")
    train = make_route(2, A_COORDS, route_type="train")
    network = TransitRouteNetwork({1: bus, 2: train})

    assert network._compute_compatible_candidates("12", "FOO") == [bus, train]
    assert network._compute_compatible_candidates("12", "") == [bus, train]


def test_compute_compatible_candidates_unknown_label_and_ref_is_empty():
    bus = make_route(1, A_COORDS, route_type="bus")
    network = TransitRouteNetwork({1: bus})

    assert network._compute_compatible_candidates("999", "FOO") == []


def test_compute_compatible_candidates_falls_back_to_type():
    bus = make_route(1, A_COORDS, route_type="bus", ref="12")
    train = make_route(2, A_COORDS, route_type="train", ref="IC5")
    network = TransitRouteNetwork({1: bus, 2: train})

    # Ref 999 exists nowhere -> every compatible-type route is returned.
    assert network._compute_compatible_candidates("999", "BUS") == [bus]

    # Fallback pool is cached per GTFS label.
    assert network._type_routes["BUS"] == [bus]


def test_compute_compatible_candidates_fallback_applies_ref_filter():
    ic5 = make_route(1, A_COORDS, route_type="train", ref="IC5A")
    ir35 = make_route(2, A_COORDS, route_type="train", ref="IR35")
    network = TransitRouteNetwork({1: ic5, 2: ir35})

    # "IC5" is not an exact ref, so the fallback narrows by substring.
    assert network._compute_compatible_candidates(
        _normalize_ref("IC5"), "RAIL"
    ) == [ic5]


def test_compute_compatible_candidates_wrong_type_only_is_empty():
    train = make_route(1, A_COORDS, route_type="train", ref="12")
    network = TransitRouteNetwork({1: train})

    assert network._compute_compatible_candidates("12", "BUS") == []


def test_compatible_candidates_are_cached_per_ref_and_type():
    bus = make_route(1, A_COORDS, route_type="bus")
    network = TransitRouteNetwork({1: bus})

    first = network._compatible_candidates(
        {"route": "12", "routeType": "bus"}
    )
    second = network._compatible_candidates(
        {"route": "12", "routeType": "BUS"}
    )

    assert first == [bus]
    assert second is first
    assert list(network._cand_cache) == [(_normalize_ref("12"), "BUS")]


# (existing tests)


def test_compatible_candidates_filters_by_gtfs_type():
    bus = TransitRoute(
        relation_id=1,
        route_type="bus",
        ref="12",
    )

    train = TransitRoute(
        relation_id=2,
        route_type="train",
        ref="12",
    )

    network = TransitRouteNetwork(
        {
            1: bus,
            2: train,
        }
    )

    leg = {
        "route": "12",
        "routeType": "BUS",
    }

    candidates = network._compatible_candidates(leg)

    assert candidates == [bus]


def test_compatible_candidates_supports_gtfs_rail():
    train = TransitRoute(
        relation_id=1,
        route_type="train",
        ref="IC5",
    )

    network = TransitRouteNetwork({1: train})

    leg = {
        "route": "IC5",
        "routeType": "RAIL",
    }

    candidates = network._compatible_candidates(leg)

    assert candidates == [train]


# ----------------------------------------------------------------------
# Spatial indexes
# ----------------------------------------------------------------------


def test_typed_bbox_index_filters_by_type_and_is_cached():
    bus = make_route(1, A_COORDS, route_type="bus")
    train = make_route(2, B_COORDS, route_type="train")
    network = TransitRouteNetwork({1: bus, 2: train})

    ids, bb = network._typed_bbox_index("BUS")

    assert ids == [1]
    assert bb.shape == (1, 4)
    assert bb[0] == pytest.approx([46.0, 46.0, 7.0, 7.1])

    assert network._typed_bbox_index("BUS")[1] is bb


def test_typed_bbox_index_unknown_label_applies_no_filter():
    bus = make_route(1, A_COORDS, route_type="bus")
    train = make_route(2, B_COORDS, route_type="train")
    network = TransitRouteNetwork({1: bus, 2: train})

    ids, bb = network._typed_bbox_index("FOO")

    assert ids == [1, 2]
    assert bb.shape == (2, 4)


def test_typed_bbox_index_skips_routes_without_stops_and_handles_empty():
    empty = TransitRoute(relation_id=1, route_type="bus", ref="12")
    network = TransitRouteNetwork({1: empty})

    ids, bb = network._typed_bbox_index("BUS")

    assert ids == []
    assert bb.shape == (0, 4)


def test_spatial_candidates_require_both_terminals_inside_bbox():
    a = make_route(1, A_COORDS)
    b = make_route(2, B_COORDS)
    network = TransitRouteNetwork({1: a, 2: b})

    assert network._spatial_candidates("BUS", [A_COORDS[0], A_COORDS[-1]], 750) == [a]
    assert network._spatial_candidates("BUS", [B_COORDS[0], B_COORDS[-1]], 750) == [b]

    # Terminals on two different routes: no single route covers both.
    assert (
        network._spatial_candidates("BUS", [A_COORDS[0], B_COORDS[-1]], 750)
        == []
    )


def test_spatial_candidates_respects_route_type_and_ignores_ref():
    bus = make_route(1, A_COORDS, route_type="bus", ref="1")
    train = make_route(2, A_COORDS, route_type="train", ref="2")
    network = TransitRouteNetwork({1: bus, 2: train})

    assert network._spatial_candidates("BUS", [A_COORDS[0], A_COORDS[-1]], 750) == [bus]
    assert network._spatial_candidates("RAIL", [A_COORDS[0], A_COORDS[-1]], 750) == [train]


def test_spatial_candidates_with_no_routes():
    network = TransitRouteNetwork({})

    assert network._spatial_candidates("BUS", [(46.0, 7.0), (46.1, 7.1)], 750) == []


def test_routes_with_stop_near_sorted_by_distance():
    far = make_route(3, [(46.4, 7.0), (46.5, 7.0)])
    near = make_route(1, [(46.0, 7.0), (46.001, 7.0)])  # 0 m
    mid = make_route(2, [(46.002, 7.0), (46.003, 7.0)])  # ~220 m
    network = TransitRouteNetwork({3: far, 1: near, 2: mid})

    result = network._routes_with_stop_near("BUS", (46.0, 7.0), 750)

    assert result == [near, mid]


def test_routes_with_stop_near_respects_distance_and_limit():
    near = make_route(1, [(46.0, 7.0), (46.001, 7.0)])
    mid = make_route(2, [(46.002, 7.0), (46.003, 7.0)])
    network = TransitRouteNetwork({1: near, 2: mid})

    assert network._routes_with_stop_near("BUS", (46.0, 7.0), 100) == [near]
    assert network._routes_with_stop_near("BUS", (46.0, 7.0), 750, limit=1) == [near]
    assert network._routes_with_stop_near("BUS", (47.0, 8.0), 750) == []


def test_routes_with_stop_near_respects_type_and_empty_network():
    train = make_route(1, A_COORDS, route_type="train")
    network = TransitRouteNetwork({1: train})

    assert network._routes_with_stop_near("BUS", A_COORDS[0], 750) == []
    assert TransitRouteNetwork({})._routes_with_stop_near("BUS", A_COORDS[0], 750) == []


# ----------------------------------------------------------------------
# Best route for a simple leg
# ----------------------------------------------------------------------


def test_best_stop_route_prefers_lowest_score():
    exact = make_route(1, A_COORDS)
    shifted = make_route(2, [(lat + 0.001, lon) for lat, lon in A_COORDS])
    network = TransitRouteNetwork({1: exact, 2: shifted})

    result = network._best_stop_route(
        [shifted, exact], A_COORDS[0], A_COORDS[-1], 750
    )

    assert result is exact


def test_best_stop_route_returns_none_without_match():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    # wrong direction
    assert (
        network._best_stop_route([route], A_COORDS[-1], A_COORDS[0], 750)
        is None
    )
    # nothing candidate
    assert network._best_stop_route([], A_COORDS[0], A_COORDS[-1], 750) is None


def test_best_geometry_route_prefers_closest_geometry():
    exact = make_route(1, A_COORDS)
    shifted = make_route(2, [(lat + 0.01, lon) for lat, lon in A_COORDS])
    network = TransitRouteNetwork({1: exact, 2: shifted})

    result = network._best_geometry_route(
        [shifted, exact], A_COORDS[0], A_COORDS[-1], 5000
    )

    assert result is exact


def test_best_geometry_route_skips_missing_or_empty_geometry():
    none_geom = TransitRoute(relation_id=1, route_type="bus", ref="12", geometry=None)
    empty_geom = TransitRoute(
        relation_id=2, route_type="bus", ref="12", geometry=LineString()
    )
    network = TransitRouteNetwork({1: none_geom, 2: empty_geom})

    assert (
        network._best_geometry_route(
            [none_geom, empty_geom], A_COORDS[0], A_COORDS[-1], 5000
        )
        is None
    )


def test_best_geometry_route_skips_routes_outside_bounds():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    assert (
        network._best_geometry_route([route], (47.0, 8.0), (47.0, 8.1), 5000)
        is None
    )


def test_best_geometry_route_skips_routes_beyond_distance():
    # Diagonal route; the points sit on the opposite bbox corners, so they
    # pass the cheap bbox rejection but are hundreds of metres from the line.
    route = make_route(1, [(46.0, 7.0), (46.01, 7.01)])
    network = TransitRouteNetwork({1: route})

    assert (
        network._best_geometry_route(
            [route], (46.0, 7.01), (46.01, 7.0), 300
        )
        is None
    )


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------


def test_describe_no_match_when_ref_unknown():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    message = network._describe_no_match(
        {"route": "999", "routeType": "BUS"}, 750
    )

    assert "No route with ref '999'" in message


def test_describe_no_match_when_type_incompatible():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS, route_type="bus")})

    message = network._describe_no_match(
        {"route": "12", "routeType": "RAIL"}, 750
    )

    assert "Ref '12' exists tagged in OSM as" in message
    assert "bus" in message
    assert "only accepts OSM types" in message
    assert "train" in message
    assert "750" in message


def test_describe_no_match_when_gtfs_type_unknown():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    message = network._describe_no_match(
        {"route": "12", "routeType": "FOO"}, 750
    )

    assert "not in GTFS_TYPE_TO_OSM_ROUTES" in message


def test_diagnose_no_candidates():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS, route_type="train")})

    info = network.diagnose_stop_sequence(BUS_LEG, A_COORDS)

    assert info["reason"] == "NO_CANDIDATES"
    assert "BUS" in info["detail"]
    assert info["n_candidates"] == 0


def test_diagnose_reports_candidate_source_ref():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    info = network.diagnose_stop_sequence(BUS_LEG, A_COORDS)

    assert info["normalized_ref"] == _normalize_ref("12")
    assert info["osm_routes_with_same_ref"] == 1
    assert info["osm_types_of_same_ref"] == "bus"
    assert info["n_candidates"] == 1
    assert info["candidate_source"] == "ref"


def test_diagnose_reports_candidate_source_type_fallback():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    info = network.diagnose_stop_sequence(
        {"route": "999", "routeType": "BUS"}, A_COORDS
    )

    assert info["osm_routes_with_same_ref"] == 0
    assert info["candidate_source"] == "type_fallback (ref not matched)"


def test_diagnose_reports_candidate_source_none():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    info = network.diagnose_stop_sequence(
        {"route": "999", "routeType": "FOO"}, A_COORDS
    )

    assert info["candidate_source"] == "none"


def test_diagnose_no_route_near_either_end():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    info = network.diagnose_stop_sequence(
        BUS_LEG, [(47.0, 8.0), (47.1, 8.1)]
    )

    assert info["reason"] == "NO_ROUTE_NEAR_EITHER_END"
    assert info["routes_near_first_stop"] == 0
    assert info["routes_near_last_stop"] == 0
    assert info["routes_near_both_ends"] == 0
    assert "best_relation" not in info


def test_diagnose_no_route_near_first_stop():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    info = network.diagnose_stop_sequence(
        BUS_LEG, [(40.0, 0.0), A_COORDS[-1]]
    )

    assert info["reason"] == "NO_ROUTE_NEAR_FIRST_STOP"
    assert info["routes_near_first_stop"] == 0
    assert info["routes_near_last_stop"] == 1


def test_diagnose_no_route_near_last_stop():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    info = network.diagnose_stop_sequence(
        BUS_LEG, [A_COORDS[0], (40.0, 0.0)]
    )

    assert info["reason"] == "NO_ROUTE_NEAR_LAST_STOP"
    assert info["routes_near_first_stop"] == 1
    assert info["routes_near_last_stop"] == 0


def test_diagnose_no_single_route_near_both_ends():
    r1 = make_route(1, [(46.0, 7.0), (46.001, 7.001)])
    r2 = make_route(2, [(47.0, 8.0), (47.001, 8.001)])
    network = TransitRouteNetwork({1: r1, 2: r2})

    info = network.diagnose_stop_sequence(
        BUS_LEG, [(46.0, 7.0), (46.5, 7.5), (47.0, 8.0)]
    )

    assert info["reason"] == "NO_SINGLE_ROUTE_NEAR_BOTH_ENDS"
    assert info["routes_near_first_stop"] == 1
    assert info["routes_near_last_stop"] == 1
    assert info["routes_near_both_ends"] == 0


def test_diagnose_stop_too_far():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    # Terminals match, the middle stop is ~2 km away from the route.
    stops = [(46.0, 7.0), (46.02, 7.05), (46.0, 7.1)]

    info = network.diagnose_stop_sequence(BUS_LEG, stops)

    assert info["reason"] == "STOP_TOO_FAR"
    assert info["routes_near_both_ends"] == 1
    assert info["best_relation"] == "https://www.openstreetmap.org/relation/1"
    assert info["best_osm_ref"] == "12"
    assert info["best_direction"] == "forward"
    assert info["matched_stops"] == "1/3"
    assert info["fail_stop_index"] == 1
    assert info["fail_distance_m"] > 750
    assert "750" in info["detail"]


def test_diagnose_stop_order_mismatch():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    # The GTFS trip has more stops than the OSM relation can absorb.
    stops = A_COORDS + [A_COORDS[-1]]

    info = network.diagnose_stop_sequence(BUS_LEG, stops)

    assert info["reason"] == "STOP_ORDER_MISMATCH"
    assert info["matched_stops"] == "3/4"
    assert info["fail_stop_index"] == 3
    assert "fail_distance_m" not in info


def test_diagnose_geometry_empty_when_stops_match():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    info = network.diagnose_stop_sequence(BUS_LEG, A_COORDS)

    assert info["reason"] == "GEOMETRY_EMPTY"
    assert info["matched_stops"] == "3/3"
    assert "fail_stop_index" not in info


def test_diagnose_fail_stop_index_uses_original_order_when_reversed():
    route = make_route(1, [(46.0, 7.0 + 0.025 * i) for i in range(5)])
    network = TransitRouteNetwork({1: route})

    # The trip runs against the OSM direction and its third stop (GTFS
    # index 2) is ~2 km off the line. Read forwards the matcher dies at
    # index 1 (it runs out of OSM stops); read backwards it gets to index 2,
    # so the reversed orientation wins. The reported index must be
    # translated back to GTFS order.
    stops = [
        (46.0, 7.1),
        (46.0, 7.075),
        (46.02, 7.05),
        (46.0, 7.025),
        (46.0, 7.0),
    ]

    info = network.diagnose_stop_sequence(BUS_LEG, stops)

    assert info["reason"] == "STOP_TOO_FAR"
    assert info["best_direction"] == "reversed"
    assert info["matched_stops"] == "2/5"
    assert info["fail_stop_index"] == 2


def test_diagnose_prefers_forward_direction_on_ties():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    # Both orientations fail at index 1; forward is evaluated first and a
    # tie does not replace it.
    stops = [(46.0, 7.1), (46.02, 7.05), (46.0, 7.0)]

    info = network.diagnose_stop_sequence(BUS_LEG, stops)

    assert info["reason"] == "STOP_ORDER_MISMATCH"
    assert info["best_direction"] == "forward"


def test_diagnose_respects_max_eval():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    info = network.diagnose_stop_sequence(BUS_LEG, A_COORDS, max_eval=0)

    assert info["routes_near_both_ends"] == 1
    assert "best_relation" not in info


# ----------------------------------------------------------------------
# Single-leg API
# ----------------------------------------------------------------------


def test_find_route_for_leg(route, leg):
    network = TransitRouteNetwork({route.relation_id: route})

    result = network.find_route_for_leg(leg)

    assert result is route


def test_find_route_for_leg_rejects_wrong_type(route, leg):
    leg = dict(leg)
    leg["routeType"] = "RAIL"

    network = TransitRouteNetwork({route.relation_id: route})

    result = network.find_route_for_leg(leg)

    assert result is None


def test_find_route_for_leg_geometry_fallback(route):
    network = TransitRouteNetwork({route.relation_id: route})

    leg = {
        "route": "12",
        "routeType": "BUS",
        "from": {
            "lat": 46.001,
            "lon": 7.001,
        },
        "to": {
            "lat": 46.009,
            "lon": 7.009,
        },
    }

    # These points are intentionally not close enough to the OSM stops
    # for the stop matcher, but are still close to the route geometry.
    result = network.find_route_for_leg(
        leg,
        max_stop_distance_m=1,
        max_geometry_distance_m=5000,
    )

    assert result is route


def test_find_route_for_leg_without_candidates():
    network = TransitRouteNetwork({})

    assert (
        network.find_route_for_leg(make_leg((46.0, 7.0), (46.0, 7.1))) is None
    )


def test_find_route_for_leg_returns_none_when_nothing_matches():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    leg = make_leg((47.0, 8.0), (47.0, 8.1))

    assert network.find_route_for_leg(leg) is None


def test_find_route_for_leg_wrong_direction_falls_back_to_geometry():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    # Stop matcher refuses the reversed leg, geometry matcher accepts it.
    leg = make_leg(A_COORDS[-1], A_COORDS[0])

    assert network.find_route_for_leg(leg) is route


def test_get_geometry_for_leg(route, leg):
    network = TransitRouteNetwork({route.relation_id: route})

    geometry = network.get_geometry_for_leg(leg)

    assert geometry
    assert geometry[0] == pytest.approx([46.0000, 7.0000])
    assert geometry[-1] == pytest.approx([46.0100, 7.0100])


def test_get_geometry_for_leg_reverse_direction():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    geometry = network.get_geometry_for_leg(
        make_leg(A_COORDS[-1], A_COORDS[0])
    )

    assert geometry[0] == pytest.approx(list(A_COORDS[-1]))
    assert geometry[-1] == pytest.approx(list(A_COORDS[0]))


def test_get_geometry_for_leg_raises_for_no_match(route):
    network = TransitRouteNetwork({1: route})

    leg = {
        "type": "vehicle",
        "route": "999",
        "routeType": "BUS",
        "from": {
            "name": "Nowhere",
            "lat": 47.0,
            "lon": 8.0,
            "platform": None,
        },
        "to": {
            "name": "Still nowhere",
            "lat": 47.01,
            "lon": 8.01,
            "platform": None,
        },
    }

    with pytest.raises(ValueError):
        network.get_geometry_for_leg(leg)


def test_get_geometry_for_leg_error_is_descriptive():
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    with pytest.raises(ValueError, match="No route with ref '999'"):
        network.get_geometry_for_leg(
            make_leg((47.0, 8.0), (47.0, 8.1), route="999")
        )


# ----------------------------------------------------------------------
# Geometry extraction
# ----------------------------------------------------------------------


def test_geometry_between_forward(route):
    network = TransitRouteNetwork({route.relation_id: route})

    geometry = network.geometry_between(
        route,
        (46.0000, 7.0000),
        (46.0100, 7.0100),
    )

    assert geometry[0] == pytest.approx([46.0000, 7.0000])
    assert geometry[-1] == pytest.approx([46.0100, 7.0100])


def test_geometry_between_reverse(route):
    network = TransitRouteNetwork({route.relation_id: route})

    geometry = network.geometry_between(
        route,
        (46.0100, 7.0100),
        (46.0000, 7.0000),
    )

    assert geometry[0] == pytest.approx([46.0100, 7.0100])
    assert geometry[-1] == pytest.approx([46.0000, 7.0000])


def test_geometry_between_returns_lat_lon(route):
    network = TransitRouteNetwork({route.relation_id: route})

    geometry = network.geometry_between(
        route,
        (46.0000, 7.0000),
        (46.0050, 7.0050),
    )

    # Shapely stores x/y = lon/lat.
    # Public API returns lat/lon.
    assert geometry[0] == pytest.approx([46.0000, 7.0000])
    assert geometry[-1] == pytest.approx([46.0050, 7.0050])


def test_geometry_between_none_geometry():
    route = TransitRoute(
        relation_id=1,
        route_type="bus",
        ref="12",
        geometry=None,
    )

    network = TransitRouteNetwork({1: route})

    assert network.geometry_between(
        route,
        (46.0, 7.0),
        (46.1, 7.1),
    ) == []


def test_geometry_between_empty_geometry():
    route = TransitRoute(
        relation_id=1,
        route_type="bus",
        ref="12",
        geometry=LineString(),
    )

    network = TransitRouteNetwork({1: route})

    assert network.geometry_between(
        route,
        (46.0, 7.0),
        (46.1, 7.1),
    ) == []


def test_geometry_between_multilinestring():
    route = TransitRoute(
        relation_id=1,
        route_type="bus",
        ref="12",
        geometry=MultiLineString(
            [
                [
                    (7.0000, 46.0000),
                    (7.0050, 46.0050),
                ],
                [
                    (7.1000, 46.1000),
                    (7.1050, 46.1050),
                ],
            ]
        ),
    )

    network = TransitRouteNetwork({1: route})

    geometry = network.geometry_between(
        route,
        (46.0000, 7.0000),
        (46.0050, 7.0050),
    )

    assert geometry[0] == pytest.approx([46.0000, 7.0000])
    assert geometry[-1] == pytest.approx([46.0050, 7.0050])


def _two_line_route():
    return TransitRoute(
        relation_id=1,
        route_type="bus",
        ref="12",
        geometry=MultiLineString(
            [
                [(7.0000, 46.0000), (7.0050, 46.0050)],
                [(7.1000, 46.1000), (7.1050, 46.1050)],
            ]
        ),
    )


def test_geometry_between_multilinestring_picks_closest_line():
    route = _two_line_route()
    network = TransitRouteNetwork({1: route})

    geometry = network.geometry_between(
        route,
        (46.1000, 7.1000),
        (46.1050, 7.1050),
    )

    assert_coords(geometry, [[46.1000, 7.1000], [46.1050, 7.1050]])


def test_geometry_between_multilinestring_reverse():
    route = _two_line_route()
    network = TransitRouteNetwork({1: route})

    geometry = network.geometry_between(
        route,
        (46.0050, 7.0050),
        (46.0000, 7.0000),
    )

    assert_coords(geometry, [[46.0050, 7.0050], [46.0000, 7.0000]])


def test_geometry_between_identical_points_returns_single_point():
    route = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: route})

    geometry = network.geometry_between(route, (46.0, 7.05), (46.0, 7.05))

    assert len(geometry) == 1
    assert_coords(geometry, [[46.0, 7.05]])


def test_segment_forward_and_reversed():
    geom = LineString([(7.0, 46.0), (7.01, 46.01)])

    forward = TransitRouteNetwork._segment(geom, 0.0, geom.length)
    backward = TransitRouteNetwork._segment(geom, geom.length, 0.0)

    assert_coords(forward, [[46.0, 7.0], [46.01, 7.01]])
    assert_coords(backward, [[46.01, 7.01], [46.0, 7.0]])


def test_segment_partial():
    geom = LineString([(7.0, 46.0), (7.01, 46.01)])

    result = TransitRouteNetwork._segment(geom, 0.0, geom.length / 2)

    assert_coords(result[0], [46.0, 7.0])
    assert_coords(result[-1], [46.005, 7.005])


# ----------------------------------------------------------------------
# Stitching
# ----------------------------------------------------------------------


def test_stitch_without_geometry():
    none_geom = TransitRoute(relation_id=1, route_type="bus", ref="12", geometry=None)
    empty_geom = TransitRoute(
        relation_id=2, route_type="bus", ref="12", geometry=LineString()
    )
    network = TransitRouteNetwork({1: none_geom, 2: empty_geom})

    assert network._stitch(none_geom, A_COORDS) == []
    assert network._stitch(empty_geom, A_COORDS) == []


def test_stitch_concatenates_segments_without_duplicates():
    coords = [(46.0, 7.0), (46.005, 7.005), (46.01, 7.01)]
    route = make_route(1, coords)
    network = TransitRouteNetwork({1: route})

    result = network._stitch(route, coords)

    assert_coords(result, [list(c) for c in coords])
    assert all(a != b for a, b in zip(result, result[1:]))


def test_stitch_in_reverse_direction():
    coords = [(46.0, 7.0), (46.005, 7.005), (46.01, 7.01)]
    route = make_route(1, coords)
    network = TransitRouteNetwork({1: route})

    result = network._stitch(route, coords[::-1])

    assert_coords(result, [list(c) for c in coords[::-1]])


def test_stitch_with_skipped_intermediate_vertices():
    coords = [(46.0, 7.0), (46.005, 7.005), (46.01, 7.01)]
    route = make_route(1, coords)
    network = TransitRouteNetwork({1: route})

    result = network._stitch(route, [coords[0], coords[-1]])

    # The OSM vertex between the two GTFS stops is kept in the geometry.
    assert_coords(result, [list(c) for c in coords])


def test_stitch_multilinestring_uses_geometry_between():
    route = _two_line_route()
    network = TransitRouteNetwork({1: route})

    result = network._stitch(
        route,
        [(46.0000, 7.0000), (46.0050, 7.0050)],
    )

    assert_coords(result, [[46.0000, 7.0000], [46.0050, 7.0050]])


# ----------------------------------------------------------------------
# Chaining two routes
# ----------------------------------------------------------------------


def test_chain_two_joins_routes_at_shared_station():
    a = make_route(1, A_COORDS)
    b = make_route(2, B_COORDS)
    network = TransitRouteNetwork({1: a, 2: b})

    p, q = A_COORDS[0], B_COORDS[-1]

    result = network._chain_two("BUS", p, q, 750)

    assert result is not None
    assert len(result) >= 2
    assert_coords(result[0], list(p))
    assert_coords(result[-1], list(q))
    # The junction stop of route A is part of the chained geometry.
    assert any(np.allclose(pt, [46.0, 7.1]) for pt in result)


def test_chain_two_returns_none_without_route_near_endpoint():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    assert network._chain_two("BUS", A_COORDS[0], (47.0, 8.0), 750) is None
    assert network._chain_two("BUS", (47.0, 8.0), A_COORDS[0], 750) is None


def test_chain_two_ignores_a_single_route_covering_both_points():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    assert network._chain_two("BUS", A_COORDS[0], A_COORDS[-1], 750) is None


def test_chain_two_requires_stops_within_link_distance():
    a = make_route(1, A_COORDS)
    gap = make_route(2, [(46.0, 7.12), (46.0, 7.16), (46.0, 7.2)])
    network = TransitRouteNetwork({1: a, 2: gap})

    assert (
        network._chain_two("BUS", A_COORDS[0], (46.0, 7.2), 750) is None
    )


def test_chain_two_link_distance_is_configurable():
    a = make_route(1, A_COORDS)
    gap = make_route(2, [(46.0, 7.12), (46.0, 7.16), (46.0, 7.2)])
    network = TransitRouteNetwork({1: a, 2: gap})

    result = network._chain_two(
        "BUS", A_COORDS[0], (46.0, 7.2), 750, link_distance_m=3000
    )

    assert result is not None
    assert_coords(result[0], list(A_COORDS[0]))
    assert_coords(result[-1], [46.0, 7.2])


# ----------------------------------------------------------------------
# Segmented matching
# ----------------------------------------------------------------------


def test_segmented_geometry_single_route_covers_everything():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    result = network._segmented_geometry("BUS", A_COORDS, 750)

    assert result is not None
    assert_coords(result[0], list(A_COORDS[0]))
    assert_coords(result[-1], list(A_COORDS[-1]))


def test_segmented_geometry_splits_run_over_two_routes():
    a = make_route(1, A_COORDS)
    b = make_route(2, B_COORDS)
    network = TransitRouteNetwork({1: a, 2: b})

    stops = [
        (46.0, 7.0),
        (46.0, 7.05),
        (46.0, 7.1),
        (46.0, 7.15),
        (46.0, 7.2),
    ]

    result = network._segmented_geometry("BUS", stops, 750)

    assert result is not None
    assert_coords(result[0], [46.0, 7.0])
    assert_coords(result[-1], [46.0, 7.2])
    # Both OSM routes contribute geometry.
    assert any(np.allclose(pt, [46.0, 7.05]) for pt in result)
    assert any(np.allclose(pt, [46.0, 7.15]) for pt in result)


def test_segmented_geometry_chains_two_routes_for_a_single_hop():
    a = make_route(1, A_COORDS)
    b = make_route(2, B_COORDS)
    network = TransitRouteNetwork({1: a, 2: b})

    result = network._segmented_geometry(
        "BUS", [A_COORDS[0], B_COORDS[-1]], 750
    )

    assert result is not None
    assert_coords(result[0], list(A_COORDS[0]))
    assert_coords(result[-1], list(B_COORDS[-1]))
    assert any(np.allclose(pt, [46.0, 7.1]) for pt in result)


def test_segmented_geometry_bridges_gaps_with_straight_lines():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    x, y = (47.0, 8.0), (47.01, 8.01)

    result = network._segmented_geometry(
        "BUS", [A_COORDS[0], A_COORDS[1], x, y], 750
    )

    assert result is not None
    assert_coords(result[0], list(A_COORDS[0]))
    # The uncovered stops are joined with straight segments.
    assert any(np.allclose(pt, list(x)) for pt in result)
    assert_coords(result[-1], list(y))


def test_segmented_geometry_returns_none_if_nothing_matches():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    assert (
        network._segmented_geometry(
            "BUS", [(47.0, 8.0), (47.01, 8.01), (47.02, 8.02)], 750
        )
        is None
    )


# ----------------------------------------------------------------------
# Stop-sequence API
# ----------------------------------------------------------------------


def test_get_geometry_for_stop_sequence(route, leg, stop_sequence):
    network = TransitRouteNetwork({route.relation_id: route})

    geometry = network.get_geometry_for_stop_sequence(
        leg,
        stop_sequence,
    )

    assert geometry
    assert geometry[0] == pytest.approx([46.0000, 7.0000])
    assert geometry[-1] == pytest.approx([46.0100, 7.0100])


def test_get_geometry_for_stop_sequence_with_two_stops(route, leg):
    network = TransitRouteNetwork({route.relation_id: route})

    geometry = network.get_geometry_for_stop_sequence(
        leg,
        [
            (46.0000, 7.0000),
            (46.0100, 7.0100),
        ],
    )

    assert geometry
    assert geometry[0] == pytest.approx([46.0000, 7.0000])
    assert geometry[-1] == pytest.approx([46.0100, 7.0100])


def test_get_geometry_for_stop_sequence_requires_two_stops(route, leg):
    network = TransitRouteNetwork({route.relation_id: route})

    assert network.get_geometry_for_stop_sequence(
        leg,
        [(46.0000, 7.0000)],
    ) == []


def test_get_geometry_for_stop_sequence_raises_without_candidates():
    network = TransitRouteNetwork({})

    with pytest.raises(ValueError):
        network.get_geometry_for_stop_sequence(
            {
                "route": "999",
                "routeType": "BUS",
            },
            [
                (46.0000, 7.0000),
                (46.0100, 7.0100),
            ],
        )


def test_stop_sequence_stage1_follows_the_ref_candidate():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    geometry = network.get_geometry_for_stop_sequence(BUS_LEG, A_COORDS)

    assert_coords(geometry[0], list(A_COORDS[0]))
    assert_coords(geometry[-1], list(A_COORDS[-1]))
    assert any(np.allclose(pt, [46.0, 7.05]) for pt in geometry)


def test_stop_sequence_stage1_matches_reversed_trip():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    geometry = network.get_geometry_for_stop_sequence(
        BUS_LEG, A_COORDS[::-1]
    )

    assert_coords(geometry[0], list(A_COORDS[-1]))
    assert_coords(geometry[-1], list(A_COORDS[0]))


def test_stop_sequence_stage2_ignores_ref_when_ref_route_does_not_fit():
    elsewhere = make_route(
        1, [(47.0, 8.0), (47.0, 8.05), (47.0, 8.1)], ref="12"
    )
    right_place = make_route(2, A_COORDS, ref="34")
    network = TransitRouteNetwork({1: elsewhere, 2: right_place})

    geometry = network.get_geometry_for_stop_sequence(BUS_LEG, A_COORDS)

    assert_coords(geometry[0], list(A_COORDS[0]))
    assert_coords(geometry[-1], list(A_COORDS[-1]))


def test_stop_sequence_stage2_when_gtfs_type_unknown():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    geometry = network.get_geometry_for_stop_sequence(
        {"route": "999", "routeType": "FOO"}, A_COORDS
    )

    assert_coords(geometry[0], list(A_COORDS[0]))
    assert_coords(geometry[-1], list(A_COORDS[-1]))


def test_stop_sequence_stage3_segments_across_routes():
    a = make_route(1, A_COORDS)
    b = make_route(2, B_COORDS)
    network = TransitRouteNetwork({1: a, 2: b})

    stops = [
        (46.0, 7.0),
        (46.0, 7.05),
        (46.0, 7.1),
        (46.0, 7.15),
        (46.0, 7.2),
    ]

    geometry = network.get_geometry_for_stop_sequence(
        {"route": "X", "routeType": "BUS"}, stops
    )

    assert_coords(geometry[0], [46.0, 7.0])
    assert_coords(geometry[-1], [46.0, 7.2])
    assert any(np.allclose(pt, [46.0, 7.05]) for pt in geometry)
    assert any(np.allclose(pt, [46.0, 7.15]) for pt in geometry)


def test_stop_sequence_stage4_geometry_fallback():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    # Stops are ~110 m off the OSM stops, but a 1 m tolerance rules out
    # stop matching; the geometry is still close enough (5 km).
    stops = [(46.001, 7.0), (46.001, 7.05), (46.001, 7.1)]

    geometry = network.get_geometry_for_stop_sequence(
        BUS_LEG,
        stops,
        max_stop_distance_m=1,
        max_geometry_distance_m=5000,
    )

    assert len(geometry) >= 2
    # Stops are projected onto the OSM line.
    assert_coords(geometry[0], [46.0, 7.0])
    assert_coords(geometry[-1], [46.0, 7.1])


def test_stop_sequence_raises_descriptive_error_when_nothing_matches():
    a = make_route(1, A_COORDS)
    network = TransitRouteNetwork({1: a})

    with pytest.raises(ValueError, match="Ref '12' exists tagged in OSM as"):
        network.get_geometry_for_stop_sequence(
            BUS_LEG,
            [(47.0, 8.0), (47.05, 8.05), (47.1, 8.1)],
        )

# ----------------------------------------------------------------------
# Merging
# ----------------------------------------------------------------------

LONG_COORDS = A_COORDS + B_COORDS  # 6 stops, geometry spans 7.0 -> 7.2


def network_of(*routes):
    return TransitRouteNetwork({r.relation_id: r for r in routes})


def with_geometry(route, geometry):
    return replace(route, geometry=geometry)


# -- completeness ranking ------------------------------------------------


def test_completeness_counts_stops_and_measures_geometry():
    n_stops, length = TransitRouteNetwork._completeness(
        make_route(1, A_COORDS)
    )

    assert n_stops == 3
    assert length == pytest.approx(0.1)


def test_completeness_without_geometry_has_zero_length():
    stops = [TransitStop(1, 46.0, 7.0, "A")]
    no_geom = TransitRoute(
        relation_id=1, route_type="bus", ref="12", stops=stops
    )
    empty_geom = replace(no_geom, geometry=LineString())

    assert TransitRouteNetwork._completeness(no_geom) == (1, 0.0)
    assert TransitRouteNetwork._completeness(empty_geom) == (1, 0.0)


def test_completeness_prefers_stops_over_geometry_length():
    many_stops = make_route(1, LONG_COORDS)
    long_line = with_geometry(
        make_route(1, A_COORDS), LineString([(0.0, 0.0), (10.0, 10.0)])
    )

    assert TransitRouteNetwork._completeness(
        many_stops
    ) > TransitRouteNetwork._completeness(long_line)


# -- disjoint networks / basic behavior ----------------------------------


def test_merge_disjoint_networks_keeps_all_routes(capsys):
    a = make_route(1, A_COORDS)
    b = make_route(2, B_COORDS)
    net_a, net_b = network_of(a), network_of(b)

    merged = net_a.merge(net_b)  # default arguments

    assert isinstance(merged, TransitRouteNetwork)
    assert merged is not net_a and merged is not net_b
    assert list(merged.routes) == [1, 2]  # self first, then other
    # Route objects are shared, not copied.
    assert merged.routes[1] is a
    assert merged.routes[2] is b


def test_merge_does_not_modify_inputs():
    mine = make_route(1, A_COORDS, name="mine")
    theirs = make_route(1, B_COORDS, name="theirs")
    extra = make_route(2, B_COORDS)
    net_a, net_b = network_of(mine), network_of(theirs, extra)

    merged = net_a.merge(net_b, on_conflict="keep_other", verbose=False)

    assert merged.routes[1] is theirs
    assert list(net_a.routes) == [1] and net_a.routes[1] is mine
    assert list(net_b.routes) == [1, 2]
    assert net_a.by_ref["12"] == [1]


def test_merge_result_has_its_own_routes_dict():
    net_a = network_of(make_route(1, A_COORDS))
    net_b = network_of(make_route(2, B_COORDS))

    merged = net_a.merge(net_b, verbose=False)
    merged.routes[99] = make_route(99, A_COORDS)

    assert 99 not in net_a.routes
    assert 99 not in net_b.routes


def test_merge_with_empty_networks():
    a = make_route(1, A_COORDS)

    assert TransitRouteNetwork({}).merge(
        TransitRouteNetwork({}), verbose=False
    ).routes == {}
    assert network_of(a).merge(
        TransitRouteNetwork({}), verbose=False
    ).routes == {1: a}
    assert TransitRouteNetwork({}).merge(
        network_of(a), verbose=False
    ).routes == {1: a}


def test_merge_result_indexes_routes_from_both_networks():
    net_a = network_of(make_route(1, A_COORDS))
    net_b = network_of(make_route(2, B_COORDS))

    merged = net_a.merge(net_b, verbose=False)

    assert merged.by_ref["12"] == [1, 2]
    assert merged._bbox[1] == pytest.approx((46.0, 46.0, 7.0, 7.1))
    assert merged._bbox[2] == pytest.approx((46.0, 46.0, 7.1005, 7.2))


def test_merge_result_starts_with_empty_caches_and_leaves_sources_warm():
    net_a = network_of(make_route(1, A_COORDS))
    net_b = network_of(make_route(2, B_COORDS))
    net_a._compatible_candidates(BUS_LEG)  # warm a cache on the source

    merged = net_a.merge(net_b, verbose=False)

    assert merged._arrays == {}
    assert merged._cand_cache == {}
    assert merged._type_routes == {}
    assert merged._type_bbox == {}
    assert net_a._cand_cache  # source untouched


def test_merge_keeps_route_without_stops_or_ref():
    stubless = TransitRoute(relation_id=3, route_type="bus")
    merged = network_of(make_route(1, A_COORDS)).merge(
        network_of(stubless), verbose=False
    )

    assert merged.routes[3] is stubless
    assert 3 not in merged._bbox
    assert merged.by_ref["12"] == [1]


# -- shared relations ----------------------------------------------------


@pytest.mark.parametrize(
    "policy", ["most_complete", "keep_self", "keep_other", "raise"]
)
def test_merge_same_route_object_is_not_a_conflict(policy, capsys):
    shared = make_route(1, A_COORDS)
    extra = make_route(2, B_COORDS)

    merged = network_of(shared).merge(
        network_of(shared, extra), on_conflict=policy
    )

    assert merged.routes == {1: shared, 2: extra}
    assert "(1 new, 0 shared, 0 shared replaced)" in capsys.readouterr().out


def test_merge_network_with_itself_under_raise():
    network = network_of(make_route(1, A_COORDS), make_route(2, B_COORDS))

    merged = network.merge(network, on_conflict="raise", verbose=False)

    assert merged.routes == network.routes
    assert merged.routes is not network.routes


def test_merge_most_complete_is_the_default():
    small = make_route(1, A_COORDS, name="small")
    big = make_route(1, LONG_COORDS, name="big")

    assert network_of(small).merge(
        network_of(big), verbose=False
    ).routes[1] is big


def test_merge_most_complete_takes_other_when_it_has_more_stops():
    small = make_route(1, A_COORDS, name="small")
    big = make_route(1, LONG_COORDS, name="big")

    merged = network_of(small).merge(
        network_of(big), on_conflict="most_complete", verbose=False
    )

    assert merged.routes[1] is big


def test_merge_most_complete_keeps_self_when_it_has_more_stops():
    small = make_route(1, A_COORDS, name="small")
    big = make_route(1, LONG_COORDS, name="big")

    merged = network_of(big).merge(
        network_of(small), on_conflict="most_complete", verbose=False
    )

    assert merged.routes[1] is big


def test_merge_most_complete_breaks_stop_ties_with_geometry_length():
    short_line = make_route(1, A_COORDS, name="short")
    long_line = with_geometry(
        make_route(1, A_COORDS, name="long"),
        LineString([(7.0, 46.0), (7.0, 47.0)]),
    )

    assert network_of(short_line).merge(
        network_of(long_line), verbose=False
    ).routes[1] is long_line
    assert network_of(long_line).merge(
        network_of(short_line), verbose=False
    ).routes[1] is long_line


def test_merge_most_complete_prefers_route_with_geometry_over_none():
    no_geom = with_geometry(make_route(1, A_COORDS, name="none"), None)
    with_geom = make_route(1, A_COORDS, name="geom")

    assert network_of(no_geom).merge(
        network_of(with_geom), verbose=False
    ).routes[1] is with_geom
    assert network_of(with_geom).merge(
        network_of(no_geom), verbose=False
    ).routes[1] is with_geom


def test_merge_most_complete_prefers_route_with_stops_over_stubless():
    stubless = TransitRoute(relation_id=1, route_type="bus", ref="12")
    full = make_route(1, A_COORDS)

    assert network_of(stubless).merge(
        network_of(full), verbose=False
    ).routes[1] is full
    assert network_of(full).merge(
        network_of(stubless), verbose=False
    ).routes[1] is full


def test_merge_most_complete_full_tie_keeps_self():
    mine = make_route(1, A_COORDS, name="mine")
    theirs = make_route(1, A_COORDS, name="theirs")

    merged = network_of(mine).merge(network_of(theirs), verbose=False)

    assert merged.routes[1] is mine


def test_merge_keep_self_ignores_a_more_complete_other():
    mine = make_route(1, A_COORDS, name="mine")
    theirs = make_route(1, LONG_COORDS, name="theirs")

    merged = network_of(mine).merge(
        network_of(theirs), on_conflict="keep_self", verbose=False
    )

    assert merged.routes[1] is mine


def test_merge_keep_self_still_adds_new_relations():
    mine = make_route(1, A_COORDS)
    new = make_route(2, B_COORDS)

    merged = network_of(mine).merge(
        network_of(make_route(1, LONG_COORDS), new),
        on_conflict="keep_self",
        verbose=False,
    )

    assert merged.routes[1] is mine
    assert merged.routes[2] is new


def test_merge_keep_other_ignores_a_less_complete_other():
    mine = make_route(1, LONG_COORDS, name="mine")
    theirs = make_route(1, A_COORDS, name="theirs")

    merged = network_of(mine).merge(
        network_of(theirs), on_conflict="keep_other", verbose=False
    )

    assert merged.routes[1] is theirs


def test_merge_raise_reports_the_relation_id():
    net_a = network_of(make_route(1, A_COORDS), make_route(7, A_COORDS))
    net_b = network_of(make_route(7, B_COORDS))

    with pytest.raises(ValueError, match="Relation 7 exists in both"):
        net_a.merge(net_b, on_conflict="raise")


def test_merge_raise_does_not_modify_self_when_inplace():
    mine = make_route(1, A_COORDS)
    network = network_of(mine)
    # Route 2 is processed (added) before the conflict on route 1.
    other = TransitRouteNetwork(
        {2: make_route(2, B_COORDS), 1: make_route(1, LONG_COORDS)}
    )

    with pytest.raises(ValueError):
        network.merge(other, on_conflict="raise", inplace=True)

    assert list(network.routes) == [1]
    assert network.routes[1] is mine
    assert network.by_ref["12"] == [1]


@pytest.mark.parametrize("policy", ["newest", "MOST_COMPLETE", "", None])
def test_merge_rejects_unknown_policy(policy):
    net_a = network_of(make_route(1, A_COORDS))
    net_b = network_of(make_route(2, B_COORDS))  # no conflict needed

    with pytest.raises(ValueError, match="Unknown on_conflict policy"):
        net_a.merge(net_b, on_conflict=policy)


def test_merge_unknown_policy_does_not_modify_self_when_inplace():
    network = network_of(make_route(1, A_COORDS))

    with pytest.raises(ValueError):
        network.merge(
            network_of(make_route(2, B_COORDS)),
            on_conflict="bogus",
            inplace=True,
        )

    assert list(network.routes) == [1]


# -- inplace ---------------------------------------------------------------


def test_merge_inplace_returns_self_and_adds_routes():
    a = make_route(1, A_COORDS)
    b = make_route(2, B_COORDS)
    network, other = network_of(a), network_of(b)

    result = network.merge(other, inplace=True, verbose=False)

    assert result is network
    assert list(network.routes) == [1, 2]
    assert network.by_ref["12"] == [1, 2]
    assert 2 in network._bbox


def test_merge_inplace_does_not_alias_or_modify_other():
    other = network_of(make_route(2, B_COORDS))
    network = network_of(make_route(1, A_COORDS))

    network.merge(other, inplace=True, verbose=False)
    network.routes[99] = make_route(99, A_COORDS)

    assert network.routes is not other.routes
    assert list(other.routes) == [2]


def test_merge_inplace_rebuilds_indexes_for_replaced_route():
    old = make_route(1, A_COORDS, ref="12")
    new = make_route(1, LONG_COORDS, ref="34")
    network = network_of(old)

    network.merge(
        network_of(new), on_conflict="keep_other", inplace=True, verbose=False
    )

    assert network.routes[1] is new
    assert "12" not in network.by_ref
    assert network.by_ref["34"] == [1]
    assert network._bbox[1] == pytest.approx((46.0, 46.0, 7.0, 7.2))


def test_merge_inplace_clears_stale_caches():
    old = make_route(1, A_COORDS)
    new = make_route(1, LONG_COORDS)
    network = network_of(old)

    # Warm every lazy cache.
    network._route_arrays(old)
    network._compatible_candidates(BUS_LEG)
    network._compute_compatible_candidates("999", "BUS")
    network._typed_bbox_index("BUS")
    assert network._arrays and network._cand_cache
    assert network._type_routes and network._type_bbox

    network.merge(
        network_of(new), on_conflict="keep_other", inplace=True, verbose=False
    )

    assert network._arrays == {}
    assert network._cand_cache == {}
    assert network._type_routes == {}
    assert network._type_bbox == {}

    # `_arrays` is keyed by relation id: it must reflect the NEW route.
    lat_r, _, _ = network._route_arrays(new)
    assert lat_r.shape == (6,)

    ids, bb = network._typed_bbox_index("BUS")
    assert ids == [1]
    assert bb[0] == pytest.approx([46.0, 46.0, 7.0, 7.2])


# -- verbose summary -------------------------------------------------------


def test_merge_prints_summary_counts(capsys):
    net_a = network_of(
        make_route(1, A_COORDS), make_route(2, B_COORDS)
    )
    net_b = network_of(
        make_route(1, LONG_COORDS), make_route(3, B_COORDS)
    )

    net_a.merge(net_b)

    out = capsys.readouterr().out

    assert (
        "Merged networks: 2 + 2 -> 3 routes "
        "(1 new, 1 shared, 1 shared replaced)"
    ) in out


def test_merge_summary_counts_unreplaced_conflicts(capsys):
    net_a = network_of(make_route(1, A_COORDS))
    net_b = network_of(make_route(1, LONG_COORDS))

    net_a.merge(net_b, on_conflict="keep_self")

    assert (
        "(0 new, 1 shared, 0 shared replaced)" in capsys.readouterr().out
    )


def test_merge_verbose_false_is_silent(capsys):
    network_of(make_route(1, A_COORDS)).merge(
        network_of(make_route(1, LONG_COORDS)), verbose=False
    )

    assert capsys.readouterr().out == ""


# -- the merged network is usable -------------------------------------------


def test_merged_network_matches_legs_from_either_source():
    net_a = network_of(make_route(1, A_COORDS))
    net_b = network_of(make_route(2, B_COORDS))
    leg_b = make_leg(B_COORDS[0], B_COORDS[-1])

    # Alone, the first network cannot serve a leg on the second one.
    assert net_a.find_route_for_leg(leg_b) is None
    with pytest.raises(ValueError):
        net_a.get_geometry_for_leg(leg_b)

    merged = net_a.merge(net_b, verbose=False)

    assert merged.find_route_for_leg(leg_b).relation_id == 2

    geometry = merged.get_geometry_for_leg(leg_b)
    assert_coords(geometry[0], list(B_COORDS[0]))
    assert_coords(geometry[-1], list(B_COORDS[-1]))

    # ... and the legs of the first network still work.
    assert merged.find_route_for_leg(
        make_leg(A_COORDS[0], A_COORDS[-1])
    ).relation_id == 1


def test_merge_can_be_chained_with_reduce():
    nets = [
        network_of(make_route(1, A_COORDS)),
        network_of(make_route(2, B_COORDS)),
        network_of(make_route(3, LONG_COORDS, ref="34")),
    ]

    merged = reduce(lambda x, y: x.merge(y, verbose=False), nets)

    assert set(merged.routes) == {1, 2, 3}
    assert merged.by_ref["12"] == [1, 2]
    assert merged.by_ref["34"] == [3]


def test_merged_network_survives_save_and_load(tmp_path):
    merged = network_of(make_route(1, A_COORDS)).merge(
        network_of(make_route(2, B_COORDS)), verbose=False
    )

    path = tmp_path / "merged.pkl"
    merged.save(path)

    loaded = TransitRouteNetwork.load(path)

    assert set(loaded.routes) == {1, 2}
    assert loaded.by_ref["12"] == [1, 2]
    assert loaded.find_route_for_leg(
        make_leg(B_COORDS[0], B_COORDS[-1])
    ).relation_id == 2


# ----------------------------------------------------------------------
# Persistence (existing)
# ----------------------------------------------------------------------


def test_save_and_load(tmp_path, route):
    network = TransitRouteNetwork({route.relation_id: route})

    path = tmp_path / "nested" / "network.pkl"

    network.save(path)

    assert path.exists()

    loaded = TransitRouteNetwork.load(path)

    assert isinstance(loaded, TransitRouteNetwork)
    assert loaded.routes.keys() == network.routes.keys()
    assert loaded.routes[1].ref == "12"
    assert loaded.routes[1].name == "Test route"
    assert loaded.routes[1].stops[0].name == "A"
    assert loaded.routes[1].geometry.equals(network.routes[1].geometry)


def test_loaded_network_is_fully_usable(tmp_path):
    network = TransitRouteNetwork({1: make_route(1, A_COORDS)})

    path = tmp_path / "network.pkl"
    network.save(path)

    loaded = TransitRouteNetwork.load(path)

    geometry = loaded.get_geometry_for_stop_sequence(BUS_LEG, A_COORDS)

    assert_coords(geometry[0], list(A_COORDS[0]))
    assert_coords(geometry[-1], list(A_COORDS[-1]))
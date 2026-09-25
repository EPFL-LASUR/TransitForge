from pathlib import Path

import pytest
from shapely.geometry import LineString, MultiLineString

from transitforge.dataclasses import TransitRoute, TransitStop
from transitforge.network import TransitRouteNetwork


def test_network_indexes_routes_by_normalized_ref(route):
    network = TransitRouteNetwork({route.relation_id: route})

    assert route.relation_id in network.routes
    assert network.by_ref["12"] == [route.relation_id]


def test_network_ignores_routes_without_ref(route):
    route.ref = None

    network = TransitRouteNetwork({route.relation_id: route})

    assert network.by_ref == {}


def test_nearest_stop(route):
    network = TransitRouteNetwork({route.relation_id: route})

    index, distance = network._nearest_stop(
        route.stops,
        (46.0050, 7.0050),
    )

    assert index == 1
    assert distance == pytest.approx(0.0)


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


def test_get_geometry_for_leg(route, leg):
    network = TransitRouteNetwork({route.relation_id: route})

    geometry = network.get_geometry_for_leg(leg)

    assert geometry
    assert geometry[0] == pytest.approx([46.0000, 7.0000])
    assert geometry[-1] == pytest.approx([46.0100, 7.0100])


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
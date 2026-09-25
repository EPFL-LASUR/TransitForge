from pathlib import Path

import pytest
from shapely.geometry import LineString

from transitforge.builder import TransitRouteBuilder
from transitforge.dataclasses import TransitRoute, TransitStop
from transitforge.network import TransitRouteNetwork


def test_default_route_types():
    builder = TransitRouteBuilder("test.osm.pbf")

    assert "train" in builder.route_types
    assert "bus" in builder.route_types
    assert "tram" in builder.route_types
    assert "metro" in builder.route_types
    assert "subway" in builder.route_types
    assert "trolleybus" in builder.route_types
    assert "light_rail" in builder.route_types
    assert "funicular" in builder.route_types
    assert "monorail" in builder.route_types
    assert "aerialway" in builder.route_types
    assert "ferry" in builder.route_types



def test_custom_route_types():
    builder = TransitRouteBuilder(
        "test.osm.pbf",
        route_types=["bus", "tram"],
    )

    assert builder.route_types == ["bus", "tram"]


def test_platform_roles():
    builder = TransitRouteBuilder("test.osm.pbf")

    assert "platform" in builder.PLATFORM_ROLES
    assert "platform_entry_only" in builder.PLATFORM_ROLES
    assert "platform_exit_only" in builder.PLATFORM_ROLES


def test_build_route_geometry_connects_ways():
    way_geometries = {
        1: LineString(
            [
                (0.0, 0.0),
                (1.0, 0.0),
            ]
        ),
        2: LineString(
            [
                (2.0, 0.0),
                (1.0, 0.0),
            ]
        ),
        3: LineString(
            [
                (2.0, 0.0),
                (3.0, 0.0),
            ]
        ),
    }

    geometry = TransitRouteBuilder._build_route_geometry(
        [1, 2, 3],
        way_geometries,
    )

    assert geometry is not None
    assert list(geometry.coords) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
    ]


def test_build_route_geometry_reverses_way_when_needed():
    way_geometries = {
        1: LineString(
            [
                (0.0, 0.0),
                (1.0, 0.0),
            ]
        ),
        2: LineString(
            [
                (2.0, 0.0),
                (1.0, 0.0),
            ]
        ),
    }

    geometry = TransitRouteBuilder._build_route_geometry(
        [1, 2],
        way_geometries,
    )

    assert geometry is not None

    assert list(geometry.coords) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ]


def test_build_route_geometry_deduplicates_boundary_points():
    way_geometries = {
        1: LineString(
            [
                (0.0, 0.0),
                (1.0, 0.0),
            ]
        ),
        2: LineString(
            [
                (1.0, 0.0),
                (2.0, 0.0),
            ]
        ),
    }

    geometry = TransitRouteBuilder._build_route_geometry(
        [1, 2],
        way_geometries,
    )

    assert geometry is not None

    assert list(geometry.coords) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ]


def test_build_route_geometry_deduplicates_nearly_identical_boundary():
    way_geometries = {
        1: LineString(
            [
                (0.0, 0.0),
                (1.0, 1.0),
            ]
        ),
        2: LineString(
            [
                (1.0 + 1e-10, 1.0 - 1e-10),
                (2.0, 2.0),
            ]
        ),
    }

    geometry = TransitRouteBuilder._build_route_geometry(
        [1, 2],
        way_geometries,
    )

    assert geometry is not None
    assert len(geometry.coords) == 3



def test_build_route_geometry_returns_none_without_valid_ways():
    geometry = TransitRouteBuilder._build_route_geometry(
        [1, 2, 3],
        {},
    )

    assert geometry is None


def test_build_route_geometry_skips_missing_ways():
    way_geometries = {
        1: LineString([(0, 0), (1, 0)]),
        3: LineString([(1, 0), (2, 0)]),
    }

    result = TransitRouteBuilder._build_route_geometry(
        [1, 999, 3],
        way_geometries,
    )

    assert result is not None
    assert list(result.coords) == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]

def test_remove_repeated_small_loop():
    coords = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (1.0, 0.0),
        (3.0, 0.0),
    ]

    result = TransitRouteBuilder._remove_repeated_loops(coords)

    assert result == [
        (0.0, 0.0),
        (1.0, 0.0),
        (3.0, 0.0),
    ]


def test_remove_repeated_loops_preserves_large_repeat():
    coords = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (4.0, 0.0),
        (5.0, 0.0),
        (2.0, 0.0),
        (6.0, 0.0),
    ]

    result = TransitRouteBuilder._remove_repeated_loops(coords)

    # The repeated point represents a sufficiently large section,
    # so it should not be aggressively removed.
    assert len(result) >= 7


def test_build_route_geometry_removes_small_loop():
    way_geometries = {
        1: LineString(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (1.0, 0.0),
                (3.0, 0.0),
            ]
        )
    }

    geometry = TransitRouteBuilder._build_route_geometry(
        [1],
        way_geometries,
    )

    assert geometry is not None

    assert list(geometry.coords) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (3.0, 0.0),
    ]

def test_build_orchestrates_collectors(monkeypatch, tmp_path):
    pbf = tmp_path / "test.osm.pbf"
    pbf.touch()

    route_id = 123

    stop = TransitStop(
        osm_id=1001,
        lat=46.0,
        lon=7.0,
        name="Test Stop",
        role="platform",
    )

    metadata = {
        route_id: {
            "route_type": "bus",
            "ref": "12",
            "name": "Bus 12",
            "from_name": "A",
            "to_name": "B",
            "operator": "Test Operator",
            "network": "Test Network",
        }
    }

    class FakeRouteMetadataCollector:
        def __init__(self, route_types):
            self.route_types = route_types
            self.routes = metadata

        def apply_file(self, filename, locations=False):
            assert filename == str(pbf)
            assert locations is False

    class FakeStopCollector:
        def __init__(self, stop_ids):
            assert stop_ids == {1001}
            self.stops = {1001: stop}

        def apply_file(self, filename, locations=False):
            assert filename == str(pbf)
            assert locations is True

    class FakeWayGeometryCollector:
        def __init__(self, way_ids):
            assert way_ids == {5001}
            self.geometries = {
                5001: LineString(
                    [
                        (7.0, 46.0),
                        (7.01, 46.01),
                    ]
                )
            }

        def apply_file(self, filename, locations=False, idx=None):
            assert filename == str(pbf)
            assert locations is True
            assert idx == "sparse_file_array"

    monkeypatch.setattr(
        "transitforge.builder.RouteMetadataCollector",
        FakeRouteMetadataCollector,
    )

    monkeypatch.setattr(
        "transitforge.builder.StreamingStopCollector",
        FakeStopCollector,
    )

    monkeypatch.setattr(
        "transitforge.builder.StreamingWayGeometryCollector",
        FakeWayGeometryCollector,
    )

    monkeypatch.setattr(
        "transitforge.builder.get_route_stop_mapping",
        lambda **kwargs: {
            route_id: [1001],
        },
    )

    monkeypatch.setattr(
        "transitforge.builder.get_route_way_roles",
        lambda **kwargs: {
            route_id: [
                (5001, ""),
            ],
        },
    )

    builder = TransitRouteBuilder(pbf)

    network = builder.build()

    assert isinstance(network, TransitRouteNetwork)

    assert list(network.routes) == [route_id]

    route = network.routes[route_id]

    assert route.relation_id == route_id
    assert route.route_type == "bus"
    assert route.ref == "12"
    assert route.name == "Bus 12"

    assert len(route.stops) == 1
    assert route.stops[0] is stop

    assert route.geometry is not None
    assert list(route.geometry.coords) == [
        (7.0, 46.0),
        (7.01, 46.01),
    ]

def test_build_filters_platform_ways(monkeypatch, tmp_path):
    pbf = tmp_path / "test.osm.pbf"
    pbf.touch()

    route_id = 123

    metadata = {
        route_id: {
            "route_type": "bus",
            "ref": "12",
            "name": "Bus 12",
            "from_name": "A",
            "to_name": "B",
            "operator": None,
            "network": None,
        }
    }

    class FakeMetadataCollector:
        def __init__(self, route_types):
            self.routes = metadata

        def apply_file(self, filename, locations=False):
            pass

    class FakeStopCollector:
        def __init__(self, stop_ids):
            self.stops = {}

        def apply_file(self, filename, locations=False):
            pass

    captured_way_ids = None

    class FakeGeometryCollector:
        def __init__(self, way_ids):
            nonlocal captured_way_ids
            captured_way_ids = way_ids
            self.geometries = {}

        def apply_file(self, filename, locations=False, idx=None):
            pass

    monkeypatch.setattr(
        "transitforge.builder.RouteMetadataCollector",
        FakeMetadataCollector,
    )

    monkeypatch.setattr(
        "transitforge.builder.StreamingStopCollector",
        FakeStopCollector,
    )

    monkeypatch.setattr(
        "transitforge.builder.StreamingWayGeometryCollector",
        FakeGeometryCollector,
    )

    monkeypatch.setattr(
        "transitforge.builder.get_route_stop_mapping",
        lambda **kwargs: {route_id: []},
    )

    monkeypatch.setattr(
        "transitforge.builder.get_route_way_roles",
        lambda **kwargs: {
            route_id: [
                (100, ""),
                (101, "platform"),
                (102, "platform_entry_only"),
                (103, "platform_exit_only"),
                (104, "stop"),
            ]
        },
    )

    builder = TransitRouteBuilder(pbf)

    builder.build()

    assert captured_way_ids == {100, 104}
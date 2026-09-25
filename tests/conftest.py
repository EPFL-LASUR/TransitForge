import pytest
from shapely.geometry import LineString

from transitforge.dataclasses import TransitRoute, TransitStop


@pytest.fixture
def stops():
    return [
        TransitStop(
            osm_id=101,
            lat=46.0000,
            lon=7.0000,
            name="A",
            role="platform",
        ),
        TransitStop(
            osm_id=102,
            lat=46.0050,
            lon=7.0050,
            name="B",
            role="platform",
        ),
        TransitStop(
            osm_id=103,
            lat=46.0100,
            lon=7.0100,
            name="C",
            role="platform",
        ),
    ]


@pytest.fixture
def route(stops):
    return TransitRoute(
        relation_id=1,
        route_type="bus",
        ref="12",
        name="Test route",
        from_name="A",
        to_name="C",
        operator="Test operator",
        network="Test network",
        stops=stops,
        geometry=LineString(
            [
                (7.0000, 46.0000),
                (7.0050, 46.0050),
                (7.0100, 46.0100),
            ]
        ),
    )


@pytest.fixture
def reverse_route():
    stops = [
        TransitStop(
            osm_id=201,
            lat=46.0100,
            lon=7.0100,
            name="C",
            role="platform",
        ),
        TransitStop(
            osm_id=202,
            lat=46.0050,
            lon=7.0050,
            name="B",
            role="platform",
        ),
        TransitStop(
            osm_id=203,
            lat=46.0000,
            lon=7.0000,
            name="A",
            role="platform",
        ),
    ]

    return TransitRoute(
        relation_id=2,
        route_type="bus",
        ref="12",
        name="Test route reverse",
        from_name="C",
        to_name="A",
        stops=stops,
        geometry=LineString(
            [
                (7.0100, 46.0100),
                (7.0050, 46.0050),
                (7.0000, 46.0000),
            ]
        ),
    )


@pytest.fixture
def leg():
    return {
        "type": "vehicle",
        "route": "12",
        "routeType": "BUS",
        "from": {
            "name": "A",
            "lat": 46.0000,
            "lon": 7.0000,
            "platform": None,
        },
        "to": {
            "name": "C",
            "lat": 46.0100,
            "lon": 7.0100,
            "platform": None,
        },
        "departureTime": "08:00",
        "arrivalTime": "08:15",
    }


@pytest.fixture
def stop_sequence():
    return [
        (46.0000, 7.0000),
        (46.0050, 7.0050),
        (46.0100, 7.0100),
    ]
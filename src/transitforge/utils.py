from math import asin, cos, radians, sin, sqrt
from typing import Any
from shapely.geometry import LineString, MultiLineString
import re


def _haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:


    R = 6_371_000.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)


    dlat = lat2 - lat1
    dlon = lon2 - lon1


    a = (
        sin(dlat / 2) ** 2
        +
        cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )


    return (
        2
        * R
        * asin(sqrt(a))
    )


def _point_to_route_distance_m(
    point: tuple[float, float],
    geometry: LineString | MultiLineString,
) -> float:
    """
    Approximate minimum distance between a WGS84 point and
    a route geometry in metres.

    Geometry vertices rather than Shapely's .distance() are used
    because the geometry is stored in lon/lat degrees.
    """

    lat, lon = point

    if isinstance(geometry, LineString):
        lines = [geometry]

    elif isinstance(geometry, MultiLineString):
        lines = list(geometry.geoms)

    else:
        return float("inf")

    best_distance = float("inf")

    for line in lines:

        for lon2, lat2 in line.coords:

            distance = _haversine(lat, lon, lat2, lon2)

            if distance < best_distance:
                best_distance = distance

    return best_distance

def _normalize_ref(
    value: Any,
) -> str:

    if value is None:
        return ""

    value = str(value).strip().upper()

    value = re.sub(r"\s+", "", value)

    return value
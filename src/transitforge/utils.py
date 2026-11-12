from math import asin, cos, radians, sin, sqrt
from typing import Any
from shapely.geometry import LineString, MultiLineString
import re


import re
from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiLineString

_R = 6_371_000.0


def _haversine(
    lat1,
    lon1,
    lat2,
    lon2,
    radians: bool = False,
):
    """
    Great-circle distance in metres.

    Works on scalars and on numpy arrays (broadcasting). If radians is
    True the inputs are already in radians.
    """

    if not radians:
        lat1, lon1, lat2, lon2 = (
            np.deg2rad(v) for v in (lat1, lon1, lat2, lon2)
        )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * _R * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


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
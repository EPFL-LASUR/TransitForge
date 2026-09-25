from dataclasses import dataclass, field
from shapely.geometry import LineString, MultiLineString


@dataclass(slots=True)
class TransitStop:
    """
    Lightweight representation of a route stop.

    slots=True reduces Python object memory overhead.
    """

    osm_id: int

    lat: float
    lon: float

    name: str | None = None

    role: str = ""

@dataclass(slots=True)
class TransitRoute:
    """
    One concrete OSM route relation / variant.

    Examples:

        Bus 1 A -> B
        Bus 1 B -> A

    => two different TransitRoute objects.

    To save RAM, we deliberately do not store:
        - node references
        - individual OSM way geometries
        - raw OSM objects
    """

    relation_id: int
    route_type: str

    ref: str | None = None
    name: str | None = None

    from_name: str | None = None
    to_name: str | None = None

    operator: str | None = None
    network: str | None = None

    nat_ref: str | None = None
    old_ref: str | None = None

    service: str | None = None
    highspeed: bool = False

    stops: list[TransitStop] = field(default_factory=list)
    geometry: LineString | MultiLineString | None = None
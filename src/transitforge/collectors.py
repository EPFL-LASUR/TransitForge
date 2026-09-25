import osmium
import osmium.geom
from shapely import wkb
from shapely.geometry import LineString

from .dataclasses import TransitStop


class RouteMetadataCollector(osmium.SimpleHandler):
    """
    Lightweight pass collecting only relation metadata.

    Geometry is collected later on
    """

    def __init__(
        self,
        route_types: set[str],
    ):

        super().__init__()
        self.route_types = route_types
        self.routes: dict[int, dict] = {}


    def relation(self, relation):

        if relation.tags.get("type") != "route":
            return

        route_type = relation.tags.get("route")

        if route_type not in self.route_types:
            return


        self.routes[relation.id] = {
            "route_type": route_type,
            "ref": relation.tags.get("ref"),
            "name": relation.tags.get("name"),
            "from_name": relation.tags.get("from"),
            "to_name": relation.tags.get("to"),
            "operator": relation.tags.get("operator"),
            "network": relation.tags.get("network"),
        }



class StreamingWayGeometryCollector(osmium.SimpleHandler):
    """
    Extract geometries only for the required way IDs.

    Osmium resolves node locations while streaming the PBF.
    Shapely LineStrings are constructed directly from the resolved
    node coordinates.
    """

    def __init__(
        self,
        required_way_ids: set[int],
    ):
        super().__init__()

        self.required_way_ids = required_way_ids
        self.geometries: dict[int, LineString] = {}

        self.seen_required = 0
        self.failed = 0

    def way(self, way):

        if way.id not in self.required_way_ids:
            return

        self.seen_required += 1

        try:
            coords = [
                (node.lon, node.lat)
                for node in way.nodes
                if node.location.valid()
            ]

            if len(coords) < 2:
                self.failed += 1
                return

            self.geometries[way.id] = LineString(coords)

        except Exception as e:
            self.failed += 1

            if self.failed <= 10:
                print(
                    f"Failed way {way.id}: "
                    f"{type(e).__name__}: {e}"
                )



class StreamingStopCollector(osmium.SimpleHandler):
    """
    Extract coordinates only for route stop nodes.

    Unlike extracting all transit stops from the PBF,
    this only stores nodes actually referenced by routes.
    """

    def __init__(
        self,
        required_stop_ids: set[int],
    ):

        super().__init__()
        self.required_stop_ids = required_stop_ids

        self.stops: dict[int, TransitStop] = {}


    def node(self, node):

        if node.id not in self.required_stop_ids:
            return

        try:
            lat = node.location.lat
            lon = node.location.lon

        except Exception:
            return


        self.stops[node.id] = TransitStop(
            osm_id=node.id,
            lat=lat,
            lon=lon,
            name=node.tags.get("name"),
        )
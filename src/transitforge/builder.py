from pathlib import Path
from shapely.geometry import LineString, MultiLineString, Point

from .dataclasses import TransitRoute
from .collectors import StreamingStopCollector, StreamingWayGeometryCollector, RouteMetadataCollector
from .network import TransitRouteNetwork

from extractosm.transit import get_route_stop_mapping, get_route_way_roles



class TransitRouteBuilder:
    """
    Memory-conscious builder for exact OSM transit routes.

    Main design:

        1. Read lightweight route metadata.
        2. Use the package extractosm to get:
            route -> ordered stops
            route -> ordered ways
        3. Stream through the PBF once for stop coordinates.
        4. Stream through the PBF once for required way geometry.
        5. Immediately discard raw OSM data.
        6. Build final route geometries.
        7. Serialize only the final network.

    Example
    -------

        builder = TransitRouteBuilder(
            "switzerland.osm.pbf"
        )

        network = builder.build()

        network.save(
            "transit_network.pkl"
        )
    """

    def __init__(
        self,
        pbf_file: str | Path,
        route_types: list[str] | None = None,
    ):

        # Constants
        self.DEFAULT_ROUTE_TYPES = [
            "train",
            "bus",
            "tram",
            "metro",
            "subway",
            "trolleybus",
            "light_rail",
            "funicular",
            "monorail",
            "aerialway",
            "ferry",
        ]

        self.PLATFORM_ROLES = {
            "platform",
            "platform_entry_only",
            "platform_exit_only",
        }

        self.pbf_file = str(pbf_file)
        self.route_types = (
            route_types
            if route_types is not None
            else self.DEFAULT_ROUTE_TYPES
        )



    # Main build
    def build(self) -> "TransitRouteNetwork":

        # Pass 1: route metadata
        print("1/4 Collecting route metadata...")

        metadata_collector = RouteMetadataCollector(set(self.route_types))

        metadata_collector.apply_file(self.pbf_file, locations=False)

        route_metadata = metadata_collector.routes

        print(f"Found {len(route_metadata)} route variants")


        # Pass 2: use  existing lightweight extractors
        print("2/4 Collecting route members...")

        route_stop_mapping = get_route_stop_mapping(
                osm_pbf_path=self.pbf_file,
                route_types=self.route_types,
            )
        

        route_way_roles = get_route_way_roles(
                osm_pbf_path=self.pbf_file,
                route_types=self.route_types,
            )
        

        # Filter platform ways
        route_way_ids: dict[int, list[int]] = {}

        all_way_ids: set[int] = set()

        for route_id, members in route_way_roles.items():
            way_ids = [way_id for way_id, role in members if role not in self.PLATFORM_ROLES]
            route_way_ids[route_id] = way_ids
            all_way_ids.update(way_ids)



        # All required stop IDs

        all_stop_ids = set()

        for stop_ids in route_stop_mapping.values():
            all_stop_ids.update(stop_ids)


        print(f"Required ways: {len(all_way_ids)}")
        print(f"Required stop nodes: {len(all_stop_ids)}")



        # Pass 3: stream only route stops
        print("3/4 Extracting stop coordinates...")

        stop_collector = StreamingStopCollector(all_stop_ids)


        stop_collector.apply_file(
            self.pbf_file,
            locations=True,
        )

        stop_lookup = stop_collector.stops

        print(f"Collected {len(stop_lookup)} stop coordinates")

        # Free this potentially large set early.
        del all_stop_ids


        # Pass 4: Stream way geometries
        print("4/4 Extracting route geometries...")

        geometry_collector = StreamingWayGeometryCollector(all_way_ids)

        geometry_collector.apply_file(
            self.pbf_file,
            locations=True,
            idx="sparse_file_array",
        )


        way_geometries = geometry_collector.geometries


        print(f"Collected {len(way_geometries)} way geometries")

        missing_ways = all_way_ids - way_geometries.keys()
        

        if missing_ways:
            print(f"WARNING: {len(missing_ways)} required ways had no geometry (outside extract bbox / incomplete). Affected routes will have gaps.")


        # We no longer need the global set.
        del all_way_ids


        # Build final routes
        print("Building final route objects...")

        routes: dict[int, TransitRoute] = {}

        for route_id, metadata in route_metadata.items():

            # Stops
            stops = []

            for stop_id in route_stop_mapping.get(route_id, []):
                stop = stop_lookup.get(stop_id)

                if stop is not None:
                    stops.append(stop)

            # Geometry
            ordered_way_ids = route_way_ids.get(route_id, [])
            
            geometry = self._build_route_geometry(ordered_way_ids, way_geometries)
            

            # Store route
            routes[route_id] = TransitRoute(
                relation_id=route_id,
                route_type=metadata["route_type"],
                ref=metadata["ref"],
                name=metadata["name"],
                from_name=metadata["from_name"],
                to_name=metadata["to_name"],
                operator=metadata["operator"],
                network=metadata["network"],
                stops=stops,
                geometry=geometry,
            )

        # Free intermediate data
        del route_metadata
        del route_stop_mapping
        del route_way_roles
        del route_way_ids
        del stop_lookup
        del way_geometries


        print(f"Built {len(routes)} route variants")

        return TransitRouteNetwork(routes)

    # BUILD ROUTE GEOMETRY
    # --------------------------------------------------------

    @staticmethod
    def _build_route_geometry(
        ordered_way_ids: list[int],
        way_geometries: dict[int, LineString,],
        _DEDUP_EPS: int = 1e-9
    ) -> LineString | MultiLineString | None:

        """
        Build a route geometry from ordered OSM way members.

        Preserves relation order and tries to orient each way to
        continue from where the previous one left off.

        Small gaps (smaller than _DEDUP_EPS) between segments are concatenated into
        one continuous line, which prevents silently dropping whole segments

        """

        lines = []

        previous_end = None


        for way_id in ordered_way_ids:

            line = way_geometries.get(way_id)

            if line is None:
                continue

            coords = list(line.coords)

            if len(coords) < 2:
                continue

            # Orient the line according to the previous way
            if previous_end is not None:
                first = coords[0]
                last = coords[-1]

                distance_to_first = Point(previous_end).distance(Point(first))
                distance_to_last = Point(previous_end).distance(Point(last))

                if distance_to_last < distance_to_first:
                    coords.reverse()

            lines.append(coords)
            previous_end = coords[-1]


        if not lines:
            return None

        # Concatenate all oriented segments into one line.
        #
        # The point at each boundary are deduped if segments share almost
        # the same coordinates
        merged_coords: list[tuple[float, float]] = []

        for coords in lines:
            if merged_coords:

                last_pt = merged_coords[-1]
                first_pt = coords[0]

                if (
                    abs(last_pt[0] - first_pt[0]) < _DEDUP_EPS
                    and abs(last_pt[1] - first_pt[1]) < _DEDUP_EPS
                ):
                    coords = coords[1:]

            merged_coords.extend(coords)


        if len(merged_coords) < 2:
            return None

        # Collapse repeated loops (roundabouts, etc.)
        merged_coords = TransitRouteBuilder._remove_repeated_loops(merged_coords)

        if len(merged_coords) < 2:
            return None

        return LineString(merged_coords)


    @staticmethod
    def _remove_repeated_loops(
        coords: list[tuple[float, float]],
        decimals: int = 6,
        max_loop_fraction: float = 0.2,   # never cut more than 20% of accumulated points as a "loop"
    ) -> list[tuple[float, float]]:

        seen: dict[tuple[float, float], int] = {}
        result: list[tuple[float, float]] = []

        for pt in coords:
            key = (round(pt[0], decimals), round(pt[1], decimals))

            if key in seen:
                cut_index = seen[key]
                loop_len = len(result) - cut_index

                # Only treat it as a loop if it's small relative to what has been built so far
                if loop_len <= max(3, int(len(result) * max_loop_fraction)):
                    result = result[: cut_index + 1]
                    seen = {(round(p[0], decimals), round(p[1], decimals)): i for i, p in enumerate(result)}
                    continue

            seen[key] = len(result)
            result.append(pt)

        return result
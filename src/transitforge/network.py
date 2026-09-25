import pickle
from collections import defaultdict
from pathlib import Path
from shapely.geometry import MultiLineString, Point
from shapely.ops import substring

from .dataclasses import TransitRoute, TransitStop
from .utils import _haversine, _normalize_ref, _point_to_route_distance_m


class TransitRouteNetwork:
    """
    Lightweight serialized transit network.

    Stores final route variants and indexes them by normalized OSM ref.

    Can match a simple start/end leg, and entire ordered GTFS stop sequence
    against one OSM TransitRoute to reconstruct geometry between every
    consecutive GTFS stop.
     """

    def __init__(
        self,
        routes: dict[int, TransitRoute],
    ):
        self.routes = routes
        self._build_indexes()

        # Mapping from GTFS route types to OSM route types.
        self.GTFS_TYPE_TO_OSM_ROUTES: dict[str, set[str]] = {
            "TRAM": {"tram", "light_rail"},
            "CABLE_TRAM": {"tram"},
            "SUBWAY": {"subway", "light_rail", "monorail"},
            "METRO": {"subway", "light_rail", "monorail"},
            "RAIL": {"train"},
            "TRAIN": {"train"},
            "BUS": {"bus", "trolleybus"},
            "TROLLEYBUS": {"trolleybus", "bus"},
            "FERRY": {"ferry"},
            "CABLE_CAR": {"aerialway"},
            "GONDOLA": {"aerialway"},
            "FUNICULAR": {"funicular"},
        }

    def _build_indexes(self):
        """
        Build indexes used for route matching.

        Routes are indexed by normalized ref. Route-type filtering is
        performed later because GTFS and OSM mode taxonomies don't align
        exactly 1:1.
        """

        self.by_ref = defaultdict(list)

        for route_id, route in self.routes.items():
            if not route.ref:
                continue

            self.by_ref[_normalize_ref(route.ref)].append(route_id)

    def save(
        self,
        path: str | Path,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved network to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "TransitRouteNetwork":

        with open(path, "rb") as f:
            network = pickle.load(f)

        if not hasattr(network, "by_ref"):
            network._build_indexes()

        return network


    def _nearest_stop(
        self,
        stops: list[TransitStop],
        coordinate: tuple[float, float],
    ) -> tuple[int, float]:
        """
        Return the nearest stop and its distance.

        Returns:
            (index, distance_m)
        """

        lat, lon = coordinate

        best_index = -1
        best_distance = float("inf")

        for i, stop in enumerate(stops):
            distance = _haversine(lat, lon, stop.lat, stop.lon)

            if distance < best_distance:
                best_distance = distance
                best_index = i

        return best_index, best_distance

    def _match_stop_sequence(
        self,
        route: TransitRoute,
        stop_coords: list[tuple[float, float]],
        max_stop_distance_m: float,
    ) -> tuple[list[int], float] | None:
        """
        Match an ordered GTFS stop sequence against an OSM route.

        Important:

        GTFS stops do NOT have to correspond one-to-one with every OSM
        stop. OSM may contain additional stops.

        Example:

            GTFS:
                A -> C -> E

            OSM:
                A -> B -> C -> D -> E

        This is valid and produces:

            A(index 0) -> C(index 2) -> E(index 4)

        The matched OSM stop indexes must be strictly increasing.

        Returns:
            ([osm_index_1, osm_index_2, ...], total_distance)

        or None if the sequence cannot be matched.
        """

        if not route.stops:
            return None

        if len(stop_coords) < 2:
            return None

        matched_indices: list[int] = []
        total_distance = 0.0

        previous_index = -1

        for coordinate in stop_coords:

            best_index = -1
            best_distance = float("inf")

            # Only search forward. This guarantees that the
            # GTFS stop order agrees with the direction of the OSM route.
            for i in range(previous_index + 1, len(route.stops),):
                osm_stop = route.stops[i]

                distance = _haversine(
                    coordinate[0],
                    coordinate[1],
                    osm_stop.lat,
                    osm_stop.lon,
                )

                if distance < best_distance:
                    best_distance = distance
                    best_index = i

            if best_index == -1:
                return None

            if best_distance > max_stop_distance_m:
                return None

            matched_indices.append(best_index)
            total_distance += best_distance
            previous_index = best_index

        return matched_indices, total_distance

    def _score_route_for_leg(
        self,
        route: TransitRoute,
        start: tuple[float, float],
        end: tuple[float, float],
        max_distance: float,
    ) -> float | None:
        """
        Score a route using only its first and last endpoints. Prefers:

        1. stops close to start/end
        2. correct stop order

        This is used for single leg route matching.
        """

        if not route.stops:
            return None

        start_index, start_distance = self._nearest_stop(route.stops, start)
        end_index, end_distance = self._nearest_stop(route.stops, end)

        if start_distance > max_distance:
            return None

        if end_distance > max_distance:
            return None

        if start_index > end_index:
            return None

        return start_distance + end_distance

    def _score_route_for_stop_sequence(
        self,
        route: TransitRoute,
        stop_coords: list[tuple[float, float]],
        max_stop_distance_m: float,
    ) -> float | None:
        """
        Score an OSM route against an entire ordered GTFS stop sequence.

        Lower is better.

        The score is the sum of the distances between every GTFS stop
        and its matched OSM stop.
        """

        result = self._match_stop_sequence(route, stop_coords, max_stop_distance_m)

        if result is None:
            return None

        _, total_distance = result

        return total_distance


    # Route matching
    def _filter_fallback_candidates(
        self,
        candidates: list[TransitRoute],
        requested_ref: str,
    ) -> list[TransitRoute]:
        """
        Narrow candidates when the routing engine's route identifier
        does not exactly match an OSM ref.

        Examples:

            TER  -> TER 10
            TGV  -> TGV InOui 071A
            IC5  -> IC 5
            IR35 -> IR 35
        """

        if not requested_ref:
            return candidates

        generic_refs = {
            "TER",
            "TGV",
            "IC",
            "IR",
            "RE",
            "EC",
            "R",
        }

        if requested_ref in generic_refs:

            matching = []

            for route in candidates:

                metadata = " ".join(
                    [
                        route.ref or "",
                        route.name or "",
                        route.network or "",
                        route.operator or "",
                        route.nat_ref or "",
                        route.old_ref or "",
                    ]
                )

                if requested_ref in _normalize_ref(metadata):
                    matching.append(route)

            return matching or candidates

        matching = []

        for route in candidates:

            route_refs = {
                _normalize_ref(route.ref),
                _normalize_ref(route.nat_ref),
                _normalize_ref(route.old_ref),
            }

            route_refs.discard("")

            if requested_ref in route_refs:
                matching.append(route)
                continue

            for candidate_ref in route_refs:

                if (
                    requested_ref in candidate_ref
                    or candidate_ref in requested_ref
                ):
                    matching.append(route)
                    break

        return matching or candidates

    def _compatible_candidates(
        self,
        leg: dict,
    ) -> list[TransitRoute]:
        """
        Return OSM route candidates based on route ref and GTFS route type.

        Can be used for single leg and full stop sequence matching.

        """

        ref = _normalize_ref(leg.get("route"))

        gtfs_label = str(
            leg.get("routeType") or ""
        ).strip().upper()

        accepted_osm_types = self.GTFS_TYPE_TO_OSM_ROUTES.get(
            gtfs_label
        )


        # Exact ref candidates
        candidate_ids = self.by_ref.get(ref, [])

        if accepted_osm_types is not None:
            candidate_ids = [
                route_id
                for route_id in candidate_ids
                if (
                    self.routes[route_id].route_type or ""
                ).strip().lower() in accepted_osm_types
            ]

        candidates = [
            self.routes[route_id]
            for route_id in candidate_ids
        ]

        if candidates:
            return candidates

        # Fallback by compatible route type
        if accepted_osm_types is None:
            return []

        fallback_candidates = [
            route
            for route in self.routes.values()
            if (
                route.route_type or ""
            ).strip().lower() in accepted_osm_types
        ]

        if not fallback_candidates:
            return []

        return self._filter_fallback_candidates(
            fallback_candidates,
            ref,
        )

    def _best_stop_route(
        self,
        candidates: list[TransitRoute],
        start: tuple[float, float],
        end: tuple[float, float],
        max_stop_distance_m: float,
    ) -> TransitRoute | None:
        """
        Find the best route for a simple start/end leg.
        """

        best_route = None
        best_score = float("inf")

        for route in candidates:

            score = self._score_route_for_leg(
                route,
                start,
                end,
                max_stop_distance_m,
            )

            if score is None:
                continue

            if score < best_score:
                best_score = score
                best_route = route

        return best_route

    def _best_stop_sequence_route(
        self,
        candidates: list[TransitRoute],
        stop_coords: list[tuple[float, float]],
        max_stop_distance_m: float,
    ) -> tuple[TransitRoute, list[int]] | None:
        """
        Find the best OSM route for an entire ordered GTFS stop sequence.

        Returns:
            (route, matched_osm_stop_indices)

        or None.
        """

        best_route = None
        best_indices = None
        best_score = float("inf")

        for route in candidates:

            result = self._match_stop_sequence(
                route,
                stop_coords,
                max_stop_distance_m,
            )

            if result is None:
                continue

            matched_indices, score = result

            if score < best_score:
                best_score = score
                best_route = route
                best_indices = matched_indices

        if best_route is None or best_indices is None:
            return None

        return best_route, best_indices

    def _best_geometry_route(
        self,
        candidates: list[TransitRoute],
        start: tuple[float, float],
        end: tuple[float, float],
        max_geometry_distance_m: float,
    ) -> TransitRoute | None:
        """
        Find the best geometry for a specific route using only
        the start/end points.
        """

        best_route = None
        best_score = float("inf")

        for route in candidates:

            if route.geometry is None:
                continue

            if route.geometry.is_empty:
                continue

            start_distance = _point_to_route_distance_m(start, route.geometry)
            end_distance = _point_to_route_distance_m(end, route.geometry)

            if start_distance > max_geometry_distance_m:
                continue

            if end_distance > max_geometry_distance_m:
                continue

            score = start_distance + end_distance

            if score < best_score:
                best_score = score
                best_route = route

        return best_route


    # Diagnostics
    def _describe_no_match(
        self,
        leg: dict,
        max_stop_distance_m: float,
    ) -> str:
        """
        Build a diagnostic message when no route matched a leg.
        """

        ref = _normalize_ref(leg.get("route"))

        gtfs_label = str(
            leg.get("routeType") or ""
        ).strip().upper()

        same_ref_ids = self.by_ref.get(ref, [])

        if not same_ref_ids:
            return (
                f"No route with ref '{leg.get('route')}' exists in "
                "this network at all (check the ref, or whether "
                "this route is inside your PBF extract)."
            )

        existing_types = sorted(
            {
                self.routes[rid].route_type
                for rid in same_ref_ids
            }
        )

        accepted = self.GTFS_TYPE_TO_OSM_ROUTES.get(
            gtfs_label
        )

        return (
            f"Ref '{leg.get('route')}' exists tagged in OSM as "
            f"{existing_types}, but GTFS routeType "
            f"'{leg.get('routeType')}' "
            + (
                f"only accepts OSM types {sorted(accepted)}"
                if accepted is not None
                else (
                    "is not in GTFS_TYPE_TO_OSM_ROUTES, so no type "
                    "filtering was applied"
                )
            )
            + f". None of the candidate stops were within "
            f"{max_stop_distance_m}m of the leg's start/end, or the "
            "stop order didn't match the travel direction."
        )

    # Single-leg API
    def find_route_for_leg(
        self,
        leg: dict,
        max_stop_distance_m: float = 750.0,
        max_geometry_distance_m: float = 5000.0,
    ) -> TransitRoute | None:
        """
        Find the best OSM route for a simple start -> end leg.

        For reconstructing GTFS shapes from a complete stop sequence,
        use get_geometry_for_stop_sequence() instead.
        """

        start = (
            leg["from"]["lat"],
            leg["from"]["lon"],
        )

        end = (
            leg["to"]["lat"],
            leg["to"]["lon"],
        )

        candidates = self._compatible_candidates(leg)

        if candidates:

            # First try stop-based matching.
            best_route = self._best_stop_route(
                candidates,
                start,
                end,
                max_stop_distance_m,
            )

            if best_route is not None:
                return best_route

            # Then geometry-based fallback.
            best_route = self._best_geometry_route(
                candidates,
                start,
                end,
                max_geometry_distance_m,
            )

            if best_route is not None:
                return best_route

        return None

    # Geometry extraction
    def geometry_between(
        self,
        route: TransitRoute,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> list[list[float]]:
        """
        Return route geometry between start and end.

        Coordinates are returned as:

            [[lat, lon], [lat, lon], ...]
        """

        geometry = route.geometry

        if geometry is None:
            return []

        if geometry.is_empty:
            return []

        # MultiLineString
        if isinstance(geometry, MultiLineString):

            lines = list(geometry.geoms)

            if not lines:
                return []

            start_point = Point(start[1], start[0])
            end_point = Point(end[1], end[0])

            geometry = min(
                lines,
                key=lambda line: (
                    line.distance(start_point)
                    + line.distance(end_point)
                ),
            )


        # Project endpoints onto route geometry
        start_point = Point(start[1], start[0])
        end_point = Point(end[1], end[0])

        start_distance = geometry.project(start_point)
        end_distance = geometry.project(end_point)


        # Extract in route direction
        if start_distance <= end_distance:

            result = substring(geometry, start_distance, end_distance,)

        else:
            result = substring(geometry, end_distance, start_distance)

            if result.is_empty:
                return []

            coords = list(result.coords)
            coords.reverse()

            return [
                [lat, lon]
                for lon, lat in coords
            ]

        if result.is_empty:
            return []

        if result.geom_type == "Point":
            lon, lat = result.coords[0]

            return [
                [lat, lon]
            ]

        return [
            [lat, lon]
            for lon, lat in result.coords
        ]


    def get_geometry_for_leg(
        self,
        leg: dict,
        max_stop_distance_m: float = 750,
    ) -> list[list[float]]:
        """
        Given one routing leg, return OSM route geometry between origin
        and destination.

        This remains a simple start -> end API.

        Example:

            {
                "route": "RE33",
                "routeType": "RAIL",
                "from": {
                    "lat": 46.0,
                    "lon": 7.0
                },
                "to": {
                    "lat": 46.1,
                    "lon": 7.1
                }
            }
        """

        route = self.find_route_for_leg(
            leg,
            max_stop_distance_m,
        )

        if route is None:
            raise ValueError(
                self._describe_no_match(
                    leg,
                    max_stop_distance_m,
                )
            )

        start = (
            leg["from"]["lat"],
            leg["from"]["lon"],
        )

        end = (
            leg["to"]["lat"],
            leg["to"]["lon"],
        )

        return self.geometry_between(
            route,
            start,
            end,
        )


    def get_geometry_for_stop_sequence(
        self,
        leg: dict,
        stops: list[tuple[float, float]],
        max_stop_distance_m: float = 750.0,
        max_geometry_distance_m: float = 5000.0,
    ) -> list[list[float]]:
        """
        Reconstruct geometry for an entire ordered GTFS stop sequence.

        Example:

            stops = [
                (lat_A, lon_A),
                (lat_B, lon_B),
                (lat_C, lon_C),
                (lat_D, lon_D),
            ]

        The method:

            1. Finds candidate OSM route variants.
            2. Matches the COMPLETE GTFS stop sequence to one OSM route.
            3. Requires the matched OSM stops to occur in the same order.
            4. Extracts geometry:
                   A -> B
                   B -> C
                   C -> D
            5. Concatenates those pieces.

        Importantly, all segments use the SAME OSM route.

        If stop-based sequence matching fails, a geometry fallback is
        attempted using the first and last GTFS stops.
        """

        if len(stops) < 2:
            return []


        # Candidate routes
        candidates = self._compatible_candidates(leg)

        if not candidates:
            raise ValueError(
                self._describe_no_match(
                    leg,
                    max_stop_distance_m,
                )
            )

        # First choice: match the complete stop sequence
        result = self._best_stop_sequence_route(
            candidates,
            stops,
            max_stop_distance_m,
        )

        if result is not None:

            route, matched_indices = result

            geometry_parts: list[list[float]] = []

            for i in range(len(stops) - 1):

                start = stops[i]
                end = stops[i + 1]

                part = self.geometry_between(route, start, end)

                if not part:
                    continue

                if not geometry_parts:
                    geometry_parts.extend(part)
                    continue

                # Avoid duplicating the boundary coordinate.
                if geometry_parts[-1] == part[0]:
                    geometry_parts.extend(part[1:])
                else:
                    geometry_parts.extend(part)

            if len(geometry_parts) >= 2:
                return geometry_parts


        # Fallback: select route based on first/last geometry
        #
        # This is deliberately a fallback rather than the primary
        # matching mechanism.
        start = stops[0]
        end = stops[-1]

        route = self._best_geometry_route(
            candidates,
            start,
            end,
            max_geometry_distance_m,
        )

        if route is None:
            raise ValueError(
                "No OSM route matched the complete GTFS stop sequence "
                f"within {max_stop_distance_m}m per stop, and no "
                f"candidate route geometry was within "
                f"{max_geometry_distance_m}m of both endpoints."
            )

        geometry_parts = []

        for i in range(len(stops) - 1):

            part = self.geometry_between(route, stops[i], stops[i + 1])

            if not part:
                continue

            if not geometry_parts:
                geometry_parts.extend(part)
                continue

            if geometry_parts[-1] == part[0]:
                geometry_parts.extend(part[1:])
            else:
                geometry_parts.extend(part)

        return geometry_parts
import pickle
from collections import defaultdict
from math import asin, cos, radians, sqrt
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import MultiLineString, Point
from shapely.ops import substring

from .dataclasses import TransitRoute, TransitStop
from .utils import _haversine, _normalize_ref, _point_to_route_distance_m

_R = 6_371_000.0


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

    # ------------------------------------------------------------------
    # Indexes / serialization
    # ------------------------------------------------------------------

    def _build_indexes(self):
        """
        Build indexes used for route matching.

        Routes are indexed by normalized ref. Route-type filtering is
        performed later because GTFS and OSM mode taxonomies don't align
        exactly 1:1.

        Also builds a bounding box per route (from its stops) for cheap
        rejection, and empty caches that are filled lazily.
        """

        self.by_ref = defaultdict(list)
        self._bbox: dict[int, tuple[float, float, float, float]] = {}
        self._arrays: dict[int, tuple] = {}  # filled lazily
        self._cand_cache: dict[tuple, list] = {}
        self._type_routes: dict[str, list] = {}
        self._type_bbox: dict[str, tuple] = {}  # filled lazily

        for route_id, route in self.routes.items():
            if route.ref:
                self.by_ref[_normalize_ref(route.ref)].append(route_id)

            if route.stops:
                lats = [s.lat for s in route.stops]
                lons = [s.lon for s in route.stops]
                self._bbox[route_id] = (
                    min(lats),
                    max(lats),
                    min(lons),
                    max(lons),
                )

    def __getstate__(self):
        # Derived indexes/caches are rebuilt on load, not stored.
        state = self.__dict__.copy()
        for k in (
            "by_ref",
            "_bbox",
            "_arrays",
            "_cand_cache",
            "_type_routes",
            "_type_bbox",
        ):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_indexes()

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

    # ------------------------------------------------------------------
    # Simple helpers (single leg)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Stop sequence matching
    # ------------------------------------------------------------------

    def _match_stop_sequence(
        self,
        route: TransitRoute,
        stop_coords: list[tuple[float, float]],
        max_stop_distance_m: float,
    ) -> tuple[list[int], float] | None:
        """
        Pure-Python reference implementation (no longer used on the main
        path, kept for single-route checks and debugging).

        Match an ordered GTFS stop sequence against an OSM route. GTFS
        stops do not have to correspond one-to-one with every OSM stop;
        matched OSM stop indexes are strictly increasing.

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

            for i in range(previous_index + 1, len(route.stops)):
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

    def _route_arrays(self, route: TransitRoute):
        """
        Radian lat / lon / cos(lat) arrays of a route's stops, cached.
        """

        arr = self._arrays.get(route.relation_id)

        if arr is None:
            ll = np.radians(
                np.array(
                    [(s.lat, s.lon) for s in route.stops],
                    dtype=float,
                )
            )
            arr = (ll[:, 0].copy(), ll[:, 1].copy(), np.cos(ll[:, 0]))
            self._arrays[route.relation_id] = arr

        return arr

    def _match_prepared(
        self,
        route: TransitRoute,
        g: np.ndarray,
        g_cos: np.ndarray,
        max_stop_distance_m: float,
        return_progress: bool = False,
    ) -> tuple[list[int], float] | tuple[int, float | None, str] | None:

        lat_r, lon_r, cos_r = self._route_arrays(route)
        n_osm = lat_r.shape[0]

        matched: list[int] = []
        total = 0.0
        prev = -1

        for k in range(g.shape[0]):

            s = prev + 1

            if s >= n_osm:
                if return_progress:
                    return k, None, "out_of_osm_stops"
                return None

            # The haversine "a" term is monotonic in distance.
            a = (
                np.sin((lat_r[s:] - g[k, 0]) * 0.5) ** 2
                + g_cos[k]
                * cos_r[s:]
                * np.sin((lon_r[s:] - g[k, 1]) * 0.5) ** 2
            )

            i = int(a.argmin())
            d = 2 * _R * asin(sqrt(min(1.0, float(a[i]))))

            if d > max_stop_distance_m:
                if return_progress:
                    return k, d, "too_far"
                return None

            prev = s + i
            matched.append(prev)
            total += d

        if return_progress:
            return g.shape[0], None, "ok"

        return matched, total

    def _best_stop_sequence_route(
        self,
        candidates: list[TransitRoute],
        stop_coords: list[tuple[float, float]],
        max_stop_distance_m: float,
    ) -> tuple[TransitRoute, list[int]] | None:
        """
        Find the best OSM route for an entire ordered GTFS stop sequence.

        Each route is tried in the direction of the trip first. If that
        fails, the reversed stop sequence is tried, because OSM often only
        maps one direction of a line. The geometry cut afterwards
        (_stitch) handles the direction automatically.

        Routes whose stop bounding box is farther than max_stop_distance_m
        from the first or the last GTFS stop cannot match, and are skipped
        without running the detailed matching.

        Returns:
            (route, matched_osm_stop_indices)

        or None.
        """

        if len(stop_coords) < 2 or not candidates:
            return None

        g_deg = np.asarray(stop_coords, dtype=float)
        g = np.radians(g_deg)
        g_cos = np.cos(g[:, 0])

        g_rev = g[::-1]
        g_cos_rev = g_cos[::-1]

        lat_m = max_stop_distance_m / 111_000.0
        lon_m = lat_m / max(
            cos(radians(min(float(np.abs(g_deg[:, 0]).max()) + 1.0, 89.0))),
            1e-6,
        )

        (f_lat, f_lon), (l_lat, l_lon) = stop_coords[0], stop_coords[-1]

        best_route = None
        best_indices = None
        best_score = float("inf")

        for route in candidates:

            bb = self._bbox.get(route.relation_id)

            if bb is None:
                continue

            if not (
                bb[0] - lat_m <= f_lat <= bb[1] + lat_m
                and bb[2] - lon_m <= f_lon <= bb[3] + lon_m
            ):
                continue

            if not (
                bb[0] - lat_m <= l_lat <= bb[1] + lat_m
                and bb[2] - lon_m <= l_lon <= bb[3] + lon_m
            ):
                continue

            result = self._match_prepared(
                route,
                g,
                g_cos,
                max_stop_distance_m,
            )

            if result is None:
                result = self._match_prepared(
                    route,
                    g_rev,
                    g_cos_rev,
                    max_stop_distance_m,
                )

            if result is None:
                continue

            indices, score = result

            if score < best_score:
                best_route = route
                best_indices = indices
                best_score = score

        if best_route is None or best_indices is None:
            return None

        return best_route, best_indices

    def _score_route_for_stop_sequence(
        self,
        route: TransitRoute,
        stop_coords: list[tuple[float, float]],
        max_stop_distance_m: float,
    ) -> float | None:
        """
        Score an OSM route against an entire ordered GTFS stop sequence.
        Lower is better.
        """

        result = self._match_stop_sequence(
            route,
            stop_coords,
            max_stop_distance_m,
        )

        if result is None:
            return None

        _, total_distance = result

        return total_distance

    # ------------------------------------------------------------------
    # Route matching (candidates)
    # ------------------------------------------------------------------

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

        generic_refs = {"TER", "TGV", "IC", "IR", "RE", "EC", "R"}

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

    def _compute_compatible_candidates(
        self,
        ref: str,
        gtfs_label: str,
    ) -> list[TransitRoute]:
        """
        Return OSM route candidates based on normalized route ref and GTFS
        route type label (uncached).
        """

        accepted_osm_types = self.GTFS_TYPE_TO_OSM_ROUTES.get(gtfs_label)

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

        candidates = [self.routes[route_id] for route_id in candidate_ids]

        if candidates:
            return candidates

        # Fallback by compatible route type
        if accepted_osm_types is None:
            return []

        fallback_candidates = self._type_routes.get(gtfs_label)

        if fallback_candidates is None:
            fallback_candidates = [
                route
                for route in self.routes.values()
                if (
                    route.route_type or ""
                ).strip().lower() in accepted_osm_types
            ]
            self._type_routes[gtfs_label] = fallback_candidates

        if not fallback_candidates:
            return []

        return self._filter_fallback_candidates(
            fallback_candidates,
            ref,
        )

    def _compatible_candidates(
        self,
        leg: dict,
    ) -> list[TransitRoute]:
        """
        Return OSM route candidates based on route ref and GTFS route type.
        Results are cached per (ref, route type), since many stop patterns
        share the same line.
        """

        ref = _normalize_ref(leg.get("route"))

        gtfs_label = str(leg.get("routeType") or "").strip().upper()

        key = (ref, gtfs_label)

        hit = self._cand_cache.get(key)

        if hit is None:
            hit = self._compute_compatible_candidates(ref, gtfs_label)
            self._cand_cache[key] = hit

        return hit

    def _typed_bbox_index(
        self,
        label: str,
    ) -> tuple[list[int], np.ndarray]:
        """
        (route_ids, bbox array) for all routes with a stop bounding box
        whose OSM type is compatible with a GTFS route type label.
        Cached per label. If the label is unknown, no type filter is
        applied.

        bbox columns: min_lat, max_lat, min_lon, max_lon
        """

        idx = self._type_bbox.get(label)

        if idx is None:

            accepted = self.GTFS_TYPE_TO_OSM_ROUTES.get(label)

            ids = [
                rid for rid, route in self.routes.items()
                if rid in self._bbox
                and (
                    accepted is None
                    or (route.route_type or "").strip().lower() in accepted
                )
            ]

            arr = np.array(
                [self._bbox[rid] for rid in ids],
                dtype=float,
            ).reshape(-1, 4)

            idx = (ids, arr)
            self._type_bbox[label] = idx

        return idx

    def _spatial_candidates(
        self,
        label: str,
        stops: list[tuple[float, float]],
        max_stop_distance_m: float,
    ) -> list[TransitRoute]:
        """
        Routes of a compatible type whose stop bounding box contains both
        the first and the last GTFS stop. The ref is ignored.
        """

        ids, bb = self._typed_bbox_index(label)

        if not ids:
            return []

        lat_m = max_stop_distance_m / 111_000.0
        lon_m = lat_m / max(
            cos(radians(min(max(abs(s[0]) for s in stops) + 1.0, 89.0))),
            1e-6,
        )

        def inside(p):
            return (
                (bb[:, 0] - lat_m <= p[0])
                & (p[0] <= bb[:, 1] + lat_m)
                & (bb[:, 2] - lon_m <= p[1])
                & (p[1] <= bb[:, 3] + lon_m)
            )

        mask = inside(stops[0]) & inside(stops[-1])

        return [self.routes[ids[i]] for i in np.flatnonzero(mask)]

    def _routes_with_stop_near(
        self,
        label: str,
        point: tuple[float, float],
        max_distance_m: float,
        limit: int = 30,
    ) -> list[TransitRoute]:
        """
        Routes of a compatible type that have an OSM stop within
        max_distance_m of a point, closest first (at most `limit`).
        """

        ids, bb = self._typed_bbox_index(label)

        if not ids:
            return []

        lat_m = max_distance_m / 111_000.0
        lon_m = lat_m / max(
            cos(radians(min(abs(point[0]) + 1.0, 89.0))),
            1e-6,
        )

        mask = (
            (bb[:, 0] - lat_m <= point[0])
            & (point[0] <= bb[:, 1] + lat_m)
            & (bb[:, 2] - lon_m <= point[1])
            & (point[1] <= bb[:, 3] + lon_m)
        )

        p_lat = radians(point[0])
        p_lon = radians(point[1])

        found: list[tuple[float, TransitRoute]] = []

        for i in np.flatnonzero(mask):

            route = self.routes[ids[i]]
            la, lo, _ = self._route_arrays(route)

            d = float(_haversine(p_lat, p_lon, la, lo, radians=True).min())

            if d <= max_distance_m:
                found.append((d, route))

        found.sort(key=lambda t: t[0])

        return [route for _, route in found[:limit]]

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

        Routes whose geometry bounds are clearly farther than the
        threshold are skipped before the expensive projection.
        """

        dlat = max_geometry_distance_m / 111_000.0 * 1.1
        dlon = dlat / max(
            cos(
                radians(
                    min(max(abs(start[0]), abs(end[0])) + 1.0, 89.0)
                )
            ),
            1e-6,
        )

        best_route = None
        best_score = float("inf")

        for route in candidates:

            geom = route.geometry

            if geom is None or geom.is_empty:
                continue

            minx, miny, maxx, maxy = geom.bounds

            if any(
                not (
                    miny - dlat <= lat <= maxy + dlat
                    and minx - dlon <= lon <= maxx + dlon
                )
                for lat, lon in (start, end)
            ):
                continue

            start_distance = _point_to_route_distance_m(start, geom)
            end_distance = _point_to_route_distance_m(end, geom)

            if start_distance > max_geometry_distance_m:
                continue

            if end_distance > max_geometry_distance_m:
                continue

            score = start_distance + end_distance

            if score < best_score:
                best_score = score
                best_route = route

        return best_route

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

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


    def diagnose_stop_sequence(
        self,
        leg: dict,
        stops: list[tuple[float, float]],
        max_stop_distance_m: float = 750.0,
        max_eval: int = 2000,
    ) -> dict:
        """
        Explain why a stop sequence found no route. Only meant to be
        called for patterns that already failed.

        The analysed pool is the same as the spatial fallback stage:
        every route of a compatible type, ignoring the ref. Every route
        near the first OR last stop is tried in both directions, and the
        one that gets furthest through the stop sequence is reported.
        """

        ref_raw = leg.get("route")
        ref = _normalize_ref(ref_raw)

        label = str(
            leg.get("routeType") or ""
        ).strip().upper()

        exact_ids = self.by_ref.get(ref, [])

        info = {
            "normalized_ref": ref,
            "osm_routes_with_same_ref": len(exact_ids),
            "osm_types_of_same_ref": ", ".join(
                sorted(
                    {
                        self.routes[r].route_type or "?"
                        for r in exact_ids
                    }
                )
            ),
        }

        ref_candidates = self._compatible_candidates(leg)

        info["n_candidates"] = len(ref_candidates)
        info["candidate_source"] = (
            "none"
            if not ref_candidates
            else "ref"
            if ref_candidates[0].relation_id in set(exact_ids)
            else "type_fallback (ref not matched)"
        )

        ids, _ = self._typed_bbox_index(label)
        candidates = [self.routes[i] for i in ids]

        if not candidates:
            info["reason"] = "NO_CANDIDATES"
            info["detail"] = (
                f"no OSM route compatible with route type '{label}' "
                "in the network"
            )
            return info

        n = len(stops)

        g_deg = np.asarray(stops, dtype=float)
        g = np.radians(g_deg)
        g_cos = np.cos(g[:, 0])

        orientations = (
            ("forward", g, g_cos),
            ("reversed", g[::-1], g_cos[::-1]),
        )

        lat_m = max_stop_distance_m / 111_000.0
        lon_m = lat_m / max(
            cos(radians(min(float(np.abs(g_deg[:, 0]).max()) + 1.0, 89.0))),
            1e-6,
        )

        def near(bb, p):
            return (
                bb[0] - lat_m <= p[0] <= bb[1] + lat_m
                and bb[2] - lon_m <= p[1] <= bb[3] + lon_m
            )

        first, last = stops[0], stops[-1]
        n_first = n_last = n_both = evaluated = 0
        best = None

        for route in candidates:

            bb = self._bbox.get(route.relation_id)

            if bb is None:
                continue

            nf, nl = near(bb, first), near(bb, last)
            n_first += int(nf)
            n_last += int(nl)
            n_both += int(nf and nl)

            if not (nf or nl) or evaluated >= max_eval:
                continue

            evaluated += 1

            for name, gg, gc in orientations:

                k, d, status = self._match_prepared(
                    route,
                    gg,
                    gc,
                    max_stop_distance_m,
                    return_progress=True
                )

                key = (k, int(nf and nl))

                if best is None or key > best[0]:
                    best = (key, route, k, d, status, name)

        info.update(
            routes_near_first_stop=n_first,
            routes_near_last_stop=n_last,
            routes_near_both_ends=n_both,
        )

        if best is not None:
            _, route, k, d, status, direction = best

            info["best_relation"] = (f"https://www.openstreetmap.org/relation/{route.relation_id}")
            info["best_osm_ref"] = route.ref
            info["best_osm_name"] = route.name
            info["best_direction"] = direction
            info["matched_stops"] = f"{k}/{n}"

            if status != "ok":
                # Index in the original GTFS stop order.
                info["fail_stop_index"] = (
                    k if direction == "forward" else n - 1 - k
                )

                if d is not None:
                    info["fail_distance_m"] = round(d)

        if n_both == 0:

            if n_first == 0 and n_last == 0:
                info["reason"] = "NO_ROUTE_NEAR_EITHER_END"
            elif n_first == 0:
                info["reason"] = "NO_ROUTE_NEAR_FIRST_STOP"
            elif n_last == 0:
                info["reason"] = "NO_ROUTE_NEAR_LAST_STOP"
            else:
                info["reason"] = "NO_SINGLE_ROUTE_NEAR_BOTH_ENDS"

            info["detail"] = (
                f"no route of a compatible type has stops within "
                f"{max_stop_distance_m:.0f} m of the terminal stop(s) "
                "(outside the PBF extract, or a type mismatch)"
            )

        elif best is not None and best[4] == "too_far":
            info["reason"] = "STOP_TOO_FAR"
            info["detail"] = (
                "furthest route breaks at this GTFS stop: nearest "
                f"remaining OSM stop is {info.get('fail_distance_m')} m "
                f"away (limit {max_stop_distance_m:.0f} m)"
            )

        elif best is not None and best[4] == "out_of_osm_stops":
            info["reason"] = "STOP_ORDER_MISMATCH"
            info["detail"] = (
                "ran out of OSM stops: the OSM relation ends before the "
                "GTFS trip does"
            )

        else:
            info["reason"] = "GEOMETRY_EMPTY"
            info["detail"] = (
                "stops matched but the route has no usable geometry "
                "(ways missing in the PBF?)"
            )

        return info

    # ------------------------------------------------------------------
    # Single-leg API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Geometry extraction
    # ------------------------------------------------------------------

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

            result = substring(geometry, start_distance, end_distance)

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

    @staticmethod
    def _segment(
        geom,
        a: float,
        b: float,
    ) -> list[list[float]]:
        """
        Geometry between two already-projected distances along a
        LineString, returned in travel direction as [[lat, lon], ...].
        """

        rev = a > b

        result = substring(geom, b, a) if rev else substring(geom, a, b)

        if result.is_empty:
            return []

        coords = [[lat, lon] for lon, lat in result.coords]

        if rev:
            coords.reverse()

        return coords

    def _stitch(
        self,
        route: TransitRoute,
        stops: list[tuple[float, float]],
    ) -> list[list[float]]:
        """
        Concatenate geometry between every consecutive GTFS stop along
        the SAME OSM route. Each stop is projected only once (all at once,
        vectorised) instead of twice. Works in either direction of the
        OSM route.
        """

        geom = route.geometry

        if geom is None or geom.is_empty:
            return []

        if isinstance(geom, MultiLineString):
            parts = (
                self.geometry_between(route, stops[i], stops[i + 1])
                for i in range(len(stops) - 1)
            )
        else:
            pts = shapely.points(
                np.array([(s[1], s[0]) for s in stops], dtype=float)
            )
            d = shapely.line_locate_point(geom, pts)

            parts = (
                self._segment(geom, float(d[i]), float(d[i + 1]))
                for i in range(len(stops) - 1)
            )

        out: list[list[float]] = []

        for part in parts:

            if not part:
                continue

            # Avoid duplicating the boundary coordinate.
            if out and out[-1] == part[0]:
                out.extend(part[1:])
            else:
                out.extend(part)

        return out

    def _chain_two(
        self,
        label: str,
        p: tuple[float, float],
        q: tuple[float, float],
        max_stop_distance_m: float,
        link_distance_m: float = 300.0,
    ) -> list[list[float]] | None:
        """
        Connect two points that no single OSM route covers, using TWO
        routes that meet at a shared station.

        Example: Zweisimmen -> Bern, where OSM has Zweisimmen -> Spiez
        and Spiez -> Bern as separate relations.

        Route A must have a stop near p, route B a stop near q, and one
        stop of A must lie within link_distance_m of a stop of B (the
        junction). Among all such combinations, the junction giving the
        shortest straight-line detour p -> junction -> q is used.
        """

        routes_a = self._routes_with_stop_near(label, p, max_stop_distance_m)
        routes_b = self._routes_with_stop_near(label, q, max_stop_distance_m)

        if not routes_a or not routes_b:
            return None

        p_lat, p_lon = radians(p[0]), radians(p[1])
        q_lat, q_lon = radians(q[0]), radians(q[1])

        best = None  # (cost, route_a, route_b, junction_stop)

        for a in routes_a:

            a_lat, a_lon, _ = self._route_arrays(a)

            for b in routes_b:

                if b.relation_id == a.relation_id:
                    continue

                b_lat, b_lon, _ = self._route_arrays(b)

                # Distance between every stop of A and every stop of B.
                dist = _haversine(
                    a_lat[:, None],
                    a_lon[:, None],
                    b_lat[None, :],
                    b_lon[None, :],
                    radians=True
                )

                ok = np.flatnonzero(dist.min(axis=1) <= link_distance_m)

                if ok.size == 0:
                    continue

                cost = (
                    _haversine(p_lat, p_lon, a_lat[ok], a_lon[ok], radians=True)
                    + _haversine(a_lat[ok], a_lon[ok], q_lat, q_lon, radians=True)
                )

                k = int(cost.argmin())

                if best is None or cost[k] < best[0]:
                    best = (float(cost[k]), a, b, a.stops[int(ok[k])])

        if best is None:
            return None

        _, route_a, route_b, junction = best
        j = (junction.lat, junction.lon)

        first = self._stitch(route_a, [p, j])
        second = self._stitch(route_b, [j, q])

        out: list[list[float]] = []

        for part in (first, second):

            if not part:
                continue

            if out and out[-1] == part[0]:
                out.extend(part[1:])
            else:
                out.extend(part)

        return out if len(out) >= 2 else None

    def _segmented_geometry(
        self,
        label: str,
        stops: list[tuple[float, float]],
        max_stop_distance_m: float,
    ) -> list[list[float]] | None:
        """
        Partial matching for stop sequences that no single OSM route
        covers (e.g. a train running through several OSM relations, or a
        relation that ends before the trip does).

        Starting at the first stop, find the longest run of consecutive
        GTFS stops that one OSM route (of a compatible type, ref ignored)
        matches, cut the geometry for that run, then continue from its
        last stop. If not even two consecutive stops share a route, two
        routes meeting at a station are chained (_chain_two). Anything
        still uncovered is joined with a straight line.

        Returns None if nothing at all could be matched on OSM routes.
        """

        n = len(stops)
        out: list[list[float]] = []
        n_matched = 0

        def join(part: list[list[float]]):
            if not part:
                return
            if out and out[-1] == part[0]:
                out.extend(part[1:])
            else:
                out.extend(part)

        def match(i: int, j: int):
            sub = stops[i : j + 1]
            pool = self._spatial_candidates(label, sub, max_stop_distance_m)
            return self._best_stop_sequence_route(pool, sub, max_stop_distance_m)

        i = 0

        while i < n - 1:

            best = None

            result = match(i, i + 1)

            if result is not None:

                best = (i + 1, result[0])
                lo, hi = i + 2, n - 1

                # Prefix property: if a route matches stops i..j it also
                # matches i..j' for j' < j, so binary search works.
                while lo <= hi:
                    mid = (lo + hi) // 2
                    result = match(i, mid)

                    if result is not None:
                        best = (mid, result[0])
                        lo = mid + 1
                    else:
                        hi = mid - 1

            else:
                chained = self._chain_two(
                    label,
                    stops[i],
                    stops[i + 1],
                    max_stop_distance_m,
                )

                if chained is not None:
                    join(chained)
                    n_matched += 1
                    i += 1
                    continue

            part: list[list[float]] = []

            if best is not None:
                j, route = best
                part = self._stitch(route, stops[i : j + 1])

            if len(part) >= 2:
                join(part)
                n_matched += 1
                i = j
            else:
                join([list(stops[i]), list(stops[i + 1])])
                i += 1

        if n_matched == 0:
            return None

        return out

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
                "from": {"lat": 46.0, "lon": 7.0},
                "to": {"lat": 46.1, "lon": 7.1}
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

        Stage 1: routes selected by ref/type (cached per ref/type), full
                 stop-sequence match, either direction.
        Stage 2: every route of a compatible type with stops near BOTH
                 terminals, ref ignored, full stop-sequence match.
        Stage 3: segmented matching: the stop sequence is split into runs,
                 each run matched to its own OSM route; two routes meeting
                 at a station are chained; remaining gaps are bridged with
                 straight lines.
        Stage 4: geometry fallback using the first/last GTFS stops.

        In stages 1, 2 and 4 the geometry between consecutive GTFS stops
        is cut from the SAME OSM route and concatenated.
        """

        if len(stops) < 2:
            return []

        label = str(leg.get("routeType") or "").strip().upper()

        ref_candidates = self._compatible_candidates(leg)

        # Stage 1: ref-based candidates
        if ref_candidates:

            result = self._best_stop_sequence_route(ref_candidates, stops, max_stop_distance_m)

            if result is not None:

                coords = self._stitch(result[0], stops)

                if len(coords) >= 2:
                    return coords

        # Stage 2: spatial candidates, ref ignored
        spatial = self._spatial_candidates(label, stops, max_stop_distance_m)

        result = self._best_stop_sequence_route(spatial, stops, max_stop_distance_m)

        if result is not None:

            coords = self._stitch(result[0], stops)

            if len(coords) >= 2:
                return coords

        # Stage 3: segmented / partial matching
        coords = self._segmented_geometry(label, stops, max_stop_distance_m)

        if coords is not None and len(coords) >= 2:
            return coords

        # Stage 4: geometry fallback
        route = None

        for pool in (ref_candidates, spatial):

            if not pool:
                continue

            route = self._best_geometry_route(
                pool,
                stops[0],
                stops[-1],
                max_geometry_distance_m,
            )

            if route is not None:
                break

        if route is None:
            raise ValueError(self._describe_no_match(leg, max_stop_distance_m))

        return self._stitch(route, stops)

        # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    @staticmethod
    def _completeness(route: TransitRoute) -> tuple[int, float]:
        """
        Rank a route variant by how much of it is present: number of
        stops first, then geometry length. Used to pick between two
        versions of the same relation (e.g. clipped by different extracts).
        """

        geom = route.geometry
        length = 0.0 if geom is None or geom.is_empty else float(geom.length)

        return len(route.stops), length

    def merge(
        self,
        other: "TransitRouteNetwork",
        on_conflict: str = "most_complete",
        inplace: bool = False,
        verbose: bool = True,
    ) -> "TransitRouteNetwork":
        """
        Merge another network into this one.

        Routes are keyed by OSM relation id. Relations present in only
        one network are always kept. For a relation present in both,
        `on_conflict` decides:

            "most_complete" : keep the version with more stops (ties
                              broken by longer geometry). Default.
            "keep_self"     : keep the version already in this network.
            "keep_other"    : take the version from `other`.
            "raise"         : raise ValueError.

        Args:
            other: network to merge in (not modified).
            on_conflict: conflict policy, see above.
            inplace: modify and return self instead of returning a new
                network. Indexes and caches are rebuilt either way.
            verbose: print a short summary.

        Note: route objects are shared, not copied, between the inputs
        and the result.
        """

        if on_conflict not in {
            "most_complete",
            "keep_self",
            "keep_other",
            "raise",
        }:
            raise ValueError(f"Unknown on_conflict policy: {on_conflict!r}")

        merged = dict(self.routes)

        n_added = n_conflicts = n_replaced = 0

        for route_id, route in other.routes.items():

            current = merged.get(route_id)

            if current is None:
                merged[route_id] = route
                n_added += 1
                continue

            if current is route:
                continue

            n_conflicts += 1

            if on_conflict == "raise":
                raise ValueError(f"Relation {route_id} exists in both networks")

            if on_conflict == "keep_self":
                continue

            if (
                on_conflict == "keep_other"
                or self._completeness(route) > self._completeness(current)
            ):
                merged[route_id] = route
                n_replaced += 1

        if verbose:
            print(
                f"Merged networks: {len(self.routes)} + {len(other.routes)} "
                f"-> {len(merged)} routes "
                f"({n_added} new, {n_conflicts} shared, "
                f"{n_replaced} shared replaced)"
            )

        if inplace:
            self.routes = merged
            self._build_indexes()  # also clears the lazy caches
            return self

        return TransitRouteNetwork(merged)
# TransitForge

TransitForge is a Python package for extracting and reconstructing public transport networks from OpenStreetMap (OSM) data.

It takes an OSM `.osm.pbf` file as input and produces a serialized transit network containing route metadata, stops, and route geometries. The resulting network can then be used to match public transport legs to OSM routes and reconstruct their geometry.

## Network extraction

The network is built using `TransitRouteBuilder`:

```python
from transitforge import TransitRouteBuilder

builder = TransitRouteBuilder(path_to_osm_pbf)
network = builder.build()

network.save(path_to_pickle)
```

The extraction is performed in several passes to limit memory usage:

1. **Route metadata**  
   Public transport route metadata is extracted from the OSM PBF.

2. **Route members**  
   The ordered stops and ways belonging to each route are extracted using [`extractosm`](https://github.com/EPFL-ENAC/extractosm).

3. **Stop coordinates**  
   Coordinates are extracted only for the stop nodes required by the selected routes.

4. **Route geometries**  
   Geometry is extracted only for the ways required by the selected routes.

The package [`osmium`](https://osmcode.org/pyosmium/) is used for streaming through the PBF and keeping memory usage low.

### Route types

Route types can be specified when creating the builder:

```python
builder = TransitRouteBuilder(
    path_to_osm_pbf,
    route_types=["train", "tram", "bus"],
)
```

If no route types are specified, the following types are extracted:

* `train`
* `bus`
* `tram`
* `metro`
* `subway`
* `trolleybus`
* `light_rail`
* `funicular`
* `monorail`
* `aerialway`
* `ferry`

## Route geometry construction

Route geometries are reconstructed from the ordered OSM ways belonging to each route relation.

The builder:

* preserves the order of the ways in the OSM relation;
* orients consecutive ways so that they connect correctly;
* removes duplicate points at way boundaries;
* skips ways for which geometry is unavailable;
* removes small repeated loops, such as loops caused by roundabouts;
* returns the resulting geometry as a Shapely `LineString`.

If some required way geometries are unavailable, the resulting route may contain gaps or incomplete geometry.

### Repeated loops

OSM route relations can occasionally produce repeated loops in the extracted geometry, for example around roundabouts. TransitForge uses an internal `_remove_repeated_loops()` function to remove small repeated loops while avoiding the removal of larger sections of the route.

## Saving and loading

Once a network has been built, it can be serialized using `pickle`:

```python
network.save("network.pkl")
```

It can later be loaded without processing the original PBF again:

```python
from transitforge import TransitRouteNetwork

network = TransitRouteNetwork.load("network.pkl")
```

The parent directory is created automatically when saving.

> **Note:** `pickle` should only be loaded from trusted sources.

## Route matching

`TransitRouteNetwork` can match routing legs to OSM transit routes and reconstruct their geometry.

Route matching uses:

* OSM route references (`ref`, `nat_ref`, and `old_ref`);
* compatibility between routing/GTFS route types and OSM route types;
* stop proximity;
* stop ordering;
* route geometry as a fallback.

The default maximum distance between a requested stop and an OSM stop is **750 m**.

When stop-based matching is not sufficient, route geometry can be used as a fallback. The default maximum endpoint-to-route-geometry distance is **5000 m**.

These thresholds can be adjusted through the relevant method parameters.

### Route-type compatibility

GTFS and OSM use different route-type classifications. TransitForge maps common GTFS route types to compatible OSM route types.

For example:

| GTFS route type | Compatible OSM route types |
|---|---|
| `TRAM` | `tram`, `light_rail` |
| `SUBWAY`, `METRO` | `subway`, `light_rail`, `monorail` |
| `RAIL`, `TRAIN` | `train` |
| `BUS` | `bus`, `trolleybus` |
| `TROLLEYBUS` | `trolleybus`, `bus` |
| `FERRY` | `ferry` |
| `CABLE_CAR`, `GONDOLA` | `aerialway` |
| `FUNICULAR` | `funicular` |

## Getting geometry for a routing leg

For a simple origin-to-destination public transport leg:

```python
geometry = network.get_geometry_for_leg(leg)
```

The method is designed to work with the routing-leg structure produced by [minotor](https://minotor.dev/).

For example:

```python
leg = {
    "type": "vehicle",
    "route": "362",
    "routeType": "BUS",
    "from": {
        "name": "Haute-Nendaz, télécabine",
        "lat": 46.18051528930664,
        "lon": 7.29151725769043,
        "platform": None,
    },
    "to": {
        "name": "Sion, poste/gare",
        "lat": 46.227813720703125,
        "lon": 7.3583879470825195,
        "platform": "H",
    },
    "departureTime": "08:19",
    "arrivalTime": "08:58",
}
```

The returned geometry is a list of `[lat, lon]` coordinates:

```python
[
    [46.1805, 7.2915],
    [46.1901, 7.3102],
    [46.2103, 7.3356],
    [46.2278, 7.3584],
]
```

## Getting geometry for a complete stop sequence

For a journey containing multiple stops, geometry can be reconstructed from the complete ordered stop sequence:

```python
geometry = network.get_geometry_for_stop_sequence(
    leg,
    stops,
)
```

Here, `stops` contains `(lat, lon)` coordinates in travel order:

```python
stops = [
    (46.00, 7.00),
    (46.05, 7.05),
    (46.10, 7.10),
]
```

Unlike the simple origin-to-destination method, this method attempts to reconstruct the geometry of the entire journey, following the supplied stop sequence.

### Matching strategy

The method uses several successive matching stages:

1. **Route-reference matching**  
   TransitForge first searches for OSM routes matching the routing leg's route reference and compatible route type. It attempts to match the complete ordered stop sequence to a single OSM route, in either direction.

2. **Spatial matching**  
   If no route-reference candidate matches, TransitForge searches all OSM routes of a compatible type, ignoring the route reference. Routes whose stop bounding boxes are sufficiently close to both terminal stops are considered. The complete stop sequence is then tested against these candidates.

3. **Segmented matching**  
   If no single OSM route covers the complete journey, TransitForge attempts to reconstruct the geometry using multiple route segments.

   It progressively identifies the longest consecutive sections of the stop sequence that can be matched to individual OSM routes. If two consecutive stops cannot be covered by a single route, it can also attempt to connect two different OSM routes meeting at a shared station.

   Any remaining sections that cannot be matched to OSM routes are connected using straight-line segments between consecutive stops.

4. **Geometry fallback**  
   As a final fallback, TransitForge attempts to reconstruct the geometry using the first and last stops and the available route geometries.

The resulting geometry is assembled in travel order, with duplicate coordinates removed at segment boundaries.

This approach makes it possible to reconstruct journeys that cross multiple OSM route relations, for example when a train service is represented by several separate OSM relations.

If no usable route geometry can be reconstructed, the method raises a `ValueError`.

This functionality can be used to reconstruct missing shapes from a GTFS feed (see the package [GTFSmith](https://github.com/EPFL-LASUR/GTFSmith)).

## Diagnosing unmatched stop sequences

When `get_geometry_for_stop_sequence()` cannot find a suitable route, it raises a `ValueError`. To investigate why a sequence could not be matched, TransitForge provides:

```python
diagnosis = network.diagnose_stop_sequence(leg, stops)
```

This method analyses the available OSM routes of a compatible type, independently of the route reference, and reports how far the best candidate route can match the supplied stop sequence.

It is intended as a diagnostic tool for sequences that have already failed reconstruction.

### Diagnostic information

The returned dictionary includes information such as:

| Field | Description |
|---|---|
| `normalized_ref` | Normalized route reference from the routing leg. |
| `osm_routes_with_same_ref` | Number of OSM routes with the same normalized reference. |
| `osm_types_of_same_ref` | OSM route types associated with that reference. |
| `n_candidates` | Number of candidates found using route-reference matching. |
| `candidate_source` | Whether candidates came from the route reference or type-based fallback. |
| `routes_near_first_stop` | Number of compatible routes whose bounding boxes are near the first stop. |
| `routes_near_last_stop` | Number of compatible routes whose bounding boxes are near the last stop. |
| `routes_near_both_ends` | Number of compatible routes whose bounding boxes are near both terminal stops. |
| `best_relation` | OpenStreetMap relation URL for the best diagnostic candidate, when available. |
| `best_osm_ref` | Route reference of the best diagnostic candidate. |
| `best_osm_name` | Name of the best diagnostic candidate. |
| `best_direction` | Whether the best match was found in forward or reversed stop order. |
| `matched_stops` | Number of consecutive GTFS stops matched by the best candidate. |
| `fail_stop_index` | Index of the first stop where matching failed, when applicable. |
| `fail_distance_m` | Distance to the nearest remaining OSM stop at the failure point, when applicable. |
| `reason` | Machine-readable explanation of the failure. |
| `detail` | Human-readable explanation of the failure. |

The diagnostic method can also be configured with a maximum stop distance and a limit on the number of routes evaluated:

```python
diagnosis = network.diagnose_stop_sequence(leg, stops, max_stop_distance_m=750, max_eval=2000)
```

Possible failure reasons include:

- `NO_CANDIDATES`: no OSM route of a compatible type exists in the network.
- `NO_ROUTE_NEAR_EITHER_END`: no compatible route is near either terminal stop.
- `NO_ROUTE_NEAR_FIRST_STOP`: no compatible route is near the first stop.
- `NO_ROUTE_NEAR_LAST_STOP`: no compatible route is near the last stop.
- `NO_SINGLE_ROUTE_NEAR_BOTH_ENDS`: no single compatible route is near both terminal stops.
- `STOP_TOO_FAR`: a candidate route was found, but a stop in the sequence is too far from the next available OSM stop.
- `STOP_ORDER_MISMATCH`: the candidate OSM route does not contain enough stops in the required order.
- `GEOMETRY_EMPTY`: the stops could be matched, but the route has no usable geometry.

For example, the diagnostic result can be inspected directly:

```python
diagnosis = network.diagnose_stop_sequence(leg, stops)

print(diagnosis["reason"])
print(diagnosis["detail"])
print(diagnosis.get("best_relation"))
```

The `best_relation` field can be used to inspect the corresponding OSM relation and investigate missing or incomplete route data.

## Visualization

TransitForge provides visualization utilities based on `folium` and `matplotlib`.

### Visualizing a leg or trip

`plot_leg_folium()` can be used to visualize a single public transport leg, while `plot_trip_folium()` can visualize an entire trip, including transfers.

Both functions extract the required route geometry from the network.

For example:

```python
network = TransitRouteNetwork.load(path_to_pickle)

m = plot_trip_folium(network, route_pt)
m.save("route_mapped.html")
```

### Visualizing the network

The complete network can be visualized either as a static or interactive map using `plot_network()`.

The default is a static map:

```python
fig = plot_network(network)
fig.show()
```

An interactive `folium` map can be requested with `mode="dynamic"`:

```python
m = plot_network(network, mode="dynamic")
m.save(path_to_html)
```

## Testing

TransitForge uses `pytest` for automated testing.

The test suite is located in the `tests/` directory.

Run the complete test suite with:

```bash
uv run pytest
```

For more detailed output:

```bash
uv run pytest -v
```

### What is tested

The test suite covers the main functionality of the package, including:

- `TransitRouteBuilder`
  - default and custom route types;
  - platform role handling;
  - route geometry construction;
  - connecting and orienting OSM ways;
  - deduplicating geometry points;
  - handling missing ways;
  - removing small repeated loops;
  - complete builder orchestration.

- `TransitRouteNetwork`
  - route indexing by reference;
  - nearest-stop matching;
  - stop-sequence matching;
  - route direction handling;
  - GTFS route-type compatibility;
  - route matching by stops;
  - geometry-based route matching fallback;
  - `LineString` and `MultiLineString` geometries;
  - forward and reverse geometry extraction;
  - public latitude/longitude geometry output;
  - segmented reconstruction across multiple OSM routes;
  - chaining routes through shared stations;
  - straight-line bridging of unmatched sections;
  - stop-sequence diagnostics;
  - error handling when no route can be matched;
  - saving and loading networks.

The tests use small in-memory fixtures rather than requiring a real OpenStreetMap PBF file. Collector and `extractosm` operations are mocked where appropriate, which keeps the test suite fast and deterministic.

## Public API

The main public classes and methods are:

| Method | Purpose |
|---|---|
| `TransitRouteBuilder(pbf_file)` | Create a network builder from an OSM PBF. |
| `TransitRouteBuilder.build()` | Extract and build a `TransitRouteNetwork`. |
| `TransitRouteNetwork(routes)` | Create a transit network. |
| `TransitRouteNetwork.load(path)` | Load a serialized network. |
| `network.save(path)` | Serialize a network. |
| `network.get_geometry_for_leg(leg)` | Reconstruct geometry for a simple origin-to-destination leg. |
| `network.get_geometry_for_stop_sequence(leg, stops)` | Reconstruct geometry for a complete ordered stop sequence, including journeys spanning multiple OSM routes. |
| `network.diagnose_stop_sequence(leg, stops)` | Diagnose why a complete stop sequence could not be matched. |
| `network.find_route_for_leg(leg)` | Find an OSM route matching a routing leg. |
| `network.geometry_between(route, start, end)` | Extract geometry between two coordinates along an OSM route. |

Methods prefixed with `_` are internal implementation details and are not intended to be part of the public API.
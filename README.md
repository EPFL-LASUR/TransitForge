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

where `stops` contains `(lat, lon)` coordinates in travel order:

```python
stops = [
    (46.00, 7.00),
    (46.05, 7.05),
    (46.10, 7.10),
]
```

The method attempts to match the complete ordered stop sequence to a single OSM route and extracts the geometry between consecutive stops.

If the complete stop sequence cannot be matched, a geometry-based fallback using the first and last stops is attempted.

This can be later used to reconstruct missing shapes from a GTFS feed (see the package [GTFSmith](https://github.com/EPFL-LASUR/GTFSmith)).

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

## Public API

The main public classes and methods are:

| Method                                     | Purpose                                           |
| ------------------------------------------ | ------------------------------------------------- |
| `TransitRouteBuilder(pbf_file)`            | Create a network builder from an OSM PBF          |
| `TransitRouteBuilder.build()`              | Extract and build a `TransitRouteNetwork`         |
| `TransitRouteNetwork(routes)`              | Create a transit network                          |
| `TransitRouteNetwork.load()`               | Load a serialized network                         |
| `network.save()`                           | Serialize a network                               |
| `network.get_geometry_for_leg()`           | Reconstruct geometry for a routing leg            |
| `network.get_geometry_for_stop_sequence()` | Reconstruct geometry for an ordered stop sequence |
| `network.find_route_for_leg()`             | Find an OSM route matching a routing leg          |
| `network.geometry_between()`               | Extract geometry between two coordinates          |

Methods prefixed with `_` are internal implementation details and are not intended to be part of the public API.

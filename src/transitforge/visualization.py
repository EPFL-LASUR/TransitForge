import folium
from folium.features import DivIcon
import matplotlib.pyplot as plt

from .network import TransitRouteNetwork


def _normalize_visual_mode(value: str | None) -> str:
    """
    Normalize routing-engine / OSM mode names for visualization.
    """

    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    mapping = {
        "rail": "train",
        "train": "train",

        "bus": "bus",

        "tram": "tram",

        "subway": "subway",
        "metro": "subway",

        "trolleybus": "trolleybus",

        "light rail": "light_rail",
        "light_rail": "light_rail",

        "funicular": "funicular",

        "monorail": "monorail",

        "aerialway": "aerialway",
        "cable_car": "aerialway",
        "gondola": "aerialway",
        "chair_lift": "aerialway",

        "ferry": "ferry",

        # Walking / transfer modes
        "walk": "transfer",
        "walking": "transfer",
        "foot": "transfer",
        "transfer": "transfer",
    }

    return mapping.get(value, value)


def plot_leg_folium(
    network: TransitRouteNetwork,
    leg: dict,
    m: "folium.Map | None" = None,
    max_stop_distance_m: float = 750,
    color: str = "#e6194B",
    weight: int = 5,
):
    """
    Plot the exact OSM geometry for a single routing leg onto a
    folium map, returning the map. If `m` is None, a new map is
    created and centered/fit on the route.

    Example
    -------

        network = TransitRouteNetwork.load("transit_routes.pkl")

        m = plot_leg_folium(network, leg)

        m.save("route.html")
    """

    coords = network.get_geometry_for_leg(leg, max_stop_distance_m=max_stop_distance_m,)

    if not coords:
        raise ValueError("No geometry could be extracted for this leg.")

    if m is None:
        m = folium.Map(tiles="Cartodb Positron")

    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=0.9,
        tooltip=(
            f"{leg.get('routeType', '')} "
            f"{leg.get('route', '')}"
        ),
    ).add_to(m)

    folium.Marker(
        coords[0],
        tooltip=leg.get("from", {}).get("name", "Start"),
        icon=folium.Icon(color="green"),
    ).add_to(m)

    folium.Marker(
        coords[-1],
        tooltip=leg.get("to", {}).get("name", "End"),
        icon=folium.Icon(color="red"),
    ).add_to(m)

    m.fit_bounds(coords)

    return m


def plot_trip_folium(
    network,
    trip: dict,
    max_stop_distance_m: float = 750,
    ):
    """
    Plot a complete trip on a Folium map.

    Vehicle legs:
        - Exact OSM route geometry
        - Fixed color based on transport mode

    Transfer legs:
        - Straight dashed line between from/to coordinates
        - Gray color

    Example
    -------

        m = plot_trip_folium(
            network,
            trip,
        )

        m.save("trip.html")
    """

    # Color settings
    MODE_COLORS = {
        "train": "#1f77b4",
        "bus": "#ff7f0e",
        "tram": "#d62728",
        "subway": "#9467bd",
        "trolleybus": "#e377c2",
        "light_rail": "#2ca02c",
        "funicular": "#8c564b",
        "monorail": "#17becf",
        "aerialway": "#bcbd22",
        "ferry": "#1f9ed1",

        # Fallback
        "unknown": "#444444",
    }

    TRANSFER_COLOR = "#666666"

    m = folium.Map(tiles="Cartodb Positron")

    all_coords: list[list[float]] = []

    legs = trip.get("legs", [])

    for i, leg in enumerate(legs):

        leg_type = leg.get("type", "")

        # Vehicle legs
        if leg_type == "vehicle":
            try:
                coords = network.get_geometry_for_leg(leg, max_stop_distance_m=max_stop_distance_m)

            except ValueError as exc:
                print(f"Skipping vehicle leg {i}: {exc}")
                continue

            if not coords:
                continue

            mode = _normalize_visual_mode(leg.get("routeType"))
            color = MODE_COLORS.get(mode, MODE_COLORS["unknown"],)

            folium.PolyLine(
                coords,
                color=color,
                weight=6,
                opacity=0.9,
                tooltip=(
                    f"{mode.upper()} "
                    f"{leg.get('route', '')}"
                ),
            ).add_to(m)

            # Obtaining the geographical middle of the trip
            middle = coords[len(coords)//2]
    
            # Plotting the name of the transport mode
            folium.Marker(
                location=middle,
                icon=DivIcon(
                    html=f"""
                    <div style="
                        font-size:14px;
                        font-weight:bold;
                        color:white;
                        -webkit-text-stroke: 2px black;
                        paint-order: stroke fill;
                        white-space: nowrap;
                        transform: translate(-50%, -50%);
                    ">
                        {leg["route"]}
                    </div>
                    """
                ),
            ).add_to(m)

            # Start marker
            from_data = leg.get("from", {})

            if (
                from_data.get("lat") is not None
                and from_data.get("lon") is not None
            ):

                folium.CircleMarker(
                    location=[
                        from_data["lat"],
                        from_data["lon"],
                    ],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=1,
                    tooltip=from_data.get(
                        "name",
                        "Start",
                    ),
                ).add_to(m)

            # End marker
            to_data = leg.get("to", {})

            if (
                to_data.get("lat") is not None
                and to_data.get("lon") is not None
            ):

                folium.CircleMarker(
                    location=[
                        to_data["lat"],
                        to_data["lon"],
                    ],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=1,
                    tooltip=to_data.get(
                        "name",
                        "End",
                    ),
                ).add_to(m)


            all_coords.extend(coords)


        # Transfer legs
        else:
            from_data = leg.get("from", {})
            to_data = leg.get("to", {})

            if not (
                from_data.get("lat") is not None
                and from_data.get("lon") is not None
                and to_data.get("lat") is not None
                and to_data.get("lon") is not None
            ):
                continue


            transfer_coords = [
                [from_data["lat"], from_data["lon"]],
                [to_data["lat"], to_data["lon"]],
            ]


            # Dashed straight transfer line.
            folium.PolyLine(
                transfer_coords,
                color=TRANSFER_COLOR,
                weight=4,
                opacity=0.8,
                dash_array="8, 8",
                tooltip=(
                    f"Transfer: "
                    f"{from_data.get('name', '')} → "
                    f"{to_data.get('name', '')}"
                ),
            ).add_to(m)


            # Transfer endpoints.
            folium.CircleMarker(
                location=transfer_coords[0],
                radius=4,
                color=TRANSFER_COLOR,
                fill=True,
                fill_opacity=1,
                tooltip=from_data.get(
                    "name",
                    "Transfer start",
                ),
            ).add_to(m)


            folium.CircleMarker(
                location=transfer_coords[1],
                radius=4,
                color=TRANSFER_COLOR,
                fill=True,
                fill_opacity=1,
                tooltip=to_data.get(
                    "name",
                    "Transfer end",
                ),
            ).add_to(m)


            all_coords.extend(
                transfer_coords
            )

    # Fit map
    if all_coords:
        m.fit_bounds(all_coords)

    return m


def plot_network(network: "TransitRouteNetwork", mode: str = "static", color_map: dict = None) -> plt.Figure | folium.Map:
    """
    Plots the whole network.
    Two modes are possible:
        - static: plots with matplotlib.pyplot (faster)
        - dynamic: plots with folium, allows for visualization as html
        
    Args:
        - network (TransitRouteNetwork) : network that has been extracted beforehand
        - mode (str) : Optional. Static or dynamic map. Default: static
        - color_map (dict): Optional. Color mapping for the PT modes.
        
    Returns:
        - map (Figure | Map): map with the plotted network
    
    """
    if color_map is None:
        route_colors = {
            # Bus family
            "bus": "#0419FA",
            "trolleybus": "#26B2F4",

            # Rail / tram family
            "tram": "#90C940",
            "light_rail": "#FB6A4A",
            "subway": "#228B22",

            # Cable transport family
            "funicular": "#984EA3",
            "aerialway": "#C994C7",

            # Other modes
            "train": "#F1040F",
            "ferry": "#00A6A6",
            "monorail": "#FF7F00",
        }
    else:
        route_colors = color_map

    # Static map
    if mode == "static":
        map, ax = plt.subplots(figsize=(12, 12))

        plotted_route_types = set()

        for route_id, route in network.routes.items():
            if route.geometry is None or route.geometry.is_empty or route.name is None:
                continue

            lons, lats = route.geometry.xy

            route_type = route.route_type
            color = route_colors.get(route_type, "#808080")

            ax.plot(
                lons,
                lats,
                color=color,
                linewidth=1,
                alpha=0.5,
            )

            plotted_route_types.add(route_type)

        # Legend
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                color=route_colors[route_type],
                linewidth=3,
                label=route_type.replace("_", " ").title(),
            )
            for route_type in route_colors
            if route_type in plotted_route_types
        ]

        ax.legend(
            handles=legend_handles,
            title="Route type",
            loc="upper right",
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Transit Route Network")
        ax.set_aspect("equal")

        return map


    # Dynamic plotting
    elif mode == "dynamic":

        # Find a reasonable center for the map
        first_route = next(
            r for r in network.routes.values()
            if r.geometry is not None and not r.geometry.is_empty
        )

        lon, lat = first_route.geometry.centroid.coords[0]

        map = folium.Map(
            location=[lat, lon],
            zoom_start=12,
            tiles=None
        )

        # Base maps
        folium.TileLayer(
            tiles="CartoDB positron",
            name="CartoDB Positron",
            control=True,
        ).add_to(map)
        

        folium.TileLayer(
            tiles="CartoDB dark_matter",
            name="CartoDB Dark",
            control=True,
        ).add_to(map)


        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            control=True,
        ).add_to(map)

        plotted_route_types = set()

        for route_id, route in network.routes.items():

            if route.geometry is None or route.geometry.is_empty or route.name is None:
                continue

            # Shapely: (lon, lat)
            coords = list(route.geometry.coords)

            # Folium: [lat, lon]
            folium_coords = [
                [lat, lon]
                for lon, lat in coords
            ]

            route_type = route.route_type
            color = route_colors.get(route_type, "#808080")

            folium.PolyLine(
                locations=folium_coords,
                color=color,
                weight=2,
                opacity=0.6,
                tooltip=f"{route.ref} — {route.name}",
            ).add_to(map)

            plotted_route_types.add(route_type)

        # Legend
        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 9999;
            background-color: white;
            border: 2px solid grey;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
        ">
        <b>Route type</b><br>
        """

        for route_type in route_colors:
            if route_type in plotted_route_types:
                label = route_type.replace("_", " ").title()
                color = route_colors[route_type]

                legend_html += f"""
                <div style="margin-top: 4px;">
                    <span style="
                        display: inline-block;
                        width: 25px;
                        height: 4px;
                        background-color: {color};
                        margin-right: 6px;
                        vertical-align: middle;
                    "></span>
                    {label}
                </div>
                """

        legend_html += "</div>"

        map.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl().add_to(map)

        return map

    else:
        raise ValueError(f"Mode {mode} not available. Please choose between 'static' and 'dynamic'.")
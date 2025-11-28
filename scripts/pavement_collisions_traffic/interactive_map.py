"""
--- DETAILED DESCRIPTION IN README.md (scripts/pavement_collisions_traffic/README.md) ---
"""

import pandas as pd
import folium

ZONE_TYPE_MAP = {
    "RS": "Residential Single-Family",
    "RM": "Residential Multi-Family",
    "CC": "Commercial",
    "CN": "Commercial Neighborhood",
    "CCPD": "Commercial Planned Development",
    "IL": "Industrial Limited",
    "IG": "Industrial General",
    "AR": "Agricultural",
    "AG": "Agricultural",
    "OP": "Open Space",
    "EMX": "Employment Mixed Use",
    "CUPD": "Community Plan Update",
    "PD": "Planned Development",
    "SP": "Special Purpose",
    "MU": "Mixed Use",
    "MX": "Mixed Use",
}

ZONE_COLORS = {
    "Residential Single-Family": "#a6cee3",
    "Residential Multi-Family": "#1f78b4",
    "Commercial": "#b2df8a",
    "Commercial Neighborhood": "#33a02c",
    "Commercial Planned Development": "#fb9a99",
    "Industrial Limited": "#e31a1c",
    "Industrial General": "#fdbf6f",
    "Agricultural": "#ff7f00",
    "Open Space": "#cab2d6",
    "Employment Mixed Use": "#6a3d9a",
    "Community Plan Update": "#ffff99",
    "Planned Development": "#b15928",
    "Special Purpose": "#8dd3c7",
    "Mixed Use": "#bebada",
    "Other/Unknown": "#d9d9d9",
}

# Match seaborn RdYlGn_r palette from other scripts
PCI_COLOR_MAP = {
    "Good": "#1a9850",
    "Satisfactory": "#91cf60",
    "Fair": "#d9ef8b",
    "Poor": "#fee08b",
    "Very Poor": "#fc8d59",
    "Serious": "#d73027",
    "Failed": "#a50026",
}


def get_zone_category(code):
    if not isinstance(code, str):
        return "Other/Unknown"
    code_upper = code.upper()
    sorted_keys = sorted(ZONE_TYPE_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if code_upper.startswith(key):
            return ZONE_TYPE_MAP[key]
    return "Other/Unknown"


def style_zones(feature):
    cat = feature["properties"]["category"]
    color = ZONE_COLORS.get(cat, "#d9d9d9")
    return {"fillColor": color, "color": color, "weight": 0.5, "fillOpacity": 0.35}


def style_roads(feature):
    pci_desc = feature["properties"].get("pci_desc")
    crashes = feature["properties"].get("total_crashes", 0)  # can change

    color = PCI_COLOR_MAP.get(pci_desc, "#808080")

    weight = 3
    if pd.notna(crashes):
        if crashes > 0:
            weight = 5.0
        if crashes > 2:
            weight = 7.0
        if crashes > 5:
            weight = 10.0

    return {"color": color, "weight": weight, "opacity": 1.0}


def generate_map_for_year(df, roads_geo, zoning_geo, target_year):
    """
    Generates a Folium map object for inline display.

    Parameters:
        df: DataFrame with paving analysis segments yearly data
        roads_geo: GeoDataFrame with road segments geometry
        zoning_geo: GeoDataFrame with zoning geometry
        target_year: Year to filter data for
    """
    # Add category column to zoning
    zoning_geo = zoning_geo.copy()
    zoning_geo["category"] = zoning_geo["zone_name"].apply(get_zone_category)

    # Initialize Map
    m = folium.Map(location=[32.7157, -117.1611], zoom_start=12, tiles="CartoDB dark_matter")

    # Add Zoning Layer
    folium.GeoJson(
        zoning_geo,
        name="Zoning Districts",
        style_function=style_zones,
        overlay=True,
        show=True,
        tooltip=folium.GeoJsonTooltip(fields=["zone_name", "category"], aliases=["Zone Code:", "Type:"]),
    ).add_to(m)

    # Process Road Data
    df_year = df[df["year"] == target_year].copy()

    if not df_year.empty:
        # Round PCI
        df_year["avg_pci"] = df_year["avg_pci"].round(2)

        # Filter Columns
        cols_to_keep = ["iamfloc", "avg_pci", "total_crashes", "injured", "killed", "pci_desc"]
        df_year = df_year[cols_to_keep]

        # Merge Geometry
        map_data_year = roads_geo.merge(df_year, on="iamfloc", how="inner")

        # Add Road Layer
        folium.GeoJson(
            map_data_year,
            name=f"Road Conditions ({target_year})",
            style_function=style_roads,
            overlay=True,
            show=True,
            tooltip=folium.GeoJsonTooltip(
                fields=["rd20full", "cpname", "avg_pci", "pci_desc", "total_crashes", "injured", "killed"],
                aliases=["Street:", "Community:", "PCI Score:", "Condition:", "Total Crashes:", "Injured:", "Killed:"],
                labels=True,
            ),
        ).add_to(m)

    # Add Control
    folium.LayerControl(collapsed=False).add_to(m)

    return m

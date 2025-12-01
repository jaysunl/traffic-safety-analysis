"""
Spatial operations for joining road segments to zoning districts.
"""

import geopandas as gpd
import pandas as pd


def join_segments_to_zoning(
    gdf_paving: gpd.GeoDataFrame,
    gdf_zoning: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Perform spatial join between road segments and zoning polygons.

    Matches road segments to zoning districts based on geometric intersection.
    If a segment intersects multiple zones, keeps the first match processed.

    Parameters
    ----------
    gdf_paving : gpd.GeoDataFrame
        Road segments with line geometries.
    gdf_zoning : gpd.GeoDataFrame
        Zoning polygons with zone information.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['roadsegid', 'zone_name', 'imp_date', 'ordnum'].
        All segments are included, with NaN for zone fields if no match found.
    """
    print("--- Joining Road Segments to Zoning ---")
    
    if gdf_paving.crs != gdf_zoning.crs:
        gdf_paving = gdf_paving.to_crs(gdf_zoning.crs)
    
    gdf_joined = gpd.sjoin(
        gdf_paving[['roadsegid', 'geometry']],
        gdf_zoning[['zone_name', 'imp_date', 'ordnum', 'geometry']],
        how='left',
        predicate='intersects'
    )
    
    gdf_joined = gdf_joined.drop_duplicates(subset='roadsegid', keep='first')
    
    matched_count = gdf_joined['zone_name'].notna().sum()
    print(f"Segments matched to zones: {matched_count} / {len(gdf_joined)}")
    
    return gdf_joined[['roadsegid', 'zone_name', 'imp_date', 'ordnum']]


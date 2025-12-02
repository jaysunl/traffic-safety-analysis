"""
Data loading functions for collisions and zoning analysis.
"""

import pandas as pd
import geopandas as gpd
from typing import Optional, Tuple

from .config import FILES, DATE_RANGE
from .street_utils import clean_street_name


def load_and_filter_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare paving segments and collision data.

    Loads CSV files, standardizes street names, computes address ranges for paving
    segments, and filters collisions to the configured date range.

    Returns
    -------
    df_paving : pd.DataFrame
        Paving segments with cleaned street names and address ranges.
    df_coll : pd.DataFrame
        Filtered collision data for the specified date range.

    Notes
    -----
    Removes directional codes like "(SB)", "(NB)", "(FTG)" from street names.
    """
    print("--- Loading Data ---")
    
    df_paving = pd.read_csv(FILES['paving'])
    df_paving['clean_street'] = clean_street_name(df_paving['rd20full'])
    df_paving['clean_street'] = df_paving['clean_street'].str.replace(
        r'\s*\([^)]*\)', '', regex=True
    ).str.strip()
    
    address_cols = ['llowaddr', 'lhighaddr', 'rlowaddr', 'rhighaddr']
    for col in address_cols:
        df_paving[col] = pd.to_numeric(df_paving[col], errors='coerce')
    
    df_paving['seg_min'] = df_paving[['llowaddr', 'rlowaddr']].min(axis=1)
    df_paving['seg_max'] = df_paving[['lhighaddr', 'rhighaddr']].max(axis=1)
    
    df_coll = pd.read_csv(FILES['collisions'], dtype={'report_id': str})
    df_coll['date_time'] = pd.to_datetime(df_coll['date_time'])
    date_mask = (
        (df_coll['date_time'] >= DATE_RANGE['start']) & 
        (df_coll['date_time'] <= DATE_RANGE['end'])
    )
    df_coll = df_coll[date_mask].copy()
    
    print(f"Loaded Paving Segments: {len(df_paving)}")
    print(f"Loaded Collisions ({DATE_RANGE['start']} to {DATE_RANGE['end']}): {len(df_coll)}")
    
    return df_paving, df_coll


def load_zoning_data() -> Optional[gpd.GeoDataFrame]:
    """
    Load zoning GeoJSON and prepare for spatial operations.

    Loads zoning polygons, sets CRS if missing, reprojects to a projected
    coordinate system, and calculates zone areas.

    Returns
    -------
    gpd.GeoDataFrame or None
        Zoning polygons with geometry and area calculations. Returns None if loading fails.

    Notes
    -----
    Attempts to reproject to EPSG:2230 (California State Plane Zone 6) or
    EPSG:32611 (UTM Zone 11N) for better spatial accuracy.
    Calculates zone areas in square meters and square kilometers.
    """
    print("--- Loading Zoning Data ---")
    
    try:
        gdf_zoning = gpd.read_file(FILES['zoning_geojson'])
        print(f"Loaded Zoning Polygons: {len(gdf_zoning)}")
        
        if gdf_zoning.crs is None:
            gdf_zoning.set_crs('EPSG:4326', inplace=True)
        
        target_crs_options = ['EPSG:2230', 'EPSG:32611']
        for crs in target_crs_options:
            try:
                gdf_zoning = gdf_zoning.to_crs(crs)
                break
            except Exception:
                continue
        else:
            print("Warning: Could not reproject zoning data. Using original CRS.")
        
        gdf_zoning['zone_area_m2'] = gdf_zoning.geometry.area
        gdf_zoning['zone_area_km2'] = gdf_zoning['zone_area_m2'] / 1_000_000
        
        return gdf_zoning
    except Exception as e:
        print(f"Error loading zoning data: {e}")
        return None


def load_paving_geometries() -> Optional[gpd.GeoDataFrame]:
    """
    Load paving segments GeoJSON with geometry for spatial joins.

    Returns
    -------
    gpd.GeoDataFrame or None
        Paving segments with line geometries. Returns None if loading fails.

    Notes
    -----
    Reprojects to match zoning CRS for consistent spatial operations.
    """
    print("--- Loading Paving Segment Geometries ---")
    
    try:
        gdf_paving = gpd.read_file(FILES['paving_geojson'])
        print(f"Loaded Paving Segments with Geometry: {len(gdf_paving)}")
        
        # set crs if not set
        # EPSG:4326 is San Diego's default crs
        if gdf_paving.crs is None:
            gdf_paving.set_crs('EPSG:4326', inplace=True)
        
        target_crs_options = ['EPSG:2230', 'EPSG:32611']
        for crs in target_crs_options:
            try:
                gdf_paving = gdf_paving.to_crs(crs)
                break
            except Exception:
                continue
        else:
            print("Warning: Could not reproject paving data. Using original CRS.")
        
        return gdf_paving
    except Exception as e:
        print(f"Error loading paving geometries: {e}")
        return None


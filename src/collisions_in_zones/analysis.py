"""
Analysis functions for collisions by zone.
"""

import pandas as pd
import geopandas as gpd

from .config import FILES, DATE_RANGE
from .zone_utils import extract_zone_type


def calculate_zone_density(
    df_zoning_stats: pd.DataFrame,
    gdf_zoning: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Calculate crash density metrics per zone.

    Parameters
    ----------
    df_zoning_stats : pd.DataFrame
        Zone statistics with columns including 'zone_name', 'total_crashes', etc.
    gdf_zoning : gpd.GeoDataFrame
        Zoning polygons with area calculations.

    Returns
    -------
    pd.DataFrame
        Zone statistics with added density metrics.
    """
    print("--- Calculating Zone Density Metrics ---")
    
    zone_areas = gdf_zoning.groupby('zone_name').agg({
        'zone_area_km2': 'sum',
        'zone_area_m2': 'sum'
    }).reset_index()
    
    df_with_density = pd.merge(df_zoning_stats, zone_areas, on='zone_name', how='left')
    
    df_with_density['crashes_per_km2'] = (
        df_with_density['total_crashes'] / 
        df_with_density['zone_area_km2'].replace(0, pd.NA)
    )
    df_with_density['injuries_per_km2'] = (
        df_with_density['injured'] / 
        df_with_density['zone_area_km2'].replace(0, pd.NA)
    )
    df_with_density['fatalities_per_km2'] = (
        df_with_density['killed'] / 
        df_with_density['zone_area_km2'].replace(0, pd.NA)
    )
    
    print(f"Calculated density metrics for {df_with_density['zone_area_km2'].notna().sum()} zones")
    
    return df_with_density


def analyze_detailed_collisions_by_zone(
    df_collisions_detailed: pd.DataFrame,
    all_matches: pd.DataFrame,
    segment_zoning: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze detailed collision characteristics by zone.

    Parameters
    ----------
    df_collisions_detailed : pd.DataFrame
        Detailed collision data with person and vehicle information.
    all_matches : pd.DataFrame
        Collision-to-segment matches with columns ['report_id', 'roadsegid'].
    segment_zoning : pd.DataFrame
        Segment-to-zone mapping with columns ['roadsegid', 'zone_name'].

    Returns
    -------
    pd.DataFrame
        Detailed statistics by zone including vehicle types, person roles, injury levels.
    """
    print("--- Analyzing Detailed Collisions by Zone ---")
    
    df_detailed = df_collisions_detailed.copy()
    df_detailed['date_time'] = pd.to_datetime(df_detailed['date_time'], errors='coerce')
    date_mask = (
        (df_detailed['date_time'] >= DATE_RANGE['start']) & 
        (df_detailed['date_time'] <= DATE_RANGE['end'])
    )
    df_detailed = df_detailed[date_mask].copy()
    
    df_detailed = pd.merge(df_detailed, all_matches, on='report_id', how='inner')
    df_detailed = pd.merge(df_detailed, segment_zoning, on='roadsegid', how='left')
    df_detailed = df_detailed[df_detailed['zone_name'].notna()].copy()
    
    if df_detailed.empty:
        return pd.DataFrame()
    
    detailed_stats = df_detailed.groupby('zone_name').agg({
        'report_id': 'nunique',
        'person_role': lambda x: x.value_counts().head(5).to_dict() if len(x.dropna()) > 0 else {},
        'person_injury_lvl': lambda x: x.value_counts().head(5).to_dict() if len(x.dropna()) > 0 else {},
        'veh_type': lambda x: x.value_counts().head(5).to_dict() if len(x.dropna()) > 0 else {},
        'person_veh_type': lambda x: x.value_counts().head(5).to_dict() if len(x.dropna()) > 0 else {},
        'hit_run_lvl': lambda x: (x.notna() & (x != '')).sum()
    }).reset_index()
    
    detailed_stats.rename(columns={'report_id': 'unique_collisions'}, inplace=True)
    
    print(f"Analyzed detailed collisions for {len(detailed_stats)} zones")
    
    return detailed_stats


def analyze_violations_by_zone(
    df_collisions: pd.DataFrame,
    all_matches: pd.DataFrame,
    segment_zoning: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze violation types by zone.

    Parameters
    ----------
    df_collisions : pd.DataFrame
        Collision data with violation information.
    all_matches : pd.DataFrame
        Collision-to-segment matches with columns ['report_id', 'roadsegid'].
    segment_zoning : pd.DataFrame
        Segment-to-zone mapping with columns ['roadsegid', 'zone_name'].

    Returns
    -------
    pd.DataFrame
        Violation statistics by zone.
    """
    print("--- Analyzing Violations by Zone ---")
    
    df_coll = df_collisions.copy()
    df_coll['date_time'] = pd.to_datetime(df_coll['date_time'], errors='coerce')
    date_mask = (
        (df_coll['date_time'] >= DATE_RANGE['start']) & 
        (df_coll['date_time'] <= DATE_RANGE['end'])
    )
    df_coll = df_coll[date_mask].copy()
    
    df_coll = pd.merge(df_coll, all_matches, on='report_id', how='inner')
    df_coll = pd.merge(df_coll, segment_zoning, on='roadsegid', how='left')
    df_coll = df_coll[df_coll['zone_name'].notna()].copy()
    
    violation_stats = df_coll.groupby('zone_name').agg({
        'report_id': 'count',
        'violation_section': lambda x: x.value_counts().head(5).to_dict() if len(x.dropna()) > 0 else {},
        'charge_desc': lambda x: x.value_counts().head(5).to_dict() if len(x.dropna()) > 0 else {},
        'hit_run_lvl': lambda x: (x.notna() & (x != '')).sum()
    }).reset_index()
    
    violation_stats.rename(columns={
        'report_id': 'total_crashes_viol',
        'hit_run_lvl': 'hit_and_run_count'
    }, inplace=True)
    
    violation_stats['hit_and_run_rate'] = (
        violation_stats['hit_and_run_count'] / 
        violation_stats['total_crashes_viol'].replace(0, pd.NA)
    ) * 100
    
    print(f"Analyzed violations for {len(violation_stats)} zones")
    
    return violation_stats


def create_zone_type_analysis(df_zoning_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze crash patterns by zone type category.

    Parameters
    ----------
    df_zoning_stats : pd.DataFrame
        Zone statistics with 'zone_name' column.

    Returns
    -------
    pd.DataFrame
        Aggregated statistics by zone type category.
    """
    print("--- Analyzing by Zone Type Category ---")
    
    df_with_type = df_zoning_stats.copy()
    df_with_type['zone_type'] = df_with_type['zone_name'].apply(extract_zone_type)
    
    zone_type_stats = df_with_type.groupby('zone_type').agg({
        'total_crashes': 'sum',
        'injured': 'sum',
        'killed': 'sum',
        'segments_with_crashes': 'sum',
        'total_segments': 'sum',
        'zone_name': 'count'
    }).reset_index()
    
    zone_type_stats.rename(columns={'zone_name': 'num_zones'}, inplace=True)
    
    zone_type_stats['crashes_per_zone'] = (
        zone_type_stats['total_crashes'] / 
        zone_type_stats['num_zones'].replace(0, pd.NA)
    )
    zone_type_stats['crashes_per_segment'] = (
        zone_type_stats['total_crashes'] / 
        zone_type_stats['total_segments'].replace(0, pd.NA)
    )
    zone_type_stats['injury_rate'] = (
        zone_type_stats['injured'] / 
        zone_type_stats['total_crashes'].replace(0, pd.NA)
    )
    zone_type_stats['fatality_rate'] = (
        zone_type_stats['killed'] / 
        zone_type_stats['total_crashes'].replace(0, pd.NA)
    )
    
    zone_type_stats = zone_type_stats.sort_values('total_crashes', ascending=False)
    
    print(f"Analyzed {len(zone_type_stats)} zone type categories")
    
    return zone_type_stats


def create_zoning_collision_analysis(
    df_coll: pd.DataFrame,
    all_matches: pd.DataFrame,
    segment_zoning: pd.DataFrame,
    gdf_zoning: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Aggregate collision statistics by zoning type and generate comprehensive output.

    Calculates crash counts, injuries, fatalities, density metrics, detailed collision
    patterns, and violation statistics for each zoning district.

    Parameters
    ----------
    df_coll : pd.DataFrame
        Full collision dataset.
    all_matches : pd.DataFrame
        Collision-to-segment matches with columns ['report_id', 'roadsegid'].
    segment_zoning : pd.DataFrame
        Segment-to-zoning mapping with columns ['roadsegid', 'zone_name', ...].
    gdf_zoning : gpd.GeoDataFrame
        Zoning polygons with area calculations.

    Returns
    -------
    pd.DataFrame
        Comprehensive aggregated statistics by zone including all metrics.
    """
    print("--- Creating Zoning Collision Analysis ---")
    
    df_linked = pd.merge(df_coll, all_matches, on='report_id', how='inner')
    df_linked['injured'] = pd.to_numeric(df_linked['injured'], errors='coerce').fillna(0)
    df_linked['killed'] = pd.to_numeric(df_linked['killed'], errors='coerce').fillna(0)
    
    df_with_zoning = pd.merge(df_linked, segment_zoning, on='roadsegid', how='left')
    
    zone_stats = df_with_zoning.groupby('zone_name').agg({
        'report_id': 'count',
        'injured': 'sum',
        'killed': 'sum',
        'roadsegid': 'nunique'
    }).reset_index()
    
    zone_stats.rename(columns={
        'report_id': 'total_crashes',
        'roadsegid': 'segments_with_crashes'
    }, inplace=True)
    
    zone_segment_counts = segment_zoning.groupby('zone_name').size().reset_index(
        name='total_segments'
    )
    
    zone_analysis = pd.merge(
        zone_stats,
        zone_segment_counts,
        on='zone_name',
        how='outer'
    )
    
    fill_cols = ['total_crashes', 'injured', 'killed', 'segments_with_crashes', 'total_segments']
    zone_analysis[fill_cols] = zone_analysis[fill_cols].fillna(0)
    
    zone_analysis['crashes_per_segment'] = (
        zone_analysis['total_crashes'] / 
        zone_analysis['total_segments'].replace(0, pd.NA)
    )
    zone_analysis['crash_rate_pct'] = (
        zone_analysis['segments_with_crashes'] / 
        zone_analysis['total_segments'].replace(0, pd.NA)
    ) * 100
    
    zone_analysis = calculate_zone_density(zone_analysis, gdf_zoning)
    
    zone_analysis['zone_type'] = zone_analysis['zone_name'].apply(extract_zone_type)
    
    try:
        df_collisions_detailed = pd.read_csv(FILES['collisions_detailed'], dtype={'report_id': str})
        detailed_stats = analyze_detailed_collisions_by_zone(
            df_collisions_detailed,
            all_matches,
            segment_zoning
        )
        if not detailed_stats.empty:
            zone_analysis = pd.merge(zone_analysis, detailed_stats, on='zone_name', how='left')
    except Exception as e:
        print(f"Warning: Could not load detailed collisions: {e}")
    
    violation_stats = analyze_violations_by_zone(df_coll, all_matches, segment_zoning)
    zone_analysis = pd.merge(zone_analysis, violation_stats, on='zone_name', how='left')
    
    zone_analysis = zone_analysis.sort_values('total_crashes', ascending=False)
    
    zone_analysis.to_csv(FILES['output_data'], index=False)
    print(f"File saved: {FILES['output_data']}")
    print("\nTop 10 zones by crash count:")
    display_cols = ['zone_name', 'zone_type', 'total_crashes', 'injured', 'killed', 'crashes_per_segment']
    print(zone_analysis[display_cols].head(10).to_string())
    
    zone_type_analysis = create_zone_type_analysis(zone_analysis)
    print("\n--- Zone Type Summary ---")
    print(zone_type_analysis[['zone_type', 'num_zones', 'total_crashes', 
                             'crashes_per_zone', 'injury_rate', 'fatality_rate']].to_string())
    
    return zone_analysis


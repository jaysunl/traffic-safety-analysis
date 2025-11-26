"""
This script loads traffic collision data and paving segment data, matches collisions to
paving segments, and then joins the paving segments to zoning districts using spatial
joins. It performs comprehensive analysis including zone type categorization, density
analysis, detailed collision patterns, violation type analysis, and generates
visualizations including a severity map.

This script uses the following files:
- ./data/raw/streets_repair_line_segments/sd_paving_segs_datasd.csv
- ./data/raw/streets_repair_line_segments/sd_paving_segs_datasd.geojson
- ./data/raw/traffic_collisions_basic/pd_collisions_datasd.csv
- ./data/raw/traffic_collisions_detailed/pd_collisions_details_datasd.csv
- ./data/raw/zoning/zoning_datasd.geojson
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Tuple

FILES = {
    'paving': './data/raw/streets_repair_line_segments/sd_paving_segs_datasd.csv',
    'paving_geojson': './data/raw/streets_repair_line_segments/sd_paving_segs_datasd.geojson',
    'collisions': './data/raw/traffic_collisions_basic/pd_collisions_datasd.csv',
    'collisions_detailed': './data/raw/traffic_collisions_detailed/pd_collisions_details_datasd.csv',
    'zoning_geojson': './data/raw/zoning/zoning_datasd.geojson',
    'output_data': './data/processed/collisions_analysis_with_zoning.csv',
    'output_visualization': './data/processed/collisions_analysis_with_zoning_visualization.png',
    'output_map': './data/processed/collisions_analysis_with_zoning_map.png',
    'output_debug': './misc/debug_unmatched_collisions.csv'
}

DATE_RANGE = {
    'start': '2023-01-01',
    'end': '2024-12-31'
}

SUFFIX_MAP = {
    'AVENUE': 'AV',
    'STREET': 'ST',
    'ROAD': 'RD',
    'DRIVE': 'DR',
    'BOULEVARD': 'BL',
    'PLACE': 'PL',
    'WAY': 'WY',
    'COURT': 'CT',
    'LANE': 'LN',
    'TERRACE': 'TER',
    'CIRCLE': 'CR',
    'MOUNTAIN': 'MTN',
    'MOUNT': 'MT',
    'NORTH': 'N',
    'SOUTH': 'S',
    'EAST': 'E',
    'WEST': 'W',
    'CAMINO': 'CAM',
    'PARKWAY': 'PY',
    'HIGHWAY': 'HY',
    'MALL': 'ML',
    'EXTENSION': 'EX',
    'VALLEY': 'VLY',
    'WALK': 'WK'
}

ZONE_TYPE_MAP = {
    'RS': 'Residential Single-Family',
    'RM': 'Residential Multi-Family',
    'CC': 'Commercial',
    'CN': 'Commercial Neighborhood',
    'CCPD': 'Commercial Planned Development',
    'IL': 'Industrial Limited',
    'IG': 'Industrial General',
    'AR': 'Agricultural',
    'AG': 'Agricultural',
    'OP': 'Open Space',
    'EMX': 'Employment Mixed Use',
    'CUPD': 'Community Plan Update',
    'PD': 'Planned Development',
    'SP': 'Special Purpose',
    'MU': 'Mixed Use',
    'MX': 'Mixed Use'
}


def clean_street_name(
    name_series: pd.Series,
    suffix_series: Optional[pd.Series] = None,
    prefix_series: Optional[pd.Series] = None
) -> pd.Series:
    """
    Standardize street names by combining components and applying canonical abbreviations.

    Combines prefix, name, and suffix into a single standardized string. Applies
    uppercase formatting and replaces long street name versions with short abbreviations.

    Parameters
    ----------
    name_series : pd.Series
        Series containing street names.
    suffix_series : pd.Series, optional
        Series containing street suffixes.
    prefix_series : pd.Series, optional
        Series containing directional prefixes.

    Returns
    -------
    pd.Series
        Series of standardized street names.
    """
    full_name = name_series.fillna('')
    
    if prefix_series is not None:
        full_name = prefix_series.fillna('') + ' ' + full_name
        
    if suffix_series is not None:
        full_name = full_name + ' ' + suffix_series.fillna('')

    full_name = full_name.str.upper().str.strip()
    
    for long_ver, short_ver in SUFFIX_MAP.items():
        full_name = full_name.str.replace(fr'\b{long_ver}\b', short_ver, regex=True)

    full_name = full_name.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return full_name


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


def match_intersections(
    df_paving: pd.DataFrame,
    df_coll: pd.DataFrame
) -> pd.DataFrame:
    """
    Match intersection collisions to paving segments.

    Links intersection collisions (address_no_primary = 0) to paving segments by
    matching the primary road and intersecting cross-street against the paving
    data's cross-street columns.

    Parameters
    ----------
    df_paving : pd.DataFrame
        Paving segments with cleaned street names and cross-streets.
    df_coll : pd.DataFrame
        Collision data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['report_id', 'roadsegid'] for matched intersections.
        Empty DataFrame if no intersections found.
    """
    print("--- Matching Intersections ---")
    
    intersections = df_coll[
        (df_coll['address_no_primary'] == 0) & 
        (df_coll['address_name_intersecting'].notna())
    ].copy()
    
    if intersections.empty:
        return pd.DataFrame()

    intersections['clean_street_1'] = clean_street_name(
        intersections['address_road_primary'],
        intersections['address_sfx_primary'],
        intersections['address_pd_primary']
    )
    intersections['clean_street_2'] = clean_street_name(
        intersections['address_name_intersecting'],
        intersections['address_sfx_intersecting'],
        intersections['address_pd_intersecting']
    )
    
    df_paving['clean_xstrt1'] = clean_street_name(df_paving['xstrt1'])
    df_paving['clean_xstrt2'] = clean_street_name(df_paving['xstrt2'])
    df_paving['clean_xstrt1'] = df_paving['clean_xstrt1'].str.replace(
        r'\s*\([^)]*\)', '', regex=True
    ).str.strip()
    df_paving['clean_xstrt2'] = df_paving['clean_xstrt2'].str.replace(
        r'\s*\([^)]*\)', '', regex=True
    ).str.strip()

    matches_1 = pd.merge(
        intersections,
        df_paving[['roadsegid', 'clean_street', 'clean_xstrt1']],
        left_on=['clean_street_1', 'clean_street_2'],
        right_on=['clean_street', 'clean_xstrt1'],
        how='inner'
    )
    
    matches_2 = pd.merge(
        intersections,
        df_paving[['roadsegid', 'clean_street', 'clean_xstrt2']],
        left_on=['clean_street_1', 'clean_street_2'],
        right_on=['clean_street', 'clean_xstrt2'],
        how='inner'
    )
    
    matched = pd.concat([
        matches_1[['report_id', 'roadsegid']],
        matches_2[['report_id', 'roadsegid']]
    ])
    
    print(f"Intersections Matched: {len(matched)}")
    return matched


def match_segments(
    df_paving: pd.DataFrame,
    df_coll: pd.DataFrame
) -> pd.DataFrame:
    """
    Match mid-block collisions to paving segments by address range.

    Links non-intersection collisions to paving segments by matching street name
    and verifying the crash address falls within the segment's numeric address range.

    Parameters
    ----------
    df_paving : pd.DataFrame
        Paving segments with cleaned street names and address ranges.
    df_coll : pd.DataFrame
        Collision data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['report_id', 'roadsegid'] for matched segments.
        Empty DataFrame if no segments found.
    """
    print("--- Matching Segments ---")
    
    segments = df_coll[df_coll['address_no_primary'] != 0].copy()
    
    if segments.empty:
        return pd.DataFrame()

    segments['clean_street'] = clean_street_name(
        segments['address_road_primary'],
        segments['address_sfx_primary'],
        segments['address_pd_primary']
    )
    segments['addr_num'] = pd.to_numeric(segments['address_no_primary'], errors='coerce')
    
    merged = pd.merge(
        segments,
        df_paving[['roadsegid', 'clean_street', 'seg_min', 'seg_max']],
        on='clean_street',
        how='inner'
    )
    
    range_mask = (
        (merged['addr_num'] >= merged['seg_min']) & 
        (merged['addr_num'] <= merged['seg_max'])
    )
    valid_matches = merged[range_mask].copy()
    valid_matches = valid_matches.drop_duplicates(subset=['report_id'])
    
    print(f"Segments Matched: {len(valid_matches)}")
    return valid_matches[['report_id', 'roadsegid']]


def consolidate_matches(
    df_coll: pd.DataFrame,
    matches_intersection: pd.DataFrame,
    matches_segment: pd.DataFrame
) -> pd.DataFrame:
    """
    Consolidate and deduplicate collision-to-segment matches.

    Combines intersection and segment matches, removes duplicates to ensure
    one segment per crash, and exports unmatched records for debugging.

    Parameters
    ----------
    df_coll : pd.DataFrame
        Full collision dataset.
    matches_intersection : pd.DataFrame
        Intersection matches with columns ['report_id', 'roadsegid'].
    matches_segment : pd.DataFrame
        Segment matches with columns ['report_id', 'roadsegid'].

    Returns
    -------
    pd.DataFrame
        Deduplicated matches with columns ['report_id', 'roadsegid'].
    """
    print("--- Auditing Results ---")
    
    combined = pd.concat([matches_intersection, matches_segment])
    
    total_rows = len(combined)
    unique_reports = combined['report_id'].nunique()
    duplicates = total_rows - unique_reports
    
    print(f"Raw Matches (Intersection + Segment): {total_rows}")
    print(f"Duplicate Matches Found: {duplicates}")
    
    if duplicates > 0:
        print("...Resolving duplicates (keeping first match per ID)...")
    
    all_matches = combined.drop_duplicates(subset='report_id')
    
    matched_ids = set(all_matches['report_id'])
    all_ids = set(df_coll['report_id'])
    unmatched_ids = all_ids - matched_ids
    
    match_rate = (len(matched_ids) / len(all_ids)) * 100
    print(f"Total Unique Collisions in Data: {len(all_ids)}")
    print(f"Successfully Linked Unique Collisions: {len(matched_ids)}")
    print(f"Match Rate: {match_rate:.2f}%")
    
    if unmatched_ids:
        unmatched_df = df_coll[df_coll['report_id'].isin(unmatched_ids)].copy()
        unmatched_df['debug_clean_name'] = clean_street_name(
            unmatched_df['address_road_primary'],
            unmatched_df['address_sfx_primary']
        )
        unmatched_df.to_csv(FILES['output_debug'], index=False)
        print(f"Debug file saved: {FILES['output_debug']}")
    
    return all_matches


def extract_zone_type(zone_name: str) -> str:
    """
    Extract zone type category from zone name.

    Parameters
    ----------
    zone_name : str
        Zone name (e.g., 'RS-1-7', 'CC-3-6').

    Returns
    -------
    str
        Zone type category or 'Unknown' if not found.
    """
    if pd.isna(zone_name) or zone_name == '':
        return 'Unknown'
    
    zone_name = str(zone_name).strip()
    
    for prefix, category in ZONE_TYPE_MAP.items():
        if zone_name.startswith(prefix):
            return category
    
    if '-' in zone_name:
        prefix = zone_name.split('-')[0]
        if prefix in ZONE_TYPE_MAP:
            return ZONE_TYPE_MAP[prefix]
    
    return 'Unknown'


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


def visualize_zoning_collisions(df_zoning: pd.DataFrame) -> None:
    """
    Create visualizations for zoning collision analysis.

    Generates bar charts showing:
    - Top zones by total crashes
    - Top zones by crash rate (crashes per segment)
    - Top zones by injuries
    - Zone type comparison
    - Density analysis

    Parameters
    ----------
    df_zoning : pd.DataFrame
        Zone statistics DataFrame.
    """
    print("--- Creating Visualizations ---")
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3, height_ratios=[1, 1, 1.2])
    fig.suptitle('Traffic Collision Analysis by Zoning District', fontsize=16, fontweight='bold', y=0.995)
    
    top_n = 15
    
    # Top zones by total crashes
    ax1 = fig.add_subplot(gs[0, 0])
    top_crashes = df_zoning.nlargest(top_n, 'total_crashes')
    bars1 = ax1.barh(range(len(top_crashes)), top_crashes['total_crashes'], 
                     color=sns.color_palette("Blues_r", len(top_crashes)))
    ax1.set_yticks(range(len(top_crashes)))
    ax1.set_yticklabels(top_crashes['zone_name'], fontsize=9)
    ax1.set_xlabel('Total Crashes', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} Zones by Total Crashes', fontsize=12, fontweight='bold', pad=10)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    for i, (idx, val) in enumerate(zip(top_crashes.index, top_crashes['total_crashes'])):
        ax1.text(val + 20, i, f'{int(val)}', va='center', fontsize=9, fontweight='bold')
    
    # Top zones by crash rate
    ax2 = fig.add_subplot(gs[0, 1])
    top_rate = df_zoning[df_zoning['total_segments'] > 0].nlargest(top_n, 'crashes_per_segment')
    bars2 = ax2.barh(range(len(top_rate)), top_rate['crashes_per_segment'],
                     color=sns.color_palette("Reds_r", len(top_rate)))
    ax2.set_yticks(range(len(top_rate)))
    ax2.set_yticklabels(top_rate['zone_name'], fontsize=9)
    ax2.set_xlabel('Crashes per Segment', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top {top_n} Zones by Crash Rate', fontsize=12, fontweight='bold', pad=10)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    for i, (idx, val) in enumerate(zip(top_rate.index, top_rate['crashes_per_segment'])):
        ax2.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
    
    # Top zones by injuries
    ax3 = fig.add_subplot(gs[1, 0])
    top_injured = df_zoning.nlargest(top_n, 'injured')
    bars3 = ax3.barh(range(len(top_injured)), top_injured['injured'],
                     color=sns.color_palette("Oranges_r", len(top_injured)))
    ax3.set_yticks(range(len(top_injured)))
    ax3.set_yticklabels(top_injured['zone_name'], fontsize=9)
    ax3.set_xlabel('Total Injuries', fontsize=11, fontweight='bold')
    ax3.set_title(f'Top {top_n} Zones by Total Injuries', fontsize=12, fontweight='bold', pad=10)
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    for i, (idx, val) in enumerate(zip(top_injured.index, top_injured['injured'])):
        ax3.text(val + 5, i, f'{int(val)}', va='center', fontsize=9, fontweight='bold')
    
    # Zone type comparison
    ax4 = fig.add_subplot(gs[1, 1])
    if 'zone_type' in df_zoning.columns:
        zone_type_stats = df_zoning.groupby('zone_type').agg({
            'total_crashes': 'sum',
            'injured': 'sum',
            'killed': 'sum'
        }).reset_index().sort_values('total_crashes', ascending=False)
        
        x_pos = np.arange(len(zone_type_stats))
        width = 0.25
        
        bars1 = ax4.bar(x_pos - width, zone_type_stats['total_crashes'], width,
                       label='Crashes', color='steelblue', alpha=0.8)
        bars2 = ax4.bar(x_pos, zone_type_stats['injured'], width,
                       label='Injuries', color='coral', alpha=0.8)
        bars3 = ax4.bar(x_pos + width, zone_type_stats['killed'], width,
                       label='Fatalities', color='darkred', alpha=0.8)
        
        ax4.set_xlabel('Zone Type', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax4.set_title('Collisions by Zone Type Category', fontsize=12, fontweight='bold', pad=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(zone_type_stats['zone_type'], rotation=45, ha='right', fontsize=9)
        ax4.legend(loc='upper right')
        ax4.grid(axis='y', alpha=0.3)
    
    # Density analysis
    ax5 = fig.add_subplot(gs[2, :])
    if 'crashes_per_km2' in df_zoning.columns:
        density_data = df_zoning[df_zoning['crashes_per_km2'].notna() & 
                                 (df_zoning['crashes_per_km2'] > 0)].nlargest(20, 'crashes_per_km2')
        if len(density_data) > 0:
            bars = ax5.barh(range(len(density_data)), density_data['crashes_per_km2'],
                          color=sns.color_palette("YlOrRd", len(density_data)))
            ax5.set_yticks(range(len(density_data)))
            ax5.set_yticklabels(density_data['zone_name'], fontsize=8)
            ax5.set_xlabel('Crashes per km²', fontsize=11, fontweight='bold')
            ax5.set_title('Top 20 Zones by Crash Density (per km²)', 
                         fontsize=12, fontweight='bold', pad=10)
            ax5.invert_yaxis()
            ax5.grid(axis='x', alpha=0.3)
            for i, (idx, val) in enumerate(zip(density_data.index, density_data['crashes_per_km2'])):
                ax5.text(val + 1, i, f'{val:.1f}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    plt.savefig(FILES['output_visualization'], dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {FILES['output_visualization']}")
    plt.close()


def create_severity_map(
    df_zoning_stats: pd.DataFrame,
    gdf_zoning: gpd.GeoDataFrame
) -> None:
    """
    Create a map visualization of zones color-coded by crash severity.

    Parameters
    ----------
    df_zoning_stats : pd.DataFrame
        Zone statistics with severity metrics.
    gdf_zoning : gpd.GeoDataFrame
        Zoning polygons with geometry.
    """
    print("--- Creating Severity Map ---")
    
    gdf_with_stats = gdf_zoning.merge(
        df_zoning_stats[['zone_name', 'total_crashes', 'injured', 'killed', 'crashes_per_segment']],
        on='zone_name',
        how='left'
    )
    
    gdf_with_stats['total_crashes'] = gdf_with_stats['total_crashes'].fillna(0)
    gdf_with_stats['severity_score'] = (
        gdf_with_stats['total_crashes'] * 1 +
        gdf_with_stats['injured'].fillna(0) * 2 +
        gdf_with_stats['killed'].fillna(0) * 10
    )
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    gdf_with_stats = gdf_with_stats.to_crs('EPSG:4326')
    
    gdf_with_stats.plot(
        column='severity_score',
        ax=ax,
        cmap='YlOrRd',
        legend=True,
        missing_kwds={'color': 'lightgray'},
        edgecolor='black',
        linewidth=0.3,
        legend_kwds={
            'label': 'Severity Score (Crashes + 2×Injuries + 10×Fatalities)',
            'shrink': 0.8,
            'orientation': 'vertical'
        }
    )
    
    ax.set_title('Traffic Collision Severity by Zoning District', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(FILES['output_map'], dpi=300, bbox_inches='tight')
    print(f"Severity map saved: {FILES['output_map']}")
    plt.close()


def main() -> None:
    """Execute the complete collision zoning analysis pipeline."""
    df_paving, df_coll = load_and_filter_data()
    
    matches_int = match_intersections(df_paving, df_coll)
    matches_seg = match_segments(df_paving, df_coll)
    all_matches = consolidate_matches(df_coll, matches_int, matches_seg)
    
    gdf_zoning = load_zoning_data()
    gdf_paving_geom = load_paving_geometries()
    
    if gdf_zoning is not None and gdf_paving_geom is not None:
        segment_zoning = join_segments_to_zoning(gdf_paving_geom, gdf_zoning)
        zone_analysis = create_zoning_collision_analysis(df_coll, all_matches, segment_zoning, gdf_zoning)
        visualize_zoning_collisions(zone_analysis)
        create_severity_map(zone_analysis, gdf_zoning)
    else:
        print("Error: Could not load spatial data. Skipping zoning analysis.")


if __name__ == "__main__":
    main()

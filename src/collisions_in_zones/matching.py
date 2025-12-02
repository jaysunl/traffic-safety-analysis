"""
Collision matching functions for linking collisions to road segments.
"""

import pandas as pd

from .config import FILES
from .street_utils import clean_street_name


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


import pandas as pd

# file paths
FILES = {
    'paving': './data/raw/streets_repair_line_segments/sd_paving_segs_datasd.csv',
    'collisions': './data/raw/traffic_collisions_basic/pd_collisions_datasd.csv',
    'output_data': './data/processed/paving_analysis.csv',
    'output_debug': './misc/debug_unmatched_collisions.csv'
}

# filter collision data to match the 2023 pavement index
# THIS SHOULD BE ADJUSTED!
DATE_RANGE = {
    'start': '2023-01-01',
    'end': '2024-12-31'
}

# Map long street names (collision data) to short street names (paving data)
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

# prep
def clean_street_name(name_series, suffix_series=None, prefix_series=None):
    """
    Standardizes street names by combining prefix, name, and suffix, then applying 
    uppercase formatting and canonical abbreviations (e.g., 'STREET' -> 'ST').
    """

    full_name = name_series.fillna('')
    
    # prepend prefix if provided (e.g. "N", "S")
    if prefix_series is not None:
        full_name = prefix_series.fillna('') + ' ' + full_name
        
    # append suffix if provided (e.g. "ST", "AV")
    if suffix_series is not None:
        full_name = full_name + ' ' + suffix_series.fillna('')

    full_name = full_name.str.upper().str.strip()
    
    # replace long street name versions with short versions
    for long_ver, short_ver in SUFFIX_MAP.items():
        full_name = full_name.str.replace(fr'\b{long_ver}\b', short_ver, regex=True)

    # remove double spaces
    full_name = full_name.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return full_name

# core functions
def load_and_filter_data():
    """
    Loads raw CSV files, performs initial string cleaning and address range calculations, 
    and filters collision data to the configured date range.
    """

    print("--- Loading Data ---")
    
    df_paving = pd.read_csv(FILES['paving'])
    
    # paving data 
    # create cleaned street name column
    df_paving['clean_street'] = clean_street_name(df_paving['rd20full'])
    
    # remove paving-specific codes like "(SB)", "(NB)", "(FTG)"
    # regex: \s* matches space, \( matches '(', [^)]* matches content, \) matches ')'
    df_paving['clean_street'] = df_paving['clean_street'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()
    
    # convert address columns to numeric
    cols_to_num = ['llowaddr', 'lhighaddr', 'rlowaddr', 'rhighaddr']
    for c in cols_to_num:
        df_paving[c] = pd.to_numeric(df_paving[c], errors='coerce')
    
    # compute inclusive address ranges
    df_paving['seg_min'] = df_paving[['llowaddr', 'rlowaddr']].min(axis=1)
    df_paving['seg_max'] = df_paving[['lhighaddr', 'rhighaddr']].max(axis=1)
    
    # collision data
    df_coll = pd.read_csv(FILES['collisions'], dtype={'report_id': str})
    
    # filter by specified date range
    df_coll['date_time'] = pd.to_datetime(df_coll['date_time'])
    mask_date = (df_coll['date_time'] >= DATE_RANGE['start']) & (df_coll['date_time'] <= DATE_RANGE['end'])
    df_coll = df_coll[mask_date].copy()
    
    print(f"Loaded Paving Segments: {len(df_paving)}")
    print(f"Loaded Collisions ({DATE_RANGE['start']} to {DATE_RANGE['end']}): {len(df_coll)}")
    
    return df_paving, df_coll

def match_intersections(df_paving, df_coll):
    """
    Links intersection collisions to paving segments by matching the primary road 
    and the intersecting cross-street against the paving data's cross-street columns.
    """

    print("--- Matching Intersections ---")
    
    # filter for intersections, and drop rows where the cross street is missing as they can't be matched
    # intersections seem to always have address_no_primary = 0
    intersections = df_coll[
        (df_coll['address_no_primary'] == 0) & 
        (df_coll['address_name_intersecting'].notna())
    ].copy()
    
    if intersections.empty: 
        return pd.DataFrame()

    # collision keys for intersections
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
    
    # paving keys (cross streets)
    df_paving['clean_xstrt1'] = clean_street_name(df_paving['xstrt1'])
    df_paving['clean_xstrt2'] = clean_street_name(df_paving['xstrt2'])

    # remove paving-specific codes like "(SB)", "(NB)", "(FTG)"
    # regex: \s* matches space, \( matches '(', [^)]* matches content, \) matches ')'
    df_paving['clean_xstrt1'] = df_paving['clean_xstrt1'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()
    df_paving['clean_xstrt2'] = df_paving['clean_xstrt2'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()

    # join tables if main street is matched and a cross street matches 
    m1 = pd.merge(
        intersections, df_paving[['roadsegid', 'clean_street', 'clean_xstrt1']],
        left_on=['clean_street_1', 'clean_street_2'],
        right_on=['clean_street', 'clean_xstrt1'],
        how='inner'
    )
    
    m2 = pd.merge(
        intersections, df_paving[['roadsegid', 'clean_street', 'clean_xstrt2']],
        left_on=['clean_street_1', 'clean_street_2'],
        right_on=['clean_street', 'clean_xstrt2'],
        how='inner'
    )
    
    matched = pd.concat([m1[['report_id', 'roadsegid']], m2[['report_id', 'roadsegid']]])
    
    print(f"Intersections Matched: {len(matched)}")
    return matched

def match_segments(df_paving, df_coll):
    """
    Links mid-block collisions to paving segments by matching the street name 
    and verifying the crash address falls within the segment's numeric address range.
    """

    print("--- Matching Segments (Range Join) ---")
    
    # populate with non-intersections
    segments = df_coll[df_coll['address_no_primary'] != 0].copy()
    
    if segments.empty:
        return pd.DataFrame()

    # paving keys
    segments['clean_street'] = clean_street_name(
        segments['address_road_primary'], 
        segments['address_sfx_primary'],
        segments['address_pd_primary']
    )
    
    # make sure crash address is numeric for range comparison
    segments['addr_num'] = pd.to_numeric(segments['address_no_primary'], errors='coerce')
    
    # pair all non-intersection crashes with all segments on the same street
    merged = pd.merge(
        segments, 
        df_paving[['roadsegid', 'clean_street', 'seg_min', 'seg_max']],
        on='clean_street',
        how='inner'
    )
    
    # check if crash address falls within the segment's address range
    mask_range = (merged['addr_num'] >= merged['seg_min']) & (merged['addr_num'] <= merged['seg_max'])
    valid_matches = merged[mask_range].copy()
    
    # if a crash matches two segments we arbitrarily keep the first match to avoid double counting accidents
    # (could change this)
    valid_matches = valid_matches.drop_duplicates(subset=['report_id'])
    
    print(f"Segments Matched: {len(valid_matches)}")
    return valid_matches[['report_id', 'roadsegid']]

def consolidate_matches(df_coll, matches_intersection, matches_segment):
    """
    Consolidates intersection and segment matches, deduplicates results to ensure 
    one segment per crash, and exports unmatched records for debugging.
    """
    
    print("--- Auditing Results ---")
    
    # combine matches
    combined = pd.concat([matches_intersection, matches_segment])
    
    # check how many duplicates we found
    total_rows_before = len(combined)
    unique_reports = combined['report_id'].nunique()
    duplicates_found = total_rows_before - unique_reports
    
    print(f"Raw Matches (Intersection + Segment): {total_rows_before}")
    print(f"Duplicate Matches Found: {duplicates_found}")
    
    if duplicates_found > 0:
        print("...Resolving duplicates (keeping first match per ID)...")
    
    # remove duplicate collision reports
    all_matches = combined.drop_duplicates(subset='report_id')
    
    # find unmatched
    matched_ids = set(all_matches['report_id'])
    all_ids = set(df_coll['report_id'])
    unmatched_ids = all_ids - matched_ids
    
    # report stats
    match_rate = (len(matched_ids) / len(all_ids)) * 100
    print(f"Total Unique Collisions in Data: {len(all_ids)}")
    print(f"Successfully Linked Unique Collisions: {len(matched_ids)}")
    print(f"Match Rate: {match_rate:.2f}%")
    
    # save unmatched in a csv for troubleshooting (in misc folder)
    if unmatched_ids:
        unmatched_df = df_coll[df_coll['report_id'].isin(unmatched_ids)].copy()
        unmatched_df['debug_clean_name'] = clean_street_name(
            unmatched_df['address_road_primary'], 
            unmatched_df['address_sfx_primary']
        )
        unmatched_df.to_csv(FILES['output_debug'], index=False)
        print(f"Debug file saved: {FILES['output_debug']}")
    
    return all_matches

def create_final_dataset(df_paving, df_coll, all_matches):
    """
    Aggregates crash statistics (counts, injuries, deaths) by road segment ID 
    and merges them into the master paving dataset to produce the final CSV.
    """

    print("--- Aggregating Final Dataset ---")
    
    # construct dataframe to calculate aggregate stats for each road segment
    # this only includes matched collisions
    df_linked = pd.merge(df_coll, all_matches, on='report_id', how='inner')
    
    # convert injured and killed to numeric
    df_linked['injured'] = pd.to_numeric(df_linked['injured'], errors='coerce').fillna(0)
    df_linked['killed'] = pd.to_numeric(df_linked['killed'], errors='coerce').fillna(0)
    
    # aggregate stats by road segment
    stats = df_linked.groupby('roadsegid').agg({
        'report_id': 'count',  # total accidents
        'injured': 'sum',      # total injured
        'killed': 'sum'        # total killed
    }).reset_index()
    
    # the number of reports is the total number of crashes
    stats.rename(columns={'report_id': 'total_crashes'}, inplace=True)
    
    # merge stats back to paving data (we keep all paving segments, even those with no crashes)
    df_final = pd.merge(df_paving, stats, on='roadsegid', how='left')
    
    # ensure roads with no accidents show 0 accidents instead of NaN
    df_final[['total_crashes', 'injured', 'killed']] = df_final[['total_crashes', 'injured', 'killed']].fillna(0)
    
    # remove helper columns used for joining
    cols_to_drop = ['clean_street', 'clean_xstrt1', 'clean_xstrt2']
    df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], inplace=True)
    
    # save output
    df_final.to_csv(FILES['output_data'], index=False)
    print(f"File saved: {FILES['output_data']}")


def main():
    df_paving, df_coll = load_and_filter_data()
    
    matches_int = match_intersections(df_paving, df_coll)
    matches_seg = match_segments(df_paving, df_coll)
    
    all_matches = consolidate_matches(df_coll, matches_int, matches_seg)
    
    create_final_dataset(df_paving, df_coll, all_matches)

if __name__ == "__main__":
    main()
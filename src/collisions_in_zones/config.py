"""
Configuration constants for collisions analysis with zoning.
"""

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


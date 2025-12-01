"""
Zone type classification and utility functions.
"""

import pandas as pd

from .config import ZONE_TYPE_MAP


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


def get_general_zone_type(zone_type: str) -> str:
    """
    Map specific zone types to general zone type categories.
    
    Parameters
    ----------
    zone_type : str
        Specific zone type category (e.g., 'Residential Single-Family', 'Agricultural').
    
    Returns
    -------
    str
        General zone type category.
    """
    if pd.isna(zone_type) or zone_type == '' or zone_type == 'Unknown':
        return 'Unknown'
    
    zone_type = str(zone_type).strip()
    
    # Residential zones
    if 'Residential' in zone_type:
        return 'Residential'
    
    # Agricultural zones
    if 'Agricultural' in zone_type:
        return 'Agricultural'
    
    # Commercial zones
    if 'Commercial' in zone_type:
        return 'Commercial'
    
    # Industrial zones
    if 'Industrial' in zone_type:
        return 'Industrial'
    
    # Mixed Use zones
    if 'Mixed Use' in zone_type or 'Employment Mixed Use' in zone_type:
        return 'Mixed Use'
    
    # Open Space
    if 'Open Space' in zone_type:
        return 'Open Space'
    
    # Planned Development
    if 'Planned Development' in zone_type or 'Community Plan Update' in zone_type:
        return 'Planned Development'
    
    # Special Purpose
    if 'Special Purpose' in zone_type:
        return 'Special Purpose'
    
    return 'Unknown'


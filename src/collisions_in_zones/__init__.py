"""
Collisions analysis with zoning module.

This module provides functionality to analyze traffic collisions by zoning districts,
including data loading, matching, spatial operations, analysis, and visualization.
"""

from .main import main
from .data_loading import (
    load_and_filter_data,
    load_zoning_data,
    load_paving_geometries
)
from .matching import (
    match_intersections,
    match_segments,
    consolidate_matches
)
from .zone_utils import (
    extract_zone_type,
    get_general_zone_type
)
from .spatial_ops import join_segments_to_zoning
from .analysis import (
    calculate_zone_density,
    analyze_detailed_collisions_by_zone,
    analyze_violations_by_zone,
    create_zone_type_analysis,
    create_zoning_collision_analysis
)
from .visualization import (
    visualize_zoning_collisions,
    create_severity_map,
    create_zone_type_map
)

__all__ = [
    'main',
    'load_and_filter_data',
    'load_zoning_data',
    'load_paving_geometries',
    'match_intersections',
    'match_segments',
    'consolidate_matches',
    'extract_zone_type',
    'get_general_zone_type',
    'join_segments_to_zoning',
    'calculate_zone_density',
    'analyze_detailed_collisions_by_zone',
    'analyze_violations_by_zone',
    'create_zone_type_analysis',
    'create_zoning_collision_analysis',
    'visualize_zoning_collisions',
    'create_severity_map',
    'create_zone_type_map',
]


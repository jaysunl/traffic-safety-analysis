"""
Main execution script for collisions analysis with zoning.
"""

from .data_loading import load_and_filter_data, load_zoning_data, load_paving_geometries
from .matching import match_intersections, match_segments, consolidate_matches
from .spatial_ops import join_segments_to_zoning
from .analysis import create_zoning_collision_analysis
from .visualization import visualize_zoning_collisions, create_severity_map, create_zone_type_map


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

    create_zone_type_map('./data/processed/zone_map.png')


if __name__ == "__main__":
    main()


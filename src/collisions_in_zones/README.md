## Analysis performed by Jason Liang

This directory contains scripts that process, analyze, and visualize traffic collisions data through the lens of urban zoning districts in San Diego.

The purpose of this analysis is to determine how traffic collision patterns vary across different zoning districts and identify which zone types experience the highest collision rates, injuries, and fatalities. This helps city planners understand how land use patterns relate to traffic safety and identify high-risk zones that may need targeted safety interventions.

Below are important details about each script.

## data_loading.py

### Purpose

This script loads and prepares the necessary data for the zoning collision analysis. It handles loading paving segments, collision data, zoning polygons, and paving segment geometries from various data sources.

### Important Details

#### Data Loading
- **Paving Segments**: Loads street segment data from CSV, standardizes street names, and computes address ranges for matching collisions to segments.
- **Collision Data**: Loads traffic collision data and filters to the configured date range (default: 2023-01-01 to 2024-12-31).
- **Zoning Data**: Loads zoning polygons from GeoJSON, sets coordinate reference system (CRS), reprojects to a projected coordinate system for spatial accuracy, and calculates zone areas in square meters and square kilometers.
- **Paving Geometries**: Loads paving segment line geometries from GeoJSON for spatial operations.

#### Street Name Cleaning
- Removes directional codes like "(SB)", "(NB)", "(FTG)" from street names to improve matching accuracy.
- Standardizes street name formats for consistent matching.

## matching.py

### Purpose

This script matches traffic collisions to road segments using two different strategies: one for intersection collisions and one for mid-block collisions.

### Important Details

#### Intersection Collision Matching
- Matches collisions where `address_no_primary = 0` (indicating an intersection collision).
- Matches by comparing the primary road and intersecting cross-street against the paving segment's cross-street columns (`xstrt1` and `xstrt2`).
- Uses cleaned street names for more reliable matching.
- When multiple segments match, selects the segment with the longest length to avoid assigning collisions to "stub" or "connector" segments.

#### Mid-Block Collision Matching
- Matches collisions where `address_no_primary > 0` (indicating a mid-block collision).
- Matches by street name and verifies that the collision address falls within the segment's address range.
- Selects the segment with the "tightest" (smallest) address range around the collision address to prevent double counting collisions on overlapping segments and ensure the most precise match.

#### Match Consolidation
- Combines intersection and mid-block matches, prioritizing intersection matches when both exist for the same collision.
- Ensures each collision is matched to at most one segment.

## spatial_ops.py

### Purpose

This script performs spatial operations to link road segments to zoning districts using geometric intersection.

### Important Details

#### Segment-to-Zoning Join
- Uses geopandas spatial join operations to intersect road segment line geometries with zoning polygon geometries.
- Each segment is assigned to the zoning district(s) it intersects with.
- Creates the connection between road infrastructure and land use zoning, enabling collision analysis at the zoning district level.

## analysis.py

### Purpose

This script aggregates collision data by zoning district and calculates various statistics and metrics for each zone.

### Important Details

#### Zone Aggregation
- Aggregates collisions by zoning district, calculating:
  - Total crashes, injuries, and fatalities per zone
  - Number of segments with crashes vs. total segments
  - Crash rates per segment
  - Crash density (crashes per km²)
  - Detailed collision characteristics (vehicle types, violation types, injury levels)

#### Zone Type Classification
- Extracts zone type codes (e.g., "RS-1-7", "CC-3-6") from zone names.
- Maps specific zone codes to general categories:
  - Residential (single-family, multi-family)
  - Commercial (general, neighborhood, planned development)
  - Industrial (limited, general)
  - Agricultural
  - Mixed Use
  - Open Space
  - Planned Development
  - Special Purpose

#### Density Metrics
- Calculates crash density per square kilometer for each zone.
- Helps identify zones where collisions are concentrated relative to zone size, which may point to infrastructure or design issues.

#### Detailed Collision Analysis
- Analyzes detailed collision characteristics including:
  - Vehicle types involved
  - Person roles (driver, passenger, pedestrian, etc.)
  - Injury levels
  - Violation types

#### Output
- Saves results to `data/processed/collisions_analysis_with_zoning.csv`.
- Displays top 10 zones by crash count for quick reference.

## visualization.py

### Purpose

This script creates visualizations for the zoning collision analysis, including bar charts, maps, and zone type visualizations.

### Important Details

#### visualize_zoning_collisions()
- Creates a comprehensive dashboard with multiple bar charts:
  - Top zones by total crashes
  - Top zones by crash rate (crashes per segment)
  - Top zones by injuries
  - Zone type comparison (crashes, injuries, fatalities by zone type)
  - Top zones by crash density (crashes per km²)

#### create_severity_map()
- Creates a map visualization of zones color-coded by crash severity.
- Severity score calculated as: Crashes + 2×Injuries + 10×Fatalities
- Overlays highways and freeways on the map for geographic context.
- Includes a colorbar legend for severity scores and a separate legend for highways.

#### create_zone_type_map()
- Creates a color-coded map of zones by general zone type (Residential, Commercial, Industrial, etc.).
- Each zone type has a distinct color for easy identification.
- Overlays highways and freeways on the map for geographic context.
- Includes a legend showing all zone types and highway demarcation.

#### Highway/Freeway Overlay
- Loads highway and freeway data from roads dataset.
- Filters for highways/freeways using FUNCLASS and SEGCLASS fields:
  - Freeways (FUNCLASS='F' or SEGCLASS='1')
  - Expressways (FUNCLASS='E' or SEGCLASS='1')
  - Highways/State Routes (SEGCLASS='2')
  - Freeway ramps (FUNCLASS='R' or SEGCLASS='8' or '9')
- Displays highways in blue (#2E86AB) with distinct line styling.
- Adds highway demarcation to map legends.

## zone_utils.py

### Purpose

This script contains utility functions for extracting and classifying zone types from zone names.

### Important Details

#### Zone Type Extraction
- Extracts zone type codes from zone names (e.g., "RS-1-7" → "RS").
- Handles various zone naming formats and edge cases.

#### Zone Type Classification
- Maps specific zone codes to general zone type categories.
- Provides consistent categorization across the analysis.

## street_utils.py

### Purpose

This script contains utility functions for cleaning and standardizing street names.

### Important Details

#### Street Name Cleaning
- Standardizes street name formats for consistent matching.
- Handles abbreviations, suffixes, and directional indicators.
- Removes special characters and normalizes whitespace.

## config.py

### Purpose

This script contains configuration constants used throughout the analysis.

### Important Details

#### File Paths
- Defines paths to all input and output data files.
- Centralizes file path management for easy updates.

#### Date Range
- Configures the date range for collision data analysis (default: 2016-01-01 to 2024-12-31).

#### Zone Type Mapping
- Maps specific zone codes to general zone type categories.

#### Suffix Mapping
- Maps full street suffixes to standard abbreviations for consistent string matching.

## main.py

### Purpose

This script provides the main execution pipeline for the complete collision zoning analysis.

### Important Details

#### Pipeline Steps
1. Loads and filters paving segments and collision data
2. Matches collisions to road segments (intersections and mid-block)
3. Consolidates matches
4. Loads zoning and paving geometry data
5. Performs spatial join to link segments to zones
6. Creates zoning collision analysis
7. Generates visualizations (bar charts, severity map, zone type map)

#### Usage
- Can be run as a standalone script or imported as a module.
- The main function executes the complete pipeline from data loading to visualization.


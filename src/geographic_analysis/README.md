# Geographic Hotspot Analysis

This directory contains scripts for geographic hotspot analysis of traffic collisions and pavement conditions in San Diego.

## Overview

The hotspot analysis identifies geographic clusters of:
- High collision density areas
- Poor pavement condition areas  
- Combined risk hotspots (high collisions + poor pavement)

## Scripts

### `hotspot_detection.py`

Performs spatial hotspot analysis using two complementary methods:

1. **DBSCAN Clustering**: Identifies spatial clusters of segments with high collision counts
   - Uses density-based clustering to find groups of nearby high-collision segments
   - Parameters: `eps=1000m` (1km), `min_samples=5`

2. **Getis-Ord Gi* Statistic**: Identifies statistically significant hotspots
   - Uses spatial autocorrelation to find areas with significantly higher/lower values than expected
   - Identifies hotspots (Gi* > 1.96, p < 0.05) and coldspots (Gi* < -1.96, p < 0.05)
   - Uses k-nearest neighbors (k=10) for spatial weights

## Outputs

### Data Files

- `data/processed/geographic_hotspots.csv`: Contains hotspot analysis results for all segments
  - `cluster_id`: DBSCAN cluster assignment (-1 = noise, >=0 = cluster ID)
  - `gi_star`: Getis-Ord Gi* statistic value
  - `is_hotspot`: Boolean indicating statistically significant hotspot
  - `is_coldspot`: Boolean indicating statistically significant coldspot
  - Plus all segment statistics (collisions, injuries, PCI, etc.)

### Visualizations

- `data/processed/geographic_hotspots_map.png`: Four-panel map showing:
  1. **Collision Density**: Heat map of total collisions by segment
  2. **DBSCAN Clusters**: Spatial clusters of high-collision segments
  3. **Getis-Ord Hotspots**: Statistically significant hotspots (red) and coldspots (blue)
  4. **Combined Risk**: Risk score combining collision density and poor pavement condition

## Usage

```bash
python src/geographic_analysis/hotspot_detection.py
```

## Analysis Results (2024 Run)

- **Total segments analyzed**: 25,765
- **Segments with collisions**: 8,809
- **Total collisions (2016-2024)**: 49,601
- **DBSCAN clusters found**: 12
- **Getis-Ord hotspots**: 121
- **Getis-Ord coldspots**: 0

### Top Hotspot Segments

The analysis identified several high-risk segments:
- Highest collision segment: 134 collisions, 41 injuries
- Multiple segments with 40+ collisions and 20+ injuries
- Some hotspots have poor pavement conditions (PCI < 50), compounding risk

## Methodology

### Data Preparation

1. Loads segment geometries from GeoJSON
2. Aggregates collision and PCI data by segment across 2016-2024
3. Calculates collision rates per mile
4. Reprojects to UTM Zone 11N for accurate distance calculations

### DBSCAN Clustering

- Filters to segments with ≥5 collisions
- Calculates segment centroids
- Performs density-based clustering
- Identifies clusters of nearby high-collision segments

### Getis-Ord Gi* Analysis

- Creates k-nearest neighbor spatial weights matrix (k=10)
- Calculates Gi* statistic for each segment
- Tests statistical significance (p < 0.05)
- Identifies hotspots (high positive autocorrelation) and coldspots (high negative autocorrelation)

### Combined Risk Score

- Normalizes collision counts (0-1 scale)
- Normalizes PCI risk (lower PCI = higher risk, 0-1 scale)
- Combines: `risk = 0.6 * collision_norm + 0.4 * pci_risk_norm`

## Limitations

1. **Collision coordinates**: Collision data doesn't have direct coordinates, so analysis is based on segment-level aggregation
2. **Spatial weights**: Uses simple k-nearest neighbors; could be enhanced with distance-based weights
3. **Temporal aggregation**: Aggregates across entire time period; could analyze temporal trends
4. **Traffic normalization**: Limited traffic count data means collision rates may not be fully normalized

## Future Enhancements

- Temporal hotspot analysis (how hotspots change over time)
- Integration with zoning data for land-use context
- Distance-based spatial weights for Getis-Ord
- Interactive map visualization (Folium)
- Hotspot prediction models


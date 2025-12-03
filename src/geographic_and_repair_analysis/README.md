

# Geographic and Repair Analysis

This folder contains scripts for analyzing traffic collisions and pavement conditions in San Diego. The analysis is divided into two primary modules:

1.  **Geographic Hotspot Analysis:** Identifies clusters of high collision density and poor pavement conditions.
2.  **Street Repair Evaluation:** Assesses the effectiveness of street repair projects on pavement quality and safety.

-----

## Part 1: Geographic Hotspot Analysis

This module performs spatial analysis to identify high-risk areas based on traffic collisions and pavement condition index (PCI).

### Overview

The analysis identifies geographic clusters of:

  * High collision density areas.
  * Poor pavement condition areas.
  * **Combined Risk Hotspots:** Areas exhibiting both high collision rates and poor pavement.

### Methodology

The `hotspot_detection.py` script employs two complementary spatial statistics methods:

#### 1\. DBSCAN Clustering

Uses density-based clustering to find groups of nearby high-collision segments.

  * **Target:** Segments with ≥5 collisions.
  * **Parameters:** `eps=1000m` (1km radius), `min_samples=5`.
  * **Goal:** Identifies "neighborhoods" of high activity.

#### 2\. Getis-Ord Gi\* Statistic

Uses spatial autocorrelation to find areas with significantly higher or lower values than expected by chance.

  * **Weights:** k-nearest neighbors (k=10).
  * **Significance:**
      * **Hotspot:** Gi\* \> 1.96 (p \< 0.05) — Cluster of high values.
      * **Coldspot:** Gi\* \< -1.96 (p \< 0.05) — Cluster of low values.

#### 3\. Combined Risk Score

A composite metric calculated for every segment:

  * Collision counts are normalized (0–1).
  * PCI risk is normalized (Inverse PCI: lower PCI = higher risk).
  * **Formula:** `risk = 0.6 * collision_norm + 0.4 * pci_risk_norm`

### 2024 Analysis Results

  * **Total segments analyzed:** 25,765
  * **Segments with collisions:** 8,809
  * **DBSCAN clusters found:** 12
  * **Getis-Ord hotspots:** 121
  * **Highest Risk Segment:** 134 collisions, 41 injuries.

### Limitations

1.  **Coordinates:** Analysis is based on segment-level aggregation as raw collision coordinates are unavailable.
2.  **Normalization:** Lack of granular traffic volume data limits the ability to calculate true crash rates per vehicle-mile traveled.

-----

## Part 2: Street Repair Projects Evaluation

This module evaluates whether the city's repair projects (Slurry, Overlay, Concrete) are achieving their intended goals.

### Purpose

While Part 1 identifies where the problems are, Part 2 evaluates the solution. It answers:

  * Do repairs effectively improve PCI?
  * Do repairs lead to a reduction in traffic collisions?
  * Which repair types are most effective and how long do they last?

### Analysis Components

#### 1\. PCI Improvement Analysis

  * Compares estimated PCI immediately before and after repair.
  * **Metric:** "Effective" repairs must improve PCI by ≥10 points.

#### 2\. Repair Type Comparison

Evaluates efficacy across three categories:

  * **SLURRY:** Surface treatment (approx. 28k projects).
  * **OVERLAY:** Full asphalt overlay (approx. 9k projects).
  * **CONCRETE:** Structural repair (approx. 200 projects).

#### 3\. Collision Reduction

  * Compares average annual crashes/mile 2 years *before* vs. 2 years *after* the repair date.
  * **Interpretation:** Positive values in the output indicate a reduction in crashes.

#### 4\. Longevity Analysis

  * Estimates the time until "Significant Degradation" (a drop of \>5 PCI points from the post-repair state).

### Methodology Notes

  * **PCI Estimation:** Because inspections happen periodically (e.g., 2016, 2023), PCI values for specific repair dates are estimated using linear decay models specific to the road's functional class.
  * **Analysis Window:** 2016–2023.

-----

## Usage

### Prerequisites

Ensure your Python environment is set up with the necessary geospatial libraries (Pandas, GeoPandas, PySAL/ESDA, Scikit-learn).

### Running the Scripts

**1. Run Hotspot Analysis:**

```bash
python src/geographic_analysis/hotspot_detection.py
```

**2. Run Repair Effectiveness Evaluation:**

```bash
python -m src.repair_projects_evaluation.repair_effectiveness_analysis
```

*Or via Python:*

```python
from src.repair_projects_evaluation.repair_effectiveness_analysis import main
main()
```

-----

## Outputs

All outputs are generated in the `data/processed/` directory.

### Geographic Hotspot Outputs

| File | Description |
| :--- | :--- |
| `geographic_hotspots.csv` | Contains Cluster IDs, Gi\* statistics, Significance booleans, and segment statistics. |
| `geographic_hotspots_map.png` | 4-Panel Map: Collision Density, DBSCAN Clusters, Gi\* Hotspots, and Combined Risk. |

### Repair Evaluation Outputs

| File | Description |
| :--- | :--- |
| `repair_effectiveness_analysis.png` | Comprehensive visualization dashboard. |
| `repair_pci_analysis.csv` | Detailed PCI improvement data per repair. |
| `repair_collision_analysis.csv` | Pre/Post collision reduction metrics. |
| `repair_longevity_analysis.csv` | Degradation rates and estimated lifespan data. |

-----

## Future Enhancements

  * **Temporal Trends:** Analyze how hotspots shift over time.
  * **Contextual Data:** Integrate zoning/land-use data to explain hotspot clusters.
  * **Interactive Maps:** Generate Folium/Leaflet maps for web interaction.
  * **Cost-Benefit:** Integrate cost data to determine the financial efficiency of specific repair types.
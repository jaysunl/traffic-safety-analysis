"""
Geographic Hotspot Analysis

This script performs spatial hotspot analysis to identify geographic clusters of:
1. High collision density areas
2. Poor pavement condition areas
3. Combined risk hotspots (high collisions + poor pavement)

Uses spatial clustering (DBSCAN) and hotspot detection (Getis-Ord Gi*) to identify
statistically significant hotspots.

This analysis helps identify neighborhoods/areas needing priority attention and
supports targeted intervention strategies.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FILES = {
    'segments_geojson': PROJECT_ROOT / 'data/raw/streets_repair_line_segments/sd_paving_segs_datasd.geojson',
    'processed_data': PROJECT_ROOT / 'data/processed/segments_collisions_pci_counts.csv',
    'collisions': PROJECT_ROOT / 'data/raw/traffic_collisions_basic/pd_collisions_datasd.csv',
    'zoning_geojson': PROJECT_ROOT / 'data/raw/zoning/zoning_datasd.geojson',
    'output_hotspots': PROJECT_ROOT / 'data/processed/geographic_hotspots.csv',
    'output_map': PROJECT_ROOT / 'data/processed/geographic_hotspots_map.png',
}

# Analysis parameters
DATE_START = '2016-01-01'
DATE_END = '2024-12-31'
MIN_COLLISIONS_FOR_HOTSPOT = 5  # Minimum collisions to consider for hotspot analysis
DBSCAN_EPS = 0.01  # Distance threshold for DBSCAN (in degrees, ~1km)
DBSCAN_MIN_SAMPLES = 5  # Minimum samples for a cluster


def load_data() -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Load segment geometries and processed collision/PCI data.
    
    Returns:
        Tuple of (segments_gdf, segments_data)
    """
    
    # Load segment geometries
    print("Loading segment geometries...")
    segments_gdf = gpd.read_file(FILES['segments_geojson'])
    print(f"Loaded {len(segments_gdf)} segments with geometry")
    
    # Set CRS if not set
    if segments_gdf.crs is None:
        segments_gdf.set_crs('EPSG:4326', inplace=True)
    
    # Reproject to a projected CRS for distance calculations
    target_crs = 'EPSG:32611'  # UTM Zone 11N (San Diego)
    try:
        segments_gdf = segments_gdf.to_crs(target_crs)
        print(f"Reprojected to {target_crs}")
    except Exception as e:
        print(f"Warning: Could not reproject to {target_crs}: {e}")
        print("Using original CRS")
    
    # Load processed data with collisions and PCI
    print("Loading processed segment data...")
    segments_data = pd.read_csv(FILES['processed_data'])
    
    # Filter to date range
    segments_data['year'] = pd.to_numeric(segments_data['year'], errors='coerce')
    date_mask = (
        (segments_data['year'] >= int(DATE_START[:4])) & 
        (segments_data['year'] <= int(DATE_END[:4]))
    )
    segments_data = segments_data[date_mask].copy()
    print(f"Filtered to {len(segments_data)} segment-year records ({DATE_START[:4]}-{DATE_END[:4]})")
    
    # Aggregate by segment (sum across years)
    print("Aggregating data by segment...")
    segment_stats = segments_data.groupby('iamfloc').agg({
        'total_crashes': 'sum',
        'injured': 'sum',
        'killed': 'sum',
        'avg_pci': 'mean',  # Average PCI over time period
        'pav_length': 'first',  # Should be constant per segment
        'traffic_count': 'mean',  # Average traffic count
    }).reset_index()
    
    # Calculate collision rate per mile (if we have length)
    segment_stats['crashes_per_mile'] = (
        segment_stats['total_crashes'] / 
        (segment_stats['pav_length'] / 5280)  # Convert feet to miles
    ).replace([np.inf, -np.inf], np.nan)
    
    print(f"Aggregated to {len(segment_stats)} unique segments")
    print(f"Total collisions: {segment_stats['total_crashes'].sum():.0f}")
    print(f"Segments with collisions: {(segment_stats['total_crashes'] > 0).sum()}")
    
    return segments_gdf, segment_stats


def calculate_segment_centroids(segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate centroids for each segment.
    
    Args:
        segments_gdf: GeoDataFrame with segment line geometries
        
    Returns:
        GeoDataFrame with centroid points
    """
    print("\nCalculating segment centroids...")
    centroids = segments_gdf.copy()
    centroids['geometry'] = centroids.geometry.centroid
    return centroids


def perform_dbscan_clustering(
    segments_gdf: gpd.GeoDataFrame,
    segment_stats: pd.DataFrame,
    metric: str = 'total_crashes'
) -> pd.DataFrame:
    """
    Perform DBSCAN clustering on high-collision segments.
    
    Args:
        segments_gdf: GeoDataFrame with segment geometries
        segment_stats: DataFrame with segment statistics
        metric: Column name to use for filtering high-value segments
        
    Returns:
        DataFrame with cluster labels
    """
    print(f"\nPerforming DBSCAN clustering on {metric}...")
    
    # Merge stats with geometries
    segments_merged = segments_gdf.merge(
        segment_stats[['iamfloc', metric]],
        on='iamfloc',
        how='inner'
    )
    
    # Filter to segments with significant collisions
    high_value = segments_merged[segments_merged[metric] >= MIN_COLLISIONS_FOR_HOTSPOT].copy()
    print(f"Segments with {metric} >= {MIN_COLLISIONS_FOR_HOTSPOT}: {len(high_value)}")
    
    if len(high_value) == 0:
        print("No segments meet threshold for clustering")
        return pd.DataFrame()
    
    # Calculate centroids
    centroids = high_value.copy()
    centroids['geometry'] = centroids.geometry.centroid
    
    # Extract coordinates
    coords = np.array([
        [geom.x, geom.y] for geom in centroids.geometry
    ])
    
    # Convert DBSCAN eps from degrees to meters (approximate)
    # For UTM Zone 11N, 1 degree ≈ 111km, so 0.01° ≈ 1.1km
    # For UTM, eps should be in meters
    if segments_gdf.crs and '32611' in str(segments_gdf.crs):
        # Already in UTM, use meters directly
        eps_meters = 1000  # 1km
    else:
        # In lat/lon, need to convert
        # Rough conversion: 1 degree ≈ 111km at equator
        eps_meters = DBSCAN_EPS * 111000
    
    # Perform DBSCAN
    clustering = DBSCAN(eps=eps_meters, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean')
    cluster_labels = clustering.fit_predict(coords)
    
    # Add cluster labels
    high_value['cluster_id'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = high_value.groupby('cluster_id').agg({
        'iamfloc': 'count',
        metric: ['sum', 'mean'],
    }).reset_index()
    cluster_stats.columns = ['cluster_id', 'num_segments', f'{metric}_sum', f'{metric}_mean']
    
    print(f"Found {len(cluster_stats[cluster_stats['cluster_id'] != -1])} clusters")
    print(f"Noise points (not in clusters): {(cluster_labels == -1).sum()}")
    
    return high_value[['iamfloc', 'cluster_id']].merge(cluster_stats, on='cluster_id', how='left')


def calculate_getis_ord_gi(
    segments_gdf: gpd.GeoDataFrame,
    segment_stats: pd.DataFrame,
    metric: str = 'total_crashes',
    k_neighbors: int = 10
) -> pd.DataFrame:
    """
    Calculate Getis-Ord Gi* statistic for hotspot detection.
    
    Args:
        segments_gdf: GeoDataFrame with segment geometries
        segment_stats: DataFrame with segment statistics
        metric: Column name to analyze
        k_neighbors: Number of nearest neighbors for spatial weights
        
    Returns:
        DataFrame with Gi* statistics and p-values
    """
    print(f"\nCalculating Getis-Ord Gi* statistic for {metric}...")
    
    # Merge stats with geometries
    segments_merged = segments_gdf.merge(
        segment_stats[['iamfloc', metric]],
        on='iamfloc',
        how='inner'
    )
    
    # Filter to segments with data
    segments_merged = segments_merged[segments_merged[metric].notna()].copy()
    
    if len(segments_merged) == 0:
        print("No segments with valid data")
        return pd.DataFrame()
    
    # Calculate centroids
    centroids = segments_merged.copy()
    centroids['geometry'] = centroids.geometry.centroid
    
    # Extract coordinates
    coords = np.array([
        [geom.x, geom.y] for geom in centroids.geometry
    ])
    
    # Calculate distance matrix
    print("Calculating distance matrix...")
    distances = squareform(pdist(coords))
    
    # Create spatial weights matrix (k-nearest neighbors)
    n = len(centroids)
    weights = np.zeros((n, n))
    
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        nearest_indices = np.argsort(distances[i])[1:k_neighbors+1]
        weights[i, nearest_indices] = 1.0
        # Row normalize
        if weights[i].sum() > 0:
            weights[i] = weights[i] / weights[i].sum()
    
    # Calculate Gi* statistic
    values = segments_merged[metric].values
    mean_val = values.mean()
    std_val = values.std()
    
    if std_val == 0:
        print("Warning: Standard deviation is 0, cannot calculate Gi*")
        return pd.DataFrame()
    
    gi_star = np.zeros(n)
    p_values = np.zeros(n)
    
    print("Calculating Gi* statistics...")
    for i in range(n):
        # Weighted sum
        weighted_sum = np.dot(weights[i], values)
        # Number of neighbors
        n_neighbors = weights[i].sum()
        
        # Gi* formula
        numerator = weighted_sum - mean_val * n_neighbors
        denominator = std_val * np.sqrt(
            (n * n_neighbors - n_neighbors**2) / (n - 1)
        )
        
        if denominator > 0:
            gi_star[i] = numerator / denominator
            # Two-tailed p-value
            p_values[i] = 2 * (1 - stats.norm.cdf(abs(gi_star[i])))
        else:
            gi_star[i] = 0
            p_values[i] = 1.0
    
    # Add results to dataframe
    result = segments_merged[['iamfloc']].copy()
    result['gi_star'] = gi_star
    result['gi_p_value'] = p_values
    result['is_hotspot'] = (gi_star > 1.96) & (p_values < 0.05)  # 95% confidence
    result['is_coldspot'] = (gi_star < -1.96) & (p_values < 0.05)
    
    n_hotspots = result['is_hotspot'].sum()
    n_coldspots = result['is_coldspot'].sum()
    
    print(f"Found {n_hotspots} hotspots (Gi* > 1.96, p < 0.05)")
    print(f"Found {n_coldspots} coldspots (Gi* < -1.96, p < 0.05)")
    
    return result


def create_hotspot_map(
    segments_gdf: gpd.GeoDataFrame,
    segment_stats: pd.DataFrame,
    dbscan_results: pd.DataFrame,
    gi_results: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a comprehensive map visualization of hotspots.
    
    Args:
        segments_gdf: GeoDataFrame with segment geometries
        segment_stats: DataFrame with segment statistics
        dbscan_results: DataFrame with DBSCAN cluster labels
        gi_results: DataFrame with Getis-Ord Gi* results
        output_path: Path to save the map
    """
    
    # Merge all data
    segments_merged = segments_gdf.merge(
        segment_stats,
        on='iamfloc',
        how='left'
    )
    
    if not dbscan_results.empty:
        segments_merged = segments_merged.merge(
            dbscan_results[['iamfloc', 'cluster_id']],
            on='iamfloc',
            how='left'
        )
        segments_merged['in_cluster'] = segments_merged['cluster_id'].notna() & (segments_merged['cluster_id'] != -1)
    else:
        segments_merged['in_cluster'] = False
    
    if not gi_results.empty:
        segments_merged = segments_merged.merge(
            gi_results[['iamfloc', 'gi_star', 'is_hotspot', 'is_coldspot']],
            on='iamfloc',
            how='left'
        )
        # Fill NaN values with False
        segments_merged['is_hotspot'] = segments_merged['is_hotspot'].fillna(False).astype(bool)
        segments_merged['is_coldspot'] = segments_merged['is_coldspot'].fillna(False).astype(bool)
        segments_merged['gi_star'] = segments_merged['gi_star'].fillna(0)
    else:
        segments_merged['is_hotspot'] = False
        segments_merged['is_coldspot'] = False
        segments_merged['gi_star'] = 0
    
    # Convert to WGS84 for visualization
    if segments_merged.crs != 'EPSG:4326':
        segments_merged = segments_merged.to_crs('EPSG:4326')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Geographic Hotspot Analysis: Collision and Pavement Condition Hotspots', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Get bounds
    bounds = segments_merged.total_bounds
    padding = 0.01
    xlim = [bounds[0] - padding, bounds[2] + padding]
    ylim = [bounds[1] - padding, bounds[3] + padding]
    
    # Map 1: Collision Density
    ax1 = axes[0, 0]
    segments_merged.plot(
        column='total_crashes',
        ax=ax1,
        cmap='Reds',
        legend=True,
        linewidth=0.5,
        legend_kwds={
            'label': 'Total Collisions',
            'shrink': 0.8,
            'orientation': 'vertical'
        },
        missing_kwds={'color': 'lightgray', 'label': 'No data'}
    )
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect('equal')
    ax1.set_title('Collision Density by Segment', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Map 2: DBSCAN Clusters
    ax2 = axes[0, 1]
    # Background segments
    segments_merged[~segments_merged['in_cluster']].plot(
        ax=ax2,
        color='lightgray',
        linewidth=0.3,
        alpha=0.5
    )
    # Clustered segments
    if segments_merged['in_cluster'].any():
        clustered = segments_merged[segments_merged['in_cluster']]
        clustered.plot(
            column='cluster_id',
            ax=ax2,
            cmap='Set1',
            linewidth=1.5,
            legend=True,
            legend_kwds={
                'label': 'Cluster ID',
                'shrink': 0.8,
                'orientation': 'vertical'
            }
        )
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect('equal')
    ax2.set_title('DBSCAN Spatial Clusters (High Collision Areas)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Map 3: Getis-Ord Gi* Hotspots
    ax3 = axes[1, 0]
    # Background
    segments_merged[~segments_merged['is_hotspot'] & ~segments_merged['is_coldspot']].plot(
        ax=ax3,
        color='lightgray',
        linewidth=0.3,
        alpha=0.5
    )
    # Coldspots
    if segments_merged['is_coldspot'].any():
        segments_merged[segments_merged['is_coldspot']].plot(
            ax=ax3,
            color='blue',
            linewidth=1.5,
            alpha=0.7,
            label='Coldspot (Low)'
        )
    # Hotspots
    if segments_merged['is_hotspot'].any():
        segments_merged[segments_merged['is_hotspot']].plot(
            ax=ax3,
            color='red',
            linewidth=1.5,
            alpha=0.7,
            label='Hotspot (High)'
        )
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_aspect('equal')
    ax3.set_title('Getis-Ord Gi* Hotspot Detection', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.axis('off')
    
    # Map 4: Combined Risk (Collisions + Poor PCI)
    ax4 = axes[1, 1]
    # Calculate risk score (normalize and combine)
    segments_merged['collision_norm'] = (
        (segments_merged['total_crashes'] - segments_merged['total_crashes'].min()) /
        (segments_merged['total_crashes'].max() - segments_merged['total_crashes'].min() + 1e-10)
    )
    segments_merged['pci_risk_norm'] = (
        (100 - segments_merged['avg_pci']) / 100  # Lower PCI = higher risk
    ).fillna(0)
    segments_merged['combined_risk'] = (
        segments_merged['collision_norm'] * 0.6 + 
        segments_merged['pci_risk_norm'] * 0.4
    )
    
    segments_merged.plot(
        column='combined_risk',
        ax=ax4,
        cmap='RdYlGn_r',  # Red = high risk, Green = low risk
        legend=True,
        linewidth=0.5,
        legend_kwds={
            'label': 'Combined Risk Score',
            'shrink': 0.8,
            'orientation': 'vertical'
        },
        missing_kwds={'color': 'lightgray', 'label': 'No data'}
    )
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_aspect('equal')
    ax4.set_title('Combined Risk: Collisions + Poor Pavement', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Only save if output_path is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved map: {output_path}")
        plt.close(fig)
    # If output_path is None, return the figure without closing (for notebook display)
    
    return fig


def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("Geographic Hotspot Analysis")
    print("="*70)
    
    # Load data
    segments_gdf, segment_stats = load_data()
    
    # Perform DBSCAN clustering
    dbscan_results = perform_dbscan_clustering(segments_gdf, segment_stats, 'total_crashes')
    
    # Perform Getis-Ord Gi* analysis
    gi_results = calculate_getis_ord_gi(segments_gdf, segment_stats, 'total_crashes')
    
    # Combine results
    if not dbscan_results.empty and not gi_results.empty:
        hotspot_summary = segment_stats.merge(
            dbscan_results[['iamfloc', 'cluster_id']],
            on='iamfloc',
            how='left'
        ).merge(
            gi_results[['iamfloc', 'gi_star', 'is_hotspot', 'is_coldspot']],
            on='iamfloc',
            how='left'
        )
        
        # Save results
        hotspot_summary.to_csv(FILES['output_hotspots'], index=False)
        print(f"\nSaved hotspot results: {FILES['output_hotspots']}")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("Hotspot Summary Statistics")
        print("="*70)
        print(f"Total segments analyzed: {len(hotspot_summary)}")
        print(f"Segments in DBSCAN clusters: {hotspot_summary['cluster_id'].notna().sum()}")
        print(f"Getis-Ord hotspots: {hotspot_summary['is_hotspot'].sum()}")
        print(f"Getis-Ord coldspots: {hotspot_summary['is_coldspot'].sum()}")
        
        if hotspot_summary['is_hotspot'].sum() > 0:
            print("\nTop 10 Hotspot Segments by Collisions:")
            top_hotspots = hotspot_summary[hotspot_summary['is_hotspot']].nlargest(
                10, 'total_crashes'
            )[['iamfloc', 'total_crashes', 'injured', 'killed', 'avg_pci', 'gi_star']]
            print(top_hotspots.to_string(index=False))
    
    # Create map visualization (save if running standalone)
    create_hotspot_map(
        segments_gdf,
        segment_stats,
        dbscan_results,
        gi_results,
        FILES['output_map'] if __name__ == '__main__' else None
    )
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


def geographic_hotspot_analysis():
    """
    Entry point for notebook.ipynb.
    Checks if output CSV exists to skip processing.
    Returns figure and summary dataframe.
    """

    # 1. Always load geometry and base stats (needed for map visualization)
    segments_gdf, segment_stats = load_data()

    output_path = FILES['output_hotspots']

    # 2. Check if results already exist
    if output_path.exists():
        # Load the existing results
        hotspot_summary = pd.read_csv(output_path)
        
        # Reconstruct the inputs required for the map function from the loaded CSV
        # We assume the CSV contains the columns 'cluster_id', 'gi_star', etc.
        dbscan_results = hotspot_summary[['iamfloc', 'cluster_id']].copy()
        gi_results = hotspot_summary[['iamfloc', 'gi_star', 'is_hotspot', 'is_coldspot']].copy()
        
    else:
        # Perform DBSCAN clustering
        dbscan_results = perform_dbscan_clustering(segments_gdf, segment_stats, 'total_crashes')

        # Perform Getis-Ord Gi* analysis
        gi_results = calculate_getis_ord_gi(segments_gdf, segment_stats, 'total_crashes')

        # Combine results for saving
        if not dbscan_results.empty and not gi_results.empty:
            hotspot_summary = segment_stats.merge(
                dbscan_results[['iamfloc', 'cluster_id']],
                on='iamfloc',
                how='left'
            ).merge(
                gi_results[['iamfloc', 'gi_star', 'is_hotspot', 'is_coldspot']],
                on='iamfloc',
                how='left'
            )
            
            # Save results
            hotspot_summary.to_csv(output_path, index=False)
            hotspot_summary = pd.DataFrame()

    # 3. Create map visualization (Return figure, don't save to disk)
    fig = create_hotspot_map(
        segments_gdf,
        segment_stats,     # Base stats from load_data
        dbscan_results,    # Derived from CSV or fresh analysis
        gi_results,        # Derived from CSV or fresh analysis
        output_path=None   # None = Return object for notebook display
    )

    return fig, hotspot_summary

if __name__ == '__main__':
    main()


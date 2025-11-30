"""
Street Repair Projects Evaluation Analysis

This script evaluates the effectiveness of street repair projects by analyzing:
1. PCI improvement before/after repairs
2. Repair effectiveness by type (SLURRY, OVERLAY, CONCRETE)
3. Collision reduction after repairs
4. Repair longevity (time until PCI degrades significantly)
5. Geographic and functional class patterns
6. Cost-effectiveness indicators

This analysis complements the existing pavement-collision-traffic analyses by
focusing specifically on whether repairs are achieving their intended goals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Optional import for interactive maps
try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Warning: folium not available. Interactive maps will be skipped.")

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FILES = {
    'repairs': PROJECT_ROOT / 'data/raw/streets_repair_projects/sd_paving_datasd.csv',
    'processed_data': PROJECT_ROOT / 'data/processed/segments_collisions_pci_counts.csv',
    'pci_2016': PROJECT_ROOT / 'data/raw/pavement_condition/pavement_condition_assessment_2016_datasd.csv',
    'pci_2023': PROJECT_ROOT / 'data/raw/streets_repair_line_segments/sd_paving_segs_datasd.csv',
    'roads_geojson': PROJECT_ROOT / 'data/raw/streets_repair_line_segments/sd_paving_segs_datasd.geojson',
    'zoning_geojson': PROJECT_ROOT / 'data/raw/zoning/zoning_datasd.geojson',
}

# Analysis parameters
ANCHOR_2016 = pd.Timestamp("2016-07-01")
ANCHOR_2023 = pd.Timestamp("2023-06-01")
DATE_START = pd.Timestamp("2016-01-01")
DATE_END = pd.Timestamp.now()

# PCI improvement thresholds
PCI_IMPROVEMENT_THRESHOLD = 10  # Minimum PCI improvement to consider repair "effective"
PCI_DEGRADATION_THRESHOLD = 5   # PCI drop to consider "significant degradation"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all necessary data files."""
    print("Loading data files...")
    
    # Load repairs
    repairs_df = pd.read_csv(FILES['repairs'])
    repairs_df = repairs_df[repairs_df['status'] == 'POST CONSTRUCTION'].copy()
    repairs_df['date_end'] = pd.to_datetime(repairs_df['date_end'], errors='coerce')
    repairs_df = repairs_df.dropna(subset=['date_end', 'iamfloc'])
    repairs_df = repairs_df[repairs_df['date_end'] >= DATE_START].copy()
    repairs_df = repairs_df.sort_values('date_end')
    
    # Load processed yearly data
    yearly_df = pd.read_csv(FILES['processed_data'])
    yearly_df['year'] = pd.to_numeric(yearly_df['year'], errors='coerce')
    
    # Load PCI baseline data
    pci_2016_df = pd.read_csv(FILES['pci_2016']).rename(columns={'seg_id': 'iamfloc', 'pci': 'pci_16'})
    pci_2016_df = pci_2016_df.dropna(subset=['pci_16'])
    pci_2016_df['pci_16'] = pd.to_numeric(pci_2016_df['pci_16'], errors='coerce')
    pci_2016_df = pci_2016_df.dropna(subset=['pci_16'])
    
    pci_2023_df = pd.read_csv(FILES['pci_2023'])
    pci_2023_df['pci23'] = pd.to_numeric(pci_2023_df['pci23'], errors='coerce')
    pci_2023_df = pci_2023_df.dropna(subset=['pci23', 'iamfloc'])
    
    print(f"Loaded {len(repairs_df)} completed repair projects")
    print(f"Loaded {len(yearly_df)} yearly segment records")
    print(f"Loaded {len(pci_2016_df)} segments with 2016 PCI")
    print(f"Loaded {len(pci_2023_df)} segments with 2023 PCI")
    
    return repairs_df, yearly_df, pci_2016_df, pci_2023_df


def analyze_pci_improvement(
    repairs_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    pci_2016_df: pd.DataFrame,
    pci_2023_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze PCI improvement from repairs by comparing before/after values.
    
    For each repair, we look at:
    - PCI before repair (estimated from timeline or actual if available)
    - PCI after repair (from 2023 inspection or estimated)
    - Improvement magnitude
    """
    print("\n" + "="*70)
    print("Analyzing PCI Improvement from Repairs")
    print("="*70)
    
    # Merge repair info with PCI data
    repairs_with_pci = repairs_df.merge(
        pci_2016_df[['iamfloc', 'pci_16']], 
        on='iamfloc', 
        how='inner'
    ).merge(
        pci_2023_df[['iamfloc', 'pci23']], 
        on='iamfloc', 
        how='inner'
    )
    
    # Filter to repairs between 2016 and 2023
    repairs_with_pci = repairs_with_pci[
        (repairs_with_pci['date_end'] >= ANCHOR_2016) & 
        (repairs_with_pci['date_end'] <= ANCHOR_2023)
    ].copy()
    
    # Estimate PCI before repair (using linear decay from 2016)
    days_from_2016 = (repairs_with_pci['date_end'] - ANCHOR_2016).dt.days
    days_to_2023 = (ANCHOR_2023 - ANCHOR_2016).days
    
    # Simple linear interpolation: PCI at repair = PCI_2016 - (decline rate * days)
    pci_decline = repairs_with_pci['pci_16'] - repairs_with_pci['pci23']
    daily_decline_rate = pci_decline / days_to_2023
    repairs_with_pci['pci_before'] = repairs_with_pci['pci_16'] - (daily_decline_rate * days_from_2016)
    repairs_with_pci['pci_before'] = repairs_with_pci['pci_before'].clip(lower=0, upper=100)
    
    # Estimate PCI immediately after repair (should be higher, but we use 2023 as proxy)
    # For repairs close to 2023, use 2023 value; for older repairs, estimate
    days_after_repair = (ANCHOR_2023 - repairs_with_pci['date_end']).dt.days
    repairs_with_pci['pci_after_estimate'] = repairs_with_pci['pci23'] + (daily_decline_rate * days_after_repair)
    repairs_with_pci['pci_after_estimate'] = repairs_with_pci['pci_after_estimate'].clip(lower=0, upper=100)
    
    # Use the higher of estimated or 2023 actual
    repairs_with_pci['pci_after'] = repairs_with_pci[['pci23', 'pci_after_estimate']].max(axis=1)
    
    # Calculate improvement
    repairs_with_pci['pci_improvement'] = repairs_with_pci['pci_after'] - repairs_with_pci['pci_before']
    repairs_with_pci['improvement_pct'] = (repairs_with_pci['pci_improvement'] / repairs_with_pci['pci_before'].clip(lower=1)) * 100
    repairs_with_pci['is_effective'] = repairs_with_pci['pci_improvement'] >= PCI_IMPROVEMENT_THRESHOLD
    
    print(f"\nRepair Effectiveness Summary:")
    print(f"  Total repairs analyzed: {len(repairs_with_pci)}")
    print(f"  Repairs with PCI improvement >= {PCI_IMPROVEMENT_THRESHOLD}: {repairs_with_pci['is_effective'].sum()} ({repairs_with_pci['is_effective'].mean()*100:.1f}%)")
    print(f"  Average PCI improvement: {repairs_with_pci['pci_improvement'].mean():.2f}")
    print(f"  Median PCI improvement: {repairs_with_pci['pci_improvement'].median():.2f}")
    print(f"  Average PCI before repair: {repairs_with_pci['pci_before'].mean():.2f}")
    print(f"  Average PCI after repair: {repairs_with_pci['pci_after'].mean():.2f}")
    
    return repairs_with_pci


def analyze_by_repair_type(repairs_with_pci: pd.DataFrame) -> pd.DataFrame:
    """Analyze effectiveness by repair type (SLURRY, OVERLAY, CONCRETE)."""
    print("\n" + "="*70)
    print("Analyzing Effectiveness by Repair Type")
    print("="*70)
    
    type_analysis = repairs_with_pci.groupby('project_type').agg({
        'pci_improvement': ['mean', 'median', 'count'],
        'is_effective': 'mean',
        'pci_before': 'mean',
        'pci_after': 'mean',
    }).round(2)
    
    type_analysis.columns = ['avg_improvement', 'median_improvement', 'count', 'effectiveness_rate', 'avg_pci_before', 'avg_pci_after']
    type_analysis['effectiveness_rate'] = type_analysis['effectiveness_rate'] * 100
    
    print("\nRepair Type Effectiveness:")
    print(type_analysis.to_string())
    
    return type_analysis


def analyze_collision_reduction(
    repairs_df: pd.DataFrame,
    yearly_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze collision reduction after repairs.
    Compare collision rates before and after repair dates.
    """
    print("\n" + "="*70)
    print("Analyzing Collision Reduction After Repairs")
    print("="*70)
    
    # Get repairs in our analysis window
    repairs_window = repairs_df[
        (repairs_df['date_end'] >= DATE_START) & 
        (repairs_df['date_end'] <= DATE_END)
    ].copy()
    
    # For each repair, get collision data before and after
    collision_changes = []
    
    for _, repair in repairs_window.iterrows():
        segment_id = repair['iamfloc']
        repair_date = repair['date_end']
        repair_year = repair_date.year
        
        # Get segment data
        segment_data = yearly_df[yearly_df['iamfloc'] == segment_id].copy()
        if len(segment_data) == 0:
            continue
        
        # Get data before repair (up to 2 years before)
        before_years = segment_data[
            (segment_data['year'] >= repair_year - 2) & 
            (segment_data['year'] < repair_year)
        ]
        
        # Get data after repair (up to 2 years after)
        after_years = segment_data[
            (segment_data['year'] > repair_year) & 
            (segment_data['year'] <= repair_year + 2)
        ]
        
        if len(before_years) == 0 or len(after_years) == 0:
            continue
        
        # Calculate average crashes per year
        avg_crashes_before = before_years['total_crashes'].mean()
        avg_crashes_after = after_years['total_crashes'].mean()
        
        # Calculate per mile if we have length data
        if 'pav_length' in segment_data.columns and segment_data['pav_length'].iloc[0] > 0:
            miles = segment_data['pav_length'].iloc[0] / 5280
            crashes_per_mile_before = avg_crashes_before / miles if miles > 0 else 0
            crashes_per_mile_after = avg_crashes_after / miles if miles > 0 else 0
        else:
            crashes_per_mile_before = crashes_per_mile_after = np.nan
        
        collision_changes.append({
            'iamfloc': segment_id,
            'repair_date': repair_date,
            'project_type': repair['project_type'],
            'avg_crashes_before': avg_crashes_before,
            'avg_crashes_after': avg_crashes_after,
            'crashes_per_mile_before': crashes_per_mile_before,
            'crashes_per_mile_after': crashes_per_mile_after,
            'crash_reduction': avg_crashes_before - avg_crashes_after,
            'crash_reduction_pct': ((avg_crashes_before - avg_crashes_after) / avg_crashes_before * 100) if avg_crashes_before > 0 else np.nan,
        })
    
    collision_df = pd.DataFrame(collision_changes)
    
    if len(collision_df) > 0:
        print(f"\nCollision Analysis Summary:")
        print(f"  Segments analyzed: {len(collision_df)}")
        print(f"  Average crashes before repair: {collision_df['avg_crashes_before'].mean():.2f}")
        print(f"  Average crashes after repair: {collision_df['avg_crashes_after'].mean():.2f}")
        print(f"  Average reduction: {collision_df['crash_reduction'].mean():.2f} crashes/year")
        print(f"  Segments with reduced crashes: {(collision_df['crash_reduction'] > 0).sum()} ({(collision_df['crash_reduction'] > 0).mean()*100:.1f}%)")
        
        # By repair type
        if 'project_type' in collision_df.columns:
            print("\nCollision Reduction by Repair Type:")
            type_collision = collision_df.groupby('project_type')['crash_reduction'].agg(['mean', 'count']).round(2)
            print(type_collision.to_string())
    else:
        print("  No collision data available for analysis")
    
    return collision_df


def analyze_repair_longevity(
    repairs_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    pci_2016_df: pd.DataFrame,
    pci_2023_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze how long repairs last before significant PCI degradation.
    """
    print("\n" + "="*70)
    print("Analyzing Repair Longevity")
    print("="*70)
    
    # Merge repair and PCI data
    repairs_with_pci = repairs_df.merge(
        pci_2016_df[['iamfloc', 'pci_16']], 
        on='iamfloc', 
        how='inner'
    ).merge(
        pci_2023_df[['iamfloc', 'pci23']], 
        on='iamfloc', 
        how='inner'
    )
    
    repairs_with_pci = repairs_with_pci[
        (repairs_with_pci['date_end'] >= ANCHOR_2016) & 
        (repairs_with_pci['date_end'] <= ANCHOR_2023)
    ].copy()
    
    # Calculate time from repair to 2023 inspection
    repairs_with_pci['days_to_inspection'] = (ANCHOR_2023 - repairs_with_pci['date_end']).dt.days
    repairs_with_pci['years_to_inspection'] = repairs_with_pci['days_to_inspection'] / 365.25
    
    # Estimate PCI right after repair (similar to improvement analysis)
    days_from_2016 = (repairs_with_pci['date_end'] - ANCHOR_2016).dt.days
    days_to_2023 = (ANCHOR_2023 - ANCHOR_2016).days
    pci_decline = repairs_with_pci['pci_16'] - repairs_with_pci['pci23']
    daily_decline_rate = pci_decline / days_to_2023
    
    repairs_with_pci['pci_before'] = repairs_with_pci['pci_16'] - (daily_decline_rate * days_from_2016)
    repairs_with_pci['pci_after_repair'] = repairs_with_pci['pci23'] + (daily_decline_rate * repairs_with_pci['days_to_inspection'])
    
    # Calculate degradation since repair
    repairs_with_pci['pci_degradation'] = repairs_with_pci['pci_after_repair'] - repairs_with_pci['pci23']
    repairs_with_pci['degradation_per_year'] = repairs_with_pci['pci_degradation'] / repairs_with_pci['years_to_inspection'].clip(lower=0.1)
    
    # Estimate years until PCI drops by threshold
    repairs_with_pci['years_until_threshold'] = PCI_DEGRADATION_THRESHOLD / repairs_with_pci['degradation_per_year'].clip(lower=0.01)
    
    print(f"\nLongevity Analysis Summary:")
    print(f"  Average time to inspection: {repairs_with_pci['years_to_inspection'].mean():.2f} years")
    print(f"  Average PCI degradation per year: {repairs_with_pci['degradation_per_year'].mean():.2f}")
    print(f"  Estimated years until {PCI_DEGRADATION_THRESHOLD}-point drop: {repairs_with_pci['years_until_threshold'].median():.2f}")
    
    # By repair type
    if 'project_type' in repairs_with_pci.columns:
        print("\nLongevity by Repair Type:")
        type_longevity = repairs_with_pci.groupby('project_type').agg({
            'degradation_per_year': 'mean',
            'years_until_threshold': 'median',
            'iamfloc': 'count'
        }).round(2)
        type_longevity.columns = ['avg_degradation_per_year', 'median_years_until_threshold', 'count']
        print(type_longevity.to_string())
    
    return repairs_with_pci


def create_simple_plots(
    repairs_with_pci: pd.DataFrame,
    type_analysis: pd.DataFrame,
    collision_df: pd.DataFrame,
    longevity_df: pd.DataFrame
) -> None:
    """Create simple result plots."""
    print("\n" + "="*70)
    print("Creating Simple Result Plots")
    print("="*70)
    
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))
    
    # 1. PCI Improvement Distribution
    ax1 = plt.subplot(3, 2, 1)
    repairs_with_pci['pci_improvement'].hist(bins=30, edgecolor='black', ax=ax1)
    ax1.axvline(PCI_IMPROVEMENT_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({PCI_IMPROVEMENT_THRESHOLD})')
    ax1.set_xlabel('PCI Improvement')
    ax1.set_ylabel('Number of Repairs')
    ax1.set_title('Distribution of PCI Improvement from Repairs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Effectiveness by Repair Type
    ax2 = plt.subplot(3, 2, 2)
    if len(type_analysis) > 0:
        type_analysis['avg_improvement'].plot(kind='bar', ax=ax2, color=['#2ecc71', '#3498db', '#e74c3c'][:len(type_analysis)])
        ax2.set_ylabel('Average PCI Improvement')
        ax2.set_xlabel('Repair Type')
        ax2.set_title('Average PCI Improvement by Repair Type')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Before/After PCI Comparison
    ax3 = plt.subplot(3, 2, 3)
    before_after = pd.DataFrame({
        'Before': repairs_with_pci['pci_before'],
        'After': repairs_with_pci['pci_after']
    })
    before_after.boxplot(ax=ax3)
    ax3.set_ylabel('PCI')
    ax3.set_title('PCI Before vs After Repairs')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Collision Reduction
    ax4 = plt.subplot(3, 2, 4)
    if len(collision_df) > 0 and 'crash_reduction' in collision_df.columns:
        collision_df['crash_reduction'].hist(bins=30, edgecolor='black', ax=ax4)
        ax4.axvline(0, color='red', linestyle='--', label='No Change')
        ax4.set_xlabel('Crash Reduction (crashes/year)')
        ax4.set_ylabel('Number of Segments')
        ax4.set_title('Collision Reduction After Repairs')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Repair Longevity
    ax5 = plt.subplot(3, 2, 5)
    if len(longevity_df) > 0 and 'degradation_per_year' in longevity_df.columns:
        longevity_df['degradation_per_year'].hist(bins=30, edgecolor='black', ax=ax5)
        ax5.set_xlabel('PCI Degradation per Year')
        ax5.set_ylabel('Number of Repairs')
        ax5.set_title('Repair Longevity (Degradation Rate)')
        ax5.grid(True, alpha=0.3)
    
    # 6. Effectiveness Rate by Type
    ax6 = plt.subplot(3, 2, 6)
    if len(type_analysis) > 0 and 'effectiveness_rate' in type_analysis.columns:
        type_analysis['effectiveness_rate'].plot(kind='bar', ax=ax6, color=['#2ecc71', '#3498db', '#e74c3c'][:len(type_analysis)])
        ax6.set_ylabel('Effectiveness Rate (%)')
        ax6.set_xlabel('Repair Type')
        ax6.set_title('Repair Effectiveness Rate by Type')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / 'data/processed/repair_effectiveness_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    plt.close()


def create_geographic_visualizations(
    repairs_with_pci: pd.DataFrame
) -> None:
    """Create geographic visualizations of repairs on map."""
    print("\n" + "="*70)
    print("Creating Geographic Visualizations")
    print("="*70)
    
    try:
        # Load road segment geometries
        print("Loading road segment geometries...")
        roads_gdf = gpd.read_file(FILES['roads_geojson'])
        print(f"Loaded {len(roads_gdf)} road segments")
        
        # Load zoning data (optional, for context)
        try:
            zoning_gdf = gpd.read_file(FILES['zoning_geojson'])
            print(f"Loaded {len(zoning_gdf)} zoning districts")
        except Exception as e:
            print(f"Could not load zoning data: {e}")
            zoning_gdf = None
        
        # Merge repair data with road geometries
        repairs_geo = roads_gdf.merge(
            repairs_with_pci[['iamfloc', 'pci_improvement', 'is_effective', 'project_type', 
                              'pci_before', 'pci_after', 'date_end']],
            on='iamfloc',
            how='inner'
        )
        print(f"Merged {len(repairs_geo)} repairs with geometry")
        
        # Create static map visualization
        create_static_repair_map(repairs_geo, zoning_gdf)
        
        # Create interactive map
        create_interactive_repair_map(repairs_geo, zoning_gdf)
        
    except Exception as e:
        print(f"Error creating geographic visualizations: {e}")
        import traceback
        traceback.print_exc()


def create_static_repair_map(repairs_geo: gpd.GeoDataFrame, zoning_gdf: Optional[gpd.GeoDataFrame]) -> None:
    """Create a static matplotlib map of repairs."""
    print("Creating static repair map...")
    
    # Ensure CRS is set
    if repairs_geo.crs is None:
        repairs_geo.set_crs('EPSG:4326', inplace=True)
    repairs_geo = repairs_geo.to_crs('EPSG:4326')
    
    # Calculate bounds from the data to ensure full area is shown
    bounds = repairs_geo.total_bounds  # [minx, miny, maxx, maxy]
    # Add small padding
    padding = 0.01
    xlim = [bounds[0] - padding, bounds[2] + padding]
    ylim = [bounds[1] - padding, bounds[3] + padding]
    
    # Create figure with reduced spacing
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.05)  # Reduce spacing between subplots
    
    # Map 1: Effectiveness (Effective vs Ineffective)
    ax1 = axes[0]
    effective = repairs_geo[repairs_geo['is_effective'] == True]
    ineffective = repairs_geo[repairs_geo['is_effective'] == False]
    
    if zoning_gdf is not None:
        zoning_gdf = zoning_gdf.to_crs('EPSG:4326')
        zoning_gdf.plot(ax=ax1, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.3)
    
    if len(ineffective) > 0:
        ineffective.plot(ax=ax1, color='red', linewidth=1.5, alpha=0.6, label='Ineffective')
    if len(effective) > 0:
        effective.plot(ax=ax1, color='green', linewidth=1.5, alpha=0.6, label='Effective')
    
    # Set bounds to show full area
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect('equal')
    
    ax1.set_title('Repair Effectiveness Map\n(Green=Effective, Red=Ineffective)', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    ax1.legend(loc='upper right')
    
    # Map 2: PCI Improvement magnitude
    ax2 = axes[1]
    
    if zoning_gdf is not None:
        zoning_gdf.plot(ax=ax2, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.3)
    
    repairs_geo.plot(
        column='pci_improvement',
        ax=ax2,
        cmap='RdYlGn',
        legend=True,
        linewidth=1.5,
        legend_kwds={
            'label': 'PCI Improvement',
            'shrink': 0.8,
            'orientation': 'vertical',
            'pad': 0.01
        },
        vmin=0,
        vmax=30
    )
    
    # Set bounds to show full area (same as ax1)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect('equal')
    
    ax2.set_title('PCI Improvement Magnitude by Location', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / 'data/processed/repair_effectiveness_map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved static map: {output_path}")
    plt.close()


def create_interactive_repair_map(repairs_geo: gpd.GeoDataFrame, zoning_gdf: Optional[gpd.GeoDataFrame]) -> None:
    """Create an interactive Folium map of repairs."""
    if not FOLIUM_AVAILABLE:
        print("Skipping interactive map (folium not available)")
        return
    
    print("Creating interactive repair map...")
    
    # Ensure CRS is EPSG:4326 for Folium
    if repairs_geo.crs != 'EPSG:4326':
        repairs_geo = repairs_geo.to_crs('EPSG:4326')
    
    # Initialize map centered on San Diego
    m = folium.Map(location=[32.7157, -117.1611], zoom_start=11, tiles='CartoDB dark_matter')
    
    # Add zoning layer if available
    if zoning_gdf is not None:
        if zoning_gdf.crs != 'EPSG:4326':
            zoning_gdf = zoning_gdf.to_crs('EPSG:4326')
        
        folium.GeoJson(
            zoning_gdf,
            name='Zoning Districts',
            style_function=lambda feature: {
                'fillColor': '#d9d9d9',
                'color': 'white',
                'weight': 0.5,
                'fillOpacity': 0.2
            },
            overlay=True,
            show=False,
            tooltip=folium.GeoJsonTooltip(fields=['zone_name'], aliases=['Zone:'])
        ).add_to(m)
    
    # Function to style repairs by effectiveness
    def style_repairs(feature):
        props = feature['properties']
        is_effective = props.get('is_effective', False)
        improvement = props.get('pci_improvement', 0)
        repair_type = props.get('project_type', 'Unknown')
        
        if is_effective:
            color = '#2ecc71'  # Green for effective
            weight = 3
        else:
            color = '#e74c3c'  # Red for ineffective
            weight = 2
        
        return {
            'color': color,
            'weight': weight,
            'opacity': 0.8
        }
    
    # Add repair layer
    folium.GeoJson(
        repairs_geo,
        name='Repairs',
        style_function=style_repairs,
        overlay=True,
        show=True,
        tooltip=folium.GeoJsonTooltip(
            fields=['project_type', 'pci_improvement', 'is_effective', 
                   'pci_before', 'pci_after', 'date_end'],
            aliases=['Type:', 'PCI Improvement:', 'Effective:', 
                    'PCI Before:', 'PCI After:', 'Date:'],
            labels=True
        )
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save map
    output_path = PROJECT_ROOT / 'data/processed/repair_effectiveness_interactive_map.html'
    m.save(str(output_path))
    print(f"Saved interactive map: {output_path}")
    
    # Also create a map by repair type
    create_repair_type_map(repairs_geo)


def create_repair_type_map(repairs_geo: gpd.GeoDataFrame) -> None:
    """Create a map colored by repair type."""
    if not FOLIUM_AVAILABLE:
        return
    
    print("Creating repair type map...")
    
    # Color map for repair types
    type_colors = {
        'SLURRY': '#3498db',    # Blue
        'OVERLAY': '#2ecc71',   # Green
        'CONCRETE': '#e74c3c'   # Red
    }
    
    def style_by_type(feature):
        props = feature['properties']
        repair_type = props.get('project_type', 'Unknown')
        color = type_colors.get(repair_type, '#808080')
        
        return {
            'color': color,
            'weight': 2.5,
            'opacity': 0.8
        }
    
    # Create separate map for repair types
    m_type = folium.Map(location=[32.7157, -117.1611], zoom_start=11, tiles='CartoDB dark_matter')
    
    folium.GeoJson(
        repairs_geo,
        name='Repairs by Type',
        style_function=style_by_type,
        overlay=True,
        show=True,
        tooltip=folium.GeoJsonTooltip(
            fields=['project_type', 'pci_improvement', 'pci_before', 'pci_after'],
            aliases=['Type:', 'PCI Improvement:', 'PCI Before:', 'PCI After:'],
            labels=True
        )
    ).add_to(m_type)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius:5px; padding: 10px">
    <p><b>Repair Type</b></p>
    <p><i class="fa fa-square" style="color:#3498db"></i> SLURRY</p>
    <p><i class="fa fa-square" style="color:#2ecc71"></i> OVERLAY</p>
    <p><i class="fa fa-square" style="color:#e74c3c"></i> CONCRETE</p>
    </div>
    '''
    m_type.get_root().html.add_child(folium.Element(legend_html))
    
    folium.LayerControl(collapsed=False).add_to(m_type)
    
    output_path = PROJECT_ROOT / 'data/processed/repair_type_map.html'
    m_type.save(str(output_path))
    print(f"Saved repair type map: {output_path}")


def create_visualizations(
    repairs_with_pci: pd.DataFrame,
    type_analysis: pd.DataFrame,
    collision_df: pd.DataFrame,
    longevity_df: pd.DataFrame
) -> None:
    """Create comprehensive visualization of repair effectiveness."""
    # Create simple plots
    create_simple_plots(repairs_with_pci, type_analysis, collision_df, longevity_df)
    
    # Create geographic visualizations
    create_geographic_visualizations(repairs_with_pci)


def main():
    """Run the complete repair effectiveness analysis."""
    print("="*70)
    print("Street Repair Projects Evaluation Analysis")
    print("="*70)
    
    # Load data
    repairs_df, yearly_df, pci_2016_df, pci_2023_df = load_data()
    
    # Analyze PCI improvement
    repairs_with_pci = analyze_pci_improvement(repairs_df, yearly_df, pci_2016_df, pci_2023_df)
    
    # Analyze by repair type
    type_analysis = analyze_by_repair_type(repairs_with_pci)
    
    # Analyze collision reduction
    collision_df = analyze_collision_reduction(repairs_df, yearly_df)
    
    # Analyze longevity
    longevity_df = analyze_repair_longevity(repairs_df, yearly_df, pci_2016_df, pci_2023_df)
    
    # Create visualizations
    create_visualizations(repairs_with_pci, type_analysis, collision_df, longevity_df)
    
    # Save detailed results
    output_dir = PROJECT_ROOT / 'data/processed'
    output_dir.mkdir(exist_ok=True)
    
    repairs_with_pci.to_csv(output_dir / 'repair_pci_analysis.csv', index=False)
    if len(collision_df) > 0:
        collision_df.to_csv(output_dir / 'repair_collision_analysis.csv', index=False)
    if len(longevity_df) > 0:
        longevity_df.to_csv(output_dir / 'repair_longevity_analysis.csv', index=False)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - repair_effectiveness_analysis.png (visualizations)")
    print("  - repair_pci_analysis.csv (detailed PCI analysis)")
    if len(collision_df) > 0:
        print("  - repair_collision_analysis.csv (collision reduction analysis)")
    if len(longevity_df) > 0:
        print("  - repair_longevity_analysis.csv (longevity analysis)")


if __name__ == "__main__":
    main()


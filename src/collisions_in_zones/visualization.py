"""
Visualization functions for collisions analysis with zoning.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from typing import Optional

from .config import FILES
from .zone_utils import extract_zone_type, get_general_zone_type
from .data_loading import load_zoning_data


def visualize_zoning_collisions(df_zoning: pd.DataFrame) -> None:
    """
    Create visualizations for zoning collision analysis.

    Generates bar charts showing:
    - Top zones by total crashes
    - Top zones by crash rate (crashes per segment)
    - Top zones by injuries
    - Zone type comparison
    - Density analysis

    Parameters
    ----------
    df_zoning : pd.DataFrame
        Zone statistics DataFrame.
    """
    print("--- Creating Visualizations ---")
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3, height_ratios=[1, 1, 1.2])
    fig.suptitle('Traffic Collision Analysis by Zoning District', fontsize=16, fontweight='bold', y=0.995)
    
    top_n = 15
    
    # Top zones by total crashes
    ax1 = fig.add_subplot(gs[0, 0])
    top_crashes = df_zoning.nlargest(top_n, 'total_crashes')
    bars1 = ax1.barh(range(len(top_crashes)), top_crashes['total_crashes'], 
                     color=sns.color_palette("Blues_r", len(top_crashes)))
    ax1.set_yticks(range(len(top_crashes)))
    ax1.set_yticklabels(top_crashes['zone_name'], fontsize=9)
    ax1.set_xlabel('Total Crashes', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} Zones by Total Crashes', fontsize=12, fontweight='bold', pad=10)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    for i, (idx, val) in enumerate(zip(top_crashes.index, top_crashes['total_crashes'])):
        ax1.text(val + 20, i, f'{int(val)}', va='center', fontsize=9, fontweight='bold')
    
    # Top zones by crash rate
    ax2 = fig.add_subplot(gs[0, 1])
    top_rate = df_zoning[df_zoning['total_segments'] > 0].nlargest(top_n, 'crashes_per_segment')
    bars2 = ax2.barh(range(len(top_rate)), top_rate['crashes_per_segment'],
                     color=sns.color_palette("Reds_r", len(top_rate)))
    ax2.set_yticks(range(len(top_rate)))
    ax2.set_yticklabels(top_rate['zone_name'], fontsize=9)
    ax2.set_xlabel('Crashes per Segment', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top {top_n} Zones by Crash Rate', fontsize=12, fontweight='bold', pad=10)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    for i, (idx, val) in enumerate(zip(top_rate.index, top_rate['crashes_per_segment'])):
        ax2.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
    
    # Top zones by injuries
    ax3 = fig.add_subplot(gs[1, 0])
    top_injured = df_zoning.nlargest(top_n, 'injured')
    bars3 = ax3.barh(range(len(top_injured)), top_injured['injured'],
                     color=sns.color_palette("Oranges_r", len(top_injured)))
    ax3.set_yticks(range(len(top_injured)))
    ax3.set_yticklabels(top_injured['zone_name'], fontsize=9)
    ax3.set_xlabel('Total Injuries', fontsize=11, fontweight='bold')
    ax3.set_title(f'Top {top_n} Zones by Total Injuries', fontsize=12, fontweight='bold', pad=10)
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    for i, (idx, val) in enumerate(zip(top_injured.index, top_injured['injured'])):
        ax3.text(val + 5, i, f'{int(val)}', va='center', fontsize=9, fontweight='bold')
    
    # Zone type comparison
    ax4 = fig.add_subplot(gs[1, 1])
    if 'zone_type' in df_zoning.columns:
        zone_type_stats = df_zoning.groupby('zone_type').agg({
            'total_crashes': 'sum',
            'injured': 'sum',
            'killed': 'sum'
        }).reset_index().sort_values('total_crashes', ascending=False)
        
        x_pos = np.arange(len(zone_type_stats))
        width = 0.25
        
        bars1 = ax4.bar(x_pos - width, zone_type_stats['total_crashes'], width,
                       label='Crashes', color='steelblue', alpha=0.8)
        bars2 = ax4.bar(x_pos, zone_type_stats['injured'], width,
                       label='Injuries', color='coral', alpha=0.8)
        bars3 = ax4.bar(x_pos + width, zone_type_stats['killed'], width,
                       label='Fatalities', color='darkred', alpha=0.8)
        
        ax4.set_xlabel('Zone Type', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax4.set_title('Collisions by Zone Type Category', fontsize=12, fontweight='bold', pad=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(zone_type_stats['zone_type'], rotation=45, ha='right', fontsize=9)
        ax4.legend(loc='upper right')
        ax4.grid(axis='y', alpha=0.3)
    
    # Density analysis
    ax5 = fig.add_subplot(gs[2, :])
    if 'crashes_per_km2' in df_zoning.columns:
        density_data = df_zoning[df_zoning['crashes_per_km2'].notna() & 
                                 (df_zoning['crashes_per_km2'] > 0)].nlargest(20, 'crashes_per_km2')
        if len(density_data) > 0:
            bars = ax5.barh(range(len(density_data)), density_data['crashes_per_km2'],
                          color=sns.color_palette("YlOrRd", len(density_data)))
            ax5.set_yticks(range(len(density_data)))
            ax5.set_yticklabels(density_data['zone_name'], fontsize=8)
            ax5.set_xlabel('Crashes per km²', fontsize=11, fontweight='bold')
            ax5.set_title('Top 20 Zones by Crash Density (per km²)', 
                         fontsize=12, fontweight='bold', pad=10)
            ax5.invert_yaxis()
            ax5.grid(axis='x', alpha=0.3)
            for i, (idx, val) in enumerate(zip(density_data.index, density_data['crashes_per_km2'])):
                ax5.text(val + 1, i, f'{val:.1f}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    plt.savefig(FILES['output_visualization'], dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {FILES['output_visualization']}")
    plt.close()


def create_severity_map(
    df_zoning_stats: pd.DataFrame,
    gdf_zoning: gpd.GeoDataFrame
) -> None:
    """
    Create a map visualization of zones color-coded by crash severity.

    Parameters
    ----------
    df_zoning_stats : pd.DataFrame
        Zone statistics with severity metrics.
    gdf_zoning : gpd.GeoDataFrame
        Zoning polygons with geometry.
    """
    print("--- Creating Severity Map ---")
    
    gdf_with_stats = gdf_zoning.merge(
        df_zoning_stats[['zone_name', 'total_crashes', 'injured', 'killed', 'crashes_per_segment']],
        on='zone_name',
        how='left'
    )
    
    gdf_with_stats['total_crashes'] = gdf_with_stats['total_crashes'].fillna(0)
    gdf_with_stats['severity_score'] = (
        gdf_with_stats['total_crashes'] * 1 +
        gdf_with_stats['injured'].fillna(0) * 2 +
        gdf_with_stats['killed'].fillna(0) * 10
    )
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    gdf_with_stats = gdf_with_stats.to_crs('EPSG:4326')
    
    gdf_with_stats.plot(
        column='severity_score',
        ax=ax,
        cmap='YlOrRd',
        legend=True,
        missing_kwds={'color': 'lightgray'},
        edgecolor='black',
        linewidth=0.3,
        legend_kwds={
            'label': 'Severity Score (Crashes + 2×Injuries + 10×Fatalities)',
            'shrink': 0.8,
            'orientation': 'vertical'
        }
    )
    
    ax.set_title('Traffic Collision Severity by Zoning District', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(FILES['output_map'], dpi=300, bbox_inches='tight')
    print(f"Severity map saved: {FILES['output_map']}")
    plt.close()


def create_zone_type_map(output_path: Optional[str] = None) -> None:
    """
    Create a map visualization of zones color-coded by general zone type.
    
    This function loads zoning data, extracts general zone type categories
    (Residential, Agricultural, Commercial, Industrial, etc.), and generates
    a color-coded map saved as a PNG file.
    
    Parameters
    ----------
    output_path : str, optional
        Path to save the output PNG file. If None, uses default path.
    """
    print("--- Creating Zone Type Map ---")
    
    # Load zoning data
    gdf_zoning = load_zoning_data()
    
    if gdf_zoning is None:
        print("Error: Could not load zoning data.")
        return
    
    gdf_zoning['zone_type'] = gdf_zoning['zone_name'].apply(extract_zone_type)
    
    gdf_zoning['general_zone_type'] = gdf_zoning['zone_type'].apply(get_general_zone_type)
    
    zone_type_colors = {
        'Residential': '#1f78b4',           # Blue
        'Agricultural': '#ff7f00',          # Orange
        'Commercial': '#33a02c',            # Green
        'Industrial': '#e31a1c',            # Red
        'Mixed Use': '#6a3d9a',             # Purple
        'Open Space': '#cab2d6',            # Light Purple
        'Planned Development': '#b15928',   # Brown
        'Special Purpose': '#8dd3c7',       # Teal
        'Unknown': '#d9d9d9'                # Gray
    }
    
    gdf_zoning['zone_color'] = gdf_zoning['general_zone_type'].map(zone_type_colors)
    gdf_zoning['zone_color'] = gdf_zoning['zone_color'].fillna(zone_type_colors['Unknown'])
    
    gdf_zoning = gdf_zoning.to_crs('EPSG:4326')
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    unique_zone_types = sorted([zt for zt in gdf_zoning['general_zone_type'].unique() 
                                if pd.notna(zt)])
    
    for zone_type in unique_zone_types:
        zone_data = gdf_zoning[gdf_zoning['general_zone_type'] == zone_type]
        color = zone_type_colors.get(zone_type, zone_type_colors['Unknown'])
        zone_data.plot(
            ax=ax,
            color=color,
            edgecolor='black',
            linewidth=0.3,
            label=zone_type,
            alpha=0.7
        )
    
    ax.set_title('San Diego Zones', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.axis('off')
    
    handles = []
    labels = []
    for zone_type in unique_zone_types:
        color = zone_type_colors.get(zone_type, zone_type_colors['Unknown'])
        handles.append(Patch(facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7))
        labels.append(zone_type)
    
    ax.legend(
        handles=handles,
        labels=labels,
        title='Zone Type',
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title_fontsize=12,
        framealpha=0.9
    )
    
    if output_path is None:
        output_path = './data/processed/zone_type_map.png'
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Zone type map saved: {output_path}")
    plt.close()
    
    print("\n--- Zone Type Summary ---")
    zone_type_counts = gdf_zoning['general_zone_type'].value_counts()
    for zone_type, count in zone_type_counts.items():
        print(f"{zone_type}: {count} zones")


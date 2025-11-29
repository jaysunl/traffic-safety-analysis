"""
--- DETAILED DESCRIPTION IN README.md (scripts/pavement_collisions_traffic/README.md) ---
"""

import pandas as pd
from matplotlib.figure import Figure
from . import analysis_utils


def analyze_pavement_safety_vmt(df: pd.DataFrame) -> Figure:
    """
    Analyzes pavement safety metrics normalized by Vehicle Miles Traveled (VMT).
    Filters data to only include segments with available traffic counts.

    Args:
        df (pd.DataFrame): The input DataFrame containing segment, collision, PCI, and traffic data.

    Returns:
        Figure: A matplotlib Figure object containing the generated plots.
    """
    # funclass to include
    target_funclasses = [
        "CL 2 LANE SUB-COLLECTOR",
        "CL 2 LANE COLLECTOR",
        "CL 2 LANE COLLECTOR WITH 2 WAY LEFT TURN",
        "CL 4 LN COLLECTOR WITH 2 WY LEFT TURN LN",
        "MJ SIX LANE URBAN MAJOR",
    ]

    df = analysis_utils.filter_by_funclass(df, target_funclasses)

    original_rows = len(df)

    # Filter for traffic volumes (Remove any row that doesn't have traffic count data)
    df = df.dropna(subset=["traffic_count"])
    df = df[df["traffic_count"] > 0]

    # Calculate percentage of data used
    final_rows = len(df)
    data_usage_pct = (final_rows / original_rows) * 100 if original_rows > 0 else 0

    # Calculate vehicle miles traveled
    df["segment_miles"] = df["pav_length"] / 5280

    # Annual VMT = daily traffic * segment length (miles) * 365 Days
    # traffic_count is daily traffic volume
    df["annual_vmt"] = df["traffic_count"] * df["segment_miles"] * 365

    # Text wrapping for funclass labels
    df, target_funclasses_wrapped = analysis_utils.wrap_funclass_labels(df, target_funclasses)

    # Data Aggregation
    grouped = analysis_utils.aggregate_data(df, ["funclass_wrapped", "pci_desc"], "annual_vmt")

    # Normalization (per million VMT)
    grouped["million_vmt"] = grouped["annual_vmt"] / 1_000_000

    # Avoid division by zero
    grouped = grouped[grouped["million_vmt"] > 0].copy()

    # Calculate rates per Million VMT
    grouped = analysis_utils.calculate_rates(grouped, "million_vmt", "crash_rate", "weighted_severity")

    # Add disclaimer due to low raw data
    def add_warning(fig):
        fig.text(
            0.475,
            0.98,
            f"Only using ~{data_usage_pct:.1f}% of original data! (lacking sufficient traffic data)",
            ha="center",
            va="top",
            fontsize=analysis_utils.FONT_SIZE_WARNING,
            fontweight="bold",
            color=analysis_utils.COLOR,
        )

    return analysis_utils.plot_analysis(
        grouped,
        target_funclasses_wrapped,
        y_label_freq="Crashes per Million VMT",
        y_label_sev="Severity Index per Million VMT",
        top_adjust=0.85,
        extra_plot_func=add_warning,
    )

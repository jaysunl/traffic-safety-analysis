"""
--- DETAILED DESCRIPTION IN README.md (scripts/pavement_collisions_traffic/README.md) ---
"""

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import textwrap
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional, Callable

PCI_ORDER_BEST_TO_WORST = ["Good", "Satisfactory", "Fair", "Poor", "Very Poor", "Serious", "Failed"]

FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_BAR_LABEL = 8
FONT_SIZE_ARROW_TITLE = 12
FONT_SIZE_ARROW_LABEL = 11
FONT_SIZE_WARNING = 18
FONT_SIZE_DISCLAIMER = 10

COLOR = "#000000"
CUSTOM_PALETTE = "RdYlGn_r"


def setup_theme() -> None:
    """
    Configures the global seaborn and matplotlib theme settings for consistent plotting.
    """
    sns.set_theme(
        style="whitegrid",
        rc={
            "text.color": COLOR,
            "axes.labelcolor": COLOR,
            "xtick.color": COLOR,
            "ytick.color": COLOR,
            "font.family": "sans-serif",
        },
    )


def filter_by_funclass(df: pd.DataFrame, target_funclasses: List[str]) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows matching the specified functional classes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_funclasses (List[str]): List of functional class names to keep.

    Returns:
        pd.DataFrame: A filtered copy of the DataFrame.
    """
    return df[df["funclass"].isin(target_funclasses)].copy()


def wrap_funclass_labels(
    df: pd.DataFrame, target_funclasses: List[str], width: int = 12
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Wraps long functional class names in the DataFrame and the target list for better display on plots.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'funclass' column.
        target_funclasses (List[str]): List of original functional class names.
        width (int): Maximum line width for wrapping text.

    Returns:
        Tuple[pd.DataFrame, List[str]]: The DataFrame with a new 'funclass_wrapped' column,
                                        and a list of wrapped target class names.
    """
    wrapper = textwrap.TextWrapper(width=width, break_long_words=False)
    df["funclass_wrapped"] = df["funclass"].apply(wrapper.fill)
    target_funclasses_wrapped = [wrapper.fill(x) for x in target_funclasses]
    return df, target_funclasses_wrapped


def aggregate_data(df: pd.DataFrame, group_cols: Union[str, List[str]], measure_col: str) -> pd.DataFrame:
    """
    Aggregates crash data by summing up the measure column, total crashes, injured, and killed counts.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_cols (Union[str, List[str]]): Column(s) to group by.
        measure_col (str): The column representing the normalization factor (e.g., length, VMT).

    Returns:
        pd.DataFrame: An aggregated DataFrame with summed statistics.
    """
    agg_dict = {measure_col: "sum", "total_crashes": "sum", "injured": "sum", "killed": "sum"}
    return df.groupby(group_cols, observed=False).agg(agg_dict).reset_index()


def calculate_rates(
    df: pd.DataFrame, measure_col: str, crash_col_name: str = "crash_rate", sev_col_name: str = "severity_rate"
) -> pd.DataFrame:
    """
    Calculates crash rate and weighted severity rate based on the provided measure column.

    Args:
        df (pd.DataFrame): The input DataFrame containing aggregated counts.
        measure_col (str): The column to divide by (normalization factor).
        crash_col_name (str): Name for the resulting crash rate column.
        sev_col_name (str): Name for the resulting severity rate column.

    Returns:
        pd.DataFrame: The DataFrame with added rate columns.
    """
    df[crash_col_name] = df["total_crashes"] / df[measure_col]

    # Weighted severity: 1 killed = 10 injured
    df[sev_col_name] = (df["injured"] + (df["killed"] * 10)) / df[measure_col]
    return df


def adjust_ylim(ax: Axes) -> None:
    """
    Adjusts the y-axis limit of a plot to add some headroom (25%) above the data.

    Args:
        ax (Axes): The matplotlib Axes object to adjust.
    """
    ymin, ymax = ax.get_ylim()
    if not np.isnan(ymax) and not np.isinf(ymax):
        ax.set_ylim(ymin, ymax * 1.1)


def add_street_size_arrow(fig: Figure, top_adjust: float = 0.88) -> None:
    """
    Adds a visual arrow annotation to the figure indicating the progression of street sizes.

    Args:
        fig (Figure): The matplotlib Figure object.
        top_adjust (float): The top margin adjustment used for layout, to position the arrow correctly.
    """
    plt.tight_layout()
    plt.subplots_adjust(top=top_adjust)

    offset = 0.88 - top_adjust

    arrow_y = 0.93 - offset
    text_y_title = 0.955 - offset
    text_y_labels = 0.947 - offset
    arrow_x_start = 0.07
    arrow_x_end = 0.85

    fig.text(0.46, text_y_title, "Street Size", ha="center", va="center", fontsize=FONT_SIZE_ARROW_TITLE, color=COLOR)

    fig.text(
        arrow_x_start, text_y_labels, "smallest", ha="center", va="bottom", fontsize=FONT_SIZE_ARROW_LABEL, color=COLOR
    )

    fig.text(
        arrow_x_end, text_y_labels, "largest", ha="center", va="bottom", fontsize=FONT_SIZE_ARROW_LABEL, color=COLOR
    )

    plt.annotate(
        "",
        xy=(arrow_x_end, arrow_y),
        xytext=(arrow_x_start, arrow_y),
        xycoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )


def plot_analysis(
    grouped: pd.DataFrame,
    target_funclasses_wrapped: List[str],
    y_label_freq: str,
    y_label_sev: str,
    top_adjust: float = 0.88,
    extra_plot_func: Optional[Callable[[Figure], None]] = None,
) -> Figure:
    """
    Generates the standard two-panel analysis plot (Frequency and Severity).

    Args:
        grouped (pd.DataFrame): Aggregated data containing rates and wrapped labels.
        target_funclasses_wrapped (List[str]): List of wrapped functional class names for ordering.
        y_label_freq (str): Label for the frequency y-axis.
        y_label_sev (str): Label for the severity y-axis.
        top_adjust (float): Top margin adjustment for layout.
        extra_plot_func (Optional[Callable[[Figure], None]]): Optional callback to add extra elements to the figure.

    Returns:
        Figure: The generated matplotlib Figure.
    """
    setup_theme()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    if extra_plot_func:
        extra_plot_func(fig)

    # Frequency analysis
    ax1 = sns.barplot(
        data=grouped,
        x="funclass_wrapped",
        y="crash_rate",
        hue="pci_desc",
        order=target_funclasses_wrapped,
        hue_order=PCI_ORDER_BEST_TO_WORST,
        ax=axes[0],
        palette=CUSTOM_PALETTE,
        edgecolor=COLOR,
        linewidth=0.5,
        errorbar=None,
    )

    for container in ax1.containers:
        if isinstance(container, BarContainer):
            ax1.bar_label(container, fmt="%.2f", padding=3, rotation=90, fontsize=FONT_SIZE_BAR_LABEL, color=COLOR)
    adjust_ylim(ax1)

    axes[0].set_title("Collision Frequency by Street Type", fontsize=FONT_SIZE_TITLE, fontweight="bold", color=COLOR)
    axes[0].set_ylabel(y_label_freq, fontsize=FONT_SIZE_LABEL, color=COLOR)
    axes[0].set_xlabel("")
    axes[0].legend(title="Pavement Condition", bbox_to_anchor=(1.01, 1), loc="upper left")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Severity analysis
    ax2 = sns.barplot(
        data=grouped,
        x="funclass_wrapped",
        y="weighted_severity",
        hue="pci_desc",
        order=target_funclasses_wrapped,
        hue_order=PCI_ORDER_BEST_TO_WORST,
        ax=axes[1],
        palette=CUSTOM_PALETTE,
        edgecolor=COLOR,
        linewidth=0.5,
        errorbar=None,
    )

    for container in ax2.containers:
        if isinstance(container, BarContainer):
            ax2.bar_label(container, fmt="%.2f", padding=3, rotation=90, fontsize=FONT_SIZE_BAR_LABEL, color=COLOR)
    adjust_ylim(ax2)

    axes[1].set_title("Severity Rate by Street Type", fontsize=FONT_SIZE_TITLE, fontweight="bold", color=COLOR)
    axes[1].set_ylabel(y_label_sev, fontsize=FONT_SIZE_LABEL, color=COLOR)
    axes[1].set_xlabel("Street Type", fontsize=FONT_SIZE_LABEL, color=COLOR)

    axes[1].get_legend().remove()
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=0, ha="center", fontsize=FONT_SIZE_TICK, color=COLOR)
    plt.yticks(color=COLOR)

    add_street_size_arrow(fig, top_adjust)

    plt.close(fig)
    return fig

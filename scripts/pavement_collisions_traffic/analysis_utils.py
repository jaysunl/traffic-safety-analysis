import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import seaborn as sns
import textwrap
import numpy as np

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


def setup_theme():
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


def filter_by_funclass(df, target_funclasses):
    return df[df["funclass"].isin(target_funclasses)].copy()


def wrap_funclass_labels(df, target_funclasses, width=12):
    wrapper = textwrap.TextWrapper(width=width, break_long_words=False)
    df["funclass_wrapped"] = df["funclass"].apply(wrapper.fill)
    target_funclasses_wrapped = [wrapper.fill(x) for x in target_funclasses]
    return df, target_funclasses_wrapped


def aggregate_data(df, group_cols, measure_col):
    """
    Aggregates crash data summing up the measure_col, total_crashes, injured, and killed.
    """
    agg_dict = {measure_col: "sum", "total_crashes": "sum", "injured": "sum", "killed": "sum"}
    return df.groupby(group_cols, observed=False).agg(agg_dict).reset_index()


def calculate_rates(df, measure_col, crash_col_name="crash_rate", sev_col_name="severity_rate"):
    """
    Calculates crash rate and weighted severity rate based on the measure column.
    """
    df[crash_col_name] = df["total_crashes"] / df[measure_col]
    df[sev_col_name] = (df["injured"] + (df["killed"] * 10)) / df[measure_col]
    return df


def adjust_ylim(ax):
    ymin, ymax = ax.get_ylim()
    if not np.isnan(ymax) and not np.isinf(ymax):
        ax.set_ylim(ymin, ymax * 1.25)


def add_street_size_arrow(fig, top_adjust=0.88):
    plt.tight_layout()
    plt.subplots_adjust(top=top_adjust)

    offset = 0.88 - top_adjust

    arrow_y = 0.93 - offset
    text_y_title = 0.955 - offset
    text_y_labels = 0.945 - offset
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


def plot_analysis(grouped, target_funclasses_wrapped, y_label_freq, y_label_sev, top_adjust=0.88, extra_plot_func=None):
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

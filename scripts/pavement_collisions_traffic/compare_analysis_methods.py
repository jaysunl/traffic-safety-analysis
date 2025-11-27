import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import analysis_utils


def compare_analysis_methods(df):
    """
    Generates a comparison of Safety Deviation (Per Mile) vs Safety Deviation (Per VMT).
    """

    # funclass to include
    target_funclasses = ["CL 4 LN COLLECTOR WITH 2 WY LEFT TURN LN", "MJ SIX LANE URBAN MAJOR"]

    df_main = analysis_utils.filter_by_funclass(df, target_funclasses)

    coverage_map = {}
    for fc in target_funclasses:
        subset = df_main[df_main["funclass"] == fc]
        total_rows = len(subset)
        traffic_rows = len(subset[subset["traffic_count"] > 0])
        pct = (traffic_rows / total_rows) * 100 if total_rows > 0 else 0
        coverage_map[fc] = pct

    # Per mile analysis
    df_mile = df_main.copy()
    df_mile["original_funclass"] = df_mile["funclass"]

    df_mile, _ = analysis_utils.wrap_funclass_labels(df_mile, target_funclasses, width=15)

    grouped_mile = analysis_utils.aggregate_data(
        df_mile, ["funclass_wrapped", "original_funclass", "pci_desc"], "pav_length"
    )

    grouped_mile["measure"] = grouped_mile["pav_length"] / 5280
    grouped_mile = analysis_utils.calculate_rates(grouped_mile, "measure", "crash_rate", "severity_rate")

    baseline_mile = analysis_utils.aggregate_data(df_mile, "funclass_wrapped", "pav_length")
    baseline_mile["measure"] = baseline_mile["pav_length"] / 5280
    baseline_mile = analysis_utils.calculate_rates(baseline_mile, "measure", "base_crash", "base_sev")

    merged_mile = pd.merge(
        grouped_mile, baseline_mile[["funclass_wrapped", "base_crash", "base_sev"]], on="funclass_wrapped"
    )
    merged_mile["crash_diff_pct"] = (
        (merged_mile["crash_rate"] - merged_mile["base_crash"]) / merged_mile["base_crash"]
    ) * 100
    merged_mile["sev_diff_pct"] = (
        (merged_mile["severity_rate"] - merged_mile["base_sev"]) / merged_mile["base_sev"]
    ) * 100
    merged_mile["method"] = "Per Mile"
    merged_mile["sort_key"] = 1

    # Per VMT analysis
    df_vmt = df_main.dropna(subset=["traffic_count"]).copy()
    df_vmt = df_vmt[df_vmt["traffic_count"] > 0].copy()
    df_vmt["original_funclass"] = df_vmt["funclass"]

    df_vmt, _ = analysis_utils.wrap_funclass_labels(df_vmt, target_funclasses, width=15)

    df_vmt["annual_vmt"] = df_vmt["traffic_count"] * (df_vmt["pav_length"] / 5280) * 365

    grouped_vmt = analysis_utils.aggregate_data(
        df_vmt, ["funclass_wrapped", "original_funclass", "pci_desc"], "annual_vmt"
    )

    grouped_vmt["measure"] = grouped_vmt["annual_vmt"] / 1_000_000
    grouped_vmt = grouped_vmt[grouped_vmt["measure"] > 0].copy()

    grouped_vmt = analysis_utils.calculate_rates(grouped_vmt, "measure", "crash_rate", "severity_rate")

    baseline_vmt = analysis_utils.aggregate_data(df_vmt, "funclass_wrapped", "annual_vmt")
    baseline_vmt["measure"] = baseline_vmt["annual_vmt"] / 1_000_000
    baseline_vmt = analysis_utils.calculate_rates(baseline_vmt, "measure", "base_crash", "base_sev")

    merged_vmt = pd.merge(
        grouped_vmt, baseline_vmt[["funclass_wrapped", "base_crash", "base_sev"]], on="funclass_wrapped"
    )
    merged_vmt["crash_diff_pct"] = (
        (merged_vmt["crash_rate"] - merged_vmt["base_crash"]) / merged_vmt["base_crash"]
    ) * 100
    merged_vmt["sev_diff_pct"] = ((merged_vmt["severity_rate"] - merged_vmt["base_sev"]) / merged_vmt["base_sev"]) * 100
    merged_vmt["method"] = "Per VMT"
    merged_vmt["sort_key"] = 2

    # Combine and create graphs
    combined = pd.concat([merged_mile, merged_vmt], ignore_index=True)
    combined = combined.sort_values(by=["funclass_wrapped", "sort_key"])
    combined["display_label"] = combined["funclass_wrapped"] + "\n(" + combined["method"] + ")"

    display_order = combined["display_label"].unique()

    analysis_utils.setup_theme()

    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    def format_plot(ax, y_col, title):
        ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.8)

        sns.barplot(
            data=combined,
            x="display_label",
            y=y_col,
            hue="pci_desc",
            order=display_order,
            hue_order=analysis_utils.PCI_ORDER_BEST_TO_WORST,
            ax=ax,
            palette=analysis_utils.CUSTOM_PALETTE,
            edgecolor=analysis_utils.COLOR,
            linewidth=0.5,
            errorbar=None,
            width=0.85,
        )

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%+.0f%%",
                padding=3,
                rotation=90,
                fontsize=analysis_utils.FONT_SIZE_BAR_LABEL,
                color=analysis_utils.COLOR,
            )

        ymin, ymax = ax.get_ylim()

        if ymin > 0:
            ymin = 0
        if ymax < 0:
            ymax = 0

        total_span = abs(ymax - ymin) if abs(ymax - ymin) > 0 else 10

        pad_top = pad_bottom = total_span * 0.1

        final_ymax = ymax + pad_top
        final_ymin = ymin - pad_bottom

        ax.set_ylim(final_ymin, final_ymax)

        ax.set_title(title, fontsize=analysis_utils.FONT_SIZE_TITLE, fontweight="bold", color=analysis_utils.COLOR)
        ax.set_ylabel("% Diff from Average", fontsize=analysis_utils.FONT_SIZE_LABEL, color=analysis_utils.COLOR)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Add disclaimers
        for i, label_text in enumerate(display_order):
            if "(Per VMT)" in label_text:
                row = combined[combined["display_label"] == label_text].iloc[0]
                orig_name = row["original_funclass"]
                pct_used = coverage_map.get(orig_name, 0)

                ax.text(
                    x=i,
                    y=final_ymax - (total_span * 0.05),
                    s=f"VMT (below) only \n using {pct_used:.0f}% of data!",
                    ha="center",
                    va="top",
                    fontsize=analysis_utils.FONT_SIZE_DISCLAIMER,
                    fontweight="bold",
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                )

    format_plot(axes[0], "crash_diff_pct", "Crash Frequency Deviation (Per Mile vs. Per VMT)")
    axes[0].legend(title="Pavement Condition", bbox_to_anchor=(1.01, 1), loc="upper left")

    format_plot(axes[1], "sev_diff_pct", "Crash Severity Deviation (Per Mile vs. Per VMT)")
    axes[1].get_legend().remove()
    axes[1].set_xlabel(
        "Street Type & Calculation Method", fontsize=analysis_utils.FONT_SIZE_LABEL, color=analysis_utils.COLOR
    )

    plt.xticks(rotation=0, ha="center", fontsize=analysis_utils.FONT_SIZE_TICK, color=analysis_utils.COLOR)
    plt.yticks(color=analysis_utils.COLOR)

    for ax in axes:
        ax.axvline(1.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.close(fig)
    return fig

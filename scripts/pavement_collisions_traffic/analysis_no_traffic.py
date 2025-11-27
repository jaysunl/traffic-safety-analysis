from . import analysis_utils


def analyze_pavement_safety(df):
    # funclass to include
    target_funclasses = [
        "RES CUL DE SAC",
        "RES RESIDENTIAL LOCAL STREET",
        "CL 2 LANE SUB-COLLECTOR",
        "CL 2 LANE COLLECTOR",
        "CL 2 LANE COLLECTOR WITH 2 WAY LEFT TURN",
        "CL 4 LN COLLECTOR WITH 2 WY LEFT TURN LN",
        "MJ SIX LANE URBAN MAJOR",
    ]

    df = analysis_utils.filter_by_funclass(df, target_funclasses)

    # Text wrapping for funclass labels
    df, target_funclasses_wrapped = analysis_utils.wrap_funclass_labels(df, target_funclasses)

    # Data Aggregation
    grouped = analysis_utils.aggregate_data(df, ["funclass_wrapped", "pci_desc"], "pav_length")

    # Normalization
    grouped["total_mile_years"] = grouped["pav_length"] / 5280

    # Calculate Rates
    grouped = analysis_utils.calculate_rates(grouped, "total_mile_years", "crash_rate", "weighted_severity")

    return analysis_utils.plot_analysis(
        grouped,
        target_funclasses_wrapped,
        y_label_freq="Annual Crashes per Mile",
        y_label_sev="Annual Severity Index per Mile",
        top_adjust=0.88,
    )

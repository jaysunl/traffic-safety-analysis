import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# Handle file paths for jupyter notebook and running normally
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def get_data_path(relative_path: str) -> str:
    return str(PROJECT_ROOT / relative_path)


FILES = {
    # Input data files (raw from city portal)
    "pci_2016": get_data_path("data/raw/pavement_condition/pavement_condition_assessment_2016_datasd.csv"),
    "pci_2023": get_data_path("data/raw/streets_repair_line_segments/sd_paving_segs_datasd.csv"),
    "repairs": get_data_path("data/raw/streets_repair_projects/sd_paving_datasd.csv"),
    "collisions": get_data_path("data/raw/traffic_collisions_basic/pd_collisions_datasd.csv"),
    "traffic_counts": get_data_path("data/raw/traffic_volumes/traffic_counts_datasd.csv"),
    # Output data
    "out_segments_yearly": get_data_path("data/processed/segments_collisions_pci_counts.csv"),
}

# The timeframe of the PCI analyses is unknown, so we use mid-year "anchor" dates for 2016 and 2023.
# This means we assume the PCI inspections occured at these specific dates.
# These anchors can be adjusted as needed.
ANCHOR_2016 = pd.Timestamp("2016-07-01")  # unknown exact date
ANCHOR_2023 = pd.Timestamp("2023-06-01")  # "was completed over the course of several months" in spring 2023

# Analysis date range. Collision data starts circa 2016 and goes to present day.
DATE_START = pd.Timestamp("2016-01-01")
DATE_END = pd.Timestamp.now()

# Minimum length (feet) for a segment to be considered significant for analysis
MIN_PAV_LENGTH = 50

# Mapping full street suffixes to standard abbreviations for consistent string matching
SUFFIX_MAP = {
    "AVENUE": "AV",
    "STREET": "ST",
    "ROAD": "RD",
    "DRIVE": "DR",
    "BOULEVARD": "BL",
    "PLACE": "PL",
    "WAY": "WY",
    "COURT": "CT",
    "LANE": "LN",
    "TERRACE": "TER",
    "CIRCLE": "CR",
    "MOUNTAIN": "MTN",
    "MOUNT": "MT",
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "CAMINO": "CAM",
    "PARKWAY": "PY",
    "HIGHWAY": "HY",
    "MALL": "ML",
    "EXTENSION": "EX",
    "VALLEY": "VLY",
    "WALK": "WK",
}


# --- Helper functions ---


def clean_street_name(
    name_series: pd.Series, suffix_series: Optional[pd.Series] = None, prefix_series: Optional[pd.Series] = None
) -> pd.Series:
    """
    Standardizes street names by combining components, converting to uppercase,
    abbreviating suffixes, removing extra whitespace, and normalizing numbers
    (removing ordinal suffixes and leading zeros).

    Args:
        name_series (pd.Series): The main street name.
        suffix_series (pd.Series, optional): Street suffixes (e.g., 'Ave', 'St').
        prefix_series (pd.Series, optional): Street prefixes (e.g., 'N', 'S').

    Returns:
        pd.Series: A Series of cleaned, full street names strings.
    """
    # Construct a full street name from components
    full_name = name_series.fillna("")
    if prefix_series is not None:
        full_name = prefix_series.fillna("") + " " + full_name
    if suffix_series is not None:
        full_name = full_name + " " + suffix_series.fillna("")

    # Standardize casing and strip surrounding whitespace
    full_name = full_name.str.upper().str.strip()

    # Replace full suffixes with abbreviations
    for long_suffix, short_suffix in SUFFIX_MAP.items():
        full_name = full_name.str.replace(rf"\b{long_suffix}\b", short_suffix, regex=True)

    # Remove ordinal suffixes (1ST, 2ND, 3RD, 4TH, etc.)
    full_name = full_name.str.replace(r"(\d+)(ST|ND|RD|TH)\b", r"\1", regex=True)

    # Remove leading zeros from numbers at start of string or after space ("01 AV" -> "1 AV")
    full_name = full_name.str.replace(r"\b0+(\d+)", r"\1", regex=True)

    # Normalize internal whitespace to single spaces
    full_name = full_name.str.replace(r"\s+", " ", regex=True)
    return full_name.str.strip()


def get_pci_at_date(timeline: List[Tuple[pd.Timestamp, float]], target_date: pd.Timestamp) -> float:
    """
    Interpolates PCI (Pavement Condition Index) from a sorted list of date/value tuples.
    Returns the nearest value if the date is outside the timeline range.

    Args:
        timeline (List[Tuple[pd.Timestamp, float]]): Sorted list of (date, pci) tuples.
        target_date (pd.Timestamp): The date to query.

    Returns:
        float: The interpolated PCI value, or NaN if the timeline is empty.
    """
    if not timeline:
        return np.nan

    # Clamp to the earliest or latest available PCI if the date is out of bounds
    if target_date <= timeline[0][0]:
        return max(0.0, timeline[0][1])
    if target_date >= timeline[-1][0]:
        return max(0.0, timeline[-1][1])

    # Find the specific time interval containing the target date for linear interpolation
    for i in range(len(timeline) - 1):
        date_prev, pci_prev = timeline[i]
        date_next, pci_next = timeline[i + 1]
        if date_prev <= target_date <= date_next:
            total_days = (date_next - date_prev).days
            if total_days == 0:
                return max(0.0, pci_prev)
            fraction = (target_date - date_prev).days / total_days
            return max(0.0, pci_prev + (pci_next - pci_prev) * fraction)
    return np.nan


def calculate_weighted_avg_pci(
    timeline: List[Tuple[pd.Timestamp, float]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """
    Calculates the time-weighted average PCI between two dates using the Trapezoidal rule.
    This accounts for duration at specific conditions, providing a more accurate average
    than a simple mean of endpoints when repairs occur mid-year.

    Args:
        timeline (List[Tuple[pd.Timestamp, float]]): Sorted list of (date, pci) tuples.
        start_date (pd.Timestamp): The start date of the window.
        end_date (pd.Timestamp): The end date of the window.

    Returns:
        float: The time-weighted average PCI, or NaN if the timeline is empty.
    """
    if not timeline:
        return np.nan

    total_days = (end_date - start_date).days
    if total_days <= 0:
        return get_pci_at_date(timeline, start_date)

    # Get value at exact start and end of the window
    start_val = get_pci_at_date(timeline, start_date)
    end_val = get_pci_at_date(timeline, end_date)

    # Find all intermediate points defined in the timeline (repairs, inspections) that fall inside the window
    intermediate_points = [(t, v) for t, v in timeline if start_date < t < end_date]

    # Build the integration list: Start -> Intermediates -> End
    integration_points = [(start_date, start_val)] + intermediate_points + [(end_date, end_val)]

    # Calculate Area under the curve
    total_area = 0.0
    for i in range(len(integration_points) - 1):
        t1, pci1 = integration_points[i]
        t2, pci2 = integration_points[i + 1]

        # Time delta in days
        dt = (t2 - t1).total_seconds() / 86400.0

        # Area of trapezoid = width * average_height
        segment_area = dt * (pci1 + pci2) / 2
        total_area += segment_area

    return total_area / total_days


def load_traffic_counts() -> pd.DataFrame:
    """
    Loads and cleans traffic count data.

    Parses dates, calculates the average directional count across available
    directions, and standardizes street names and limits to prepare for matching.

    Returns:
        pd.DataFrame: A DataFrame containing cleaned traffic data with columns for
                      date, averaged volume, and standardized location limits.
    """
    df = pd.read_csv(FILES["traffic_counts"])

    # Parse date
    df["date_count"] = pd.to_datetime(df["date_count"], errors="coerce")
    df = df.dropna(subset=["date_count"])

    # Calculate average directional count
    # Average only populated columns (ignore empty ones)
    dir_cols = ["northbound_count", "southbound_count", "eastbound_count", "westbound_count"]
    for col in dir_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Average of the 4 columns (skips NaNs by default)
    df["traffic_vol"] = df[dir_cols].mean(axis=1)

    # Clean street name
    df["clean_street"] = clean_street_name(df["street_name"])

    # Clean limits
    # Limits format is usually "STREET A - STREET B"
    def parse_limits(limit_str):
        if not isinstance(limit_str, str):
            return None, None
        parts = limit_str.split(" - ")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    limits = df["limits"].apply(parse_limits)
    df["limit_1"] = clean_street_name(limits.apply(lambda x: x[0]))
    df["limit_2"] = clean_street_name(limits.apply(lambda x: x[1]))

    return df


def match_traffic_to_segments(
    segments_df: pd.DataFrame, traffic_df: pd.DataFrame
) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
    """
    Matches traffic counts to segments and builds a timeline of traffic volumes for each segment.

    Matching is performed by comparing standardized street names and cross-street limits.
    The order of cross-streets (limits) is treated as interchangeable (using sets) to ensure
    matches regardless of direction.

    Args:
        segments_df (pd.DataFrame): DataFrame containing segment definitions and cross-streets.
        traffic_df (pd.DataFrame): DataFrame containing traffic counts and limit definitions.

    Returns:
        Dict[str, List[Tuple[pd.Timestamp, float]]]: A dictionary mapping segment IDs (iamfloc)
                                                     to a sorted list of (date, volume) tuples.
    """
    # Prepare segments for matching
    segments_df = segments_df.copy()
    segments_df["match_street"] = clean_street_name(segments_df["rd20full"])
    segments_df["match_x1"] = clean_street_name(segments_df["xstrt1"])
    segments_df["match_x2"] = clean_street_name(segments_df["xstrt2"])

    # Create a lookup for traffic data
    # We can group traffic data by (street, limit1, limit2) sets
    # Since order doesn't matter, we use frozenset for limits
    traffic_map = {}  # (street, frozenset(limit1, limit2)) -> list of (date, vol)

    for _, row in traffic_df.iterrows():
        if pd.isna(row["limit_1"]) or pd.isna(row["limit_2"]):
            continue

        key = (row["clean_street"], frozenset([row["limit_1"], row["limit_2"]]))
        if key not in traffic_map:
            traffic_map[key] = []
        traffic_map[key].append((row["date_count"], row["traffic_vol"]))

    # Match segments
    segment_traffic_timelines = {}
    match_count = 0
    total_segments = len(segments_df)

    for _, row in segments_df.iterrows():
        key = (row["match_street"], frozenset([row["match_x1"], row["match_x2"]]))
        if key in traffic_map:
            # Sort by date
            timeline = sorted(traffic_map[key], key=lambda x: x[0])
            segment_traffic_timelines[row["iamfloc"]] = timeline
            match_count += 1

    pct = (match_count / total_segments * 100) if total_segments > 0 else 0.0
    print(f"\nMatched traffic counts to {match_count} segments ({pct:.1f}%)")
    return segment_traffic_timelines


def get_traffic_volume_at_date(timeline: List[Tuple[pd.Timestamp, float]], target_date: pd.Timestamp) -> float:
    """
    Interpolates traffic volume for a specific date from a segment's traffic timeline.

    Performs linear interpolation between known data points. If the target date
    falls outside the range of the timeline (start or end), constant extrapolation
    is used (returning the nearest known value).

    Args:
        timeline (List[Tuple[pd.Timestamp, float]]): Sorted list of (date, volume) tuples.
        target_date (pd.Timestamp): The specific date to estimate volume for.

    Returns:
        float: The interpolated traffic volume, or NaN if the timeline is empty.
    """
    if not timeline:
        return np.nan

    # If only one point, constant
    if len(timeline) == 1:
        return timeline[0][1]

    # Extrapolation (constant)
    if target_date <= timeline[0][0]:
        return timeline[0][1]
    if target_date >= timeline[-1][0]:
        return timeline[-1][1]

    # Interpolation
    for i in range(len(timeline) - 1):
        d1, v1 = timeline[i]
        d2, v2 = timeline[i + 1]
        if d1 <= target_date <= d2:
            total_days = (d2 - d1).days
            if total_days == 0:
                return v1
            fraction = (target_date - d1).days / total_days
            return v1 + (v2 - v1) * fraction

    return np.nan


# --- Core functions ---


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV data for PCI assessments, segment definitions, repairs, and collisions.
    Performs initial cleaning, type conversion, and filtering (e.g., dropping short segments).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Returns (pci_2016_df, segments_2023_df, repairs_df, collisions_df).
    """
    # Load 2016 PCI data
    # value of 'iamfloc' is the unique segment identifier used across datasets
    pci_2016_df = pd.read_csv(FILES["pci_2016"]).rename(columns={"seg_id": "iamfloc", "pci": "pci_16"})

    # Clean 2016 data: coerce non-numeric PCI values and remove entries without valid scores
    initial_count_16 = len(pci_2016_df)
    pci_2016_df = pci_2016_df.dropna(subset=["pci_16"])
    pci_2016_df["pci_16"] = pd.to_numeric(pci_2016_df["pci_16"], errors="coerce")
    pci_2016_df = pci_2016_df.dropna(subset=["pci_16"])

    # Ensure unique 2016 snapshot per segment
    count_before_drop = len(pci_2016_df)
    pci_2016_df = pci_2016_df[["iamfloc", "pci_16"]]
    print(
        f"Loaded 2016 PCI: {len(pci_2016_df)} valid segments (Dropped {initial_count_16 - count_before_drop} NaN/Invalid)"
    )

    # Load 2023 Segment data
    segments_2023_df = pd.read_csv(FILES["pci_2023"])
    numeric_columns = ["llowaddr", "lhighaddr", "rlowaddr", "rhighaddr", "pci23", "pav_length"]
    segments_2023_df[numeric_columns] = segments_2023_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Filter out insignificant segments that are smaller than the minimum pavement length
    initial_count_23 = len(segments_2023_df)
    segments_2023_df = segments_2023_df[segments_2023_df["pav_length"] >= MIN_PAV_LENGTH]
    dropped_short_segs = initial_count_23 - len(segments_2023_df)

    # Ensure 2023 data has valid PCI scores
    count_before_pci = len(segments_2023_df)
    segments_2023_df = segments_2023_df.dropna(subset=["pci23"])
    dropped_missing_pci = count_before_pci - len(segments_2023_df)

    print(
        f"Loaded 2023 Segments: {len(segments_2023_df)} valid (Dropped {dropped_short_segs} < {MIN_PAV_LENGTH}ft, {dropped_missing_pci} missing PCI)"
    )

    # 1. Standardize basic names using the helper
    raw_names = clean_street_name(segments_2023_df["rd20full"])

    # 2. Remove construction related parenthetical info (e.g. (SB), (N FTG)) from street names
    # Regex matches: optional whitespace, literal '(', non-')' chars, literal ')'
    segments_2023_df["clean_street"] = raw_names.str.replace(r"\s*\([^)]*\)", "", regex=True).str.strip()

    # Pre-calculate address ranges (min/max) for mid-block collision matching later
    segments_2023_df["seg_min"] = segments_2023_df[["llowaddr", "rlowaddr"]].min(axis=1)
    segments_2023_df["seg_max"] = segments_2023_df[["lhighaddr", "rhighaddr"]].max(axis=1)

    # Load and filter Repairs data
    # Only consider completed projects (POST CONSTRUCTION) with valid identifiers
    repairs_df = pd.read_csv(FILES["repairs"])
    repairs_df = repairs_df[(repairs_df["status"] == "POST CONSTRUCTION") & repairs_df["iamfloc"].notna()]
    repairs_df["date_end"] = pd.to_datetime(repairs_df["date_end"], errors="coerce")
    repairs_df = repairs_df.dropna(subset=["date_end"])
    repairs_df = repairs_df[repairs_df["date_end"] > DATE_START].sort_values("date_end")

    # Load and filter Collisions data within the analysis window (this is pretty much all collisions)
    collisions_df = pd.read_csv(FILES["collisions"], dtype={"report_id": str})
    collisions_df["date_time"] = pd.to_datetime(collisions_df["date_time"])
    collisions_df = collisions_df[(collisions_df["date_time"] >= DATE_START) & (collisions_df["date_time"] <= DATE_END)]

    # Filter out segments with pwidth > 500 feet
    initial_len = len(segments_2023_df)
    segments_2023_df = segments_2023_df[segments_2023_df["pwidth"] <= 500]
    print(f"Excluded {initial_len - len(segments_2023_df)} segments with pwidth > 500 feet")

    return pci_2016_df, segments_2023_df, repairs_df, collisions_df


def calculate_cohort_decay_rates(
    pci_2016_df: pd.DataFrame, segments_2023_df: pd.DataFrame, repairs_df: pd.DataFrame
) -> Tuple[Dict[str, float], float]:
    """
    Calculates the median daily deterioration rate of PCI for different functional classes
    based on segments that were NOT repaired between 2016 and 2023.

    Args:
        pci_2016_df (pd.DataFrame): 2016 PCI data.
        segments_2023_df (pd.DataFrame): 2023 Segment/PCI data.
        repairs_df (pd.DataFrame): Repairs data.

    Returns:
        Tuple[Dict[str, float], float]: A dictionary mapping functional class to decay rate,
        and a fallback global median decay rate.
    """
    cols_2023 = ["iamfloc", "funclass", "pci23"]
    merged_pci_df = pd.merge(segments_2023_df[cols_2023], pci_2016_df, on="iamfloc", how="inner")
    repaired_segment_ids = set(repairs_df["iamfloc"].unique())

    # The control groups are segments that were not repaired between 2016 and 2023.
    # We also verify pci_16 > pci23 to ensure we only look at logical decay, not data errors ("ghost improvements")
    control_group_df = merged_pci_df[~merged_pci_df["iamfloc"].isin(repaired_segment_ids)].copy()
    valid_decay_df = control_group_df[control_group_df["pci_16"] > control_group_df["pci23"]].copy()

    days_diff = (ANCHOR_2023 - ANCHOR_2016).days
    valid_decay_df["daily_loss"] = (valid_decay_df["pci_16"] - valid_decay_df["pci23"]) / days_diff

    # Calculate median decay rates per functional class
    # Use median to avoid skew from data entry errors (e.g., massive unjustified PCI drops)
    decay_rates_by_class = valid_decay_df.groupby("funclass")["daily_loss"].median()
    median_decay_rate = valid_decay_df["daily_loss"].median()

    print(f"\nDecay Rates (Global Median: {median_decay_rate:.5f}):")
    print(decay_rates_by_class.to_string())
    return decay_rates_by_class.to_dict(), median_decay_rate


def build_pci_timelines(
    pci_2016_df: pd.DataFrame,
    segments_2023_df: pd.DataFrame,
    repairs_df: pd.DataFrame,
    decay_rates: Dict[str, float],
    median_decay_rate: float,
) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
    """
    Constructs a continuous PCI timeline for each segment. Handles two cases:
    1. Repaired segments: Decay -> Repair Jump -> Decay.
    2. Non-repaired segments: Linear interpolation between 2016 and 2023.

    Args:
        pci_2016_df (pd.DataFrame): 2016 PCI data.
        segments_2023_df (pd.DataFrame): 2023 Segment/PCI data.
        repairs_df (pd.DataFrame): Repairs data.
        decay_rates (Dict[str, float]): Decay rates by functional class.
        median_decay_rate (float): Fallback decay rate.

    Returns:
        Dict[str, List[Tuple[pd.Timestamp, float]]]: A dictionary where keys are segment IDs
        and values are sorted lists of (date, pci_value) tuples.
    """
    # Only process segments that exist in both 2016 and 2023 to ensure a valid baseline
    pci_history_df = pd.merge(segments_2023_df, pci_2016_df, on="iamfloc", how="inner")
    print(
        f"\nSegments with history (2016 & 2023 match): {len(pci_history_df)} / {len(segments_2023_df)} (Dropped {len(segments_2023_df) - len(pci_history_df)} unmatched)"
    )

    repair_dates_by_segment = repairs_df.groupby("iamfloc")["date_end"].apply(list).to_dict()
    pci_timelines = {}

    # Interpolated segments are those that do not have any recorded repairs between 2016 and 2023.
    # Ghost improvements (where pci_2023 > pci_2016) are excluded as they imply missing repair data.
    stats = {"interpolated": 0, "ghost": 0}

    for _, row in pci_history_df.iterrows():
        segment_id, pci_2023, pci_2016, func_class = row["iamfloc"], row["pci23"], row["pci_16"], row["funclass"]

        # Determine specific decay rate for back-casting logic
        cohort_rate = decay_rates.get(func_class, median_decay_rate)
        relevant_repairs = [
            r for r in sorted(repair_dates_by_segment.get(segment_id, [])) if ANCHOR_2016 < r <= ANCHOR_2023
        ]

        points = []

        if relevant_repairs:
            # Case 1: Segment was repaired.
            # Strategy: Start at 2016 value, decay until repair date, jump to new value, decay to 2023.
            # This creates a "sawtooth" pattern.

            # Estimate the PCI right after the last repair so it decays exactly to the 2023 observed value
            last_repair_date = relevant_repairs[-1]
            days_to_23 = (ANCHOR_2023 - last_repair_date).days
            target_pci = min(100.0, max(pci_2023, pci_2023 + (cohort_rate * days_to_23)))

            # Back-cast start: Estimate the PCI prior to the 2016 inspection
            pci_start = min(100.0, pci_2016 + (cohort_rate * (ANCHOR_2016 - DATE_START).days))
            points.extend([(DATE_START, pci_start), (ANCHOR_2016, pci_2016)])

            current_date, current_pci = ANCHOR_2016, pci_2016
            for repair_date in relevant_repairs:
                # Calculate condition immediately before repair (Decay forward)
                pci_before_repair = max(0.0, current_pci - (cohort_rate * (repair_date - current_date).days))
                points.append((repair_date - pd.Timedelta(seconds=1), pci_before_repair))

                # Apply the repair (Instant jump)
                points.append((repair_date, target_pci))
                current_date, current_pci = repair_date, target_pci

            points.append((ANCHOR_2023, pci_2023))

            # Forecast future decay beyond the 2023 inspection
            points.append((DATE_END, pci_2023 - (cohort_rate * (DATE_END - ANCHOR_2023).days)))

        else:
            # Case 2: No Repairs recorded.
            # Strategy: Linear interpolation between 2016 and 2023 values.

            if pci_2023 > pci_2016:
                # "Ghost improvement": Data shows quality improved, but no repair was logged.
                # Excluding these from analysis as they imply missing repair data.
                stats["ghost"] += 1
                continue

            stats["interpolated"] += 1

            # Calculate the specific realized daily loss for this individual segment
            daily_loss = (pci_2016 - pci_2023) / (ANCHOR_2023 - ANCHOR_2016).days

            # Back-cast start: Estimate the PCI prior to the 2016 inspection using calculated rate
            days_back = (ANCHOR_2016 - DATE_START).days
            pci_start = min(100.0, pci_2016 + (daily_loss * days_back))

            points.append((DATE_START, pci_start))
            points.extend([(ANCHOR_2016, pci_2016), (ANCHOR_2023, pci_2023)])

            # Forward cast to present day
            points.append((DATE_END, pci_2023 - (daily_loss * (DATE_END - ANCHOR_2023).days)))

        points.sort(key=lambda x: x[0])
        pci_timelines[segment_id] = points

    print(
        f"\nTimeline Generation Stats:\n  Standard Interpolation: {stats['interpolated']}\n  Ghost Improvements (Removed): {stats['ghost']}"
    )
    return pci_timelines


def match_collisions_to_segments(segments_df: pd.DataFrame, collisions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Geocodes collisions to road segments. Uses intersection matching (primary + intersecting streets)
    and mid-block matching (address number ranges).

    Args:
        segments_df (pd.DataFrame): The street segments.
        collisions_df (pd.DataFrame): The traffic collision reports.

    Returns:
        pd.DataFrame: A DataFrame containing 'report_id' and 'roadsegid' for matched collisions.
    """
    # Prepare Segment Data for matching: Clean street names and calculate address range lengths
    raw_xstrt1 = clean_street_name(segments_df["xstrt1"])
    segments_df["clean_xstrt1"] = raw_xstrt1.str.replace(r"\s*\([^)]*\)", "", regex=True).str.strip()

    raw_xstrt2 = clean_street_name(segments_df["xstrt2"])
    segments_df["clean_xstrt2"] = raw_xstrt2.str.replace(r"\s*\([^)]*\)", "", regex=True).str.strip()

    # Range length is used to prefer more specific segments over wider ones if multiple match
    segments_df["addr_range_len"] = segments_df["seg_max"] - segments_df["seg_min"]

    # Match intersection collisions:
    # Identify collisions occurring at intersections (address_no_primary == 0)
    intersections_df = collisions_df[
        (collisions_df["address_no_primary"] == 0) & (collisions_df["address_name_intersecting"].notna())
    ].copy()

    # Clean collision street names for matching
    intersections_df["s1"] = clean_street_name(
        intersections_df["address_road_primary"],
        intersections_df["address_sfx_primary"],
        intersections_df["address_pd_primary"],
    )
    intersections_df["s2"] = clean_street_name(
        intersections_df["address_name_intersecting"],
        intersections_df["address_sfx_intersecting"],
        intersections_df["address_pd_intersecting"],
    )

    # Match against segment start cross-street (xstrt1) and end cross-street (xstrt2)
    match_primary = pd.merge(
        intersections_df, segments_df, left_on=["s1", "s2"], right_on=["clean_street", "clean_xstrt1"]
    )
    match_intersecting = pd.merge(
        intersections_df, segments_df, left_on=["s1", "s2"], right_on=["clean_street", "clean_xstrt2"]
    )

    # Combine matches and remove duplicates, prioritizing segments with longer paved lengths
    # This helps to prevent assigning collisions to "stubs" or "connector" segments.
    all_intersections = pd.concat([match_primary, match_intersecting])
    sorted_intersections = all_intersections.sort_values("pav_length", ascending=False)
    intersection_matches = sorted_intersections.drop_duplicates("report_id")
    intersection_matches = intersection_matches[["report_id", "roadsegid"]]

    # Match mid-block collisions:
    # Identify collisions with a specific house address number
    midblock_collisions_df = collisions_df[collisions_df["address_no_primary"] != 0].copy()
    midblock_collisions_df["s1"] = clean_street_name(
        midblock_collisions_df["address_road_primary"],
        midblock_collisions_df["address_sfx_primary"],
        midblock_collisions_df["address_pd_primary"],
    )
    midblock_collisions_df["addr_num"] = pd.to_numeric(midblock_collisions_df["address_no_primary"])

    # Match collision street to segment street, then filter by address range
    midblock_candidates = pd.merge(midblock_collisions_df, segments_df, left_on="s1", right_on="clean_street")

    # Filter by valid address range
    is_after_start = midblock_candidates["addr_num"] >= midblock_candidates["seg_min"]
    is_before_end = midblock_candidates["addr_num"] <= midblock_candidates["seg_max"]
    midblock_matches = midblock_candidates[is_after_start & is_before_end]

    # Select best match: smallest address range usually implies a more accurate segment
    sorted_midblocks = midblock_matches.sort_values("addr_range_len", ascending=True)
    midblock_final_matches = sorted_midblocks.drop_duplicates("report_id")
    midblock_final_matches = midblock_final_matches[["report_id", "roadsegid"]]

    combined_matches = pd.concat([intersection_matches, midblock_final_matches])
    all_matches = combined_matches.drop_duplicates("report_id")

    total_int, total_int_matched = len(intersections_df), len(intersection_matches)
    total_seg, total_seg_matched = len(midblock_collisions_df), len(midblock_final_matches)

    print(f"\nCollision Matching:")
    print(f"  Intersections: {total_int_matched} / {total_int} ({total_int_matched / total_int * 100:.1f}%)")
    print(f"  Mid-Blocks:    {total_seg_matched} / {total_seg} ({total_seg_matched / total_seg * 100:.1f}%)")
    print(
        f"  Total:         {len(all_matches)} / {len(collisions_df)} ({len(all_matches) / len(collisions_df) * 100:.2f}%)"
    )
    return all_matches


def generate_data():
    # Load and process raw data
    pci_2016_df, segments_2023_df, repairs_df, collisions_df = load_data()

    # Load traffic counts
    traffic_counts_df = load_traffic_counts()

    # Determine deterioration rates (baseline for interpolation)
    decay_rates, median_decay_rate = calculate_cohort_decay_rates(pci_2016_df, segments_2023_df, repairs_df)

    # Create PCI histories (Timelines) for valid segments
    segment_id_map = segments_2023_df.set_index("roadsegid")["iamfloc"].to_dict()
    pci_timelines = build_pci_timelines(pci_2016_df, segments_2023_df, repairs_df, decay_rates, median_decay_rate)

    # Match traffic counts to segments
    traffic_timelines = match_traffic_to_segments(segments_2023_df, traffic_counts_df)

    # Associate collisions with specific road segments
    collision_segment_matches = match_collisions_to_segments(segments_2023_df, collisions_df)
    collisions_with_pci_df = pd.merge(collisions_df, collision_segment_matches, on="report_id", how="inner")

    # Filter collisions to only include those on segments where we successfully generated a PCI timeline
    segments_with_timelines = set(pci_timelines.keys())
    mapped_ids = collisions_with_pci_df["roadsegid"].map(segment_id_map)
    is_valid_segment = mapped_ids.isin(segments_with_timelines)
    collisions_with_pci_df = collisions_with_pci_df[is_valid_segment].copy()

    # Categorize numeric PCI into qualitative descriptions (Good, Poor, Failed, etc.)
    # These categories come from the city (same as used in raw data).
    bins = [-1, 10, 25, 40, 55, 70, 85, 101]
    labels = ["Failed", "Serious", "Very Poor", "Poor", "Fair", "Satisfactory", "Good"]

    # Aggregate stats: Create a Yearly time-series dataset per segment
    print("\nAggregating Yearly Stats...")
    years_range = range(DATE_START.year, DATE_END.year + 1)
    valid_segments_df = segments_2023_df[segments_2023_df["iamfloc"].isin(segments_with_timelines)].copy()

    # Columns to preserve from the segment data in the yearly stats
    keep_cols = [
        "iamfloc",
        "roadsegid",
        "rd20full",
        "xstrt1",
        "xstrt2",
        "llowaddr",
        "rhighaddr",
        "zip",
        "cpname",
        "pwidth",
        "pav_length",
        "paveclass",
        "funclass",
    ]

    # Create base rows for each combination of segment and year
    yearly_rows = []
    for _, row in valid_segments_df.iterrows():
        segment_id = row["iamfloc"]
        timeline = pci_timelines[segment_id]
        traffic_timeline = traffic_timelines.get(segment_id)

        for year in years_range:
            date_year_start = pd.Timestamp(f"{year}-01-01")
            date_year_end = min(pd.Timestamp(f"{year}-12-31"), DATE_END)
            date_mid_year = pd.Timestamp(f"{year}-07-01")

            # Calculate weighted average PCI for the year
            pci_average = calculate_weighted_avg_pci(timeline, date_year_start, date_year_end)

            # Get start/end for reference columns
            pci_start = get_pci_at_date(timeline, date_year_start)
            pci_end = get_pci_at_date(timeline, date_year_end)

            # Calculate traffic volume
            traffic_vol = get_traffic_volume_at_date(traffic_timeline, date_mid_year)

            base_data = row[keep_cols].to_dict()
            base_data["year"] = year
            base_data["avg_pci"] = pci_average
            base_data["pci_start"] = pci_start
            base_data["pci_end"] = pci_end
            base_data["traffic_count"] = traffic_vol
            yearly_rows.append(base_data)

    yearly_stats_df = pd.DataFrame(yearly_rows)
    yearly_stats_df["pci_desc"] = (
        pd.cut(yearly_stats_df["avg_pci"], bins=bins, labels=labels, right=False).astype(str).replace("nan", "")
    )

    # Print traffic matching statistics
    print("\nTraffic Count Matching Statistics:")

    # Filter for rows with valid traffic counts
    matched_df = yearly_stats_df[yearly_stats_df["traffic_count"].notna()]

    total_rows = len(yearly_stats_df)
    matched_rows = len(matched_df)
    pct_total = (matched_rows / total_rows * 100) if total_rows > 0 else 0.0
    print(f"\nTotal Rows with Traffic Data: {matched_rows}/{total_rows} ({pct_total:.1f}%)")

    def print_stats(title, total, matched, sort_by_pci=False):
        print(f"\n{title}:")
        df = pd.DataFrame({"Total": total, "Matched": matched})
        df["Matched"] = df["Matched"].fillna(0).astype(int)
        df["Total"] = df["Total"].fillna(0).astype(int)
        df["Pct"] = (df["Matched"] / df["Total"] * 100).fillna(0)

        if sort_by_pci:
            pci_order = {l: i for i, l in enumerate(labels)}
            # Handle MultiIndex (funclass, pci_desc)
            if isinstance(df.index, pd.MultiIndex):
                df["pci_rank"] = df.index.map(lambda x: pci_order.get(x[1], 99))
                df = df.sort_values(["funclass", "pci_rank"]).drop(columns=["pci_rank"])
            else:
                df["pci_rank"] = df.index.map(lambda x: pci_order.get(x, 99))
                df = df.sort_values("pci_rank").drop(columns=["pci_rank"])

        print(df.to_string(formatters={"Pct": "{:.1f}%".format}))

    # Row matches by funclass
    print_stats(
        "Row matches by Functional Class (Total Rows)",
        yearly_stats_df["funclass"].value_counts(),
        matched_df["funclass"].value_counts(),
    )

    # Matches by PCI description per funclass
    print_stats(
        "Row matches by PCI Description per Functional Class",
        yearly_stats_df.groupby(["funclass", "pci_desc"]).size(),
        matched_df.groupby(["funclass", "pci_desc"]).size(),
        sort_by_pci=True,
    )

    # De-duplicated matches by PCI description per funclass (using most recent year for PCI desc)
    print_stats(
        "Unique Segments contributing to PCI Description per Functional Class",
        yearly_stats_df.groupby(["funclass", "pci_desc"])["iamfloc"].nunique(),
        matched_df.groupby(["funclass", "pci_desc"])["iamfloc"].nunique(),
        sort_by_pci=True,
    )

    # Merge crash counts into the yearly segment stats
    collisions_with_pci_df["year"] = collisions_with_pci_df["date_time"].dt.year
    agg_funcs = {
        "report_id": "count",
        "injured": lambda x: pd.to_numeric(x, errors="coerce").sum(),
        "killed": lambda x: pd.to_numeric(x, errors="coerce").sum(),
    }
    grouped_crashes = collisions_with_pci_df.groupby(["roadsegid", "year"]).agg(agg_funcs)
    crash_aggregations = grouped_crashes.rename(columns={"report_id": "total_crashes"}).reset_index()

    final_yearly_df = pd.merge(yearly_stats_df, crash_aggregations, on=["roadsegid", "year"], how="left")
    final_yearly_df = final_yearly_df.fillna({"total_crashes": 0, "injured": 0, "killed": 0})
    final_yearly_df.sort_values(["iamfloc", "year"]).to_csv(FILES["out_segments_yearly"], index=False)
    print(f"\nSaved: {FILES['out_segments_yearly']}")


if __name__ == "__main__":
    generate_data()

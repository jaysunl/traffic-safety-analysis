import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

COLLISIONS_CSV = "data/processed/collisions_at_intersections.csv"
VOLUMES_CSV    = "data/processed/paving_segments_with_traffic_volume.csv"

coll = pd.read_csv(COLLISIONS_CSV, low_memory=False)
vol  = pd.read_csv(VOLUMES_CSV, low_memory=False)

def norm_street(s: str) -> str:
    """Normalize street strings WITHOUT mapping SAINT->ST (ST is street)."""
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    if s in {"", "NAN", "NONE", "NULL", "NA", "N/A"}:
        return ""

    # standardize separators
    s = s.replace("&", " AND ").replace("/", " ")

    # standardize route formatting
    s = re.sub(r"\bI-\s*(\d+)\b", r"I \1", s)
    s = re.sub(r"\bSR-\s*(\d+)\b", r"SR \1", s)
    s = re.sub(r"\bCA-\s*(\d+)\b", r"CA \1", s)

    # drop punctuation
    s = re.sub(r"[.,'#()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # normalize direction words
    dir_map = {
        "NORTH":"N","SOUTH":"S","EAST":"E","WEST":"W",
        "NORTHEAST":"NE","NORTHWEST":"NW","SOUTHEAST":"SE","SOUTHWEST":"SW",
    }
    toks = [dir_map.get(t, t) for t in s.split()]
    s = " ".join(toks)

    # normalize common street types 
    type_map = {
        "STREET":"ST","ST":"ST",
        "AVENUE":"AVE","AV":"AVE","AVE":"AVE",
        "BOULEVARD":"BLVD","BLVD":"BLVD",
        "ROAD":"RD","RD":"RD",
        "DRIVE":"DR","DR":"DR",
        "LANE":"LN","LN":"LN",
        "COURT":"CT","CT":"CT",
        "PLACE":"PL","PL":"PL",
        "PARKWAY":"PKWY","PKWY":"PKWY",
        "CIRCLE":"CIR","CIR":"CIR",
        "TERRACE":"TER","TER":"TER",
        "HIGHWAY":"HWY","HWY":"HWY",
        "FREEWAY":"FWY","FWY":"FWY",
        "EXPRESSWAY":"EXPY","EXPY":"EXPY",
        "ALLEY":"ALY","ALY":"ALY",
        "PLAZA":"PLZ","PLZ":"PLZ",
        "WAY":"WAY"
    }

    toks = [type_map.get(t, t) for t in s.split()]
    return " ".join(toks).strip()

def build_full(pd_dir, road, sfx):
    """Build 'PD ROAD SFX' (skips blanks; avoids duplicate suffix)."""
    parts = [norm_street(pd_dir), norm_street(road), norm_street(sfx)]
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        road_tokens = parts[-2].split()
        sfx_tokens  = parts[-1].split()
        if sfx_tokens and road_tokens and road_tokens[-1] == sfx_tokens[-1]:
            parts = parts[:-1]
    return " ".join(parts)

def pair_key(a, b):
    """Orderless intersection key 'A @ B' so A&B == B&A."""
    a = norm_street(a); b = norm_street(b)
    if not a or not b:
        return None
    return f"{a} @ {b}" if a <= b else f"{b} @ {a}"

# Build normalized intersection keys from collisions
coll["roadA"] = coll.apply(lambda r: build_full(
    r.get("address_pd_primary",""),
    r.get("address_road_primary",""),
    r.get("address_sfx_primary","")
), axis=1)

coll["roadB"] = coll.apply(lambda r: build_full(
    r.get("address_pd_intersecting",""),
    r.get("address_name_intersecting",""),
    r.get("address_sfx_intersecting","")
), axis=1)

coll["pair_key"] = coll.apply(lambda r: pair_key(r["roadA"], r["roadB"]), axis=1)
coll = coll[coll["pair_key"].notna()].copy()

# time span in collisions dataset -> collisions per day
coll["date_time_parsed"] = pd.to_datetime(coll.get("date_time"), errors="coerce")
min_dt = coll["date_time_parsed"].min()
max_dt = coll["date_time_parsed"].max()
days_span = (max_dt - min_dt).days + 1 if pd.notna(min_dt) and pd.notna(max_dt) else np.nan

# Count collisions per intersection. 
if "report_id" in coll.columns:
    coll_counts = (coll.groupby("pair_key")["report_id"]
                      .nunique()
                      .reset_index(name="collisions"))
else:
    coll_counts = (coll.groupby("pair_key")
                      .size()
                      .reset_index(name="collisions"))

# Build normalized intersection exposure keys from volume segments
vol["rd"] = vol["rd20full"].map(norm_street)
vol["x1"] = vol["xstrt1"].map(norm_street)
vol["x2"] = vol["xstrt2"].map(norm_street)

# Each segment contributes exposure to two intersections
parts = []
for cross_col in ["x1","x2"]:
    tmp = vol[["rd", cross_col, "vmt_day_recent", "adt_recent"]].copy()
    tmp = tmp.rename(columns={cross_col:"cross"})
    tmp["pair_key"] = tmp.apply(lambda r: pair_key(r["rd"], r["cross"]), axis=1)
    parts.append(tmp)

vol_long = pd.concat(parts, ignore_index=True)
vol_long = vol_long[vol_long["pair_key"].notna()].copy()

exposure = (
    vol_long.groupby("pair_key", as_index=False)
            .agg(
                vmt_day=("vmt_day_recent","sum"),
                adt_sum=("adt_recent","sum")
            )
)

# Merge + compute rate
df = coll_counts.merge(exposure, on="pair_key", how="left")

# crashes per million VMT
df["collisions_per_day"]      = df["collisions"] / days_span
df["crashes_per_million_vmt"] = df["collisions_per_day"] / (df["vmt_day"] / 1e6)

df_clean = df[(df["vmt_day"] > 0) & (df["collisions"] >= 5) & (df["vmt_day"] >= 500)].copy()

#print(f"Collision date span: {min_dt.date()} to {max_dt.date()}  ({days_span} days)")
#print(f"Matched exposure for {df['vmt_day'].notna().mean():.1%} of intersections")

#print("\nTop 20 by raw collisions:")
#print(df[df["vmt_day"].notna()].sort_values("collisions", ascending=False)
#        .head(20)[["pair_key","collisions","vmt_day","crashes_per_million_vmt"]])

#print("\nTop 20 by crashes per million VMT (filtered for stability):")
#print(df_clean.sort_values("crashes_per_million_vmt", ascending=False)
#        .head(20)[["pair_key","collisions","vmt_day","crashes_per_million_vmt"]])


TOP_N = 20

top_raw = (df[df["vmt_day"].notna()]
           .sort_values("collisions", ascending=False)
           .head(TOP_N)
           .loc[:, ["pair_key", "collisions"]]
           .copy())

top_rate = (df_clean.sort_values("crashes_per_million_vmt", ascending=False)
            .head(TOP_N)
            .loc[:, ["pair_key", "crashes_per_million_vmt", "collisions", "vmt_day"]]
            .copy())

def plot_barh(title: str, x_col: str, df_plot: pd.DataFrame, xlabel: str, fname: str):
#    if df_plot.empty:
#        print(f"[plot] No data to plot for: {title}")
#        return

    # Reverse so the largest appears at the top in barh
    df_plot = df_plot.iloc[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(df_plot["pair_key"], df_plot[x_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Intersection")
    plt.tight_layout()

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
#    print("Saved plot:", out_path)

    plt.show()

plot_barh(
    title=f"Top {TOP_N} Intersections by Total Collisions ({min_dt.date()} to {max_dt.date()})",
    x_col="collisions",
    df_plot=top_raw,
    xlabel="Collisions",
    fname="top_intersections_by_collisions.png",
)

plot_barh(
    title=f"Top {TOP_N} Intersections by Crashes per Million VMT (filtered)",
    x_col="crashes_per_million_vmt",
    df_plot=top_rate,
    xlabel="Crashes per million VMT (per day normalized)",
    fname="top_intersections_by_crash_rate_per_million_vmt.png",
)
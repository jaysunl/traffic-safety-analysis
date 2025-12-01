import os
import re
import numpy as np
import pandas as pd

# Normalization helpers 
RE_MULTISPACE = re.compile(r"\s+")
RE_PARENS = re.compile(r"\([^)]*\)")
RE_SD_ROUTE = re.compile(r"\bSD\s*0*(\d{1,3})\b")  # e.g. "SD 052"
RE_ROUTE_I = re.compile(r"\bI\s*-\s*(\d{1,3})\b")  # e.g. "I-8"
RE_ROUTE_SR = re.compile(r"\bSR\s*-\s*(\d{1,3})\b")
RE_RAMP = re.compile(r"\bR-[A-Z]\b")

STOP_TOKENS = {"SB","NB","EB","WB","OFF","ON","RAMP","RAMPS","RMP","OFFRAMP","ONRAMP","EXIT","ENTRANCE"}

TYPE_MAP = {
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

def ordinal_suffix(n: int) -> str:
    if 10 <= (n % 100) <= 13:
        return "TH"
    return {1:"ST",2:"ND",3:"RD"}.get(n % 10, "TH")

def protect_routes(s: str) -> str:
    def sd_repl(m):
        num = int(m.group(1))
        if num in {5, 8, 15}:
            return f"I_{num}"
        if num == 805:
            return "I_805"
        return f"SR_{num}"

    s = RE_SD_ROUTE.sub(sd_repl, s)
    s = RE_ROUTE_I.sub(lambda m: f"I_{int(m.group(1))}", s)
    s = RE_ROUTE_SR.sub(lambda m: f"SR_{int(m.group(1))}", s)
    return s

def restore_routes(s: str) -> str:
    s = re.sub(r"\bI_(\d{1,3})\b", lambda m: f"I-{int(m.group(1))}", s)
    s = re.sub(r"\bSR_(\d{1,3})\b", lambda m: f"SR-{int(m.group(1))}", s)
    return s

def collapse_adjacent_dupes(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out

def norm_street(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).upper().strip()
    if not s:
        return ""

    s = RE_PARENS.sub("", s)
    s = s.replace("&", " AND ")
    s = protect_routes(s)

    s = s.replace("/", " ")
    s = re.sub(r"[.,]", " ", s)
    s = s.replace("-", " ")                     
    s = RE_MULTISPACE.sub(" ", s).strip()

    parts = s.split()

    if len(parts) >= 2 and parts[-1] in {"N","S","E","W"}:
        parts[-1] = {"N":"NORTH","S":"SOUTH","E":"EAST","W":"WEST"}[parts[-1]]

    parts = [("CAM" if p == "CAMINO" else p) for p in parts]

    parts = [p for p in parts if p not in STOP_TOKENS and not RE_RAMP.fullmatch(p)]

    # normalize street-type tokens
    parts = [TYPE_MAP.get(p, p) for p in parts]
    parts = collapse_adjacent_dupes(parts)

    s = " ".join(parts)

    # Fix "01 AV" to "01ST AV" 
    m = re.fullmatch(r"0*(\d+)\s+(ST|AV|RD|DR|BL|LN|CT|PL|PY|HY|WY|TR)", s)
    if m:
        num = int(m.group(1))
        typ = m.group(2)
        num_str = f"{num:02d}" if num < 10 else str(num)
        s = f"{num_str}{ordinal_suffix(num)} {typ}"

    # Fix "05TH AV"
    m = re.fullmatch(r"0*(\d+)(ST|ND|RD|TH)\s+(ST|AV|RD|DR|BL|LN|CT|PL|PY|HY|WY|TR)", s)
    if m:
        num = int(m.group(1))
        typ = m.group(3)
        num_str = f"{num:02d}" if num < 10 else str(num)
        s = f"{num_str}{ordinal_suffix(num)} {typ}"

    s = restore_routes(s)
    return s.strip()

def parse_limits(lim):
    # "A ST - ASH ST" => ("A ST","ASH ST")
    if lim is None or (isinstance(lim, float) and np.isnan(lim)) or str(lim).strip() == "":
        return ("","")
    raw = str(lim).upper()
    if " - " in raw:
        a, b = raw.split(" - ", 1)
    elif "-" in raw:
        a, b = raw.split("-", 1)
    else:
        return (norm_street(raw), "")
    return (norm_street(a), norm_street(b))

def make_key(road, a, b):
    # treat as unordered segment endpoints
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{road}|{lo}|{hi}"


segs = pd.read_csv("data/raw/streets_repair_line_segments/sd_paving_segs_datasd.csv")
counts = pd.read_csv("data/raw/traffic_volumes/traffic_counts_datasd.csv")

# Normalize paving segments
segs["road_norm"] = segs["rd20full"].map(norm_street)
segs["c1"] = segs["xstrt1"].map(norm_street)
segs["c2"] = segs["xstrt2"].map(norm_street)

# drop BEGIN/END segments 
segs.loc[segs["c1"].isin({"BEGIN","END",""}), "c1"] = ""
segs.loc[segs["c2"].isin({"BEGIN","END",""}), "c2"] = ""

segs["seg_key"] = segs.apply(lambda r: make_key(r["road_norm"], r["c1"], r["c2"]), axis=1)
segs_valid = segs[(segs["c1"] != "") & (segs["c2"] != "")].copy()

# Normalize traffic counts 
counts["road_norm"] = counts["street_name"].map(norm_street)
ab = counts["limits"].apply(parse_limits)
counts["c1"] = ab.map(lambda t: t[0])
counts["c2"] = ab.map(lambda t: t[1])
counts["seg_key"] = counts.apply(lambda r: make_key(r["road_norm"], r["c1"], r["c2"]), axis=1)

counts["date_count"] = pd.to_datetime(counts["date_count"], errors="coerce")
counts["total_count"] = pd.to_numeric(counts["total_count"], errors="coerce")


counts_recent = (
    counts.dropna(subset=["total_count"])
          .sort_values("date_count")
          .groupby("seg_key", as_index=False)
          .tail(1)
          .loc[:, ["seg_key", "total_count", "date_count"]]
          .rename(columns={"total_count": "adt_recent", "date_count": "adt_date"})
)

counts_mean = (
    counts.dropna(subset=["total_count"])
          .groupby("seg_key", as_index=False)
          .agg(adt_mean=("total_count","mean"), n_counts=("total_count","size"))
)

# Join onto paving segments
segs_with_volume = (
    segs_valid.merge(counts_recent, on="seg_key", how="left")
              .merge(counts_mean, on="seg_key", how="left")
)

#print("Segments with cross streets:", len(segs_valid))
#print("Matched to at least one count:", segs_with_volume["adt_recent"].notna().sum())
#print("Match rate:", segs_with_volume["adt_recent"].notna().mean())


segs_with_volume["seg_miles"] = pd.to_numeric(segs_with_volume["pav_length"], errors="coerce") / 5280.0
segs_with_volume["vmt_day_recent"] = segs_with_volume["adt_recent"] * segs_with_volume["seg_miles"]

out_dir = "data/processed"
os.makedirs(out_dir, exist_ok=True)

matched_out = f"{out_dir}/paving_segments_with_traffic_volume.csv"


matched_only = segs_with_volume[segs_with_volume["adt_recent"].notna()].copy()
matched_only.to_csv(matched_out, index=False)

#print("Saved matched-only output to:", matched_out)
#print("Matched-only rows:", len(matched_only))
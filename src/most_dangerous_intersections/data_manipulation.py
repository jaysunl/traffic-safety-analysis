import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "data/raw/traffic_collisions_detailed/pd_collisions_details_datasd.csv",
    low_memory=False
)

def present(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    return ~s.str.lower().isin({"", "nan", "none", "null", "na", "n/a"})

is_intersection = present(df["address_name_intersecting"])

df_intersections = df.loc[is_intersection].copy()

out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

df_intersections.to_csv(out_dir / "collisions_at_intersections.csv", index=False)



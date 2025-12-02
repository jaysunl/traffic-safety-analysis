## Intersection Crash Risk (Collisions Normalized by VMT)

This directory contains scripts that (1) extract **intersection collisions**, (2) estimate
**traffic exposure (VMT/day)** at intersections using traffic count data, and (3) rank
intersections by **raw collisions** and by **crash rate per million VMT**.

The goal is to answer:

> **Which intersections are “most dangerous,” both by total collisions and relative to traffic volume?**

---

## What this produces

After running the pipeline, you’ll get:

- `data/processed/collisions_at_intersections.csv`  
  Filtered collision records where an intersecting street is present.

- `data/processed/paving_segments_with_traffic_volume.csv`  
  Paving/road segments matched to traffic count records, including estimated segment VMT/day.

- `data/processed/top_intersections_by_collisions.png`  
  Top intersections by total collisions.

- `data/processed/top_intersections_by_crash_rate_per_million_vmt.png`  
  Top intersections by crash rate per million VMT (with stability filters).

---

## Repo/Data assumptions

These scripts expect the following files to exist at these paths:

### Collisions (raw)
- `data/raw/traffic_collisions_detailed/pd_collisions_details_datasd.csv`

### Street segments (raw)
- `data/raw/streets_repair_line_segments/sd_paving_segs_datasd.csv`

### Traffic counts (raw)
- `data/raw/traffic_volumes/traffic_counts_datasd.csv`

Outputs are written to:
- `data/processed/`

---

## Scripts

### 1) `data_manipulation.py`
**Purpose:** Extract only collisions that occurred at intersections.

**How it works:**
- Reads the detailed collision dataset
- Flags rows where `address_name_intersecting` is present
- Writes filtered output to `data/processed/collisions_at_intersections.csv`

---

### 2) `traffic_per_intersection.py`
**Purpose:** Match road segments to traffic counts and estimate exposure (ADT + VMT/day) per segment.

**How it works (high level):**
- Normalizes street names + cross streets (direction cleanup, suffix standardization, route formatting)
- Builds an **order-independent segment key**: `ROAD|CROSS1|CROSS2`
- For each segment, pulls:
  - **Most recent** traffic count (`adt_recent`, `adt_date`)
  - **Mean** traffic count across all observations (`adt_mean`, `n_counts`)
- Computes:
  - `seg_miles = pav_length / 5280`
  - `vmt_day_recent = adt_recent * seg_miles`
- Writes matched segments to `data/processed/paving_segments_with_traffic_volume.csv`

---

### 3) `collisions_by_vmt_perinter.py`
**Purpose:** Aggregate collisions by intersection and normalize by traffic exposure (VMT/day).

**Methodology:**
1. **Intersection keys from collisions**
   - Builds two street strings from collision fields (primary + intersecting)
   - Creates an **orderless** key: `A @ B` so `A @ B == B @ A`

2. **Intersection exposure from segments**
   - Each road segment contributes exposure to **two intersections** (one for each endpoint cross street)
   - Sums `vmt_day_recent` across all segments that “touch” that intersection

3. **Crash rate**
   - Computes collisions per day over the collision dataset time span
   - Then:
     \[
     \text{crashes\_per\_million\_vmt}
     = \frac{\text{collisions\_per\_day}}{\text{vmt\_day}/10^6}
     \]

4. **Stability filter (for rate ranking)**
   - Keeps intersections where:
     - `vmt_day > 0`
     - `collisions >= 5`
     - `vmt_day >= 500`

**Outputs:**
- `top_intersections_by_collisions.png`
- `top_intersections_by_crash_rate_per_million_vmt.png`

---

## Recommended run order (pipeline)

```bash
python data_manipulation.py
python traffic_per_intersection.py
python collisions_by_vmt_perinter.py
```

---

## Notes / Limitations

- **Matching is string-based.** Street normalization helps, but misspellings/alt-names can still reduce match rates.
- **Exposure is approximated.** VMT at an intersection is estimated by summing segment VMT touching that intersection (a reasonable proxy, but not perfect turning-movement volume).
- **Traffic counts are sparse in space/time.** The “most recent count” may not represent the collision period; interpret rates cautiously.
- The “Top by rate” list uses thresholds (`collisions >= 5`, `vmt_day >= 500`) to reduce noisy “small denominator” effects.


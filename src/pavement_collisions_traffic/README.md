## Analysis performed by Evan Robert

This directory contains scripts that process, generate, and analyze data linking streets, pavement conditions, traffic volums, and traffic collisions.

The purpose of this analysis is to determine if pavement condition has an impact on traffic collision frequency and/or severity.

Below are important details about each script.

## generate_data.py

### Purpose

This script parses, cleans, and merges data pavement condition index (PCI) data from the 2016 and 2023 inspections, street repair data, traffic collision data, and traffic volume data to create a unified dataset for analysis.

The output csv (data/processed/segments_collisions_pci_counts.csv) contains the above data for all street segments for the years 2016 - 2025.

### Important Details

#### PCI Estimation

The city only conducted PCI inspections in 2016 and 2023, however, traffic collisions data is available from circa 2016 to present day. To enrich the data, PCI values for the years 2017-2022 and 2024-2025 are estimated following this logic:
- Street segments without repairs: PCI is linearly interpolated between the two known inspection years (2016 and 2023) to estimate annual PCI values.
- Street segments with repairs: PCI is estimated by assuming a linear decline from 2016 to the point up until just prior to the repair date, and then a linear decline from the repair date to 2023 (passing through the 2023 inspection PCI) and beyond. The peak The rate of decline is calculated by taking the median of the decline rates of all road segments of the same type. This results in a "sawtooth" pattern for PCI over time for repaired segments.

#### Collision Matching

A main challenge when implementing this script was finding a reliable way to match street segment data to traffic collision data, especially regarding mid-block collisions and intersections. The logic for matching is:
- Mid-block collisions: the street segment containing the "tightest" (smallest) address range around the collision address is selected. This prevents double counting collisions on overlapping segments and helps to ensure the most precise match.
- Intersection collisions: the street segment with the longest length is selected. This helps to prevent collisions from being assigned to "stubs" or "connector" segments. For instance, if a collision occurs at the intersection of a large, arterial road and an alley, the collision is assigned to the arterial road segment rather than the alley.

#### Traffic Volume Estimation

The total amount of traffic volume data available that matches a street segment is limited. Additionally, there is sometimes only volume data in one direction (e.g. Northbound). To best estimate traffic volume for each street segment over all years, the following logic is used:
- A potential traffic volume for the North/South axis and the East/West axis is calculated separately (this is to prevent double counting). For each axis:
    - If both directions have data, they are summed.
    - If only one direction has data, that value is doubled.
- The maximum of the two axis values is taken as the estimated traffic volume for that street segment.

#### Other Notes
Some other less notable details:
- Street segments whose PCI increases between 2016 and 2023 WITHOUT a repair ("ghost improvement") are removed.
- In the output csv, the final avg_pci is a time-weighted average. This means that the avg_pci will accurately reflect if a street was in poor condition/good condition for a majority of the year.


## Analysis Scripts

#### analysis_no_traffic.py

This script creates a visualization for the annual crashes and severity index per mile for each street type. In this analysis, street types with neglible data are not included. 

Notably, in this script, traffic volume is not considered. This means that traffic volume could be a confounding variable.

#### analysis_with_traffic.py

This script creates a visualization for the annual crashes and severity index per million vehicle miles traveled for each street type. In this analysis, street types with neglible data are not included.

By considering traffic volume, this analysis is more robust against bias. However, the traffic volume data is limited, so this analysis should be interpreted with caution.

#### compare_analysis_methods.py

This script creates a side-by-side comparison of the two analysis methods (with and without traffic volume consideration) to highlight the differences in results.

#### interactive_map.py

This script creates an interactive map that shows zones and street segments overlaid onto the city map. Information about the zones and street segments is shown when hovering over them. PCI is color coded for easy viewing, and street width/thickness corresponds to to the number of crashes (thicker lines = more crashes).

#### analysis_utils.py

This script contains utility functions used by the analysis scripts.
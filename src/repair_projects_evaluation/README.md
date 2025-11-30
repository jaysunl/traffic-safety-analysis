# Street Repair Projects Evaluation

This module evaluates the effectiveness of street repair projects in San Diego by analyzing whether repairs achieve their intended goals of improving pavement condition and reducing traffic collisions.

## Purpose

While the main pavement-collision-traffic analysis examines the relationship between pavement condition and safety, this analysis specifically evaluates whether the city's repair projects are effective. This helps answer critical questions:

- **Do repairs actually improve pavement condition?**
- **Which repair types are most effective?**
- **Do repairs reduce traffic collisions?**
- **How long do repairs last before significant degradation?**
- **Are repairs cost-effective?**

## Analysis Components

### 1. PCI Improvement Analysis
Evaluates whether repairs improve Pavement Condition Index (PCI):
- Compares PCI before and after repairs
- Calculates improvement magnitude
- Identifies effective vs. ineffective repairs
- Analyzes by repair type (SLURRY, OVERLAY, CONCRETE)

### 2. Repair Type Comparison
Compares effectiveness across different repair types:
- **SLURRY**: Surface treatment (most common, ~28,000 projects)
- **OVERLAY**: Full overlay (moderate, ~9,000 projects)
- **CONCRETE**: Concrete repair (rare, ~200 projects)

### 3. Collision Reduction Analysis
Examines whether repairs lead to reduced traffic collisions:
- Compares collision rates before and after repairs
- Calculates crash reduction per mile
- Analyzes by repair type

### 4. Repair Longevity Analysis
Evaluates how long repairs last:
- Calculates PCI degradation rate after repair
- Estimates time until significant degradation
- Compares longevity by repair type

## Usage

Run the analysis:

```bash
python -m src.repair_projects_evaluation.repair_effectiveness_analysis
```

Or from Python:

```python
from src.repair_projects_evaluation.repair_effectiveness_analysis import main
main()
```

## Output Files

The analysis generates several output files in `data/processed/`:

1. **repair_effectiveness_analysis.png**: Comprehensive visualization dashboard
2. **repair_pci_analysis.csv**: Detailed PCI improvement data for each repair
3. **repair_collision_analysis.csv**: Collision reduction data (if available)
4. **repair_longevity_analysis.csv**: Longevity and degradation rate data

## Key Metrics

- **PCI Improvement Threshold**: 10 points (repairs improving PCI by ≥10 are considered "effective")
- **Degradation Threshold**: 5 points (significant degradation)
- **Analysis Window**: 2016-2023 (between PCI inspections)

## Methodology Notes

### PCI Estimation
- For repairs between 2016 and 2023, PCI before repair is estimated using linear decay from 2016 baseline
- PCI after repair is estimated based on 2023 inspection and decay rates
- Uses functional class-specific decay rates when available

### Collision Analysis
- Compares average annual crashes 2 years before vs. 2 years after repair
- Normalizes by segment length (crashes per mile)
- Only includes segments with sufficient data in both periods

### Limitations
- PCI values for years between inspections are estimated, not measured
- Collision data may be affected by other factors (traffic volume changes, weather, etc.)
- Some repairs may have incomplete data
- Multiple repairs on same segment may complicate analysis

## Interpretation

**Effective Repairs**: Repairs that improve PCI by ≥10 points
- Indicates successful intervention
- May justify continued investment in similar projects

**Ineffective Repairs**: Repairs with minimal PCI improvement
- May indicate:
  - Poor repair quality
  - Inappropriate repair type for condition
  - Underlying structural issues
  - Data quality issues

**Collision Reduction**: Positive values indicate fewer crashes after repair
- Suggests safety benefits beyond pavement condition
- May be confounded by other factors (traffic volume, weather, etc.)

## Future Enhancements

Potential improvements to the analysis:
- Cost-effectiveness analysis (if cost data available)
- Geographic patterns (which areas benefit most)
- Functional class patterns (which street types benefit most)
- Time series analysis of repair effectiveness over years
- Machine learning models to predict repair effectiveness


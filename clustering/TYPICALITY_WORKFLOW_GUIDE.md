# Typicality Scoring Workflow Guide

## Overview

This workflow helps you define what makes an FSC curve "typical" vs "atypical", then converts those definitions into quantitative typicality scores (0-1) that you can use for further analysis.

## The Three-Step Process

### Step 1: Cluster Inspection & Definition
**File:** `ClusterInspection_TypeicalityDefinition.py`

This script analyzes your clusters and helps you understand their characteristics:

```bash
cd clustering
python ClusterInspection_TypeicalityDefinition.py
```

**What it produces:**
- `cluster_distance_distributions_{fsc_curve_type}.png` - Histograms of distance distributions per cluster
- `cluster_statistics_{fsc_curve_type}.csv` - Statistical summaries of each cluster

**What you do:**
1. Run the inspection script
2. Look at the output visualizations
3. Look at your existing `cluster_subplots_{fsc_curve_type}.png` plots
4. Decide for each cluster: "Is this a good/typical FSC curve cluster?"

**Output to inspect:**
- `cluster_distance_distributions_{fsc_curve_type}.png` shows where each cluster's curves fall
- `cluster_statistics_{fsc_curve_type}.csv` shows suggested thresholds

---

### Step 2: Define Your Classes
**File:** `CalculateTypicalityScores.py` (top of file)

Based on your visual inspection, edit these two dictionaries:

#### Dictionary 1: Classify your clusters

```python
# Edit these based on visual inspection
TYPICAL_CLUSTERS = [0, 3, 5]        # Clusters with "good" FSC curves
ATYPICAL_CLUSTERS = [1, 2, 9]       # Clusters with "bad/noisy" FSC curves
```

**How to decide:**
- Look at `cluster_subplots_{fsc_curve_type}.png`
- Do the curves in this cluster look like good quality FSC curves?
- Are they smooth? Do they follow expected patterns?
- Mark the good ones as TYPICAL, bad ones as ATYPICAL

#### Dictionary 2: Set distance thresholds for typical clusters

```python
CLASS_THRESHOLDS = {
    0: 0.45,    # Cluster 0: curves within distance 0.45 are "typical"
    3: 0.38,    # Cluster 3: curves within distance 0.38 are "typical"
    5: 0.52,    # Cluster 5: curves within distance 0.52 are "typical"
}
```

**How to choose thresholds:**
- Use the "Suggested Threshold" from `cluster_statistics_{fsc_curve_type}.csv`
- Or use the percentile breakpoints from the histograms
- You can always adjust these later if needed
- The value represents: "Curves beyond this distance are outliers in this cluster"

---

### Step 3: Calculate Typicality Scores
**File:** `CalculateTypicalityScores.py`

Once you've defined your classes and thresholds, run:

```bash
python CalculateTypicalityScores.py
```

**What it produces:**

1. **CSV File:**
   - `typicality_scores_{fsc_curve_type}.csv`
   - Contains: cluster, distance, typicality_score, typicality_class, curve

2. **Visualizations:**
   - `typicality_distribution_{fsc_curve_type}.png` - Violin & histogram plots
   - `typicality_vs_distance_{fsc_curve_type}.png` - Scatter plot showing relationships
   - `typicality_by_cluster_{fsc_curve_type}.png` - Stacked bar chart per cluster

3. **Console output:**
   - Summary statistics
   - Per-cluster breakdown
   - Counts by typicality class

---

## Understanding the Typicality Score

### The Scoring System

**For curves in TYPICAL clusters:**
```
If distance ≤ threshold:
    typicality_score = 1.0 - (distance / threshold)
    class = "Typical"
    
If distance > threshold:
    typicality_score = 0.0
    class = "Atypical_Outlier"
```

**For curves in ATYPICAL clusters:**
```
typicality_score = 0.0
class = "Atypical"
```

**For curves in unclassified clusters:**
```
typicality_score = 0.5
class = "Mixed"
```

### What the Score Means

- **1.0**: At the cluster centroid (perfect representative)
- **0.8**: 80% of the way from centroid to threshold
- **0.5**: Halfway to threshold / or unclassified cluster
- **0.0**: Beyond threshold or in an atypical cluster / outlier

---

## Practical Example

Let's say you have this setup:

```python
TYPICAL_CLUSTERS = [0, 3]
ATYPICAL_CLUSTERS = [1, 2]
CLASS_THRESHOLDS = {0: 0.5, 3: 0.4}
```

**Example curves:**

| Curve | Cluster | Distance | Threshold | Typicality Score | Class |
|-------|---------|----------|-----------|------------------|-------|
| A     | 0       | 0.2      | 0.5       | 0.6              | Typical |
| B     | 0       | 0.6      | 0.5       | 0.0              | Atypical_Outlier |
| C     | 3       | 0.1      | 0.4       | 0.75             | Typical |
| D     | 1       | 0.3      | —         | 0.0              | Atypical |
| E     | 4       | 0.2      | —         | 0.5              | Mixed |

---

## Workflow Example

### Run 1: Inspect clusters
```bash
python ClusterInspection_TypeicalityDefinition.py
# Output: cluster_distance_distributions_fsc_masked.png
#         cluster_statistics_fsc_masked.csv
```

### Review
- Open `cluster_distance_distributions_fsc_masked.png`
- Look at each histogram
- Note which clusters look "good" and which look "bad"
- Check the suggested thresholds

### Run 2: Define classes and calculate scores
```bash
# Edit CalculateTypicalityScores.py:
# - Set TYPICAL_CLUSTERS = [0, 3, 5]
# - Set ATYPICAL_CLUSTERS = [1, 9]
# - Set CLASS_THRESHOLDS from inspection

python CalculateTypicalityScores.py
# Output: typicality_scores_fsc_masked.csv
#         typicality_distribution_fsc_masked.png
#         typicality_vs_distance_fsc_masked.png
#         typicality_by_cluster_fsc_masked.png
```

### Review Results
- Look at the new visualizations
- Do the typicality assignments match your intuition?
- Are "good looking" curves getting high scores?
- Are "bad looking" curves getting low scores?

### Refine if needed
- If thresholds are too strict/lenient, adjust and re-run
- If cluster classifications are wrong, adjust and re-run

---

## Converting to Binary Classification (Optional)

If you want binary Typical/Atypical labels:

```python
# In your analysis code:
df['is_typical'] = df['typicality_score'] > 0.5  # Simple threshold

# Or more sophisticated:
df['is_typical'] = (df['typicality_class'] == 'Typical')  # Only perfect class
```

---

## Tips & Tricks

### Choosing Good Thresholds
1. **Data-driven:** Use the "Suggested Threshold" as a baseline
2. **Visual:** Look at the histogram and pick where the distribution drops off
3. **Percentile-based:** Use 75th or 85th percentile instead of mean+std
4. **Conservative:** Start strict and loosen if you're excluding too much
5. **Domain knowledge:** Do the thresholds make sense for your application?

### Debugging Poor Classifications
If results don't match your visual inspection:
- Check that you classified the right clusters as TYPICAL/ATYPICAL
- Check that thresholds are reasonable (not too small/large)
- Look at `typicality_vs_distance_{fsc_curve_type}.png` for patterns
- Manually inspect curves at boundary scores

### Multiple FSC Curve Types
Repeat the entire workflow for each curve type:
```bash
# For fsc_masked:
python ClusterInspection_TypeicalityDefinition.py
# Edit CalculateTypicalityScores.py: fsc_curve_type = 'fsc_masked'
python CalculateTypicalityScores.py

# For fsc_unmasked:
# Edit both files: fsc_curve_type = 'fsc_unmasked'
# Repeat steps...
```

---

## Files Summary

| File | Purpose | When to use |
|------|---------|------------|
| `ClusterInspection_TypeicalityDefinition.py` | Analyze clusters and understand distances | Before defining classes |
| `CalculateTypicalityScores.py` | Calculate typicality scores | After defining classes |
| `cluster_distance_distributions_{type}.png` | Visualize distance distributions | Understanding cluster cohesion |
| `cluster_statistics_{type}.csv` | Statistical summary per cluster | Reference for thresholds |
| `typicality_scores_{type}.csv` | Main output with scores for each curve | Further analysis/filtering |
| `typicality_distribution_{type}.png` | Overall typicality score distribution | Quality control checks |
| `typicality_vs_distance_{type}.png` | Relationship between distance and score | Validation |
| `typicality_by_cluster_{type}.png` | Class distribution per cluster | Sanity checks |

---

## Next Steps

Once you have typicality scores, you can:
1. **Filter data:** Keep only "Typical" curves for downstream analysis
2. **Rank structures:** Score EMDB entries by their best FSC curve's typicality
3. **Quality metrics:** Track typicality as a quality metric over time
4. **Outlier detection:** Investigate why certain curves are atypical
5. **Predictions:** Train models to predict typicality on new curves


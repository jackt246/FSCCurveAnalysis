# Configuration Template for Typicality Scoring

## Use This Template

Copy and paste the structure below into `CalculateTypicalityScores.py` (lines 13-29) to configure your typicality definitions.

---

## For fsc_masked

### Step 1: Decide which clusters are typical vs atypical
```python
# Look at cluster_subplots_fsc_masked.png
# Ask: "Does this cluster look like good quality FSC data?"
# YES → add to TYPICAL_CLUSTERS
# NO  → add to ATYPICAL_CLUSTERS

TYPICAL_CLUSTERS = [0, 3, 5]        # Edit: your "good" clusters
ATYPICAL_CLUSTERS = [1, 2, 9]       # Edit: your "bad" clusters
```

### Step 2: Set distance thresholds for each typical cluster
```python
# Look at cluster_distance_distributions_fsc_masked.png
# Look at cluster_statistics_fsc_masked.csv "Suggested Threshold" column
# Pick a value that makes sense for each typical cluster

CLASS_THRESHOLDS = {
    0: 0.45,    # Edit: threshold for cluster 0
    3: 0.38,    # Edit: threshold for cluster 3
    5: 0.52,    # Edit: threshold for cluster 5
}
```

---

## For fsc_unmasked
(Repeat the same process)

```python
TYPICAL_CLUSTERS = []           # Edit
ATYPICAL_CLUSTERS = []          # Edit
CLASS_THRESHOLDS = {}           # Edit
```

---

## For fsc_corrected
(Repeat the same process)

```python
TYPICAL_CLUSTERS = []           # Edit
ATYPICAL_CLUSTERS = []          # Edit
CLASS_THRESHOLDS = {}           # Edit
```

---

## For fsc_phaserandom
(Repeat the same process)

```python
TYPICAL_CLUSTERS = []           # Edit
ATYPICAL_CLUSTERS = []          # Edit
CLASS_THRESHOLDS = {}           # Edit
```

---

## How to Fill This Out

### Understanding TYPICAL_CLUSTERS
Look at `cluster_subplots_fsc_masked.png`:
- Smooth curves, follow expected patterns → Typical
- Noisy, broken, random → Atypical
- Mixed quality → Skip (leave unclassified)

Example:
```
Cluster 0: All curves smooth and well-formed → TYPICAL
Cluster 1: All curves noisy and broken → ATYPICAL
Cluster 2: Half good, half bad → SKIP (unclassified)
Cluster 3: All curves smooth → TYPICAL
```

### Understanding CLASS_THRESHOLDS
For each TYPICAL cluster, look at `cluster_statistics_fsc_masked.csv`:

```
Column: "suggested_threshold"
Cluster 0: 0.45
Cluster 3: 0.38
Cluster 5: 0.52
```

Or look at `cluster_distance_distributions_fsc_masked.png`:
- Find where the histogram starts dropping off
- That's a good threshold

### What the Threshold Means
"Curves within this distance from the cluster center are typical"

Example:
- Cluster 0, threshold 0.45
- Curve at distance 0.2 → typicality = 1 - (0.2/0.45) = 0.56 ✓ Typical
- Curve at distance 0.6 → typicality = 1 - (0.6/0.45) = -0.33 → 0.0 ✗ Outlier

---

## Decision Tree

```
Looking at a cluster:

Does it look like "good" FSC data?
├─ YES → Add to TYPICAL_CLUSTERS
│   └─ Now set its threshold (use suggested value or adjust)
│
├─ NO → Add to ATYPICAL_CLUSTERS
│   └─ Done (no threshold needed)
│
└─ MAYBE (mixed) → Skip
    └─ Will get typicality_score = 0.5 (neutral)
```

---

## Example Configurations

### Conservative (Strict)
```python
TYPICAL_CLUSTERS = [0, 3]
ATYPICAL_CLUSTERS = [1, 2, 9]
CLASS_THRESHOLDS = {
    0: 0.35,
    3: 0.30,
}
# Result: Fewer curves classified as typical
```

### Moderate (Balanced)
```python
TYPICAL_CLUSTERS = [0, 3, 5]
ATYPICAL_CLUSTERS = [1, 2, 9]
CLASS_THRESHOLDS = {
    0: 0.45,
    3: 0.38,
    5: 0.52,
}
# Result: Balanced number of typical/atypical
```

### Lenient (Loose)
```python
TYPICAL_CLUSTERS = [0, 3, 5, 7]
ATYPICAL_CLUSTERS = [1, 9]
CLASS_THRESHOLDS = {
    0: 0.60,
    3: 0.50,
    5: 0.70,
    7: 0.55,
}
# Result: More curves classified as typical
```

---

## Workflow Checklist

- [ ] Run: `python ClusterInspection_TypeicalityDefinition.py`
- [ ] Review: `cluster_distance_distributions_fsc_masked.png`
- [ ] Review: `cluster_statistics_fsc_masked.csv`
- [ ] Review: Your existing `cluster_subplots_fsc_masked.png`
- [ ] Decide: Which clusters are typical/atypical
- [ ] Get thresholds from CSV or histogram
- [ ] Edit: `CalculateTypicalityScores.py` lines 13-29
- [ ] Run: `python CalculateTypicalityScores.py`
- [ ] Review: Generated `typicality_distribution_*.png`
- [ ] Check: Does it look right?
  - [ ] YES → Use the output CSV ✓
  - [ ] NO → Adjust thresholds and re-run (takes 1 minute)

---

## Tips

1. **Start simple:** Just classify the most obvious clusters first
2. **Use defaults:** Use the suggested thresholds from the CSV
3. **Test conservative:** If unsure, use strict thresholds first
4. **Iterate:** You can always adjust and re-run in 1 minute
5. **Multiple types:** Do this for each FSC curve type separately

---

## Common Values

After analyzing real FSC curves, typical thresholds are usually in this range:
```
0.30 - 0.70  (very tight to very loose)

Most common:
0.35 - 0.55  (medium range)
```

If your threshold is way outside this, double-check your input!

---

## Examples from Real Data

### Tight Cluster (Well-defined)
```
Mean distance: 0.2
Std: 0.05
Suggested threshold: 0.275
Your choice: 0.30
→ Creates strict "only best curves" classification
```

### Loose Cluster (Variable)
```
Mean distance: 0.4
Std: 0.15
Suggested threshold: 0.625
Your choice: 0.60
→ Creates lenient "good enough" classification
```

---

## FAQ

**Q: What if I pick the wrong threshold?**
A: Edit and re-run in 1 minute. No penalty.

**Q: Should all typical clusters have the same threshold?**
A: No! Each cluster can be different based on its variability.

**Q: How do I know if it worked?**
A: Look at `typicality_distribution_*.png`. Does it show the split you expected?

**Q: I want to exclude a cluster entirely, how?**
A: Put it in ATYPICAL_CLUSTERS (all curves get score 0).

**Q: Can I change thresholds later?**
A: Yes! Edit and re-run anytime.

---

## You're Ready!

Pick your configuration, edit the file, run the script, and get your typicality scores. 

Good luck! 🚀


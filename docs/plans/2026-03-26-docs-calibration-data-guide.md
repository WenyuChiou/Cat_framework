# Documentation Update: Calibration + Missing Value + User Data Guide

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add calibration methodology, missing value handling, and user data format guide to README.md and tutorials/README.md so users can understand the full pipeline without reading the .docx technical documentation.

**Architecture:** Update two existing markdown files with new sections. No code changes — documentation only. Content is already validated in CAT411_Technical_Documentation.docx (Sections 6.2, 9.5, 13).

**Tech Stack:** Markdown (GitHub-flavored)

---

### Task 1: Update README.md — Add Calibration section

**Files:**
- Modify: `README.md` (after "Validation Summary" section, ~line 216)

**Step 1: Add calibration section after Validation Summary**

Insert between "Validation Summary" and "Known Limitations":

```markdown
## MLE Fragility Calibration

Default Hazus fragility parameters over-predict Northridge bridge damage (27.2% predicted vs 10.6% observed). The framework includes an MLE calibration module (`src/calibration.py`) that fits two global parameters against Basoz & Kiremidjian (1998) observed counts (N=1,600):

| Parameter | HAZUS Default | Calibrated | Meaning |
|-----------|--------------|------------|---------|
| **k** (median scale factor) | 1.0 | 1.84 | All fragility medians x1.84; curves shift right |
| **beta** (dispersion) | 0.6 | 0.26 | Steeper curves; sharper damage transitions |
| **Damage fraction** | 27.2% | 9.5% | vs observed 10.6% |

```bash
# Run calibration
python scripts/run_calibration.py

# Run validation with calibrated parameters (default)
python scripts/anik_validation.py

# Apply to main pipeline — add to config.yaml:
#   calibration:
#     global_median_factor: 1.8432
```

Ground truth: Basoz & Kiremidjian (1998) aggregate damage counts from the 1994 Northridge earthquake, stored in `src/northridge_case.py`. See [Tutorial 04](tutorials/04_fragility.ipynb) Section 7 for details.
```

**Step 2: Update Validation Summary table to include calibrated results**

Update the L2 row to mention calibration:

```markdown
| **L2: Event** | Aggregate damage distribution vs Basoz (1998) survey of 1,600 bridges | Baseline over-predicts ~2.5x; **MLE calibration (k=1.84) reduces to 9.5% vs 10.6% observed** |
```

**Step 3: Update Known Limitations table**

Replace the "Fragility" row:

```markdown
| Fragility | Global 2-parameter calibration; intermediate DS ordering reversal (<0.5 bridges) | Per-class calibration with class-level damage counts |
```

**Step 4: Update Module Reference table**

Add calibration.py row:

```markdown
| | `calibration.py` | MLE fragility calibration (k, beta) against observed data |
```

**Step 5: Update Tutorials table**

Update NB04 description:

```markdown
| 04 | [Fragility Curves](tutorials/04_fragility.ipynb) | Vulnerability | Fragility parameters, HWB lookup, curve plotting, **MLE calibration** |
```

**Step 6: Commit**

```bash
git add README.md
git commit -m "docs: add calibration section + update validation/limitations in README"
```

---

### Task 2: Update README.md — Add Missing Value + User Data sections

**Files:**
- Modify: `README.md` (after "Known Limitations" section)

**Step 1: Add Missing Value Handling section**

```markdown
## Missing Value Handling

NBI bridge records may contain missing fields. The framework applies conservative defaults during classification:

| Field | Default | Rationale |
|-------|---------|-----------|
| `num_spans` | 1 | Single-span assumption (most conservative) |
| `year_built` | 1960 | Pre-seismic era (conventional design) |
| `structure_length_m` | 30 m | Typical short-span bridge |
| `deck_width_m` | 10 m | Standard two-lane width |
| `material_code` | "other" | Maps to HWB28 (general category) |
| `vs30` | 760 m/s | NEHRP B/C boundary (rock) |

ShakeMap interpolation NaN (outside convex hull) is filled with nearest-neighbor. See Technical Documentation Section 6.2 for full details.
```

**Step 2: Add User Data Guide section**

```markdown
## Using Your Own Data

The framework can analyze any earthquake + bridge inventory, not just Northridge.

### Required Input

1. **Bridge inventory CSV** with at minimum:

   | Column | Type | Example |
   |--------|------|---------|
   | `structure_number` | string | `53 0012` |
   | `latitude` | float | `34.0522` |
   | `longitude` | float | `-118.2437` |

   Optional: `year_built`, `material_code`, `design_code`, `num_spans`, `hwb_class` (if provided, skips auto-classification).

2. **Earthquake scenario** (choose one):
   - **ShakeMap:** `im_source: shakemap` + USGS event ID
   - **GMPE:** `im_source: gmpe` + magnitude, depth, epicenter, fault type
   - **Pre-computed Sa CSV:** with `structure_number` + `sa1s` columns

3. **Observed damage** (optional, for validation/calibration):

   | Column | Type | Allowed values |
   |--------|------|---------------|
   | `structure_number` | string | must match inventory |
   | `observed_damage` | string | none, slight, moderate, extensive, complete |

### Data Format Requirements

- CSV with UTF-8 encoding
- WGS84 coordinates (decimal degrees)
- Sa in units of g
- Damage states as lowercase strings

See Technical Documentation Section 13 for complete specifications and examples.
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add missing value handling + user data guide to README"
```

---

### Task 3: Update tutorials/README.md

**Files:**
- Modify: `tutorials/README.md`

**Step 1: Update NB04 description in table**

```markdown
| 04 | [Fragility Curves](04_fragility.ipynb) | Vulnerability | Load fragility parameter database (CSV), NBI → HWB → parameter lookup workflow, plot fragility curves, heatmap of 28 HWB classes, portfolio damage distribution, **MLE calibration (k=1.84, β=0.26) against Basoz 1998** |
```

**Step 2: Add calibration scripts section after "How to Run"**

```markdown
## Calibration & Validation Scripts

Beyond notebooks, standalone scripts provide production-ready calibration and validation:

| Script | Purpose | Command |
|--------|---------|---------|
| `scripts/run_calibration.py` | MLE calibration of fragility parameters | `python scripts/run_calibration.py` |
| `scripts/anik_validation.py` | Bridge-level validation (baseline + calibrated) | `python scripts/anik_validation.py` |
| `scripts/run_validation_real.py` | Validation with comparison plots | `python scripts/run_validation_real.py` |

Calibration results are saved to `output/calibration/` and automatically loaded by validation scripts.
```

**Step 3: Add data preparation note**

```markdown
## Data Preparation Credits

- **Sirisha Kedarsetty:** Mapped Northridge bridge damage descriptions to standardized HAZUS damage states; cross-verified HAZUS fragility parameter tables against Hazus 6.1 Technical Manual.
- **Anik Das:** Built the validation framework (confusion matrix, per-class metrics, residual analysis) and performed bridge-level validation against 2,008 Northridge bridges.
```

**Step 4: Commit**

```bash
git add tutorials/README.md
git commit -m "docs: update tutorials README with calibration + validation scripts"
```

---

### Task 4: Final verification

**Step 1: Verify markdown renders correctly**

```bash
# Check no broken links
grep -n '\[.*\](.*\.ipynb)' README.md tutorials/README.md
```

**Step 2: Push all commits**

```bash
git push
```

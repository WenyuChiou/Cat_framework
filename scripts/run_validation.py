"""Run fragility model validation against confirmed Northridge observations."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.fragility import damage_state_probabilities

# Load validation data
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
val = pd.read_csv(os.path.join(basedir, 'data/validation/northridge_validation_full.csv'))

# Only use confirmed observations
confirmed = val[val['damage_confirmed'] == True].copy()
print(f"Confirmed observations: {len(confirmed)}")
print(f"Damage distribution:")
print(confirmed['observed_damage_state'].value_counts().to_string())

# Run fragility model predictions
# NOTE: Hazus bridge fragility medians are expressed in Sa(1.0s), NOT PGA!
# Use sa1s_shakemap as the IM input.
DS_ORDER = ['none', 'slight', 'moderate', 'extensive', 'complete']
DS_INDEX = {ds: i for i, ds in enumerate(DS_ORDER)}

results = []
for _, row in confirmed.iterrows():
    pga = row['sa1s_shakemap']  # Sa(1.0s), matches Hazus fragility IM type
    hwb = row['hwb_class']

    # Get predicted probabilities
    probs = damage_state_probabilities(pga, hwb)

    # Most likely damage state (highest probability)
    pred_ds = max(DS_ORDER, key=lambda ds: probs[ds])
    pred_idx = DS_INDEX[pred_ds]

    # Expected damage index (weighted sum)
    expected_idx = sum(DS_INDEX[ds] * probs[ds] for ds in DS_ORDER)

    obs_ds = row['observed_damage_state']
    obs_idx = DS_INDEX.get(obs_ds, -1)

    results.append({
        'structure_number': row['structure_number'],
        'hwb_class': hwb,
        'pga': pga,
        'observed': obs_ds,
        'observed_idx': obs_idx,
        'predicted': pred_ds,
        'predicted_idx': pred_idx,
        'expected_idx': round(expected_idx, 2),
        'p_none': round(probs['none'], 3),
        'p_slight': round(probs['slight'], 3),
        'p_moderate': round(probs['moderate'], 3),
        'p_extensive': round(probs['extensive'], 3),
        'p_complete': round(probs['complete'], 3),
        'correct': pred_ds == obs_ds,
        'error': pred_idx - obs_idx
    })

rdf = pd.DataFrame(results)

print(f"\n{'='*70}")
print(f"VALIDATION RESULTS — Fragility Model vs Observed (N={len(rdf)})")
print(f"{'='*70}")

# Accuracy
accuracy = rdf['correct'].mean()
print(f"\nExact match accuracy: {rdf['correct'].sum()}/{len(rdf)} = {accuracy:.1%}")

# Mean absolute error in damage index
mae = abs(rdf['error']).mean()
print(f"Mean absolute error (damage index): {mae:.2f}")

# Bias (positive = over-prediction)
bias = rdf['error'].mean()
print(f"Mean bias: {bias:+.2f} ({'over-predicts' if bias > 0 else 'under-predicts'})")

# Confusion-style summary
print(f"\n{'─'*70}")
print(f"Observed vs Predicted damage state:")
print(f"{'─'*70}")
print(f"{'':>12s}  {'none':>6s}  {'slight':>6s}  {'mod':>6s}  {'ext':>6s}  {'comp':>6s}  | Total")
print(f"{'─'*70}")
for obs_ds in DS_ORDER:
    subset = rdf[rdf['observed'] == obs_ds]
    if len(subset) == 0:
        continue
    counts = [len(subset[subset['predicted'] == pd]) for pd in DS_ORDER]
    line = f"Obs={obs_ds:>8s}  " + "  ".join(f"{c:>6d}" for c in counts) + f"  | {sum(counts):>5d}"
    print(line)

# Per damage state analysis
print(f"\n{'─'*70}")
print(f"Per-bridge details (non-none observed):")
print(f"{'─'*70}")
damaged = rdf[rdf['observed'] != 'none'].sort_values('pga', ascending=False)
for _, r in damaged.iterrows():
    match = "OK" if r['correct'] else "MISS"
    print(f"  {r['structure_number']:>10s}  HWB={r['hwb_class']:>5s}  PGA={r['pga']:.3f}g  "
          f"Obs={r['observed']:>9s}  Pred={r['predicted']:>9s}  E[idx]={r['expected_idx']:.1f}  [{match}]")

# Also show PGA for context
print(f"\n  (IM used: Sa(1.0s) from ShakeMap, matching Hazus fragility convention)")

# Save results
outpath = os.path.join(basedir, 'data/validation/validation_results.csv')
rdf.to_csv(outpath, index=False, encoding='utf-8')
print(f"\nResults saved: {outpath}")

# Summary stats for report
print(f"\n{'='*70}")
print(f"SUMMARY FOR REPORT")
print(f"{'='*70}")
print(f"- Validation dataset: {len(val)} bridges from 1996 NBI")
print(f"- Confirmed observations: {len(confirmed)} bridges (from Basoz 1998, Yashinsky 1998, Mitchell 2011)")
print(f"- Confirmed damaged: {len(confirmed[confirmed['observed_damage_state']!='none'])}")
print(f"- Confirmed undamaged: {len(confirmed[confirmed['observed_damage_state']=='none'])}")
print(f"- Unknown status: {len(val[val['damage_confirmed']==False])}")
print(f"- Model accuracy (exact match): {accuracy:.1%}")
print(f"- Mean absolute error: {mae:.2f} damage states")
print(f"- Bias: {bias:+.2f}")

"""Rebuild validation dataset using 1994 NBI as base inventory."""
import sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from collections import defaultdict
from src.data_loader import load_shakemap
from src.interpolation import interpolate_im

# ── Helpers ──
def parse_latlon(line):
    try:
        lat_str = line[129:137]
        lon_str = line[137:146]
        lat = int(lat_str[0:2]) + int(lat_str[2:4])/60.0 + (int(lat_str[4:6]) + int(lat_str[6:8])/100.0)/3600.0
        lon = -(int(lon_str[0:3]) + int(lon_str[3:5])/60.0 + (int(lon_str[5:7]) + int(lon_str[7:9])/100.0)/3600.0)
        return lat, lon
    except:
        return None, None

def normalize_sn(sn):
    s = sn.strip()
    while len(s) > 2 and s[-1].isalpha() and s[-2].isdigit():
        s = s[:-1]
    s = re.sub(r'^(\d{2})C(\d{4})', r'\1 \2', s)
    return s.strip()

MAT_MAP = {
    '1': 'concrete', '2': 'concrete', '3': 'steel', '4': 'steel',
    '5': 'prestressed_concrete', '6': 'prestressed_concrete',
    '7': 'timber', '8': 'masonry', '9': 'aluminum', '0': 'other'
}

def simple_hwb(material, num_spans, design_era):
    if material == 'concrete':
        return ('HWB1' if design_era == 'conventional' else 'HWB2') if num_spans == 1 else ('HWB3' if design_era == 'conventional' else 'HWB4')
    elif material == 'steel':
        return ('HWB17' if design_era == 'conventional' else 'HWB18') if num_spans == 1 else ('HWB19' if design_era == 'conventional' else 'HWB20')
    elif material == 'prestressed_concrete':
        return ('HWB9' if design_era == 'conventional' else 'HWB10') if num_spans == 1 else ('HWB11' if design_era == 'conventional' else 'HWB12')
    return 'HWB5'

# ═══════════════════════════════════════════════════════════════
# STEP 1: Parse NBI94
# ═══════════════════════════════════════════════════════════════
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

with open(os.path.join(basedir, 'data/nbi/CA94.txt'), 'r', encoding='latin-1') as f:
    lines96 = f.readlines()

nbi94_bridges = {}
for line in lines96:
    sn_raw = line[3:18].strip()
    if not (sn_raw.startswith('53') or sn_raw.startswith('52') or sn_raw.startswith('55')):
        continue
    lat, lon = parse_latlon(line)
    if lat is None or not (33.8 <= lat <= 34.6 and -118.9 <= lon <= -118.0):
        continue
    try:
        year = int(line[156:160].strip())
    except:
        year = 0
    mat_code = line[128:129].strip()
    material = MAT_MAP.get(mat_code, 'other')
    try:
        num_spans = int(line[176:179].strip())
    except:
        num_spans = 1

    norm = normalize_sn(sn_raw)
    if norm not in nbi94_bridges:
        nbi94_bridges[norm] = {
            'structure_number': norm,
            'latitude': lat, 'longitude': lon,
            'year_built': year, 'material': material,
            'num_spans': num_spans,
            'design_era': 'conventional' if year < 1975 else 'seismic'
        }

print(f'NBI94 bridges in Northridge region: {len(nbi94_bridges)}')

# ═══════════════════════════════════════════════════════════════
# STEP 2: Assign ShakeMap IM values
# ═══════════════════════════════════════════════════════════════
grid = load_shakemap(os.path.join(basedir, 'data', 'grid.xml'))
bridge_lats = np.array([b['latitude'] for b in nbi94_bridges.values()])
bridge_lons = np.array([b['longitude'] for b in nbi94_bridges.values()])

pga_vals = interpolate_im(grid['LAT'].values, grid['LON'].values, grid['PGA'].values,
                          bridge_lats, bridge_lons, method='nearest')
sa1s_vals = interpolate_im(grid['LAT'].values, grid['LON'].values, grid['PSA10'].values,
                           bridge_lats, bridge_lons, method='nearest')

for i, (sn, b) in enumerate(nbi94_bridges.items()):
    b['pga_shakemap'] = float(pga_vals[i])
    b['sa1s_shakemap'] = float(sa1s_vals[i])

print(f'ShakeMap PGA: {pga_vals.min():.4f} - {pga_vals.max():.4f} g')

# ═══════════════════════════════════════════════════════════════
# STEP 3: Load confirmed observations
# ═══════════════════════════════════════════════════════════════
val_old = pd.read_csv(os.path.join(basedir, 'data/validation/confirmed_observations_backup.csv'))
confirmed = val_old[val_old['damage_confirmed'] == True]

obs_lookup = {}
for _, row in confirmed.iterrows():
    norm = normalize_sn(row['structure_number'])
    obs_lookup[norm] = {
        'observed_damage_state': row['observed_damage_state'],
        'observed_damage_index': int(row['observed_damage_index']),
        'damage_description': row['damage_description'] if pd.notna(row['damage_description']) else '',
        'data_source': row['data_source'],
        'pga_literature': row['pga_literature'] if pd.notna(row['pga_literature']) else ''
    }

print(f'Observed damage entries: {len(obs_lookup)}')
print(f'Non-none: {sum(1 for v in obs_lookup.values() if v["observed_damage_state"] != "none")}')

# ═══════════════════════════════════════════════════════════════
# STEP 4: HWB from 2024 NBI where available
# ═══════════════════════════════════════════════════════════════
nbi24 = pd.read_csv(os.path.join(basedir, 'data/nbi_classified_2024.csv'))
hwb_lookup = {}
for _, row in nbi24.iterrows():
    norm = normalize_sn(row['structure_number'])
    hwb_lookup[norm] = row['hwb_class']

# ═══════════════════════════════════════════════════════════════
# STEP 5: Build full validation dataset
# ═══════════════════════════════════════════════════════════════
rows = []
matched_obs = 0
for sn, b in nbi94_bridges.items():
    hwb = hwb_lookup.get(sn, simple_hwb(b['material'], b['num_spans'], b['design_era']))
    obs = obs_lookup.get(sn, None)

    if obs:
        matched_obs += 1
        row = {**b, 'hwb_class': hwb,
               'pga_shakemap': round(b['pga_shakemap'], 4),
               'sa1s_shakemap': round(b['sa1s_shakemap'], 4),
               'pga_literature': obs['pga_literature'],
               'observed_damage_state': obs['observed_damage_state'],
               'observed_damage_index': obs['observed_damage_index'],
               'damage_confirmed': True,
               'damage_description': obs['damage_description'],
               'data_source': obs['data_source']}
    else:
        row = {**b, 'hwb_class': hwb,
               'pga_shakemap': round(b['pga_shakemap'], 4),
               'sa1s_shakemap': round(b['sa1s_shakemap'], 4),
               'pga_literature': '',
               'observed_damage_state': 'unknown',
               'observed_damage_index': -1,
               'damage_confirmed': False,
               'damage_description': '',
               'data_source': 'NBI94_no_observation'}
    rows.append(row)

df = pd.DataFrame(rows)
cols = ['structure_number', 'latitude', 'longitude', 'year_built', 'hwb_class',
        'material', 'num_spans', 'design_era', 'pga_shakemap', 'sa1s_shakemap',
        'pga_literature', 'observed_damage_state', 'observed_damage_index',
        'damage_confirmed', 'damage_description', 'data_source']
df = df[cols]

print(f'\n=== REBUILT VALIDATION DATASET ===')
print(f'Total bridges (NBI94 base): {len(df)}')
print(f'Matched observations: {matched_obs}')

conf = df[df['damage_confirmed'] == True]
print(f'\nDamage distribution (confirmed):')
print(conf['observed_damage_state'].value_counts().to_string())

non_none = len(conf[conf['observed_damage_state'] != 'none'])
print(f'\nConfirmed damage (non-none): {non_none}')
print(f'Damage rate: {non_none}/{len(df)} = {non_none/len(df)*100:.2f}%')
print(f'(Basoz reported ~6.7% = 221/3318)')

# High PGA bridges without confirmed damage (potential unidentified damage)
high_pga = df[(df['pga_shakemap'] > 0.4) & (df['damage_confirmed'] == False)]
print(f'\nBridges with PGA > 0.4g but no confirmed status: {len(high_pga)}')

outpath = os.path.join(basedir, 'data/validation/northridge_1994_validation.csv')
df.to_csv(outpath, index=False, encoding='utf-8')
print(f'\nSaved: {outpath}')

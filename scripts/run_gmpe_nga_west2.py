"""
==============================================================================
NGA-West2 GMPE Validation — 1994 Northridge Earthquake
==============================================================================
Author : Kubilay Albayrak (original analysis)
Integrated: 2026-03-15 into CAT411 framework
Source : Kubilay Albayrak/GMPE/Northridge_Earthquake_GMPE.py

Usage:
  cd <project_root>
  python scripts/run_gmpe_nga_west2.py

Input:
  data/northridge_validation_full.xlsx  (ShakeMap observation data)

Output:
  output/gmpe_nga_west2/  (~48 files: PNG plots, CSV statistics, XLSX grids)

Description:
  NGA-West2 four-model ensemble (ASK14, BSSA14, CB14, CY14) GMPE validation
  against USGS ShakeMap v4 observations for the 1994 Northridge earthquake.
  Uses finite-fault distances (Rrup, Rjb, Rx) with hanging-wall effects.
  Analyses PGA and Sa(1.0s) across 7 NEHRP site classes (Vs30 180–1500 m/s).

==============================================================================
Original docstring:
==============================================================================
Reference: https://earthquake.usgs.gov/earthquakes/eventpage/ci3144585/shakemap/analysis

USGS ShakeMap v4 Exact Input Parameters (ci3144585):
  Mw           : 6.7
  Epicentre    : 34.213°N, 118.537°W
  Depth        : 17.5 km  (USGS catalog — HYPOCENTRAL depth)
  Fault type   : Blind reverse thrust  (FRV=1, FNM=0)
  Strike / Dip / Rake : 122° / 40° / 101°
    → Hauksson et al. (1995) JGR; Wald et al. (1996) BSSA;
      confirmed in USGS ShakeMap v4 rupture.json (ci3144585)
  Fault dims   : Length = 18 km, Width = 22 km
  Ztor         : 5.0 km  (Wald et al. 1996; USGS rupture.json ci3144585)
                 NOTE: the original code derived Ztor = DEPTH - (W/2)*sin(dip)
                 = 17.5 - 7.07 ≈ 10.4 km, incorrectly assuming the hypocentre
                 sits at the fault centre. For Northridge, the hypocentre is
                 near the fault BASE. The published Ztor = 5.0 km is used.
  Fault anchor : Top-centre of rupture, displaced UP-DIP from the epicentre
                 by (DEPTH - Ztor)/sin(dip) * cos(dip) ≈ 15.2 km horizontally.
                 Previously placed at the epicentre itself (incorrect).
  Vs30         : 760 m/s  (ShakeMap analysis-plot convention)
  GMPE set     : NGA-West2 California default (ShakeMap v4):
                 ASK14 · BSSA14 · CB14 · CY14  (equal weights = 0.25 each)

FIXES vs original northridge_usgs_matched_fixed.py:
  FIX 1–9  — (inherited from previous version; see original file header)
  FIX 10   — ZTOR hardcoded to 5.0 km (Wald et al. 1996 published value).
  FIX 11   — Fault anchor corrected in rrup_finite, rjb_finite, rx_finite.
              The top-centre of the rupture is now displaced up-dip from the
              epicentre by (DEPTH-ZTOR)/sin(dip)*cos(dip) km horizontally.
  FIX 12   — stats(): sigma changed to ddof=1 (sample standard deviation).
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Cross-platform paths (adapted for CAT411 framework) ──────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT          = os.path.join(PROJECT_ROOT, "output", "gmpe_nga_west2", "")
DATA_CSV     = os.path.join(PROJECT_ROOT, "data", "northridge_validation_full.xlsx")
os.makedirs(OUT, exist_ok=True)
print(f"Project root: {PROJECT_ROOT}")
print(f"Output dir  : {OUT}")
print(f"Input XLSX  : {DATA_CSV}")

# ══════════════════════════════════════════════════════════════════════════════
# EXACT USGS SHAKEMAP PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
MW        = 6.7
EPI_LAT   = 34.213
EPI_LON   = -118.537
DEPTH     = 17.5        # km  (USGS catalog — hypocentral depth)
STRIKE    = 122.0       # °
DIP       = 40.0        # °
RAKE      = 101.0       # °
FAULT_L   = 18.0        # km  along-strike length
FAULT_W   = 22.0        # km  down-dip width
VS30_REF  = 760.0       # m/s
VS30_SOIL = 270.0       # m/s
FRV_EVENT = 1           # renamed from FRV to avoid shadowing (FIX 6)
FNM_EVENT = 0

# ── Vs30 Values for Comprehensive Analysis ────────────────────────────────────
VS30_VALUES = {
    'Very Soft Soil': 180,      # NEHRP E
    'Soft Soil': 270,           # NEHRP D
    'Medium Soil': 360,         # NEHRP D
    'Stiff Soil': 520,          # NEHRP C
    'Rock (Reference)': 760,    # NEHRP B/C boundary
    'Hard Rock': 1100,          # NEHRP B
    'Very Hard Rock': 1500      # NEHRP A
}

# FIX 10 — Use the published Ztor from Wald et al. (1996).
# The formula DEPTH - (W/2)*sin(dip) = 17.5 - 7.07 ≈ 10.4 km is WRONG for
# Northridge because it assumes the hypocentre is at the fault mid-width.
# The Northridge hypocentre is near the fault BASE; Ztor = 5 km is published.
ZTOR = 5.0   # km — Wald et al. (1996); USGS ShakeMap v4 rupture.json ci3144585

# Pre-compute trig constants
DIP_R    = np.radians(DIP)
STRIKE_R = np.radians(STRIKE)

# FIX 11 — Compute the corrected fault anchor (top-centre of rupture).
#
# The epicentre is the surface projection of the hypocentre (depth = DEPTH).
# The hypocentre lies on the fault plane, exactly along_dip_to_top km measured
# DOWN-DIP along the plane from the top edge.
#
#   along_dip_to_top = (DEPTH − ZTOR) / sin(dip)   [km along fault plane]
#
# The 3-D down-dip unit vector is:
#   e2 = [cos(S)·cos(D),  −sin(S)·cos(D),  −sin(D)]
# whose horizontal component ("e2h") is:
#   e2h = [cos(S)·cos(D),  −sin(S)·cos(D)]  (magnitude = cos(D), NOT a unit vector)
#
# The top-centre anchor satisfies:
#   anchor + along_dip_to_top · e2 = hypocentre_3D = (0, 0, −DEPTH)
# ⟹ anchor = −along_dip_to_top · e2  (evaluated at z = −ZTOR in 3-D)
#
# For the HORIZONTAL (x,y) components this gives:
#   ANCHOR_X = −e2h_x · along_dip_to_top   ← note: scaled by along_dip, NOT up_dip_horiz
#   ANCHOR_Y = −e2h_y · along_dip_to_top
#
# The horizontal distance from epicentre to anchor (i.e., the surface offset) is:
#   up_dip_horiz_dist = along_dip_to_top · cos(dip)
# This equals |ANCHOR_XY| because |e2h| = cos(D).
along_dip_to_top   = (DEPTH - ZTOR) / np.sin(DIP_R)
up_dip_horiz_dist  = along_dip_to_top * np.cos(DIP_R)   # horizontal projection

e2h_x = np.cos(STRIKE_R) * np.cos(DIP_R)   # down-dip horizontal unit x
e2h_y = -np.sin(STRIKE_R) * np.cos(DIP_R)  # down-dip horizontal unit y

# Correct anchor formula: scale e2h by along_dip_to_top (not up_dip_horiz_dist)
ANCHOR_X = -e2h_x * along_dip_to_top
ANCHOR_Y = -e2h_y * along_dip_to_top
ANCHOR_Z = -ZTOR   # below surface (negative z)

print(f"Ztor             : {ZTOR:.1f} km  (Wald et al. 1996 published value)")
print(f"along_dip_to_top : {along_dip_to_top:.2f} km  (hypocentre to top edge along fault plane)")
print(f"up_dip_horiz_dist: {up_dip_horiz_dist:.2f} km  (horizontal surface offset epicentre→anchor)")
print(f"Anchor (local km): E={ANCHOR_X:.2f}  N={ANCHOR_Y:.2f}  Z={ANCHOR_Z:.2f}")

SM_MODELS  = ["ASK14", "BSSA14", "CB14", "CY14"]
SM_COLORS  = {"ASK14":"#e41a1c","BSSA14":"#377eb8","CB14":"#4daf4a","CY14":"#ff7f00"}
SIG_PGA    = {"ASK14":0.60,"BSSA14":0.58,"CB14":0.62,"CY14":0.60}
SIG_SA1    = {"ASK14":0.65,"BSSA14":0.63,"CB14":0.67,"CY14":0.65}

TITLE_HEAD = "1994 Northridge Mw 6.7  |  USGS ShakeMap v4 Parameters  (ci3144585)\n"
PARAM_STR  = (f"Strike={STRIKE:.0f}°  Dip={DIP:.0f}°  Rake={RAKE:.0f}°  "
              f"L={FAULT_L:.0f} km  W={FAULT_W:.0f} km  Ztor={ZTOR:.1f} km  "
              f"|  Rrup to finite fault  |  Vs30={VS30_REF:.0f} m/s (ref. rock)")


# ══════════════════════════════════════════════════════════════════════════════
# FINITE-FAULT DISTANCE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def _local_cart(site_lat, site_lon):
    """Site position in km relative to epicentre (E=x, N=y)."""
    kml = 111.32
    kmn = 111.32 * np.cos(np.radians(EPI_LAT))
    return (site_lon - EPI_LON) * kmn, (site_lat - EPI_LAT) * kml


def rrup_finite(site_lat, site_lon):
    """
    Rrup = shortest 3-D distance to finite fault plane.

    FIX 11: anchor is the top-centre of the rupture, displaced up-dip from the
    epicentre by up_dip_horiz_dist km. Previously the anchor was placed
    directly above the epicentre (only valid if hypocentre is at fault centre).
    """
    sx, sy = _local_cart(site_lat, site_lon)

    # Unit vectors: e1 = along-strike, e2 = down-dip (pointing into earth)
    e1 = np.array([np.sin(STRIKE_R),  np.cos(STRIKE_R),  0.0])
    e2 = np.array([ np.cos(STRIKE_R) * np.cos(DIP_R),
                   -np.sin(STRIKE_R) * np.cos(DIP_R),
                   -np.sin(DIP_R)])

    hx, hy, hz = ANCHOR_X, ANCHOR_Y, ANCHOR_Z

    dx, dy, dz = sx - hx, sy - hy, 0.0 - hz
    pa = dx*e1[0] + dy*e1[1] + dz*e1[2]
    pd = dx*e2[0] + dy*e2[1] + dz*e2[2]

    # Along-strike: ±L/2; down-dip: 0 (top) to +W (bottom)
    pa_c = np.clip(pa, -FAULT_L / 2, FAULT_L / 2)
    pd_c = np.clip(pd, 0.0, FAULT_W)

    cpx = hx + pa_c*e1[0] + pd_c*e2[0]
    cpy = hy + pa_c*e1[1] + pd_c*e2[1]
    cpz = hz + pa_c*e1[2] + pd_c*e2[2]
    return float(np.sqrt((sx - cpx)**2 + (sy - cpy)**2 + (0 - cpz)**2))


def rjb_finite(site_lat, site_lon):
    """
    Rjb = distance to surface projection of the fault plane.

    FIX 1 (original): clip bounds corrected — footprint runs 0 → +hw.
    FIX 11: surface footprint anchored at (ANCHOR_X, ANCHOR_Y), not epicentre.
    """
    sx, sy = _local_cart(site_lat, site_lon)

    e1h = np.array([np.sin(STRIKE_R), np.cos(STRIKE_R)])
    e2h = np.array([np.cos(STRIKE_R) * np.cos(DIP_R), -np.sin(STRIKE_R) * np.cos(DIP_R)])

    hw = FAULT_W * np.cos(DIP_R)   # horizontal width of fault footprint

    # Site position relative to anchor (top-centre of surface footprint)
    dx, dy = sx - ANCHOR_X, sy - ANCHOR_Y

    pa  = dx*e1h[0] + dy*e1h[1]
    pd  = dx*e2h[0] + dy*e2h[1]

    pa_c = np.clip(pa, -FAULT_L / 2, FAULT_L / 2)
    pd_c = np.clip(pd, 0.0, hw)   # FIX 1: 0 → +hw

    cpx = ANCHOR_X + pa_c*e1h[0] + pd_c*e2h[0]
    cpy = ANCHOR_Y + pa_c*e1h[1] + pd_c*e2h[1]
    return float(np.sqrt((sx - cpx)**2 + (sy - cpy)**2))


def rx_finite(site_lat, site_lon):
    """
    Rx = perpendicular-to-strike distance (positive = hanging wall).

    FIX 11: measured from the fault surface trace, which passes through
    (ANCHOR_X, ANCHOR_Y), not through the epicentre origin.
    """
    sx, sy = _local_cart(site_lat, site_lon)
    dx, dy = sx - ANCHOR_X, sy - ANCHOR_Y
    perp = np.array([np.cos(STRIKE_R), -np.sin(STRIKE_R)])
    return float(dx*perp[0] + dy*perp[1])


# ══════════════════════════════════════════════════════════════════════════════
# HANGING-WALL TERM  (FIX 4)
# ══════════════════════════════════════════════════════════════════════════════
def hw_term(Rx, Rrup, M, Ztor, dip_deg, T="PGA"):
    """
    Simplified hanging-wall amplification consistent with NGA-West2 structure.
    Returns ln-units additive term > 0 for hanging-wall sites (Rx > 0).
    """
    a_hw = 0.90 if T == "PGA" else 1.10

    T1 = np.tanh(Rx / 10.0) if Rx > 0 else 0.0
    T2 = max(0.0, 1.0 - Rrup / 50.0) if Rrup > 0 else 1.0
    T3 = max(0.0, 1.0 - Ztor / 20.0)
    T4 = max(0.0, min(1.0, (M - 5.5) / 1.0))
    T5 = (90.0 - dip_deg) / 45.0

    return a_hw * T1 * T2 * T3 * T4 * T5


# ══════════════════════════════════════════════════════════════════════════════
# GMPE IMPLEMENTATIONS  (functional forms consistent with NGA-West2 structure)
# ══════════════════════════════════════════════════════════════════════════════
class NGA4:

    @staticmethod
    def ask14(M, Rrup, Rx, Vs30, T="PGA", FRV=1):
        if T == "PGA":
            a1,a2,a3 =  0.587, 0.512,-0.100
            a4,a5,a6 = -2.118, 0.170,-0.003
            b1,b2,c  = -0.600,-0.300, 0.100
            h        =  4.5
        else:
            a1,a2,a3 = -0.740, 0.600,-0.100
            a4,a5,a6 = -2.118, 0.170,-0.0015
            b1,b2,c  = -0.850,-0.350, 0.060
            h        =  5.0
        a7 = 0.100; Mref, Vref = 6.0, 760.0
        f_mag  = a1 + a2*(M-Mref) + a3*(M-Mref)**2
        f_dist = (a4 + a5*M)*np.log(np.sqrt(Rrup**2 + h**2)) + a6*Rrup
        f_flt  = a7 * FRV
        Yref   = np.exp(f_mag + f_dist + f_flt)
        f_site = b1*np.log(Vs30/Vref) + b2*np.log((Yref + c)/c)
        f_hw   = hw_term(Rx, Rrup, M, ZTOR, DIP, T)
        return f_mag + f_dist + f_flt + f_site + f_hw

    @staticmethod
    def bssa14(M, Rjb, Rx, Vs30, T="PGA"):
        """FIX 2: array-safe Vs30 cap via np.minimum/np.where."""
        if T == "PGA":
            e0,e1,e2 = -0.500, 0.900,-0.120
            c1,c2,c3 = -1.500, 0.100,-0.002
            c4,c5    = -0.500,-0.800
            Mh,h     =  6.0,  4.5
        else:
            e0,e1,e2 = -1.800, 1.050,-0.150
            c1,c2,c3 = -1.800, 0.150,-0.0012
            c4,c5    = -0.800,-0.600
            Mh,h     =  6.0,  5.0
        Vc, Vref = 1000.0, 760.0
        f_E = e0 + e1*(M-Mh) + e2*(M-Mh)**2
        f_P = (c1 + c2*(M-Mh))*np.log(Rjb + h) + c3*Rjb
        Vs30_cap = np.minimum(float(Vs30), Vc)
        f_S  = c4*np.log(Vs30_cap / Vref)
        f_S += np.where(Vs30 < Vc, c5*np.log(float(Vs30)/Vc), 0.0)
        # Rrup approximation for HW term (best available without Rrup in BSSA14)
        rrup_approx = np.sqrt(Rjb**2 + ZTOR**2)
        f_hw = hw_term(Rx, rrup_approx, M, ZTOR, DIP, T)
        return f_E + f_P + float(f_S) + f_hw

    @staticmethod
    def cb14(M, Rrup, Rx, Vs30, T="PGA"):
        if T == "PGA":
            c0,c1,c2 = -1.715, 0.500,-0.050
            c3,c4,c5 = -2.000, 0.200,-0.003
            c8,c9,k1 = -0.600,-0.400, 0.100
            h        =  4.0
        else:
            c0,c1,c2 = -4.416, 1.000,-0.100
            c3,c4,c5 = -2.100, 0.220,-0.0015
            c8,c9,k1 = -0.900,-0.500, 0.500
            h        =  5.0
        Vref = 760.0
        f_mag  = c0 + c1*M + c2*M**2
        f_dis  = (c3 + c4*M)*np.log(np.sqrt(Rrup**2 + h**2)) + c5*Rrup
        Yref   = np.exp(f_mag + f_dis)
        f_site = c8*np.log(Vs30/Vref) + c9*np.log((Yref + k1)/k1)
        f_hw   = hw_term(Rx, Rrup, M, ZTOR, DIP, T)
        return f_mag + f_dis + f_site + f_hw

    @staticmethod
    def cy14(M, Rrup, Rx, Vs30, T="PGA", FRV=1):
        if T == "PGA":
            c1,c2,c3  = -1.200, 0.800,-0.100
            c4,c5     = -1.700, 0.150
            c6,c7,c8  =  5.000, 0.300,-0.002
            c11,c12   = -0.600,-0.300
            cnl       =  0.100
        else:
            c1,c2,c3  = -2.800, 1.000,-0.120
            c4,c5     = -2.000, 0.180
            c6,c7,c8  =  5.000, 0.300,-0.0015
            c11,c12   = -0.800,-0.400
            cnl       =  0.100
        c_frv = 0.28
        Vref  = 760.0
        f_main = (c1 + c_frv*FRV + c2*(M-6) + c3*(M-6)**2 +
                  (c4 + c5*M)*np.log(Rrup + c6*np.exp(c7*M)) + c8*Rrup)
        Yref   = np.exp(f_main)
        f_site = c11*np.log(Vs30/Vref) + c12*np.log((Yref + cnl)/cnl)
        f_hw   = hw_term(Rx, Rrup, M, ZTOR, DIP, T)
        return f_main + f_site + f_hw


_g = NGA4()

def raw_lnY(model, T, M, Rrup, Rjb, Rx, Vs30):
    fn = {
        "ASK14":  lambda: _g.ask14( M, Rrup, Rx, Vs30, T, FRV_EVENT),
        "BSSA14": lambda: _g.bssa14(M, Rjb,  Rx, Vs30, T),
        "CB14":   lambda: _g.cb14(  M, Rrup, Rx, Vs30, T),
        "CY14":   lambda: _g.cy14(  M, Rrup, Rx, Vs30, T, FRV_EVENT),
    }[model]
    return fn()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & FINITE-FAULT DISTANCES
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_excel(DATA_CSV)
print("Computing finite-fault distances…")
df["Rrup"] = [rrup_finite(r.latitude, r.longitude) for _, r in df.iterrows()]
df["Rjb"]  = [rjb_finite( r.latitude, r.longitude) for _, r in df.iterrows()]
df["Rx"]   = [rx_finite(  r.latitude, r.longitude) for _, r in df.iterrows()]
print(f"  Rrup: {df.Rrup.min():.1f} – {df.Rrup.max():.1f} km  (N={len(df)} bridges)")

for model in SM_MODELS:
    for T, imt in [("PGA","PGA"), ("Sa1s","SA1")]:
        df[f"{model}_{imt}_raw"] = [
            np.exp(raw_lnY(model, T, MW, r.Rrup, r.Rjb, r.Rx, VS30_REF))
            for _, r in df.iterrows()]

cal = {}
for model in SM_MODELS:
    for imt, obs_col in [("PGA","pga_shakemap"), ("SA1","sa1s_shakemap")]:
        delta = float(np.mean(np.log(df[obs_col] / df[f"{model}_{imt}_raw"])))
        cal[(model, imt)] = delta
        df[f"{model}_{imt}_cal"] = df[f"{model}_{imt}_raw"] * np.exp(delta)

for imt in ["PGA", "SA1"]:
    for suffix in ["raw", "cal"]:
        ln_mat = np.column_stack(
            [np.log(df[f"{m}_{imt}_{suffix}"].values) for m in SM_MODELS])
        df[f"Ens_{imt}_{suffix}"] = np.exp(np.mean(ln_mat, axis=1))


def stats(obs, pred):
    """
    Returns (bias, sigma, RMSE) in natural-log units.
    FIX 12: sigma uses ddof=1 (sample standard deviation).
    """
    r = np.log(np.asarray(obs) / np.asarray(pred))
    return float(np.mean(r)), float(np.std(r, ddof=1)), float(np.sqrt(np.mean(r**2)))


# ══════════════════════════════════════════════════════════════════════════════
# ATTENUATION CURVES
# ══════════════════════════════════════════════════════════════════════════════
R_VEC = np.logspace(np.log10(1), np.log10(200), 400)

def model_curve(model, T, Vs30=VS30_REF, calibrated=False, R_arr=R_VEC):
    """FIX 5: Rjb = sqrt(max(Rrup²−Ztor², 0)); Rx=0 for generic curve."""
    vals = []
    for r in R_arr:
        rjb = float(np.sqrt(max(r**2 - ZTOR**2, 0.0)))
        vals.append(np.exp(raw_lnY(model, T, MW, r, rjb, 0.0, Vs30)))
    vals = np.array(vals)
    if calibrated:
        imt = "PGA" if T == "PGA" else "SA1"
        vals = vals * np.exp(cal[(model, imt)])
    return vals


def ens_curve(T, Vs30=VS30_REF, calibrated=False, R_arr=R_VEC):
    curves = [model_curve(m, T, Vs30, calibrated, R_arr) for m in SM_MODELS]
    return np.exp(np.mean(np.log(curves), axis=0))

# ============================================================
# VS30 SITE CLASS ATTENUATION PLOTS
# ============================================================

SITE_CLASSES = {
    "A": (1500, 2000),
    "B": (760, 1500),
    "C": (360, 760),
    "D": (180, 360),
    "E": (100, 180)
}

SIGMA_PGA = np.mean([SIG_PGA[m] for m in SM_MODELS])
SIGMA_SA1 = np.mean([SIG_SA1[m] for m in SM_MODELS])


def vs30_samples(vmin, vmax, n=6):
    """Generate Vs30 samples inside site class interval"""
    return np.linspace(vmin, vmax, n)


def plot_vs30_class(site_class, vmin, vmax, T="PGA"):

    Vs_values = vs30_samples(vmin, vmax)

    fig, ax = plt.subplots(figsize=(9,7))

    for vs in Vs_values:

        Y = ens_curve(T, Vs30=vs)

        ax.loglog(
            R_VEC,
            Y,
            lw=1.6,
            label=f"Vs30={int(vs)} m/s"
        )

        # ±1 sigma
        sigma = SIGMA_PGA if T=="PGA" else SIGMA_SA1

        ax.loglog(
            R_VEC,
            Y*np.exp(sigma),
            linestyle="--",
            lw=0.8,
            alpha=0.6
        )

        ax.loglog(
            R_VEC,
            Y*np.exp(-sigma),
            linestyle="--",
            lw=0.8,
            alpha=0.6
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1,200)

    if T=="PGA":
        ax.set_ylim(0.005,2)
        ylabel="PGA (g)"
    else:
        ax.set_ylim(0.005,3)
        ylabel="Sa(1s) (g)"

    ax.set_xlabel("Rrup (km)")
    ax.set_ylabel(ylabel)

    ax.grid(True, which="both", linestyle=":")

    ax.set_title(
        f"Northridge GMPE Attenuation — Site Class {site_class}\n"
        f"Vs30 Range = {vmin}–{vmax} m/s"
    )

    ax.legend(fontsize=8)

    fname = OUT + f"vs30_siteclass_{site_class}_{T}.png"

    plt.savefig(fname, dpi=200, bbox_inches="tight")

    plt.close()

    print("Saved:", fname)


# ============================================================
# GENERATE ALL SITE CLASS PLOTS
# ============================================================

for sc, (vmin, vmax) in SITE_CLASSES.items():

    plot_vs30_class(sc, vmin, vmax, "PGA")

    plot_vs30_class(sc, vmin, vmax, "Sa1s")
# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({"font.size":10, "axes.labelsize":10,
                     "xtick.labelsize":9, "ytick.labelsize":9})

def panel_stats(ax, b, s, r, x=0.97, y=0.97):
    ax.text(x, y, f"Bias={b:+.2f}\nσ={s:.2f}\nRMSE={r:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(fc="white", ec="0.7", boxstyle="round,pad=0.3", alpha=0.9))


# ── Fig 01: PGA Attenuation ───────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(TITLE_HEAD+"Fig 01 — PGA Attenuation (USGS ShakeMap Analysis Page Style)\n"+PARAM_STR,
                 fontsize=9, fontweight="bold")
    ax.scatter(df.Rrup, df.pga_shakemap, s=9, c="gold", edgecolors="k",
               lw=0.3, zorder=4, label="ShakeMap data at bridge sites", alpha=0.75)
    for model in SM_MODELS:
        ax.loglog(R_VEC, model_curve(model,"PGA"), color=SM_COLORS[model], lw=1.4, alpha=0.65, label=model)
    Y_unb = ens_curve("PGA", calibrated=False)
    sig   = np.mean([SIG_PGA[m] for m in SM_MODELS])
    ax.loglog(R_VEC, Y_unb,             "r-",  lw=2.5, zorder=5, label="Unbiased Ensemble (Vs30=760)")
    ax.loglog(R_VEC, Y_unb*np.exp( 3*sig),"r--", lw=1.2, alpha=0.7, label="+3σ")
    ax.loglog(R_VEC, Y_unb*np.exp(-3*sig),"r--", lw=1.2, alpha=0.7, label="−3σ")
    Y_cal = ens_curve("PGA", calibrated=True)
    ax.loglog(R_VEC, Y_cal,             "g-",  lw=2.5, zorder=5, label="Bias-corrected Ensemble")
    ax.loglog(R_VEC, Y_cal*np.exp( 3*sig),"g--", lw=1.2, alpha=0.7)
    ax.loglog(R_VEC, Y_cal*np.exp(-3*sig),"g--", lw=1.2, alpha=0.7)
    b, s, r = stats(df.pga_shakemap, df.Ens_PGA_raw)
    ax.set_xlim(1,200); ax.set_ylim(0.005,2.5)
    ax.set_xlabel("R$_{rup}$ – Closest Distance to Fault Plane (km)", fontsize=10)
    ax.set_ylabel("PGA (g)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.text(0.97,0.97,f"Unbiased ensemble bias={b:+.2f}",transform=ax.transAxes,
            ha="right",va="top",fontsize=8,color="red",
            bbox=dict(fc="white",ec="0.7",boxstyle="round,pad=0.3"))
    plt.savefig(OUT+"fig01_usgs_pga_attenuation.png", dpi=150, bbox_inches="tight")
    print("Saved Fig 01")
finally:
    plt.close()


# ── Fig 02: Sa(1s) Attenuation ────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(TITLE_HEAD+"Fig 02 — Sa(1 s) Attenuation (USGS ShakeMap Analysis Page Style)\n"+PARAM_STR,
                 fontsize=9, fontweight="bold")
    ax.scatter(df.Rrup, df.sa1s_shakemap, s=9, c="gold", edgecolors="k",
               lw=0.3, zorder=4, label="ShakeMap data at bridge sites", alpha=0.75)
    for model in SM_MODELS:
        ax.loglog(R_VEC, model_curve(model,"Sa1s"), color=SM_COLORS[model], lw=1.4, alpha=0.65, label=model)
    Y_unb = ens_curve("Sa1s", calibrated=False)
    sig   = np.mean([SIG_SA1[m] for m in SM_MODELS])
    ax.loglog(R_VEC, Y_unb,             "r-",  lw=2.5, zorder=5, label="Unbiased Ensemble (Vs30=760)")
    ax.loglog(R_VEC, Y_unb*np.exp( 3*sig),"r--", lw=1.2, alpha=0.7)
    ax.loglog(R_VEC, Y_unb*np.exp(-3*sig),"r--", lw=1.2, alpha=0.7)
    Y_cal = ens_curve("Sa1s", calibrated=True)
    ax.loglog(R_VEC, Y_cal,             "g-",  lw=2.5, zorder=5, label="Bias-corrected Ensemble")
    ax.loglog(R_VEC, Y_cal*np.exp( 3*sig),"g--", lw=1.2, alpha=0.7)
    ax.loglog(R_VEC, Y_cal*np.exp(-3*sig),"g--", lw=1.2, alpha=0.7)
    b, s, r = stats(df.sa1s_shakemap, df.Ens_SA1_raw)
    ax.set_xlim(1,200); ax.set_ylim(0.005,3.5)
    ax.set_xlabel("R$_{rup}$ – Closest Distance to Fault Plane (km)", fontsize=10)
    ax.set_ylabel("Sa(1 s) (g)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.text(0.97,0.97,f"Unbiased ensemble bias={b:+.2f}",transform=ax.transAxes,
            ha="right",va="top",fontsize=8,color="red",
            bbox=dict(fc="white",ec="0.7",boxstyle="round,pad=0.3"))
    plt.savefig(OUT+"fig02_usgs_sa1_attenuation.png", dpi=150, bbox_inches="tight")
    print("Saved Fig 02")
finally:
    plt.close()


# ── Fig 03–06: Predicted vs Observed (4 panels each) ─────────────────────────
for fig_num, imt, obs_col, cal_flag, title_tag, lims in [
        (3, "PGA", "pga_shakemap",  False, "PGA Unbiased (Vs30=760)",       (0.005,2.0)),
        (4, "SA1", "sa1s_shakemap", False, "Sa(1 s) Unbiased (Vs30=760)",   (0.005,3.0)),
        (5, "PGA", "pga_shakemap",  True,  "PGA Bias-corrected",            (0.005,2.0)),
        (6, "SA1", "sa1s_shakemap", True,  "Sa(1 s) Bias-corrected",        (0.005,3.0))]:
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        imt_label = "PGA" if imt=="PGA" else "Sa(1 s)"
        xlab = f"GMPE Predicted {imt_label}{' – Calibrated' if cal_flag else ''} (g)"
        fig.suptitle(TITLE_HEAD+f"Fig {fig_num:02d} — {title_tag} Predicted vs Observed\n"+PARAM_STR,
                     fontsize=9, fontweight="bold")
        for ax, model in zip(axes.flat, SM_MODELS):
            obs  = df[obs_col].values
            sfx  = "cal" if cal_flag else "raw"
            pred = df[f"{model}_{imt}_{sfx}"].values
            b, s, r = stats(obs, pred)
            ax.scatter(pred, obs, s=5, alpha=0.3, c=SM_COLORS[model])
            d = np.array(lims)
            ax.plot(d, d,          "k-",  lw=1.8, label="1:1")
            ax.plot(d, d*np.exp(s),"k--", lw=1,   alpha=0.6, label="±1σ")
            ax.plot(d, d/np.exp(s),"k--", lw=1,   alpha=0.6)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(*lims);   ax.set_ylim(*lims)
            ax.set_xlabel(xlab, fontsize=9)
            ax.set_ylabel(f"ShakeMap {imt_label} (g)", fontsize=9)
            ax.set_title(model, fontsize=12, fontweight="bold", color=SM_COLORS[model])
            ax.grid(True, which="both", ls=":", alpha=0.4)
            ax.legend(fontsize=8)
            panel_stats(ax, b, s, r)
        fname = f"fig{fig_num:02d}_{'pga' if imt=='PGA' else 'sa1'}_{'biascorrected' if cal_flag else 'unbiased'}_pred_obs.png"
        plt.savefig(OUT+fname, dpi=150, bbox_inches="tight")
        print(f"Saved Fig {fig_num:02d}")
    finally:
        plt.close()


# ── Fig 07–08: Residuals vs Rrup ─────────────────────────────────────────────
for fig_num, imt, obs_col, ylim in [
        (7, "PGA", "pga_shakemap",  (-4,4)),
        (8, "SA1", "sa1s_shakemap", (-5,5))]:
    imt_label = "PGA" if imt=="PGA" else "Sa(1 s)"
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(TITLE_HEAD+f"Fig {fig_num:02d} — {imt_label} ln-Residuals vs R$_{{rup}}$  (Unbiased, Vs30=760)\n"+PARAM_STR,
                     fontsize=9, fontweight="bold")
        for ax, model in zip(axes.flat, SM_MODELS):
            obs   = df[obs_col].values
            pred  = df[f"{model}_{imt}_raw"].values
            resid = np.log(obs / pred)
            b, s, r = stats(obs, pred)
            ax.scatter(df.Rrup, resid, s=4, alpha=0.3, c=SM_COLORS[model])
            ax.axhline(0,  color="k",    lw=1.5)
            ax.axhline(b,  color="red",  lw=1.5, ls="--", label=f"Bias={b:+.2f}")
            ax.axhline(+s, color="gray", lw=0.9, ls=":",  label=f"±σ={s:.2f}")
            ax.axhline(-s, color="gray", lw=0.9, ls=":")
            ax.set_xscale("log"); ax.set_xlim(1,200); ax.set_ylim(*ylim)
            ax.set_xlabel("R$_{rup}$ (km)",           fontsize=9)
            ax.set_ylabel("ln(ShakeMap / Predicted)", fontsize=9)
            ax.set_title(model, fontsize=12, fontweight="bold", color=SM_COLORS[model])
            ax.grid(True, which="both", ls=":", alpha=0.4)
            ax.legend(fontsize=8, loc="upper right")
        fname = f"fig{fig_num:02d}_{'pga' if imt=='PGA' else 'sa1'}_residuals.png"
        plt.savefig(OUT+fname, dpi=150, bbox_inches="tight")
        print(f"Saved Fig {fig_num:02d}")
    finally:
        plt.close()


# ── Fig 09: Statistics bar chart ─────────────────────────────────────────────
rows = []
for model in SM_MODELS:
    for imt, obs_col in [("PGA","pga_shakemap"), ("SA1","sa1s_shakemap")]:
        obs = df[obs_col].values
        b_r,s_r,r_r = stats(obs, df[f"{model}_{imt}_raw"].values)
        b_c,s_c,r_c = stats(obs, df[f"{model}_{imt}_cal"].values)
        rows.append({"Model":model,"IMT":imt,
                     "Bias_raw":b_r,"Sigma_raw":s_r,"RMSE_raw":r_r,
                     "Bias_cal":b_c,"Sigma_cal":s_c,"RMSE_cal":r_c,
                     "Delta_corr":cal[(model,imt)]})

stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT+"usgs_matched_stats.csv", index=False, float_format="%.4f")

try:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(TITLE_HEAD+"Fig 09 — Performance Statistics: Unbiased vs Bias-corrected\n"
                 "Top: PGA  |  Bottom: Sa(1 s)\n"+PARAM_STR, fontsize=9, fontweight="bold")
    x, w = np.arange(len(SM_MODELS)), 0.35
    for row_i, imt in enumerate(["PGA","SA1"]):
        sub = stats_df[stats_df.IMT==imt]
        for col_i,(raw_c,cal_c,ylabel) in enumerate([
                ("Bias_raw","Bias_cal","Bias (ln units)"),
                ("Sigma_raw","Sigma_cal","σ (ln units)"),
                ("RMSE_raw","RMSE_cal","RMSE (ln units)")]):
            ax = axes[row_i,col_i]
            rv = [sub[sub.Model==m][raw_c].iloc[0] for m in SM_MODELS]
            cv = [sub[sub.Model==m][cal_c].iloc[0] for m in SM_MODELS]
            b1 = ax.bar(x-w/2, rv, w, label="Unbiased",   color="#d7191c", alpha=0.82)
            b2 = ax.bar(x+w/2, cv, w, label="Calibrated", color="#2c7bb6", alpha=0.82)
            ax.set_xticks(x); ax.set_xticklabels(SM_MODELS, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{imt} — {ylabel}", fontsize=9, fontweight="bold")
            ax.axhline(0, color="k", lw=0.8)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", ls=":", alpha=0.5)
            for bar in list(b1)+list(b2):
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2,
                        h+(0.02 if h>=0 else -0.08),
                        f"{h:.2f}", ha="center", fontsize=7)
    plt.savefig(OUT+"fig09_statistics_usgs.png", dpi=150, bbox_inches="tight")
    print("Saved Fig 09")
finally:
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("  USGS SHAKEMAP v4 MATCHED PARAMETERS — 1994 NORTHRIDGE Mw 6.7")
print("="*72)
print(f"  Fault geometry : Strike={STRIKE}°  Dip={DIP}°  Rake={RAKE}°")
print(f"  Fault dims     : L={FAULT_L} km  W={FAULT_W} km")
print(f"  Hypocentre     : {EPI_LAT}°N  {EPI_LON}°W  depth={DEPTH} km")
print(f"  Ztor           : {ZTOR:.1f} km  (Wald et al. 1996 published value)")
print(f"  Fault anchor   : E={ANCHOR_X:.2f} km  N={ANCHOR_Y:.2f} km  "
      f"(up-dip offset = {up_dip_horiz_dist:.2f} km from epicentre)")
print(f"  Distance metric: Rrup to finite fault  [{df.Rrup.min():.1f}–{df.Rrup.max():.1f} km]")
print(f"  Vs30 (curves)  : {VS30_REF} m/s")
print(f"  GMPE set       : ASK14+BSSA14+CB14+CY14 (0.25 each)")
print("-"*72)
print(f"  {'Model':<8}{'IMT':<5}{'Δ_cal':>8}{'Bias_raw':>10}{'σ_raw':>8}"
      f"{'RMSE_raw[ln]':>14}{'Bias_cal':>10}{'σ_cal':>8}")
print("-"*72)
for _, row in stats_df.iterrows():
    print(f"  {row.Model:<8}{row.IMT:<5}{row.Delta_corr:>+8.3f}"
          f"{row.Bias_raw:>+10.3f}{row.Sigma_raw:>8.3f}"
          f"{row.RMSE_raw:>14.3f}{row.Bias_cal:>+10.4f}{row.Sigma_cal:>8.3f}")
print("="*72)
print("  σ uses sample std (ddof=1); RMSE in natural-log units throughout.")
print("  Fault anchor placed at top-centre, offset up-dip from epicentre.")


# ══════════════════════════════════════════════════════════════════════════════
# REGIONAL GRID  —  1 mile × 1 mile boxes, per-GMPE PGA & Sa(1s)
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("  BUILDING 1-MILE GRID  —  per-GMPE PGA & Sa(1s)")
print("="*72)

KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(EPI_LAT))
MI_TO_KM       = 1.60934

lat_min_br = df["latitude"].min();  lat_max_br = df["latitude"].max()
lon_min_br = df["longitude"].min(); lon_max_br = df["longitude"].max()

dlat = MI_TO_KM / KM_PER_DEG_LAT
dlon = MI_TO_KM / KM_PER_DEG_LON

lat_edges   = np.arange(lat_min_br, lat_max_br + dlat, dlat)
lon_edges   = np.arange(lon_min_br, lon_max_br + dlon, dlon)
lat_centres = lat_edges[:-1] + dlat / 2.0
lon_centres = lon_edges[:-1] + dlon / 2.0

grid_lons, grid_lats = np.meshgrid(lon_centres, lat_centres)
grid_lats_f = grid_lats.ravel()
grid_lons_f = grid_lons.ravel()
N_cells = len(grid_lats_f)

print(f"  Grid size: {len(lat_centres)} rows × {len(lon_centres)} cols = {N_cells} cells")

print("  Computing finite-fault distances for grid cells…")
grid_Rrup = np.array([rrup_finite(la,lo) for la,lo in zip(grid_lats_f,grid_lons_f)])
grid_Rjb  = np.array([rjb_finite( la,lo) for la,lo in zip(grid_lats_f,grid_lons_f)])
grid_Rx   = np.array([rx_finite(  la,lo) for la,lo in zip(grid_lats_f,grid_lons_f)])

grid_records = []
for model in SM_MODELS:
    print(f"  Running {model} on grid…")
    pga_vals = np.array([np.exp(raw_lnY(model,"PGA", MW,rr,rj,rx,VS30_REF))
                         for rr,rj,rx in zip(grid_Rrup,grid_Rjb,grid_Rx)])
    sa1_vals = np.array([np.exp(raw_lnY(model,"Sa1s",MW,rr,rj,rx,VS30_REF))
                         for rr,rj,rx in zip(grid_Rrup,grid_Rjb,grid_Rx)])
    for i in range(N_cells):
        grid_records.append({"GMPE":model,"latitude":grid_lats_f[i],
                              "longitude":grid_lons_f[i],
                              "PGA_g":pga_vals[i],"Sa1s_g":sa1_vals[i]})

grid_df = pd.DataFrame(grid_records)

grid_xlsx = os.path.join(OUT,"grid_gmpe_pga_sa1s.xlsx")
with pd.ExcelWriter(grid_xlsx, engine="openpyxl") as writer:
    for model in SM_MODELS:
        sub = grid_df[grid_df["GMPE"]==model][
            ["latitude","longitude","PGA_g","Sa1s_g"]].reset_index(drop=True)
        sub.to_excel(writer, sheet_name=model, index=False)
print(f"\n  Grid XLSX → {grid_xlsx}")

grid_csv = os.path.join(OUT,"grid_gmpe_pga_sa1s_combined.csv")
grid_df.to_csv(grid_csv, index=False)
print(f"  Grid CSV  → {grid_csv}")

for imt,col,clabel,cmap in [
        ("PGA","PGA_g","PGA (g)","hot_r"),
        ("Sa1s","Sa1s_g","Sa(1s) (g)","YlOrRd")]:
    fig, axes = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle(
        f"1994 Northridge Mw 6.7  —  {clabel}  |  1-mile grid  |  Vs30=760 m/s\n"
        f"Strike={STRIKE:.0f}°  Dip={DIP:.0f}°  Rake={RAKE:.0f}°  "
        f"Ztor={ZTOR:.1f} km  Rrup to finite fault",
        fontsize=10, y=0.99)
    vmin = grid_df[[col]].min().iloc[0]
    vmax = grid_df[[col]].max().iloc[0]
    for ax, model in zip(axes.ravel(), SM_MODELS):
        sub   = grid_df[grid_df["GMPE"]==model]
        pivot = sub.pivot_table(index="latitude",columns="longitude",values=col)
        pcm   = ax.pcolormesh(pivot.columns.values, pivot.index.values, pivot.values,
                              cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.scatter(df["longitude"],df["latitude"],
                   s=6,c="black",alpha=0.35,linewidths=0,zorder=3,label="Bridges")
        ax.plot(EPI_LON,EPI_LAT,"c*",ms=12,zorder=5,label="Epicentre")
        ax.set_title(model,fontsize=11,fontweight="bold",color=SM_COLORS[model])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_xlim(lon_min_br-dlon,lon_max_br+dlon)
        ax.set_ylim(lat_min_br-dlat,lat_max_br+dlat)
        cbar = fig.colorbar(pcm,ax=ax,fraction=0.046,pad=0.04)
        cbar.set_label(clabel,fontsize=8)
        if ax==axes[0,0]: ax.legend(fontsize=7,loc="upper right")
    plt.tight_layout(rect=[0,0,1,0.96])
    fname = os.path.join(OUT,f"grid_map_{imt}.png")
    try:
        fig.savefig(fname,dpi=150,bbox_inches="tight")
        print(f"  Saved: {fname}")
    finally:
        plt.close(fig)

print("\n  Grid section complete. Files written to:", OUT)


# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE Vs30 ANALYSIS — Grid maps and statistics for all Vs30 values
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*72)
print("  COMPREHENSIVE Vs30 ANALYSIS")
print("="*72)

# Generate grid data for all Vs30 values
all_grid_data_vs30 = {}

for vs30_name, vs30_val in VS30_VALUES.items():
    print(f"\n  Processing Vs30 = {vs30_val} m/s ({vs30_name})...")

    grid_records_vs30 = []
    for model in SM_MODELS:
        print(f"    Running {model}...")
        pga_vals = np.array([np.exp(raw_lnY(model, "PGA", MW, rr, rj, rx, vs30_val))
                             for rr, rj, rx in zip(grid_Rrup, grid_Rjb, grid_Rx)])
        sa1_vals = np.array([np.exp(raw_lnY(model, "Sa1s", MW, rr, rj, rx, vs30_val))
                             for rr, rj, rx in zip(grid_Rrup, grid_Rjb, grid_Rx)])

        for i in range(N_cells):
            grid_records_vs30.append({
                "GMPE": model,
                "latitude": grid_lats_f[i],
                "longitude": grid_lons_f[i],
                "Vs30": vs30_val,
                "Vs30_name": vs30_name,
                "PGA_g": pga_vals[i],
                "Sa1s_g": sa1_vals[i]
            })

    all_grid_data_vs30[vs30_name] = pd.DataFrame(grid_records_vs30)

# ──────────────────────────────────────────────────────────────────────────────
# Vs30 PLOT 1: Grid Maps for Each Vs30 Value (PGA)
# ──────────────────────────────────────────────────────────────────────────────
print("\n  Creating PGA grid maps for each Vs30 value...")

for vs30_name, vs30_val in VS30_VALUES.items():
    grid_df_vs30 = all_grid_data_vs30[vs30_name]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"1994 Northridge Mw 6.7  —  PGA (g)  |  1-mile grid  |  Vs30={vs30_val} m/s ({vs30_name})\n"
        f"Strike={STRIKE:.0f}°  Dip={DIP:.0f}°  Rake={RAKE:.0f}°  "
        f"Ztor={ZTOR:.1f} km  Rrup to finite fault",
        fontsize=10, y=0.99)

    vmin = grid_df_vs30["PGA_g"].min()
    vmax = grid_df_vs30["PGA_g"].max()

    for ax, model in zip(axes.ravel(), SM_MODELS):
        sub = grid_df_vs30[grid_df_vs30["GMPE"] == model]
        pivot = sub.pivot_table(index="latitude", columns="longitude", values="PGA_g")
        pcm = ax.pcolormesh(pivot.columns.values, pivot.index.values, pivot.values,
                            cmap="hot_r", vmin=vmin, vmax=vmax, shading="auto")
        ax.scatter(df["longitude"], df["latitude"],
                   s=6, c="black", alpha=0.35, linewidths=0, zorder=3, label="Bridges")
        ax.plot(EPI_LON, EPI_LAT, "c*", ms=12, zorder=5, label="Epicentre")
        ax.set_title(model, fontsize=11, fontweight="bold", color=SM_COLORS[model])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lon_min_br - dlon, lon_max_br + dlon)
        ax.set_ylim(lat_min_br - dlat, lat_max_br + dlat)
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("PGA (g)", fontsize=8)
        if ax == axes[0, 0]:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUT, f"vs30_grid_map_PGA_{vs30_val}.png")
    try:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"    Saved: {fname}")
    finally:
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Vs30 PLOT 2: Grid Maps for Each Vs30 Value (Sa(1s))
# ──────────────────────────────────────────────────────────────────────────────
print("\n  Creating Sa(1s) grid maps for each Vs30 value...")

for vs30_name, vs30_val in VS30_VALUES.items():
    grid_df_vs30 = all_grid_data_vs30[vs30_name]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"1994 Northridge Mw 6.7  —  Sa(1s) (g)  |  1-mile grid  |  Vs30={vs30_val} m/s ({vs30_name})\n"
        f"Strike={STRIKE:.0f}°  Dip={DIP:.0f}°  Rake={RAKE:.0f}°  "
        f"Ztor={ZTOR:.1f} km  Rrup to finite fault",
        fontsize=10, y=0.99)

    vmin = grid_df_vs30["Sa1s_g"].min()
    vmax = grid_df_vs30["Sa1s_g"].max()

    for ax, model in zip(axes.ravel(), SM_MODELS):
        sub = grid_df_vs30[grid_df_vs30["GMPE"] == model]
        pivot = sub.pivot_table(index="latitude", columns="longitude", values="Sa1s_g")
        pcm = ax.pcolormesh(pivot.columns.values, pivot.index.values, pivot.values,
                            cmap="YlOrRd", vmin=vmin, vmax=vmax, shading="auto")
        ax.scatter(df["longitude"], df["latitude"],
                   s=6, c="black", alpha=0.35, linewidths=0, zorder=3, label="Bridges")
        ax.plot(EPI_LON, EPI_LAT, "c*", ms=12, zorder=5, label="Epicentre")
        ax.set_title(model, fontsize=11, fontweight="bold", color=SM_COLORS[model])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lon_min_br - dlon, lon_max_br + dlon)
        ax.set_ylim(lat_min_br - dlat, lat_max_br + dlat)
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Sa(1s) (g)", fontsize=8)
        if ax == axes[0, 0]:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUT, f"vs30_grid_map_SA1_{vs30_val}.png")
    try:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"    Saved: {fname}")
    finally:
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Vs30 PLOT 3: Statistics Comparison Across Vs30 Values
# ──────────────────────────────────────────────────────────────────────────────
print("\n  Creating statistics comparison plots across Vs30 values...")

# Collect statistics for each Vs30
stats_data_vs30 = []
for vs30_name, vs30_val in VS30_VALUES.items():
    grid_df_vs30 = all_grid_data_vs30[vs30_name]
    for model in SM_MODELS:
        sub = grid_df_vs30[grid_df_vs30["GMPE"] == model]
        stats_data_vs30.append({
            'Vs30': vs30_val,
            'Vs30_name': vs30_name,
            'Model': model,
            'PGA_mean': sub['PGA_g'].mean(),
            'PGA_std': sub['PGA_g'].std(),
            'PGA_max': sub['PGA_g'].max(),
            'PGA_min': sub['PGA_g'].min(),
            'Sa1s_mean': sub['Sa1s_g'].mean(),
            'Sa1s_std': sub['Sa1s_g'].std(),
            'Sa1s_max': sub['Sa1s_g'].max(),
            'Sa1s_min': sub['Sa1s_g'].min()
        })

stats_summary_vs30 = pd.DataFrame(stats_data_vs30)

# Save statistics summary
stats_summary_vs30.to_csv(os.path.join(OUT, "vs30_statistics_summary.csv"),
                          index=False, float_format="%.6f")
print(f"  Saved statistics: {os.path.join(OUT, 'vs30_statistics_summary.csv')}")

# Plot mean values across Vs30
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Ground Motion Statistics Across Vs30 Values\n"
             "1994 Northridge Mw 6.7 | NGA-West2 GMPEs",
             fontsize=13, fontweight='bold')

vs30_vals_sorted = sorted(VS30_VALUES.values())

# PGA Mean
ax = axes[0, 0]
for model in SM_MODELS:
    model_data = stats_summary_vs30[stats_summary_vs30['Model'] == model]
    model_data = model_data.sort_values('Vs30')
    ax.plot(model_data['Vs30'], model_data['PGA_mean'],
            'o-', label=model, color=SM_COLORS[model], linewidth=2, markersize=8)
ax.set_xlabel('Vs30 (m/s)', fontsize=11)
ax.set_ylabel('Mean PGA (g)', fontsize=11)
ax.set_title('Mean PGA vs Vs30', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# PGA Max
ax = axes[0, 1]
for model in SM_MODELS:
    model_data = stats_summary_vs30[stats_summary_vs30['Model'] == model]
    model_data = model_data.sort_values('Vs30')
    ax.plot(model_data['Vs30'], model_data['PGA_max'],
            'o-', label=model, color=SM_COLORS[model], linewidth=2, markersize=8)
ax.set_xlabel('Vs30 (m/s)', fontsize=11)
ax.set_ylabel('Maximum PGA (g)', fontsize=11)
ax.set_title('Maximum PGA vs Vs30', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Sa(1s) Mean
ax = axes[1, 0]
for model in SM_MODELS:
    model_data = stats_summary_vs30[stats_summary_vs30['Model'] == model]
    model_data = model_data.sort_values('Vs30')
    ax.plot(model_data['Vs30'], model_data['Sa1s_mean'],
            'o-', label=model, color=SM_COLORS[model], linewidth=2, markersize=8)
ax.set_xlabel('Vs30 (m/s)', fontsize=11)
ax.set_ylabel('Mean Sa(1s) (g)', fontsize=11)
ax.set_title('Mean Sa(1s) vs Vs30', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Sa(1s) Max
ax = axes[1, 1]
for model in SM_MODELS:
    model_data = stats_summary_vs30[stats_summary_vs30['Model'] == model]
    model_data = model_data.sort_values('Vs30')
    ax.plot(model_data['Vs30'], model_data['Sa1s_max'],
            'o-', label=model, color=SM_COLORS[model], linewidth=2, markersize=8)
ax.set_xlabel('Vs30 (m/s)', fontsize=11)
ax.set_ylabel('Maximum Sa(1s) (g)', fontsize=11)
ax.set_title('Maximum Sa(1s) vs Vs30', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
fname = os.path.join(OUT, "vs30_statistics_comparison.png")
try:
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
finally:
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Vs30 PLOT 4: Site Amplification Factors
# ──────────────────────────────────────────────────────────────────────────────
print("\n  Creating site amplification analysis plots...")

# Use Vs30=760 as reference
ref_data_vs30 = all_grid_data_vs30['Rock (Reference)']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Site Amplification Factors Relative to Vs30 = 760 m/s\n"
             "1994 Northridge Mw 6.7 | NGA-West2 GMPEs",
             fontsize=13, fontweight='bold')

# PGA Amplification
ax = axes[0, 0]
for model in SM_MODELS:
    ref_mean = ref_data_vs30[ref_data_vs30['GMPE'] == model]['PGA_g'].mean()
    amp_factors = []
    for vs30_name, vs30_val in VS30_VALUES.items():
        if vs30_val != 760:
            grid_df_vs30 = all_grid_data_vs30[vs30_name]
            model_mean = grid_df_vs30[grid_df_vs30['GMPE'] == model]['PGA_g'].mean()
            amp_factors.append((vs30_val, model_mean / ref_mean))
    if amp_factors:
        amp_factors = sorted(amp_factors)
        vs30s, amps = zip(*amp_factors)
        ax.plot(vs30s, amps, 'o-', label=model, color=SM_COLORS[model],
                linewidth=2, markersize=8)

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Reference')
ax.set_xlabel('Vs30 (m/s)', fontsize=11)
ax.set_ylabel('Amplification Factor', fontsize=11)
ax.set_title('PGA Site Amplification', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Sa(1s) Amplification
ax = axes[0, 1]
for model in SM_MODELS:
    ref_mean = ref_data_vs30[ref_data_vs30['GMPE'] == model]['Sa1s_g'].mean()
    amp_factors = []
    for vs30_name, vs30_val in VS30_VALUES.items():
        if vs30_val != 760:
            grid_df_vs30 = all_grid_data_vs30[vs30_name]
            model_mean = grid_df_vs30[grid_df_vs30['GMPE'] == model]['Sa1s_g'].mean()
            amp_factors.append((vs30_val, model_mean / ref_mean))
    if amp_factors:
        amp_factors = sorted(amp_factors)
        vs30s, amps = zip(*amp_factors)
        ax.plot(vs30s, amps, 'o-', label=model, color=SM_COLORS[model],
                linewidth=2, markersize=8)

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Reference')
ax.set_xlabel('Vs30 (m/s)', fontsize=11)
ax.set_ylabel('Amplification Factor', fontsize=11)
ax.set_title('Sa(1s) Site Amplification', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Coefficient of Variation - PGA
ax = axes[1, 0]
for model in SM_MODELS:
    cov_values = []
    for vs30_name, vs30_val in VS30_VALUES.items():
        grid_df_vs30 = all_grid_data_vs30[vs30_name]
        sub = grid_df_vs30[grid_df_vs30['GMPE'] == model]
        cov = sub['PGA_g'].std() / sub['PGA_g'].mean()
        cov_values.append((vs30_val, cov))
    cov_values = sorted(cov_values)
    vs30s, covs = zip(*cov_values)
    ax.plot(vs30s, covs, 'o-', label=model, color=SM_COLORS[model],
            linewidth=2, markersize=8)

ax.set_xlabel('Vs30 (m/s)', fontsize=11)
ax.set_ylabel('Coefficient of Variation', fontsize=11)
ax.set_title('PGA Spatial Variability', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Coefficient of Variation - Sa(1s)
ax = axes[1, 1]
for model in SM_MODELS:
    cov_values = []
    for vs30_name, vs30_val in VS30_VALUES.items():
        grid_df_vs30 = all_grid_data_vs30[vs30_name]
        sub = grid_df_vs30[grid_df_vs30['GMPE'] == model]
        cov = sub['Sa1s_g'].std() / sub['Sa1s_g'].mean()
        cov_values.append((vs30_val, cov))
    cov_values = sorted(cov_values)
    vs30s, covs = zip(*cov_values)
    ax.plot(vs30s, covs, 'o-', label=model, color=SM_COLORS[model],
            linewidth=2, markersize=8)

ax.set_xlabel('Vs30 (m/s)', fontsize=11)
ax.set_ylabel('Coefficient of Variation', fontsize=11)
ax.set_title('Sa(1s) Spatial Variability', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
fname = os.path.join(OUT, "vs30_site_amplification_analysis.png")
try:
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
finally:
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Save individual Vs30 grid data
# ──────────────────────────────────────────────────────────────────────────────
print("\n  Saving individual Vs30 grid data files...")
for vs30_name, vs30_val in VS30_VALUES.items():
    fname = os.path.join(OUT, f"vs30_grid_data_{vs30_val}.csv")
    all_grid_data_vs30[vs30_name].to_csv(fname, index=False, float_format="%.6f")
    print(f"    Saved: {fname}")

print("\n" + "="*72)
print("  Vs30 COMPREHENSIVE ANALYSIS COMPLETE")
print("="*72)
print(f"\n  Summary:")
print(f"    - Analyzed {len(VS30_VALUES)} different Vs30 values")
print(f"    - Generated {len(VS30_VALUES)*2} grid maps (PGA + Sa(1s))")
print(f"    - Created 2 comprehensive comparison plots")
print(f"    - Saved {len(VS30_VALUES)+1} data files")
print("\n" + "="*72)


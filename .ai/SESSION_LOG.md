# Session Log — 2026-03-03

## BSSA21 GMPE Integration (Complete)

Implemented full GMPE plugin architecture and ported BSSA14/21 coefficients.

### Files Created
- `src/gmpe_base.py` — GMPEModel Protocol, GMPE_REGISTRY, get_gmpe/register_gmpe, IM_TYPE_TO_PERIOD
- `src/gmpe_bssa21.py` — Full BSSA14/21 port with 108-row coefficient table (PGV + PGA + 106 SA periods)
- `src/gmpe_ba08.py` — Thin BA08 wrapper (PGA + Sa 1.0s)
- `tests/test_bssa21.py` — 22 tests: registry, coefficients, compute accuracy, multi-scenario
- `tests/test_gmpe_integration.py` — 10 tests: config validation, E2E pipeline, BA08 wrapper

### Files Modified
- `src/config.py` — Added `gmpe_model` field, GMPE-specific validation (scenario required, model in registry)
- `main.py` — Replaced NotImplementedError with full GMPE branch in `_compute_bridge_damage()`
- `config.yaml` — Added gmpe_model documentation, updated gmpe_scenario section

### Code Review Fixes Applied
- Fixed nonlinear site term: use `c[_CI["f3"]]` instead of hardcoded 0.1
- BA08 PGA: added warning when Vs30 < 700 (rock-only PGA, no site amplification)
- Config validation: narrowed ImportError catch to numpy-related only
- `_get_row()`: replaced linear scan with dict lookup for O(1) period matching
- Documented R_JB ≈ R_epi point-source approximation

### Test Results
32/32 tests pass.

### Coefficient Source
Official BSSA14 CSV from Dave Boore's website (BSSA14_Coefficients_071314_Revisedf4_071514.csv), cross-verified against OpenQuake hazardlib.

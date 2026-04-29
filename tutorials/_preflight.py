"""
Helper script: insert a preflight install cell at the top of every
tutorial notebook so a fresh-clone JupyterLab session auto-installs
any missing dependency into the kernel's own Python (avoids the
classic "I pip-installed but the notebook still says ImportError"
problem caused by a mismatched pip vs jupyter Python).
"""

import json
from pathlib import Path

NOTEBOOKS = {
    '01_config_and_data.ipynb':            ['pyyaml', 'pandas', 'numpy', 'matplotlib', 'geopandas', 'contextily'],
    '01b_config_and_exposure_demo.ipynb':  ['pandas', 'numpy', 'matplotlib', 'openpyxl'],
    '02_hazard_shakemap.ipynb':            ['pyyaml', 'pandas', 'numpy', 'matplotlib', 'geopandas', 'contextily'],
    '03_hazard_gmpe.ipynb':                ['pyyaml', 'pandas', 'numpy', 'matplotlib', 'scipy'],
    '04_fragility.ipynb':                  ['pandas', 'numpy', 'matplotlib', 'scipy'],
    '05_validation.ipynb':                 ['pandas', 'numpy', 'matplotlib', 'scipy', 'openpyxl'],
    '06_loss_and_cost.ipynb':              ['pandas', 'numpy', 'matplotlib'],
}

# Mapping from pip-install name → import name where they differ
IMPORT_NAME = {
    'pyyaml': 'yaml',
}


def make_preflight_cell(packages):
    pkgs_repr = repr(sorted(packages))
    body = (
        "# ── Preflight: auto-install any missing packages into THIS kernel ──\n"
        "# Run this cell first. If it installs anything, restart the kernel\n"
        "# (Kernel → Restart Kernel) before running the rest of the notebook.\n"
        "import sys, importlib, subprocess\n"
        f"_REQUIRED = {pkgs_repr}\n"
        f"_IMPORT_NAME = {dict(IMPORT_NAME)!r}\n"
        "_missing = []\n"
        "for _pkg in _REQUIRED:\n"
        "    _mod = _IMPORT_NAME.get(_pkg, _pkg)\n"
        "    try:\n"
        "        importlib.import_module(_mod)\n"
        "    except ImportError:\n"
        "        _missing.append(_pkg)\n"
        "\n"
        "print(f'Kernel Python: {sys.executable}')\n"
        "if _missing:\n"
        "    print(f'Installing missing packages into kernel: {_missing}')\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *_missing])\n"
        "    print('Done. RESTART THE KERNEL (Kernel → Restart Kernel) before running the next cell.')\n"
        "else:\n"
        "    print('All required packages already importable. Ready to run.')\n"
    )
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["preflight"]},
        "outputs": [],
        "source": body.splitlines(keepends=True),
    }


def make_preflight_md():
    return {
        "cell_type": "markdown",
        "metadata": {"tags": ["preflight"]},
        "source": [
            "## 0. Preflight — auto-install missing dependencies\n",
            "\n",
            "Run the cell below first. It checks which Python this notebook's "
            "kernel is using and installs any missing packages into **that "
            "same Python** — avoiding the common `pip install` vs. `jupyter` "
            "version mismatch.\n",
            "\n",
            "If the cell prints *\"Installing missing packages\"*, **restart "
            "the kernel** (top menu → `Kernel → Restart Kernel`) before "
            "running the rest of the notebook.\n"
        ],
    }


HERE = Path(__file__).resolve().parent

for nb_name, packages in NOTEBOOKS.items():
    path = HERE / nb_name
    if not path.exists():
        print(f'  SKIP (not found): {nb_name}')
        continue
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)

    # Skip if already has a preflight cell (idempotent)
    has_preflight = any(
        'preflight' in c.get('metadata', {}).get('tags', [])
        for c in nb['cells']
    )
    if has_preflight:
        print(f'  ALREADY HAS PREFLIGHT: {nb_name}')
        continue

    # Insert markdown + code preflight cells at position 1 (after title)
    preflight_md = make_preflight_md()
    preflight_code = make_preflight_cell(packages)
    nb['cells'].insert(1, preflight_md)
    nb['cells'].insert(2, preflight_code)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'  ADDED preflight to {nb_name} (now {len(nb["cells"])} cells)')

print('\nDone.')

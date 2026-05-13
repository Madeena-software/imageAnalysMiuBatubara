"""
refactor_notebooks.py
=====================
Replaces duplicated production logic in both research notebooks with clean
imports from the public/ package. Run from the repo root:

    python scripts/refactor_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "public" / "image-analysis-miu-batubara"

# ──────────────────────────────────────────────────────────────────────────────
# Cell source builders
# ──────────────────────────────────────────────────────────────────────────────

# Shared sys.path injection cell — works regardless of CWD or kernel restart.
# Uses pathlib so it is bulletproof even if the notebook is moved within the
# repo, because the path is resolved relative to the notebook itself via
# __vsc_ipynb_file__ (VS Code) with a fallback to Path.cwd().
SETUP_CELL = """\
# ── Notebook bootstrap: resolve the production module path ─────────────────
# This single cell replaces all duplicated production logic.
# Strategy: insert public/image-analysis-miu-batubara/ into sys.path so that
#   `import circle_detection` and `import block_detection` resolve to the
#   exact same source files served by the PyScript production environment.
# The path is computed from the repo root, found by walking up from this
#   notebook file.  VS Code exposes __vsc_ipynb_file__; for other kernels we
#   fall back to Path.cwd() (reliable when the server is started from the root).

import sys
from pathlib import Path

def _find_repo_root() -> Path:
    \"\"\"Walk up from the notebook location until we find pytest.ini (repo marker).\"\"\"
    # VS Code Jupyter sets __vsc_ipynb_file__ to the notebook's absolute path.
    try:
        start = Path(__vsc_ipynb_file__).resolve().parent  # noqa: F821
    except NameError:
        start = Path.cwd()  # fallback: assumes kernel was started from repo root

    for candidate in [start, *start.parents]:
        if (candidate / "pytest.ini").exists():
            return candidate
    # Last resort: return start (notebook dir / CWD)
    return start

_REPO_ROOT = _find_repo_root()
_APP_DIR = str(_REPO_ROOT / "public" / "image-analysis-miu-batubara")

if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

print(f"✅ Production module path: {_APP_DIR}")
print(f"   Resolved repo root   : {_REPO_ROOT}")
"""

# ── Circle notebook ───────────────────────────────────────────────────────────

CIRCLE_IMPORT_CELL = """\
# ── Production imports (Single Source of Truth) ────────────────────────────
# All detection logic lives exclusively in circle_detection.py.
# No code is duplicated here.
from circle_detection import (
    process_tiff_image,
    detect_grid_from_diagonal,
    analyze_grid_histograms,
    visualize_circle_invalid_roi,
    compare_diagonals,
    AIR_CV_THRESHOLD,
    AIR_DIAGONAL_VALIDATION_CODE,
    AIR_DIAGONAL_VALIDATION_ERROR,
)
print("✅ circle_detection imported from production source.")
"""

CIRCLE_HELPERS_CELL = """\
# ── Notebook-only helpers (not in production) ──────────────────────────────
# These thin wrappers let notebooks pass a file *path* (str/Path) in addition
# to raw bytes, mirroring the convenience expected in an analysis workflow.

from pathlib import Path


def _to_file_bytes(file_or_bytes):
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return bytes(file_or_bytes)
    if isinstance(file_or_bytes, Path):
        return file_or_bytes.read_bytes()
    if isinstance(file_or_bytes, str):
        return Path(file_or_bytes).read_bytes()
    raise TypeError(f"Unsupported input type: {type(file_or_bytes)!r}")


def process_image(image_path, **params):
    \"\"\"Notebook convenience: accepts a file path as well as raw bytes.\"\"\"
    return process_tiff_image(_to_file_bytes(image_path), params)


def run_full_miu_analysis(image_path, detect_params=None, compare_params=None):
    \"\"\"End-to-end production-identical circle MIU pipeline for notebook debugging.\"\"\"
    file_bytes = _to_file_bytes(image_path)
    detected  = process_tiff_image(file_bytes, detect_params or {})
    grid      = detect_grid_from_diagonal(file_bytes, detected)
    analysis  = compare_diagonals(file_bytes, grid, params=compare_params or {})
    return {
        "detected": detected,
        "grid":     grid,
        "analysis": analysis,
    }


print("✅ Notebook helpers defined (process_image, run_full_miu_analysis).")
"""

# ── Block notebook ────────────────────────────────────────────────────────────

BLOCK_IMPORT_CELL = """\
# ── Production imports (Single Source of Truth) ────────────────────────────
# All detection logic lives exclusively in block_detection.py.
# No code is duplicated here.
from block_detection import (
    process_blocks,
    analyze_block_histograms,
    subdivide_blocks,
    analyze_subdivision_histograms,
    visualize_block_invalid_roi,
    compare_blocks_1_vs_3,
    AIR_STEP_MAX_REL_DIFF,
    AIR_BLOCK_VALIDATION_CODE,
    AIR_BLOCK_VALIDATION_ERROR,
    ROI_SHRINK_RATIO,
)
print("✅ block_detection imported from production source.")
"""

BLOCK_HELPERS_CELL = """\
# ── Notebook-only helpers (not in production) ──────────────────────────────
from pathlib import Path


def _to_file_bytes(file_or_bytes):
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return bytes(file_or_bytes)
    if isinstance(file_or_bytes, Path):
        return file_or_bytes.read_bytes()
    if isinstance(file_or_bytes, str):
        return Path(file_or_bytes).read_bytes()
    raise TypeError(f"Unsupported input type: {type(file_or_bytes)!r}")


def process_image(image_path, **params):
    \"\"\"Notebook convenience: accepts a file path as well as raw bytes.\"\"\"
    return process_blocks(_to_file_bytes(image_path), params or {})


def run_full_miu_analysis(
    image_path,
    detect_params=None,
    compare_params=None,
    num_subdivisions=10,
    scale_factor=2 / 3,
):
    \"\"\"End-to-end production-identical block MIU pipeline for notebook debugging.\"\"\"
    file_bytes = _to_file_bytes(image_path)
    detected = process_blocks(file_bytes, detect_params or {})
    subs     = subdivide_blocks(
        file_bytes,
        detected["blocks"],
        num_subdivisions=num_subdivisions,
        scale_factor=scale_factor,
    )
    analysis = compare_blocks_1_vs_3(file_bytes, subs, params=compare_params or {})
    return {
        "detected":     detected,
        "subdivisions": subs,
        "analysis":     analysis,
    }


print("✅ Notebook helpers defined (process_image, run_full_miu_analysis).")
"""

# Replacement analysis cells for block notebook (old cells used obsolete API)
BLOCK_PROCESS_IMAGE_CELL = """\
# ── Step 1: Detect blocks ──────────────────────────────────────────────────
image_file = r"sample-block.tiff"  # ← update to your TIFF path

detect_params = {
    "threshold_value": 55000,
    "min_length_rectangular": 1400,
    "max_length_rectangular": 1600,
    "min_rectangularity": 0.9,
    "min_solidity": 0.9,
}

detection_result = process_image(image_file, **detect_params)
all_blocks = detection_result["blocks"]

print(f"Detected {detection_result['count']} blocks")
for b in all_blocks:
    print(f"  Block {b['id']} ({b['type']:10}) center={b['center']}  "
          f"mean={b['mean_value']:.1f}")
"""

BLOCK_SUBDIVIDE_CELL = """\
# ── Step 2: Subdivide each block into 10 step-wedge subdivisions ───────────
subdivisions = subdivide_blocks(image_file, all_blocks, num_subdivisions=10)
print(f"Total subdivisions: {subdivisions['total_count']}")
print(f"Subdivisions per block: {subdivisions['num_subdivisions']}")
"""

BLOCK_HISTOGRAM_CELL = """\
# ── Step 3: Histogram analysis for each block ──────────────────────────────
hist_result = analyze_block_histograms(image_file, all_blocks)
if hist_result:
    print("Block histogram generated successfully.")
else:
    print("Histogram analysis returned None (check image path).")
"""

BLOCK_SUBDIVIDE_HISTOGRAM_CELL = """\
# ── Step 4: Histogram for each subdivision of Block 1 ─────────────────────
sub_hist = analyze_subdivision_histograms(image_file, subdivisions, block_number=1)
if sub_hist:
    print(f"Subdivision histogram for Block {sub_hist['block_number']} generated.")
else:
    print("Subdivision histogram returned None.")
"""

BLOCK_COMPARE_CELL = """\
# ── Step 5: Full MIU analysis (differential regression) ───────────────────
comparison_result = compare_blocks_1_vs_3(image_file, subdivisions)

summary = comparison_result["summary"]
print("=== MIU Summary ===")
print(f"Orientation      : {summary['orientation']}")
print(f"μ Block 2 (coal) : {summary['mu_block2']:.5f} ± {summary['delta_mu_block2']:.5f}")
print(f"μ Block 4 (coal) : {summary['mu_block4']:.5f} ± {summary['delta_mu_block4']:.5f}")
print(f"R² Block 2       : {summary['r2_block2']:.4f}")
print(f"R² Block 4       : {summary['r2_block4']:.4f}")
if summary.get("air_validation_warning"):
    print(f"⚠️  {summary['air_validation_warning']}")
"""

# ──────────────────────────────────────────────────────────────────────────────
# Notebook manipulation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(nb: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"✅  Written: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Circle notebook refactor
# ──────────────────────────────────────────────────────────────────────────────

def refactor_circle_notebook(nb_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Refactoring: {nb_path.name}")
    print(f"{'='*60}")
    nb = _load(nb_path)

    # Preserve the original downstream analysis cells (indices 4-11)
    # They already call the right API; just keep them unchanged.
    original_cells = nb["cells"]

    # Cells to preserve (markdown + analysis cells): indices 0, 2, 4, 6, 8, 10
    # and code analysis cells: 5, 7, 9, 11
    md_import      = original_cells[0]   # "## Import Libraries"
    md_define      = original_cells[2]   # "## Define Image Processing Function"
    md_process     = original_cells[4]   # "## Process Image"
    cell_process   = original_cells[5]   # image_file = ...; process_image(...)
    md_full        = original_cells[6]   # "## Run Full MIU Analysis"
    cell_full      = original_cells[7]   # run_full_miu_analysis(...)
    md_hist        = original_cells[8]   # "## Histogram Analysis"
    cell_hist      = original_cells[9]   # analyze_grid_histograms(...)
    md_diff        = original_cells[10]  # "## Differential Attenuation"
    cell_diff      = original_cells[11]  # print summary

    nb["cells"] = [
        md_import,
        _code_cell(SETUP_CELL),
        _code_cell(CIRCLE_IMPORT_CELL),
        md_define,
        _code_cell(CIRCLE_HELPERS_CELL),
        md_process,
        cell_process,
        md_full,
        cell_full,
        md_hist,
        cell_hist,
        md_diff,
        cell_diff,
    ]

    # Clear all stale outputs
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    _save(nb, nb_path)


# ──────────────────────────────────────────────────────────────────────────────
# Block notebook refactor
# ──────────────────────────────────────────────────────────────────────────────

def refactor_block_notebook(nb_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Refactoring: {nb_path.name}")
    print(f"{'='*60}")
    nb = _load(nb_path)

    original_cells = nb["cells"]

    # Preserve only structural markdown cells from original
    md_import  = original_cells[0]  # "## Import Libraries"
    md_define  = original_cells[2]  # "## Define Image Processing Function"
    # Cells 4-14 are all old/legacy; replace with fresh production cells.

    nb["cells"] = [
        md_import,
        _code_cell(SETUP_CELL),
        _code_cell(BLOCK_IMPORT_CELL),
        md_define,
        _code_cell(BLOCK_HELPERS_CELL),
        _md_cell("## Step 1 — Detect Blocks\n\nSpecify the TIFF image path and run block detection."),
        _code_cell(BLOCK_PROCESS_IMAGE_CELL),
        _md_cell("## Step 2 — Subdivide Blocks into 10 Step-Wedge Grids\n\nDivide each block into 10 equal subdivisions along its longest side."),
        _code_cell(BLOCK_SUBDIVIDE_CELL),
        _md_cell("## Step 3 — Histogram Analysis\n\nPlot pixel-value histograms for each block and each subdivision."),
        _code_cell(BLOCK_HISTOGRAM_CELL),
        _code_cell(BLOCK_SUBDIVIDE_HISTOGRAM_CELL),
        _md_cell("## Step 4 — Differential Attenuation (MIU) Analysis\n\nCompare Block 2 vs Block 4 (coal) against Block 1 and Block 3 (air references) using the pre-log FFC differential regression model."),
        _code_cell(BLOCK_COMPARE_CELL),
    ]

    # Clear all stale outputs
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    _save(nb, nb_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    circle_nb = REPO_ROOT / "note-circle-detection.ipynb"
    block_nb  = REPO_ROOT / "note-block-detection.ipynb"

    refactor_circle_notebook(circle_nb)
    refactor_block_notebook(block_nb)

    print("\n✅  Both notebooks refactored successfully.")
    print("   Run the following to verify imports work:")
    print(f"   cd {REPO_ROOT} && python -c \"import sys; sys.path.insert(0, '{APP_DIR}'); import circle_detection, block_detection; print('OK')\"")

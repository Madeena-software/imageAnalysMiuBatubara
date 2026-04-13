"""PyScript processing orchestrator.

This module is intentionally thin: it exposes stable entry-point functions used by
index.html and delegates implementation to circle_detection.py and block_detection.py.
"""

from circle_detection import (
    analyze_grid_histograms as _analyze_grid_histograms,
    compare_diagonals as _compare_diagonals,
    detect_grid_from_diagonal as _detect_grid_from_diagonal,
    process_tiff_image as _process_tiff_image,
    visualize_circle_invalid_roi as _visualize_circle_invalid_roi,
)
from block_detection import (
    analyze_block_histograms as _analyze_block_histograms,
    analyze_subdivision_histograms as _analyze_subdivision_histograms,
    compare_blocks_1_vs_3 as _compare_blocks_1_vs_3,
    process_blocks as _process_blocks,
    subdivide_blocks as _subdivide_blocks,
    visualize_block_invalid_roi as _visualize_block_invalid_roi,
)


# Circle pipeline wrappers

def _bridge_call(operation_name, fn, *args, **kwargs):
    """Bridge Python exceptions to clean UI-readable errors."""
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"{operation_name}: {str(exc)}") from None
    if result is None:
        raise RuntimeError(f"{operation_name}: no result produced.")
    return result


def process_tiff_image(file_bytes, params):
    return _bridge_call("Circle processing failed", _process_tiff_image, file_bytes, params)


def detect_grid_from_diagonal(file_bytes, initial_results, grid_size=None):
    return _bridge_call("Grid detection failed", _detect_grid_from_diagonal, file_bytes, initial_results, grid_size)


def analyze_grid_histograms(file_bytes, grid_results):
    return _bridge_call("Grid histogram analysis failed", _analyze_grid_histograms, file_bytes, grid_results)


def compare_diagonals(file_bytes, grid_results, params=None):
    return _bridge_call("Circle physics analysis failed", _compare_diagonals, file_bytes, grid_results, params)


def visualize_circle_invalid_roi(file_bytes, grid_results):
    return _bridge_call("Circle invalid ROI visualization failed", _visualize_circle_invalid_roi, file_bytes, grid_results)


# Block pipeline wrappers

def process_blocks(file_bytes, params):
    return _bridge_call("Block processing failed", _process_blocks, file_bytes, params)


def analyze_block_histograms(file_bytes, all_blocks):
    return _bridge_call("Block histogram analysis failed", _analyze_block_histograms, file_bytes, all_blocks)


def subdivide_blocks(file_bytes, all_blocks, num_subdivisions=10, scale_factor=2 / 3):
    return _bridge_call(
        "Block subdivision failed",
        _subdivide_blocks,
        file_bytes,
        all_blocks,
        num_subdivisions,
        scale_factor,
    )


def analyze_subdivision_histograms(file_bytes, subdivisions, block_number=1):
    return _bridge_call(
        "Subdivision histogram analysis failed",
        _analyze_subdivision_histograms,
        file_bytes,
        subdivisions,
        block_number,
    )


def compare_blocks_1_vs_3(file_bytes, subdivisions, params=None):
    return _bridge_call("Block physics analysis failed", _compare_blocks_1_vs_3, file_bytes, subdivisions, params)


def visualize_block_invalid_roi(file_bytes, subdivisions):
    return _bridge_call("Block invalid ROI visualization failed", _visualize_block_invalid_roi, file_bytes, subdivisions)


# UI summary helpers

def build_circle_attenuation_summary(diagonal_result):
    """Create a normalized circle attenuation summary for UI rendering."""
    summary = (diagonal_result or {}).get("summary", {})
    upper_mu = float(summary.get("upper_mu_avg", 0.0))
    lower_mu = float(summary.get("lower_mu_avg", 0.0))
    delta_mu = abs(upper_mu - lower_mu)
    return {
        "title": "Attenuation (μ) Comparison",
        "left_label": "Upper Anti-Diagonal Sample",
        "right_label": "Lower Anti-Diagonal Sample",
        "left_mu": upper_mu,
        "right_mu": lower_mu,
        "delta_mu": float(delta_mu),
    }


def build_block_attenuation_summary(comparison_result):
    """Create a normalized block attenuation summary for UI rendering."""
    summary = (comparison_result or {}).get("summary", {})
    # Backward-compatible fallback: older payloads may expose coal μ as block1/block3.
    mu_block2 = float(summary.get("mu_block2", summary.get("mu_block1", 0.0)))
    mu_block4 = float(summary.get("mu_block4", summary.get("mu_block3", 0.0)))
    delta_mu = abs(mu_block2 - mu_block4)
    return {
        "title": "Attenuation (μ) Comparison",
        "left_label": "Block 2 (Coal)",
        "right_label": "Block 4 (Coal)",
        "left_mu": mu_block2,
        "right_mu": mu_block4,
        "delta_mu": float(delta_mu),
    }


# Export functions used by PyScript UI
__all__ = [
    "process_tiff_image",
    "detect_grid_from_diagonal",
    "analyze_grid_histograms",
    "compare_diagonals",
    "visualize_circle_invalid_roi",
    "process_blocks",
    "analyze_block_histograms",
    "subdivide_blocks",
    "analyze_subdivision_histograms",
    "compare_blocks_1_vs_3",
    "visualize_block_invalid_roi",
    "build_circle_attenuation_summary",
    "build_block_attenuation_summary",
]

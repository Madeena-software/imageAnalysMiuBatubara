"""PyScript processing orchestrator.

This module is intentionally thin: it exposes stable entry-point functions used by
index.html and delegates implementation to circle_detection.py and block_detection.py.
"""

from circle_detection import (
    analyze_grid_histograms as _analyze_grid_histograms,
    compare_diagonals as _compare_diagonals,
    detect_grid_from_diagonal as _detect_grid_from_diagonal,
    process_tiff_image as _process_tiff_image,
)
from block_detection import (
    analyze_block_histograms as _analyze_block_histograms,
    analyze_subdivision_histograms as _analyze_subdivision_histograms,
    compare_blocks_1_vs_3 as _compare_blocks_1_vs_3,
    process_blocks as _process_blocks,
    subdivide_blocks as _subdivide_blocks,
)


# Circle pipeline wrappers

def process_tiff_image(file_bytes, params):
    return _process_tiff_image(file_bytes, params)


def detect_grid_from_diagonal(file_bytes, initial_results, grid_size=None):
    return _detect_grid_from_diagonal(file_bytes, initial_results, grid_size)


def analyze_grid_histograms(file_bytes, grid_results):
    return _analyze_grid_histograms(file_bytes, grid_results)


def compare_diagonals(file_bytes, grid_results):
    return _compare_diagonals(file_bytes, grid_results)


# Block pipeline wrappers

def process_blocks(file_bytes, params):
    return _process_blocks(file_bytes, params)


def analyze_block_histograms(file_bytes, all_blocks):
    return _analyze_block_histograms(file_bytes, all_blocks)


def subdivide_blocks(file_bytes, all_blocks, num_subdivisions=10, scale_factor=2 / 3):
    return _subdivide_blocks(file_bytes, all_blocks, num_subdivisions, scale_factor)


def analyze_subdivision_histograms(file_bytes, subdivisions, block_number=1):
    return _analyze_subdivision_histograms(file_bytes, subdivisions, block_number)


def compare_blocks_1_vs_3(file_bytes, subdivisions):
    return _compare_blocks_1_vs_3(file_bytes, subdivisions)


# Export functions used by PyScript UI
__all__ = [
    "process_tiff_image",
    "detect_grid_from_diagonal",
    "analyze_grid_histograms",
    "compare_diagonals",
    "process_blocks",
    "analyze_block_histograms",
    "subdivide_blocks",
    "analyze_subdivision_histograms",
    "compare_blocks_1_vs_3",
]

"""
Circle Detection Processor for PyScript
Adapted from the original Jupyter notebook implementation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

# ============================================================
# DEBUG FLAG - Set to False to disable all console logging
# ============================================================
DEBUG = True


def process_tiff_image(file_bytes, params):
    """
    Process TIFF image from bytes with circle detection.

    Parameters:
    -----------
    file_bytes : bytes
        Image file data as bytes
    params : dict
        Processing parameters including:
        - threshold_value
        - min_diameter
        - max_diameter
        - min_circularity
        - min_solidity
        - expected_count
        - grid_cols

    Returns:
    --------
    dict with keys: 'circles', 'detection_image', 'mask_image', 'count'
    """
    try:
        # Load image from bytes
        nparr = np.frombuffer(file_bytes, np.uint8)

        # Try to decode as 16-bit TIFF
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            # Try with PIL
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        if img_16bit is None:
            if DEBUG:
                print("Error: Could not load image")
            return None

        if DEBUG:
            print(f"Image dimensions: {img_16bit.shape}")
            print(f"Image dtype: {img_16bit.dtype}")

        # Extract parameters
        threshold_value = params.get("threshold_value", 24000)
        min_diameter = params.get("min_diameter", 50)
        max_diameter = params.get("max_diameter", 357)

        # Convert diameter to area (for circular objects: area = π * (d/2)²)
        min_area = np.pi * (min_diameter / 2) ** 2
        max_area = np.pi * (max_diameter / 2) ** 2

        min_circularity = params.get("min_circularity", 0.6)
        min_solidity = params.get("min_solidity", 0.7)
        min_aspect_ratio = params.get("min_aspect_ratio", 0.7)
        max_aspect_ratio = params.get("max_aspect_ratio", 1.3)
        expected_count = params.get("expected_count", 4)
        grid_cols = params.get("grid_cols", 4)

        # Thresholding
        binary_mask = np.zeros_like(img_16bit, dtype=np.uint8)
        binary_mask[img_16bit < threshold_value] = 255

        # Clean noise with morphology
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, kernel, iterations=2
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_circles = []
        if DEBUG:
            print(f"Initial contours found: {len(contours)}")

        # Filter contours with shape validation
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter 1: Area
            if area < min_area or area > max_area:
                continue

            # Calculate basic properties
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Filter 2: Circularity
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < min_circularity:
                continue

            # Filter 3: Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue

            # Filter 4: Aspect Ratio
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue

            # All filters passed - save circle
            valid_circles.append(
                {
                    "center": center,
                    "radius": radius,
                    "contour": cnt,
                    "area": area,
                    "circularity": circularity,
                    "solidity": solidity,
                    "aspect_ratio": aspect_ratio,
                }
            )

        if DEBUG:
            print(f"After shape filtering: {len(valid_circles)} valid objects")

        # Filter duplicates based on distance
        if len(valid_circles) > expected_count:
            filtered = []
            for i, circle in enumerate(valid_circles):
                is_duplicate = False
                for j, other in enumerate(filtered):
                    dist = np.sqrt(
                        (circle["center"][0] - other["center"][0]) ** 2
                        + (circle["center"][1] - other["center"][1]) ** 2
                    )
                    avg_radius = (circle["radius"] + other["radius"]) / 2
                    if dist < avg_radius * 0.8:
                        is_duplicate = True
                        if circle["circularity"] > other["circularity"]:
                            filtered[j] = circle
                        break

                if not is_duplicate:
                    filtered.append(circle)

            valid_circles = filtered
            if DEBUG:
                print(f"After removing duplicates: {len(valid_circles)} objects")

        # Sort grid with row clustering
        if len(valid_circles) >= expected_count:
            valid_circles.sort(key=lambda c: c["center"][1])

            num_rows = expected_count // grid_cols
            sorted_final = []

            for row_idx in range(num_rows):
                start_idx = row_idx * grid_cols
                end_idx = start_idx + grid_cols
                if end_idx <= len(valid_circles):
                    row = valid_circles[start_idx:end_idx]
                    row.sort(key=lambda c: c["center"][0])
                    sorted_final.extend(row)

            if len(sorted_final) == expected_count:
                valid_circles = sorted_final
            else:
                if DEBUG:
                    print(
                        f"WARNING: Detected {len(valid_circles)} objects (not {expected_count}). Using fallback sorting."
                    )
                valid_circles.sort(
                    key=lambda c: (c["center"][1] // 200, c["center"][0])
                )
        else:
            if DEBUG:
                print(
                    f"WARNING: Detected {len(valid_circles)} objects (not {expected_count}). Sorting may not be accurate."
                )
            valid_circles.sort(key=lambda c: (c["center"][1] // 200, c["center"][0]))

        # Visualization & Classification
        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        debug_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

        if DEBUG:
            print("\n--- Final Detection Data ---")

        results = []

        for i, item in enumerate(valid_circles):
            center = item["center"]
            radius = item["radius"]

            # Get mean pixel value from 16-bit image
            mask_sampling = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.circle(mask_sampling, center, int(radius * 0.7), 255, -1)
            mean_val = cv2.mean(img_16bit, mask=mask_sampling)[0]

            # Draw on output (single color - blue)
            color = (0, 255, 0)  # Green
            cv2.circle(img_rgb, center, radius, color, 4)
            cv2.putText(
                img_rgb,
                str(i + 1),
                (center[0] - 20, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 100, 255),
                4,
            )

            # Draw on mask debug
            cv2.circle(debug_mask_rgb, center, radius, (0, 0, 255), 4)
            cv2.putText(
                debug_mask_rgb,
                str(i + 1),
                (center[0] - 20, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 100, 255),
                4,
            )

            # Save results (no classification)
            result_item = {
                "id": i + 1,
                "center": center,
                "radius": radius,
                "mean_value": mean_val,
                "circularity": item["circularity"],
                "solidity": item["solidity"],
                "aspect_ratio": item["aspect_ratio"],
            }
            results.append(result_item)

            if DEBUG:
                print(f"{i+1}: {center}, R={radius}, Mean={mean_val:.1f}")

        # Convert images to base64 for web display
        detection_base64 = numpy_to_base64(img_rgb)
        mask_base64 = numpy_to_base64(debug_mask_rgb)

        return {
            "circles": results,
            "detection_image": detection_base64,
            "mask_image": mask_base64,
            "count": len(results),
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in process_tiff_image: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


def detect_grid_from_diagonal(file_bytes, initial_results, grid_size=None):
    """
    Calculate full NxN grid positions from detected diagonal.
    Auto-detects grid size from number of detected circles.

    Parameters:
    -----------
    file_bytes : bytes
        Image file data
    initial_results : dict
        Results from initial detection
    grid_size : int, optional
        Grid size (if None, auto-detected from circle count)

    Returns:
    --------
    dict with grid information and visualization
    """
    try:
        # Load image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        if img_16bit is None:
            if DEBUG:
                print("Error: Could not load image for grid detection")
            return None

        circles = initial_results["circles"]

        # Auto-detect grid size from number of circles (assuming diagonal)
        if grid_size is None:
            grid_size = len(circles)
            if DEBUG:
                print(
                    f"Auto-detected grid size: {grid_size}x{grid_size} = {grid_size * grid_size} positions"
                )

        # Analyze diagonal
        if DEBUG:
            print("=== Diagonal Pattern Analysis ===")
        positions = [(c["center"][0], c["center"][1]) for c in circles]

        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        if DEBUG:
            print(f"Diagonal X: {x_coords}")
            print(f"Diagonal Y: {y_coords}")

        # Calculate spacing
        x_spacings = [x_coords[i + 1] - x_coords[i] for i in range(len(x_coords) - 1)]
        y_spacings = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]

        mean_x_spacing = np.mean(x_spacings)
        mean_y_spacing = np.mean(y_spacings)

        if DEBUG:
            print(f"X spacing: {mean_x_spacing:.1f}px")
            print(f"Y spacing: {mean_y_spacing:.1f}px")

        # Generate all grid positions
        if DEBUG:
            print(f"\n=== Generate {grid_size}x{grid_size} Grid ===")

        ref_x, ref_y = positions[0]

        # X for columns
        x_positions = [int(ref_x + col * mean_x_spacing) for col in range(grid_size)]
        # Y for rows
        y_positions = [int(ref_y + row * mean_y_spacing) for row in range(grid_size)]

        if DEBUG:
            print(f"X positions (columns): {x_positions}")
            print(f"Y positions (rows): {y_positions}")

        # Create all grid combinations
        grid_results = []

        for row in range(grid_size):
            for col in range(grid_size):
                center_x = x_positions[col]
                center_y = y_positions[row]

                grid_results.append(
                    {
                        "id": row * grid_size + col + 1,
                        "grid_pos": (row, col),
                        "center": (center_x, center_y),
                        "radius": 103,
                    }
                )

        # Visualization
        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

        for item in grid_results:
            row, col = item["grid_pos"]
            center = item["center"]

            # Draw circles at all positions
            cv2.circle(img_rgb, center, item["radius"], (0, 255, 255), 3)
            cv2.circle(img_rgb, center, 5, (255, 0, 0), -1)  # Red dot at center

            # Label with number
            cv2.putText(
                img_rgb,
                str(item["id"]),
                (center[0] - 20, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

        # Convert to base64
        grid_base64 = numpy_to_base64(img_rgb)

        return {
            "grid": grid_results,
            "grid_image": grid_base64,
            "x_spacing": mean_x_spacing,
            "y_spacing": mean_y_spacing,
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in detect_grid_from_diagonal: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


def analyze_grid_histograms(file_bytes, grid_results):
    """
    Analyze pixel value distribution for each grid position.

    Parameters:
    -----------
    file_bytes : bytes
        Image file data as bytes
    grid_results : dict
        Grid detection results from detect_grid_from_diagonal

    Returns:
    --------
    dict with histogram data and visualization image
    """
    try:
        # Load image from bytes
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        grid_data = grid_results["grid"]
        histogram_stats = []

        # Auto-detect grid size from grid data
        max_row = max(item["grid_pos"][0] for item in grid_data)
        max_col = max(item["grid_pos"][1] for item in grid_data)
        grid_size = max(max_row, max_col) + 1

        # Create NxN subplot for histograms
        fig_size = max(12, grid_size * 4)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(fig_size, fig_size))
        fig.suptitle(
            f"Histogram Distribution - {grid_size}x{grid_size} Grid",
            fontsize=16,
            fontweight="bold",
        )

        # Handle single row/column case (axes won't be 2D array)
        if grid_size == 1:
            axes = np.array([[axes]])
        elif grid_size > 1 and len(axes.shape) == 1:
            axes = axes.reshape(grid_size, 1)

        for idx, item in enumerate(grid_data):
            row, col = item["grid_pos"]
            center = item["center"]
            radius = item["radius"]

            # Create mask for circle area
            mask = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.circle(mask, center, int(radius * 0.7), 255, -1)

            # Extract pixel values inside circle
            pixel_values = img_16bit[mask == 255]

            # Calculate statistics
            mean_val = float(np.mean(pixel_values))
            median_val = float(np.median(pixel_values))
            std_val = float(np.std(pixel_values))
            min_val = float(np.min(pixel_values))
            max_val = float(np.max(pixel_values))

            histogram_stats.append(
                {
                    "grid_pos": (row, col),
                    "center": center,
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "pixel_count": len(pixel_values),
                }
            )

            # Plot histogram in appropriate subplot
            ax = axes[row, col]
            ax.hist(
                pixel_values, bins=50, color="steelblue", alpha=0.7, edgecolor="black"
            )

            # Add mean and median lines
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.1f}",
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.1f}",
            )

            # Title and labels
            ax.set_title(
                f"Grid [{row},{col}] - Pos {center}", fontsize=10, fontweight="bold"
            )
            ax.set_xlabel("Pixel Value (16-bit)", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

            # Add stats text box
            stats_text = f"Min: {min_val:.0f}\nMax: {max_val:.0f}\nStd: {std_val:.1f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        histogram_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        return {"histogram_stats": histogram_stats, "histogram_image": histogram_image}

    except Exception as e:
        if DEBUG:
            print(f"Error in analyze_grid_histograms: {str(e)}")
        return None


def compare_diagonals(file_bytes, grid_results):
    """
    Compare statistics between lower diagonal and upper diagonal.
    Auto-calculates diagonal positions based on grid size.
    Lower diagonal: positions below main diagonal (row > col)
    Upper diagonal: positions above main diagonal (row < col)

    Parameters:
    -----------
    file_bytes : bytes
        Image file data as bytes
    grid_results : dict
        Grid detection results from detect_grid_from_diagonal

    Returns:
    --------
    dict with diagonal comparison statistics and visualization
    """
    try:
        # Load image from bytes
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        grid_data = grid_results["grid"]

        # Auto-detect grid size
        max_row = max(item["grid_pos"][0] for item in grid_data)
        max_col = max(item["grid_pos"][1] for item in grid_data)
        grid_size = max(max_row, max_col) + 1

        # Calculate diagonal positions programmatically
        # Lower diagonal: row > col
        # Upper diagonal: row < col
        lower_positions = [
            (r, c) for r in range(grid_size) for c in range(grid_size) if r > c
        ]
        upper_positions = [
            (r, c) for r in range(grid_size) for c in range(grid_size) if r < c
        ]

        if DEBUG:
            print(f"\n=== Diagonal Comparison ({grid_size}x{grid_size}) ===")
            print(f"Lower diagonal positions: {lower_positions}")
            print(f"Upper diagonal positions: {upper_positions}")

        lower_diagonal = []
        upper_diagonal = []

        for item in grid_data:
            row, col = item["grid_pos"]
            if (row, col) in lower_positions:
                lower_diagonal.append(item)
            elif (row, col) in upper_positions:
                upper_diagonal.append(item)

        # Calculate statistics for each diagonal
        def get_diagonal_stats(diagonal_items):
            stats = []
            for item in diagonal_items:
                center = item["center"]
                radius = item["radius"]
                row, col = item["grid_pos"]

                # Create mask
                mask = np.zeros_like(img_16bit, dtype=np.uint8)
                cv2.circle(mask, center, int(radius * 0.7), 255, -1)

                # Extract pixel values
                pixel_values = img_16bit[mask == 255]

                mean_val = float(np.mean(pixel_values))
                median_val = float(np.median(pixel_values))
                std_val = float(np.std(pixel_values))

                stats.append(
                    {
                        "grid_pos": (row, col),
                        "center": center,
                        "mean": mean_val,
                        "median": median_val,
                        "std": std_val,
                        "pixel_count": len(pixel_values),
                    }
                )

            return stats

        lower_stats = get_diagonal_stats(lower_diagonal)
        upper_stats = get_diagonal_stats(upper_diagonal)

        # Calculate averages
        lower_means = [s["mean"] for s in lower_stats]
        lower_medians = [s["median"] for s in lower_stats]
        upper_means = [s["mean"] for s in upper_stats]
        upper_medians = [s["median"] for s in upper_stats]

        summary = {
            "lower_avg_mean": float(np.mean(lower_means)),
            "lower_avg_median": float(np.mean(lower_medians)),
            "lower_std_means": float(np.std(lower_means)),
            "upper_avg_mean": float(np.mean(upper_means)),
            "upper_avg_median": float(np.mean(upper_medians)),
            "upper_std_means": float(np.std(upper_means)),
            "mean_difference": float(abs(np.mean(lower_means) - np.mean(upper_means))),
        }

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Plot 1: Mean comparison
        ax1 = axes[0]
        x_lower = np.arange(len(lower_means))
        x_upper = np.arange(len(upper_means))
        width = 0.35

        ax1.bar(
            x_lower - width / 2,
            lower_means,
            width,
            label="Lower Diagonal",
            color="steelblue",
            alpha=0.8,
        )
        ax1.bar(
            x_upper + width / 2,
            upper_means,
            width,
            label="Upper Diagonal",
            color="coral",
            alpha=0.8,
        )

        ax1.set_xlabel("Grid Position", fontweight="bold")
        ax1.set_ylabel("Mean Pixel Value", fontweight="bold")
        ax1.set_title(
            "Mean Comparison: Lower vs Upper Diagonal", fontweight="bold", fontsize=14
        )
        ax1.set_xticks(np.arange(6))

        # Create labels
        lower_labels = [f"[{s['grid_pos'][0]},{s['grid_pos'][1]}]" for s in lower_stats]
        upper_labels = [f"[{s['grid_pos'][0]},{s['grid_pos'][1]}]" for s in upper_stats]
        combined_labels = [
            f"L:{lower_labels[i]}\nU:{upper_labels[i]}" for i in range(6)
        ]

        ax1.set_xticklabels(combined_labels, fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Median comparison
        ax2 = axes[1]
        ax2.bar(
            x_lower - width / 2,
            lower_medians,
            width,
            label="Lower Diagonal",
            color="steelblue",
            alpha=0.8,
        )
        ax2.bar(
            x_upper + width / 2,
            upper_medians,
            width,
            label="Upper Diagonal",
            color="coral",
            alpha=0.8,
        )

        ax2.set_xlabel("Grid Position", fontweight="bold")
        ax2.set_ylabel("Median Pixel Value", fontweight="bold")
        ax2.set_title(
            "Median Comparison: Lower vs Upper Diagonal", fontweight="bold", fontsize=14
        )
        ax2.set_xticks(np.arange(6))
        ax2.set_xticklabels(combined_labels, fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        comparison_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        return {
            "lower_stats": lower_stats,
            "upper_stats": upper_stats,
            "summary": summary,
            "comparison_image": comparison_image,
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in compare_diagonals: {str(e)}")
        return None


def numpy_to_base64(img_array):
    """
    Convert numpy array to base64 encoded PNG string.

    Parameters:
    -----------
    img_array : numpy.ndarray
        Image as numpy array (RGB or grayscale)

    Returns:
    --------
    str : Base64 encoded PNG image
    """
    # Convert to PIL Image
    if len(img_array.shape) == 2:
        # Grayscale
        pil_img = Image.fromarray(img_array, mode="L")
    else:
        # RGB
        pil_img = Image.fromarray(img_array, mode="RGB")

    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_base64


def process_blocks(file_bytes, params):
    """
    Process TIFF image for rectangular block detection.
    Detects blocks 2 and 4, then calculates positions of blocks 1 and 3.

    Parameters:
    -----------
    file_bytes : bytes
        Image file data as bytes
    params : dict
        Processing parameters including:
        - threshold_value: Threshold for separating objects from background
        - min_length_rectangular: Minimum length of rectangle side
        - max_length_rectangular: Maximum length of rectangle side
        - min_rectangularity: Minimum rectangularity (0-1)
        - min_solidity: Minimum solidity

    Returns:
    --------
    dict with keys: 'blocks', 'all_blocks', 'detection_image', 'mask_image', 'count'
    """
    try:
        # Load image from bytes
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        if img_16bit is None:
            if DEBUG:
                print("Error: Could not load image")
            return None

        if DEBUG:
            print(f"Image dimensions: {img_16bit.shape}")
            print(f"Image dtype: {img_16bit.dtype}")

        # Extract parameters
        threshold_value = params.get("threshold_value", 55000)
        min_length_rectangular = params.get("min_length_rectangular", 1400)
        max_length_rectangular = params.get("max_length_rectangular", 1600)
        min_rectangularity = params.get("min_rectangularity", 0.9)
        min_solidity = params.get("min_solidity", 0.9)

        # Thresholding
        binary_mask = np.zeros_like(img_16bit, dtype=np.uint8)
        binary_mask[img_16bit < threshold_value] = 255

        # Clean noise with morphology
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, kernel, iterations=2
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_blocks = []
        if DEBUG:
            print(f"Initial contours found: {len(contours)}")

        # Filter contours with rectangular shape validation
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Calculate basic properties for rectangle
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            center = (int(rect[0][0]), int(rect[0][1]))
            rect_width, rect_height = rect[1]

            # Normalize: width = short, height = long (for consistency)
            width = min(rect_width, rect_height)
            height = max(rect_width, rect_height)
            longest_side = height
            shortest_side = width

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Filter 1: Length constraints
            if (
                longest_side < min_length_rectangular
                or longest_side > max_length_rectangular
            ):
                continue

            # Filter 2: Rectangularity
            rect_area = width * height
            if rect_area == 0:
                continue
            rectangularity = area / rect_area
            if rectangularity < min_rectangularity:
                continue

            # Filter 3: Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue

            # All filters passed - save block
            valid_blocks.append(
                {
                    "center": center,
                    "box": box.tolist(),
                    "width": float(width),
                    "height": float(height),
                    "longest_side": float(longest_side),
                    "shortest_side": float(shortest_side),
                    "contour": cnt,
                    "area": float(area),
                    "rectangularity": float(rectangularity),
                    "solidity": float(solidity),
                }
            )

        if DEBUG:
            print(f"After shape filtering: {len(valid_blocks)} valid blocks")

        # Filter duplicates based on distance
        if len(valid_blocks) > 1:
            filtered = []
            for i, block in enumerate(valid_blocks):
                is_duplicate = False
                for j, other in enumerate(filtered):
                    dist = np.sqrt(
                        (block["center"][0] - other["center"][0]) ** 2
                        + (block["center"][1] - other["center"][1]) ** 2
                    )
                    avg_size = (
                        (block["width"] + block["height"]) / 2
                        + (other["width"] + other["height"]) / 2
                    ) / 2
                    if dist < avg_size * 0.5:
                        is_duplicate = True
                        if block["rectangularity"] > other["rectangularity"]:
                            filtered[j] = block
                        break

                if not is_duplicate:
                    filtered.append(block)

            valid_blocks = filtered
            if DEBUG:
                print(f"After removing duplicates: {len(valid_blocks)} blocks")

        # Sort by X position (left to right)
        valid_blocks.sort(key=lambda c: c["center"][0])

        # Calculate positions for blocks 1 and 3 if we have at least 2 detected blocks
        all_blocks = []
        if len(valid_blocks) >= 2:
            block_2 = valid_blocks[0]
            block_4 = valid_blocks[1]

            x2 = block_2["center"][0]
            x4 = block_4["center"][0]

            # Calculate spacing
            x3_temp = (x2 + x4) // 2
            dx = abs(((x4 - x3_temp) + (x3_temp - x2)) / 2)

            x3 = x3_temp
            x1 = x2 - dx

            # Create block 1 (calculated)
            all_blocks.append(
                {
                    "id": 1,
                    "center": (int(x1), block_2["center"][1]),
                    "box": [
                        [
                            int(x1 - block_2["width"] / 2),
                            int(block_2["center"][1] - block_2["height"] / 2),
                        ],
                        [
                            int(x1 + block_2["width"] / 2),
                            int(block_2["center"][1] - block_2["height"] / 2),
                        ],
                        [
                            int(x1 + block_2["width"] / 2),
                            int(block_2["center"][1] + block_2["height"] / 2),
                        ],
                        [
                            int(x1 - block_2["width"] / 2),
                            int(block_2["center"][1] + block_2["height"] / 2),
                        ],
                    ],
                    "width": block_2["width"],
                    "height": block_2["height"],
                    "mean_value": 0.0,
                    "rectangularity": block_2.get("rectangularity", 0.0),
                    "classification": "Calculated",
                    "type": "calculated",
                }
            )

            # Add block 2 (detected)
            all_blocks.append(
                {
                    "id": 2,
                    "center": block_2["center"],
                    "box": block_2["box"],
                    "width": block_2["width"],
                    "height": block_2["height"],
                    "mean_value": 0.0,
                    "rectangularity": block_2.get("rectangularity", 0.0),
                    "classification": "Detected",
                    "type": "detected",
                }
            )

            # Create block 3 (calculated)
            all_blocks.append(
                {
                    "id": 3,
                    "center": (int(x3), block_2["center"][1]),
                    "box": [
                        [
                            int(x3 - block_2["width"] / 2),
                            int(block_2["center"][1] - block_2["height"] / 2),
                        ],
                        [
                            int(x3 + block_2["width"] / 2),
                            int(block_2["center"][1] - block_2["height"] / 2),
                        ],
                        [
                            int(x3 + block_2["width"] / 2),
                            int(block_2["center"][1] + block_2["height"] / 2),
                        ],
                        [
                            int(x3 - block_2["width"] / 2),
                            int(block_2["center"][1] + block_2["height"] / 2),
                        ],
                    ],
                    "width": block_2["width"],
                    "height": block_2["height"],
                    "mean_value": 0.0,
                    "rectangularity": block_2.get("rectangularity", 0.0),
                    "classification": "Calculated",
                    "type": "calculated",
                }
            )

            # Add block 4 (detected)
            all_blocks.append(
                {
                    "id": 4,
                    "center": block_4["center"],
                    "box": block_4["box"],
                    "width": block_4["width"],
                    "height": block_4["height"],
                    "mean_value": 0.0,
                    "rectangularity": block_4.get("rectangularity", 0.0),
                    "classification": "Detected",
                    "type": "detected",
                }
            )

        # Calculate mean values for all blocks
        for block in all_blocks:
            mask_sampling = np.zeros_like(img_16bit, dtype=np.uint8)
            box_array = np.array(block["box"], dtype=np.int32)
            cv2.drawContours(mask_sampling, [box_array], 0, 255, -1)
            mean_val = cv2.mean(img_16bit, mask=mask_sampling)[0]
            block["mean_value"] = float(mean_val)

        # Visualization
        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        debug_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

        # Draw all blocks
        for block in all_blocks:
            # Color: magenta for calculated, green for detected
            if block["type"] == "calculated":
                color = (255, 0, 255)  # Magenta
            else:
                color = (0, 255, 0)  # Green

            box_array = np.array(block["box"], dtype=np.int32)
            cv2.drawContours(img_rgb, [box_array], 0, color, 4)
            cv2.putText(
                img_rgb,
                f"Block {block['id']}",
                (block["center"][0] - 40, block["center"][1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 100, 255),
                3,
            )

        # Convert images to base64
        detection_base64 = numpy_to_base64(img_rgb)
        mask_base64 = numpy_to_base64(debug_mask_rgb)

        # Prepare detected blocks results
        results = []
        for i, item in enumerate(valid_blocks):
            center = item["center"]
            box = item["box"]

            # Get mean pixel value
            mask_sampling = np.zeros_like(img_16bit, dtype=np.uint8)
            box_array = np.array(box, dtype=np.int32)
            cv2.drawContours(mask_sampling, [box_array], 0, 255, -1)
            mean_val = cv2.mean(img_16bit, mask=mask_sampling)[0]

            result_item = {
                "id": i + 1,
                "center": center,
                "box": box,
                "width": item["width"],
                "height": item["height"],
                "longest_side": item["longest_side"],
                "shortest_side": item["shortest_side"],
                "mean_value": float(mean_val),
                "rectangularity": item["rectangularity"],
                "solidity": item["solidity"],
            }
            results.append(result_item)

        return {
            "blocks": results,
            "all_blocks": all_blocks,
            "detection_image": detection_base64,
            "mask_image": mask_base64,
            "count": len(results),
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in process_blocks: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


def analyze_block_histograms(file_bytes, all_blocks):
    """
    Generate histogram visualization for each block.

    Parameters:
    -----------
    file_bytes : bytes
        Image file data as bytes
    all_blocks : list
        List of all blocks (blocks 1-4)

    Returns:
    --------
    dict with histogram image
    """
    try:
        # Load image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        if img_16bit is None:
            return None

        # Create figure with 2x2 subplots for 4 blocks
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        fig.suptitle(
            "Histogram Analysis for Each Block", fontsize=14, fontweight="bold"
        )

        for idx, block in enumerate(all_blocks):
            box = np.array(block["box"], dtype=np.int32)

            # Create mask for block area
            mask = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.drawContours(mask, [box], 0, 255, -1)

            # Extract pixel values
            pixel_values = img_16bit[mask == 255]

            if len(pixel_values) == 0:
                continue

            # Calculate statistics
            mean_val = np.mean(pixel_values)
            median_val = np.median(pixel_values)
            std_val = np.std(pixel_values)
            min_val = np.min(pixel_values)
            max_val = np.max(pixel_values)

            # Plot histogram
            ax = axes[idx]
            ax.hist(
                pixel_values, bins=40, color="steelblue", alpha=0.7, edgecolor="black"
            )

            # Add mean and median lines
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.0f}",
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.0f}",
            )

            # Set title based on block type
            block_type = block.get("type", "unknown")
            title_color = "blue" if block_type == "calculated" else "green"
            ax.set_title(
                f'Block {block["id"]} ({block_type.capitalize()})',
                fontsize=11,
                fontweight="bold",
                color=title_color,
            )
            ax.set_xlabel("Pixel Value", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

            # Add statistics box
            stats_text = f"Min: {min_val:.0f}\nMax: {max_val:.0f}\nStd: {std_val:.0f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        histogram_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {"histogram_image": histogram_base64}

    except Exception as e:
        if DEBUG:
            print(f"Error in analyze_block_histograms: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


def subdivide_blocks(file_bytes, all_blocks, num_subdivisions=10, scale_factor=2 / 3):
    """
    Subdivide each block into smaller grids along its longest side.

    Parameters:
    -----------
    file_bytes : bytes
        Image file data as bytes
    all_blocks : list
        List of all blocks (blocks 1-4)
    num_subdivisions : int
        Number of subdivisions (default: 10)
    scale_factor : float
        Scale factor for subdivision size (default: 2/3)

    Returns:
    --------
    dict with subdivision data and visualization
    """
    try:
        # Load image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        # Visualization
        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

        all_subdivisions = []

        for block in all_blocks:
            block_id = block["id"]
            center = block["center"]
            width = block["width"]
            height = block["height"]

            # Blocks are vertical - subdivide along HEIGHT
            actual_length = height
            actual_width = width

            # Subdivision size before scaling
            subdivision_length = actual_length / num_subdivisions
            subdivision_width = actual_width

            # Small block size after scaling
            small_length = subdivision_length * scale_factor
            small_width = subdivision_width * scale_factor

            # Get box points to determine orientation
            box = np.array(block["box"])

            # Vertical: iterate along Y
            box_sorted_y = sorted(box, key=lambda p: p[1])
            top_point = np.mean([box_sorted_y[0], box_sorted_y[1]], axis=0)
            bottom_point = np.mean([box_sorted_y[2], box_sorted_y[3]], axis=0)

            direction_vector = bottom_point - top_point
            direction_unit = direction_vector / np.linalg.norm(direction_vector)

            # Perpendicular vector for width
            perp_vector = np.array([-direction_unit[1], direction_unit[0]])

            start_point = top_point

            # Create subdivisions
            for i in range(num_subdivisions):
                # Center of subdivision i
                offset_along_length = (i + 0.5) * subdivision_length
                sub_center = start_point + direction_unit * offset_along_length
                sub_center = (int(sub_center[0]), int(sub_center[1]))

                # Calculate 4 corners of small block
                half_length = small_length / 2
                half_width = small_width / 2

                corners = [
                    sub_center
                    + direction_unit * half_length
                    + perp_vector * half_width,
                    sub_center
                    + direction_unit * half_length
                    - perp_vector * half_width,
                    sub_center
                    - direction_unit * half_length
                    - perp_vector * half_width,
                    sub_center
                    - direction_unit * half_length
                    + perp_vector * half_width,
                ]

                small_box = np.array(corners, dtype=np.int32)

                all_subdivisions.append(
                    {
                        "parent_block": block_id,
                        "subdivision_id": i + 1,
                        "center": sub_center,
                        "box": small_box.tolist(),
                        "length": float(small_length),
                        "width": float(small_width),
                    }
                )

                # Draw small block
                cv2.drawContours(img_rgb, [small_box], 0, (0, 255, 255), 2)
                cv2.circle(img_rgb, sub_center, 3, (255, 0, 0), -1)

                # Label
                label = f"{block_id}.{i + 1}"
                cv2.putText(
                    img_rgb,
                    label,
                    (sub_center[0] - 15, sub_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Convert to base64
        subdivision_base64 = numpy_to_base64(img_rgb)

        return {
            "subdivisions": all_subdivisions,
            "subdivision_image": subdivision_base64,
            "num_subdivisions": num_subdivisions,
            "total_count": len(all_subdivisions),
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in subdivide_blocks: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


def analyze_subdivision_histograms(file_bytes, subdivisions, block_number=1):
    """
    Generate histogram visualization for each subdivision of a specific block.

    Parameters:
    -----------
    file_bytes : bytes
        Image file data as bytes
    subdivisions : dict
        Result from subdivide_blocks containing subdivisions data
    block_number : int
        Block number to analyze (1-4)

    Returns:
    --------
    dict with histogram image
    """
    try:
        # Load image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        if img_16bit is None:
            return None

        # Filter subdivisions for the specific block
        subdivision_data = subdivisions.get("subdivisions", [])
        block_subs = [s for s in subdivision_data if s["parent_block"] == block_number]

        if len(block_subs) == 0:
            return None

        num_subs = len(block_subs)

        # Create figure with 2 rows x 5 cols for 10 subdivisions
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        fig.suptitle(
            f"Histogram Subdivisions for Block {block_number} - {num_subs} Grids",
            fontsize=16,
            fontweight="bold",
        )

        for idx, sub in enumerate(block_subs):
            box = np.array(sub["box"], dtype=np.int32)

            # Create mask for subdivision area
            mask = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.drawContours(mask, [box], 0, 255, -1)

            # Extract pixel values
            pixel_values = img_16bit[mask == 255]

            if len(pixel_values) == 0:
                continue

            # Calculate statistics
            mean_val = np.mean(pixel_values)
            median_val = np.median(pixel_values)
            std_val = np.std(pixel_values)
            min_val = np.min(pixel_values)
            max_val = np.max(pixel_values)

            # Plot histogram
            ax = axes[idx]
            ax.hist(
                pixel_values, bins=40, color="steelblue", alpha=0.7, edgecolor="black"
            )

            # Add mean and median lines
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {mean_val:.0f}",
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=f"Median: {median_val:.0f}",
            )

            # Title
            ax.set_title(f'Sub {sub["subdivision_id"]}', fontsize=10, fontweight="bold")
            ax.set_xlabel("Pixel Value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

            # Add statistics box
            stats_text = f"Min: {min_val:.0f}\nMax: {max_val:.0f}\nStd: {std_val:.0f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        histogram_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {
            "subdivision_histogram_image": histogram_base64,
            "block_number": block_number,
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in analyze_subdivision_histograms: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


def compare_blocks_1_vs_3(file_bytes, subdivisions):
    """
    Compare subdivisions of block 1 vs block 3.

    Parameters:
    -----------
    file_bytes : bytes
        Image file data as bytes
    subdivisions : dict
        Subdivision results from subdivide_blocks

    Returns:
    --------
    dict with comparison statistics and visualization
    """
    try:
        # Load image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_16bit is None:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_16bit = np.array(pil_img)

        subdivision_data = subdivisions["subdivisions"]

        # Separate by parent block
        block1_subs = [s for s in subdivision_data if s["parent_block"] == 1]
        block2_subs = [s for s in subdivision_data if s["parent_block"] == 2]
        block3_subs = [s for s in subdivision_data if s["parent_block"] == 3]
        block4_subs = [s for s in subdivision_data if s["parent_block"] == 4]

        # Calculate statistics for each block
        def get_block_stats(subs):
            stats = []
            for sub in subs:
                mask = np.zeros_like(img_16bit, dtype=np.uint8)
                box_array = np.array(sub["box"], dtype=np.int32)
                cv2.drawContours(mask, [box_array], 0, 255, -1)
                pixel_values = img_16bit[mask == 255]

                if len(pixel_values) > 0:
                    stats.append(
                        {
                            "subdivision_id": sub["subdivision_id"],
                            "mean": float(np.mean(pixel_values)),
                            "median": float(np.median(pixel_values)),
                            "std": float(np.std(pixel_values)),
                        }
                    )
            return stats

        block1_stats = get_block_stats(block1_subs)
        block2_stats = get_block_stats(block2_subs)
        block3_stats = get_block_stats(block3_subs)
        block4_stats = get_block_stats(block4_subs)

        # Create visualization
        num_subs = min(
            len(block1_stats), len(block2_stats), len(block3_stats), len(block4_stats)
        )
        x_pos = np.arange(num_subs)

        block1_means = [s["mean"] for s in block1_stats[:num_subs]]
        block2_means = [s["mean"] for s in block2_stats[:num_subs]]
        block3_means = [s["mean"] for s in block3_stats[:num_subs]]
        block4_means = [s["mean"] for s in block4_stats[:num_subs]]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Plot 1: Mean comparison bar chart
        ax1 = axes[0, 0]
        width = 0.35
        ax1.bar(
            x_pos - width / 2,
            block1_means,
            width,
            label="Block 1",
            alpha=0.8,
            color="steelblue",
        )
        ax1.bar(
            x_pos + width / 2,
            block3_means,
            width,
            label="Block 3",
            alpha=0.8,
            color="orange",
        )
        ax1.set_xlabel("Subdivision", fontweight="bold", fontsize=11)
        ax1.set_ylabel("Mean Pixel Value", fontweight="bold", fontsize=11)
        ax1.set_title("Mean Pixel Value - Block 1 vs 3", fontweight="bold", fontsize=13)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{i+1}" for i in range(num_subs)])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Median comparison
        ax2 = axes[0, 1]
        block1_medians = [s["median"] for s in block1_stats[:num_subs]]
        block3_medians = [s["median"] for s in block3_stats[:num_subs]]
        ax2.bar(
            x_pos - width / 2,
            block1_medians,
            width,
            label="Block 1",
            alpha=0.8,
            color="steelblue",
        )
        ax2.bar(
            x_pos + width / 2,
            block3_medians,
            width,
            label="Block 3",
            alpha=0.8,
            color="orange",
        )
        ax2.set_xlabel("Subdivision", fontweight="bold", fontsize=11)
        ax2.set_ylabel("Median Pixel Value", fontweight="bold", fontsize=11)
        ax2.set_title(
            "Median Pixel Value - Block 1 vs 3", fontweight="bold", fontsize=13
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{i+1}" for i in range(num_subs)])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: All 4 blocks trend
        ax3 = axes[1, 0]
        ax3.plot(
            x_pos,
            block1_means,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Block 1",
            color="steelblue",
        )
        ax3.plot(
            x_pos,
            block2_means,
            marker="^",
            linewidth=2,
            markersize=8,
            label="Block 2",
            color="green",
        )
        ax3.plot(
            x_pos,
            block3_means,
            marker="s",
            linewidth=2,
            markersize=8,
            label="Block 3",
            color="orange",
        )
        ax3.plot(
            x_pos,
            block4_means,
            marker="D",
            linewidth=2,
            markersize=8,
            label="Block 4",
            color="red",
        )
        ax3.set_xlabel("Subdivision", fontweight="bold", fontsize=11)
        ax3.set_ylabel("Mean Pixel Value", fontweight="bold", fontsize=11)
        ax3.set_title(
            "Trend Comparison - All Blocks Mean Values", fontweight="bold", fontsize=13
        )
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f"{i+1}" for i in range(num_subs)])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Difference plot
        ax4 = axes[1, 1]
        mean_diffs = [block1_means[i] - block3_means[i] for i in range(num_subs)]
        colors_diff = ["green" if d > 0 else "red" for d in mean_diffs]
        ax4.bar(x_pos, mean_diffs, color=colors_diff, alpha=0.7, edgecolor="black")
        ax4.axhline(0, color="black", linewidth=1, linestyle="-")
        ax4.set_xlabel("Subdivision", fontweight="bold", fontsize=11)
        ax4.set_ylabel(
            "Mean Difference (Block 1 - Block 3)", fontweight="bold", fontsize=11
        )
        ax4.set_title("Mean Value Difference", fontweight="bold", fontsize=13)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"{i+1}" for i in range(num_subs)])
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        comparison_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        # Calculate summary statistics
        summary = {
            "block1_mean_avg": float(np.mean(block1_means)),
            "block1_mean_std": float(np.std(block1_means)),
            "block2_mean_avg": float(np.mean(block2_means)),
            "block2_mean_std": float(np.std(block2_means)),
            "block3_mean_avg": float(np.mean(block3_means)),
            "block3_mean_std": float(np.std(block3_means)),
            "block4_mean_avg": float(np.mean(block4_means)),
            "block4_mean_std": float(np.std(block4_means)),
            "mean_difference_avg": float(np.mean(mean_diffs)),
            "mean_difference_std": float(np.std(mean_diffs)),
            "mean_difference_range": [
                float(np.min(mean_diffs)),
                float(np.max(mean_diffs)),
            ],
        }

        return {
            "block1_stats": block1_stats,
            "block2_stats": block2_stats,
            "block3_stats": block3_stats,
            "block4_stats": block4_stats,
            "comparison_image": comparison_image,
            "summary": summary,
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in compare_blocks_1_vs_3: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


# Export functions
__all__ = [
    "process_tiff_image",
    "detect_grid_from_diagonal",
    "analyze_grid_histograms",
    "compare_diagonals",
    "process_blocks",
    "subdivide_blocks",
    "compare_blocks_1_vs_3",
    "numpy_to_base64",
]

"""Circle detection and Beer-Lambert analysis utilities for PyScript."""

import base64
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEBUG = True
# Coefficient of Variation (CV = std/mean) threshold for anti-diagonal air circles.
# AIR_CV_THRESHOLD=0.05 means CV > 5% indicates likely ROI contamination by acrylic wall/noise.
AIR_CV_THRESHOLD = 0.05
AIR_DIAGONAL_VALIDATION_CODE = "E_CIRCLE_AIR_ROI"
AIR_DIAGONAL_VALIDATION_ERROR = (
    f"{AIR_DIAGONAL_VALIDATION_CODE}: Validation Failed: The 4 Air reference circles on the anti-diagonal show inconsistent intensities. "
    "Please adjust the Minimum/Maximum Diameter or Threshold parameters to ensure the circles fit strictly "
    "inside the empty physical holes."
)


def _load_image(file_bytes):
    """Load TIFF bytes into a NumPy array, compatible with PyScript runtime."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_16bit is None:
        pil_img = Image.open(io.BytesIO(file_bytes))
        img_16bit = np.array(pil_img)
    return img_16bit


def _load_and_validate_image(file_bytes):
    """Strict SOP validation: only 16-bit grayscale TIFF images are accepted."""
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Image format validation failed: unable to read image bytes ({exc}).") from exc

    img_format = (pil_img.format or "unknown").upper()
    if img_format != "TIFF":
        raise ValueError(
            f"Image format validation failed: expected 16-bit grayscale TIFF, got format '{img_format}'."
        )

    img = np.array(pil_img)
    if img.ndim != 2:
        raise ValueError(
            f"Image format validation failed: expected grayscale TIFF (single channel), got shape {img.shape}."
        )
    if img.dtype != np.uint16:
        raise ValueError(
            f"Image format validation failed: expected 16-bit TIFF (uint16), got dtype '{img.dtype}'."
        )
    return img


def _numpy_to_base64(img_array):
    """Convert a NumPy image array into base64 PNG for browser display."""
    if len(img_array.shape) == 2:
        pil_img = Image.fromarray(img_array, mode="L")
    else:
        pil_img = Image.fromarray(img_array, mode="RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_tiff_image(file_bytes, params):
    """Detect candidate circles from TIFF image bytes."""
    try:
        img_16bit = _load_and_validate_image(file_bytes)

        threshold_value = params.get("threshold_value", 24000)
        min_diameter = params.get("min_diameter", 50)
        max_diameter = params.get("max_diameter", 357)
        min_area = np.pi * (min_diameter / 2) ** 2
        max_area = np.pi * (max_diameter / 2) ** 2
        min_circularity = params.get("min_circularity", 0.6)
        min_solidity = params.get("min_solidity", 0.7)
        min_aspect_ratio = params.get("min_aspect_ratio", 0.7)
        max_aspect_ratio = params.get("max_aspect_ratio", 1.3)
        expected_count = params.get("expected_count", 4)
        grid_cols = params.get("grid_cols", 4)

        binary_mask = np.zeros_like(img_16bit, dtype=np.uint8)
        binary_mask[img_16bit < threshold_value] = 255
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_circles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < min_circularity:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue

            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue

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

        if len(valid_circles) > expected_count:
            filtered = []
            for circle in valid_circles:
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
                valid_circles.sort(key=lambda c: (c["center"][1] // 200, c["center"][0]))
        else:
            valid_circles.sort(key=lambda c: (c["center"][1] // 200, c["center"][0]))

        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        debug_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

        results = []
        for i, item in enumerate(valid_circles):
            center = item["center"]
            radius = item["radius"]

            mask_sampling = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.circle(mask_sampling, center, int(radius * 0.7), 255, -1)
            mean_val = cv2.mean(img_16bit, mask=mask_sampling)[0]

            cv2.circle(img_rgb, center, radius, (0, 255, 0), 4)
            cv2.putText(
                img_rgb,
                str(i + 1),
                (center[0] - 20, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 100, 255),
                4,
            )

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

            results.append(
                {
                    "id": i + 1,
                    "center": center,
                    "radius": radius,
                    "mean_value": mean_val,
                    "circularity": item["circularity"],
                    "solidity": item["solidity"],
                    "aspect_ratio": item["aspect_ratio"],
                }
            )

        return {
            "circles": results,
            "detection_image": _numpy_to_base64(img_rgb),
            "mask_image": _numpy_to_base64(debug_mask_rgb),
            "count": len(results),
        }

    except Exception as e:
        raise ValueError(f"Circle detection failed: {str(e)}") from e


def detect_grid_from_diagonal(file_bytes, initial_results, grid_size=None):
    """Extrapolate full grid positions from detected diagonal circles."""
    try:
        img_16bit = _load_and_validate_image(file_bytes)

        circles = initial_results["circles"]
        detected_count = len(circles)
        if detected_count != 4:
            raise ValueError(
                "Circle anchor validation failed: expected exactly 4 detected anchor circles "
                f"for diagonal extrapolation, found {detected_count}. "
                "Adjust circle detection parameters in the UI and retry."
            )
        if grid_size is None:
            grid_size = 4
        if int(grid_size) != 4:
            raise ValueError(f"Grid validation failed: expected 4x4 grid, received grid_size={grid_size}.")

        positions = [(c["center"][0], c["center"][1]) for c in circles]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        x_spacings = [x_coords[i + 1] - x_coords[i] for i in range(len(x_coords) - 1)]
        y_spacings = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
        mean_x_spacing = np.mean(x_spacings)
        mean_y_spacing = np.mean(y_spacings)

        ref_x, ref_y = positions[0]
        x_positions = [int(ref_x + col * mean_x_spacing) for col in range(grid_size)]
        y_positions = [int(ref_y + row * mean_y_spacing) for row in range(grid_size)]

        grid_results = []
        for row in range(grid_size):
            for col in range(grid_size):
                grid_results.append(
                    {
                        "id": row * grid_size + col + 1,
                        "grid_pos": (row, col),
                        "center": (x_positions[col], y_positions[row]),
                        "radius": 103,
                    }
                )

        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

        for item in grid_results:
            center = item["center"]
            cv2.circle(img_rgb, center, item["radius"], (0, 255, 255), 3)
            cv2.circle(img_rgb, center, 5, (255, 0, 0), -1)
            cv2.putText(
                img_rgb,
                str(item["id"]),
                (center[0] - 20, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

        if len(grid_results) != 16:
            raise ValueError(
                f"Circle grid validation failed: expected exactly 16 circles in 4x4 grid, found {len(grid_results)}."
            )

        for item in grid_results:
            cx, cy = item["center"]
            if cx < 0 or cy < 0 or cx >= img_16bit.shape[1] or cy >= img_16bit.shape[0]:
                raise ValueError(
                    f"Circle grid validation failed: extrapolated center {item['center']} is out of image bounds."
                )

        return {
            "grid": grid_results,
            "grid_image": _numpy_to_base64(img_rgb),
            "x_spacing": mean_x_spacing,
            "y_spacing": mean_y_spacing,
        }

    except Exception as e:
        raise ValueError(f"Grid extrapolation failed: {str(e)}") from e


def analyze_grid_histograms(file_bytes, grid_results):
    """Keep existing histogram output for UI compatibility."""
    try:
        img_16bit = _load_image(file_bytes)
        if img_16bit is None:
            return None

        grid_data = grid_results["grid"]
        histogram_stats = []
        max_row = max(item["grid_pos"][0] for item in grid_data)
        max_col = max(item["grid_pos"][1] for item in grid_data)
        grid_size = max(max_row, max_col) + 1

        grid_lookup = {(item["grid_pos"][0], item["grid_pos"][1]): item for item in grid_data}
        measured_pixels = {}
        global_min = None
        global_max = None

        # Collect all pixel distributions first so every subplot can use the same
        # x-axis range for fair visual comparison across positions.
        for pos, item in grid_lookup.items():
            mask = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.circle(mask, item["center"], int(item["radius"] * 0.7), 255, -1)
            pixel_values = img_16bit[mask == 255]
            if len(pixel_values) == 0:
                continue
            measured_pixels[pos] = pixel_values
            local_min = float(np.min(pixel_values))
            local_max = float(np.max(pixel_values))
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)

        fig_size = max(12, grid_size * 4)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(fig_size, fig_size))
        fig.suptitle(
            f"Histogram Distribution - {grid_size}x{grid_size} Grid (Anti-Diagonal Layout)",
            fontsize=16,
            fontweight="bold",
        )

        if grid_size == 1:
            axes = np.array([[axes]])
        elif grid_size > 1 and len(axes.shape) == 1:
            axes = axes.reshape(grid_size, 1)

        for row in range(grid_size):
            for col in range(grid_size):
                display_col = (grid_size - 1) - col
                ax = axes[row, display_col]
                grid_pos_id = row * grid_size + col + 1
                if (row, col) not in measured_pixels:
                    ax.text(
                        0.5,
                        0.5,
                        f"Position [{row},{col}]\nNo Data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="gray",
                    )
                    ax.set_title(f"Pos {grid_pos_id} [{row},{col}]", fontsize=10, fontweight="bold", color="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                item = grid_lookup[(row, col)]
                pixel_values = measured_pixels[(row, col)]

                mean_val = float(np.mean(pixel_values))
                median_val = float(np.median(pixel_values))
                std_val = float(np.std(pixel_values))
                min_val = float(np.min(pixel_values))
                max_val = float(np.max(pixel_values))

                histogram_stats.append(
                    {
                        "grid_pos": (row, col),
                        "position_id": grid_pos_id,
                        "center": item["center"],
                        "mean": mean_val,
                        "median": median_val,
                        "std": std_val,
                        "min": min_val,
                        "max": max_val,
                        "pixel_count": len(pixel_values),
                    }
                )

                ax.hist(pixel_values, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
                ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}")
                ax.axvline(
                    median_val,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {median_val:.1f}",
                )
                ax.set_title(f"Pos {grid_pos_id} [{row},{col}]", fontsize=10, fontweight="bold")
                ax.set_xlabel("Pixel Value (16-bit)", fontsize=8)
                ax.set_ylabel("Frequency", fontsize=8)
                if global_min is not None and global_max is not None:
                    if global_max == global_min:
                        ax.set_xlim([global_min - 0.5, global_max + 0.5])
                    else:
                        pad = (global_max - global_min) * 0.02
                        ax.set_xlim([global_min - pad, global_max + pad])
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, alpha=0.3)

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


def visualize_circle_invalid_roi(file_bytes, grid_results):
    """Build guidance overlay highlighting anti-diagonal air reference ROIs."""
    try:
        img_16bit = _load_and_validate_image(file_bytes)
        grid_data = grid_results["grid"]
        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

        for item in grid_data:
            row, col = item["grid_pos"]
            center = tuple(item["center"])
            radius = int(item["radius"])
            is_anti_air = (row + col) == 3
            color = (255, 0, 0) if is_anti_air else (0, 255, 0)
            label_prefix = "AIR" if is_anti_air else "ROI"
            cv2.circle(img_rgb, center, radius, color, 3)
            cv2.circle(img_rgb, center, int(radius * 0.7), color, 2)
            cv2.putText(
                img_rgb,
                f"{label_prefix}[{row},{col}]",
                (center[0] - 85, center[1] - radius - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return {
            "invalid_roi_image": _numpy_to_base64(img_rgb),
            "hint": "Anti-diagonal AIR ROIs are highlighted in red. Tune threshold and diameter so each ROI stays inside hole area.",
        }
    except Exception as e:
        raise ValueError(f"Circle invalid ROI visualization failed: {str(e)}") from e


def compare_diagonals(file_bytes, grid_results, params=None):
    """
    Beer-Lambert analysis for 4x4 circle grid.

    Physics model used (ratio method):
      μ_coal = -ln(I_coal / I_air) / x_coal

    Where:
    - I_air is taken from the average of the four main-diagonal air circles
    - x_coal = 6 mm (coal thickness only)
    - Acrylic attenuation cancels out because both paths include identical acrylic layers.
    """
    try:
        img_16bit = _load_and_validate_image(file_bytes)
        params = params or {}
        air_cv_threshold = float(params.get("air_cv_threshold", AIR_CV_THRESHOLD))

        grid_data = grid_results["grid"]
        circle_count = len(grid_data)
        if circle_count != 16:
            raise ValueError(
                "Circle count validation failed: expected exactly 16 circles before Beer-Lambert physics "
                f"calculation, found {circle_count}. "
                "Adjust circle detection parameters in the UI and retry."
            )

        diagonal_items = []
        anti_diagonal_items = []
        upper_items = []
        lower_items = []
        max_row = max(item["grid_pos"][0] for item in grid_data)
        max_col = max(item["grid_pos"][1] for item in grid_data)
        grid_size = max(max_row, max_col) + 1
        anti_diagonal_sum = grid_size - 1

        for item in grid_data:
            row, col = item["grid_pos"]
            if row == col:
                diagonal_items.append(item)
            elif (row + col) == anti_diagonal_sum:
                anti_diagonal_items.append(item)
            elif (row + col) < anti_diagonal_sum:
                upper_items.append(item)
            else:
                lower_items.append(item)

        def _measure(items):
            measured = []
            for item in items:
                mask = np.zeros_like(img_16bit, dtype=np.uint8)
                cv2.circle(mask, item["center"], int(item["radius"] * 0.7), 255, -1)
                pixel_values = img_16bit[mask == 255]
                measured.append(
                    {
                        "grid_pos": item["grid_pos"],
                        "center": item["center"],
                        "mean": float(np.mean(pixel_values)),
                        "median": float(np.median(pixel_values)),
                        "std": float(np.std(pixel_values)),
                    }
                )
            return measured

        diagonal_stats = _measure(diagonal_items)
        upper_stats = _measure(upper_items)
        lower_stats = _measure(lower_items)

        if len(diagonal_stats) == 0:
            raise ValueError("Physics validation failed: no diagonal (air reference) circles were available.")

        anti_diagonal_stats = _measure(anti_diagonal_items)
        if len(anti_diagonal_stats) != grid_size:
            raise ValueError(
                f"Circle validation failed: expected {grid_size} anti-diagonal air reference circles, "
                f"found {len(anti_diagonal_stats)}."
            )
        anti_air_means = np.array([float(s["mean"]) for s in anti_diagonal_stats], dtype=float)
        anti_air_mean = float(np.mean(anti_air_means))
        if anti_air_mean <= 0:
            raise ValueError(AIR_DIAGONAL_VALIDATION_ERROR)
        anti_air_cv = float(np.std(anti_air_means) / anti_air_mean)
        if anti_air_cv > air_cv_threshold:
            raise ValueError(AIR_DIAGONAL_VALIDATION_ERROR)

        # I0 from diagonal air circles (global reference intensity for Beer-Lambert).
        # Unit is pixel intensity from the 16-bit image.
        i0_air = float(np.mean([float(s["mean"]) for s in diagonal_stats]))
        # Coal thickness used in ratio method, in millimeters.
        x_coal_mm = 6.0
        eps = 1e-9
        if i0_air <= eps:
            raise ValueError("Invalid I0 reference: diagonal air intensity is too small.")

        def _attach_mu(stats_list):
            for s in stats_list:
                i_coal = np.clip(float(s["mean"]), eps, None)
                ratio = np.clip(i0_air / i_coal, eps, np.finfo(np.float64).max)
                # Beer-Lambert ratio method: μ = ln(I_air / I_coal) / x_coal.
                # Since x_coal_mm is in mm, resulting μ units are 1/mm.
                s["mu_coal"] = float(np.log(ratio) / x_coal_mm)
            return stats_list

        upper_stats = _attach_mu(upper_stats)
        lower_stats = _attach_mu(lower_stats)

        upper_intensity = np.array([s["mean"] for s in upper_stats], dtype=float)
        lower_intensity = np.array([s["mean"] for s in lower_stats], dtype=float)
        upper_mu = np.array([s["mu_coal"] for s in upper_stats], dtype=float)
        lower_mu = np.array([s["mu_coal"] for s in lower_stats], dtype=float)

        upper_count = len(upper_intensity)
        lower_count = len(lower_intensity)
        if upper_count == 0 or lower_count == 0 or lower_count != upper_count:
            raise ValueError(
                "Circle partition validation failed: "
                f"expected non-empty symmetric upper/lower partitions, found upper={upper_count}, "
                f"lower={lower_count}."
            )

        upper_intensity_mean = float(np.mean(upper_intensity))
        lower_intensity_mean = float(np.mean(lower_intensity))
        upper_mu_mean = float(np.mean(upper_mu))
        lower_mu_mean = float(np.mean(lower_mu))

        # Plot 1: Bar graph of mean intensity (upper anti-diagonal sample vs lower anti-diagonal sample)
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        labels = [
            f"Upper Anti-Diagonal ({upper_count})",
            f"Lower Anti-Diagonal ({lower_count})",
        ]
        x = np.arange(len(labels))
        intensity_vals = [upper_intensity_mean, lower_intensity_mean]
        bars1 = ax1.bar(x, intensity_vals, color=["#4c78a8", "#f58518"], width=0.6)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        for bar, val in zip(bars1, intensity_vals):
            ax1.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.1f}", ha="center", va="bottom", fontsize=10)
        ax1.set_xlabel("Sample Group", fontweight="bold")
        ax1.set_ylabel("Mean Intensity (I)", fontweight="bold")
        ax1.set_title("Mean Intensity Comparison (Upper vs Lower Anti-Diagonal Samples)", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        buf1 = io.BytesIO()
        plt.savefig(buf1, format="png", dpi=100, bbox_inches="tight")
        buf1.seek(0)
        intensity_plot_image = base64.b64encode(buf1.getvalue()).decode("utf-8")
        plt.close(fig1)

        # Plot 2: Bar graph of mean μ (upper anti-diagonal sample vs lower anti-diagonal sample)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        mu_vals = [upper_mu_mean, lower_mu_mean]
        bars2 = ax2.bar(x, mu_vals, color=["#54a24b", "#e45756"], width=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        for bar, val in zip(bars2, mu_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.5f}", ha="center", va="bottom", fontsize=10)
        ax2.set_xlabel("Sample Group", fontweight="bold")
        ax2.set_ylabel("μ (1/mm)", fontweight="bold")
        ax2.set_title("Mean μ Comparison (Upper vs Lower Anti-Diagonal Samples)", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
        buf2.seek(0)
        mu_plot_image = base64.b64encode(buf2.getvalue()).decode("utf-8")
        plt.close(fig2)

        summary = {
            "i0_air": i0_air,
            "x_coal_mm": x_coal_mm,
            "anti_air_cv": anti_air_cv,
            "air_cv_threshold": air_cv_threshold,
            "upper_intensity_avg": upper_intensity_mean,
            "lower_intensity_avg": lower_intensity_mean,
            "upper_mu_avg": upper_mu_mean,
            "lower_mu_avg": lower_mu_mean,
            "upper_mu_std": float(np.std(upper_mu)),
            "lower_mu_std": float(np.std(lower_mu)),
            # Compatibility keys for existing UI summary table.
            "lower_avg_mean": lower_intensity_mean,
            "upper_avg_mean": upper_intensity_mean,
            "mean_difference": float(abs(upper_intensity_mean - lower_intensity_mean)),
            "lower_avg_median": float(np.mean([float(s["median"]) for s in lower_stats])),
            "upper_avg_median": float(np.mean([float(s["median"]) for s in upper_stats])),
            "lower_std_means": float(np.std(lower_intensity)),
            "upper_std_means": float(np.std(upper_intensity)),
        }

        return {
            "diagonal_stats": diagonal_stats,
            "upper_stats": upper_stats,
            "lower_stats": lower_stats,
            "summary": summary,
            "intensity_plot_image": intensity_plot_image,
            "mu_plot_image": mu_plot_image,
            # Backward compatibility with existing UI key.
            "comparison_image": mu_plot_image,
        }

    except Exception as e:
        raise ValueError(f"Circle Beer-Lambert analysis failed: {str(e)}") from e


__all__ = [
    "process_tiff_image",
    "detect_grid_from_diagonal",
    "analyze_grid_histograms",
    "visualize_circle_invalid_roi",
    "compare_diagonals",
]

"""Block detection and Beer-Lambert regression analysis utilities for PyScript."""

import base64
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEBUG = True


def _load_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_16bit is None:
        pil_img = Image.open(io.BytesIO(file_bytes))
        img_16bit = np.array(pil_img)
    return img_16bit


def _numpy_to_base64(img_array):
    if len(img_array.shape) == 2:
        pil_img = Image.fromarray(img_array, mode="L")
    else:
        pil_img = Image.fromarray(img_array, mode="RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_blocks(file_bytes, params):
    """Detect step-wedge reference blocks and infer full block layout (1..4)."""
    try:
        img_16bit = _load_image(file_bytes)
        if img_16bit is None:
            return None

        threshold_value = params.get("threshold_value", 55000)
        min_length_rectangular = params.get("min_length_rectangular", 1400)
        max_length_rectangular = params.get("max_length_rectangular", 1600)
        min_rectangularity = params.get("min_rectangularity", 0.9)
        min_solidity = params.get("min_solidity", 0.9)

        binary_mask = np.zeros_like(img_16bit, dtype=np.uint8)
        binary_mask[img_16bit < threshold_value] = 255
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            center = (int(rect[0][0]), int(rect[0][1]))
            rect_width, rect_height = rect[1]

            width = min(rect_width, rect_height)
            height = max(rect_width, rect_height)
            longest_side = height
            shortest_side = width

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            if longest_side < min_length_rectangular or longest_side > max_length_rectangular:
                continue

            rect_area = width * height
            if rect_area == 0:
                continue
            rectangularity = area / rect_area
            if rectangularity < min_rectangularity:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue

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

        if len(valid_blocks) > 1:
            filtered = []
            for block in valid_blocks:
                is_duplicate = False
                for j, other in enumerate(filtered):
                    dist = np.sqrt(
                        (block["center"][0] - other["center"][0]) ** 2
                        + (block["center"][1] - other["center"][1]) ** 2
                    )
                    avg_size = (
                        (block["width"] + block["height"]) / 2 + (other["width"] + other["height"]) / 2
                    ) / 2
                    if dist < avg_size * 0.5:
                        is_duplicate = True
                        if block["rectangularity"] > other["rectangularity"]:
                            filtered[j] = block
                        break
                if not is_duplicate:
                    filtered.append(block)
            valid_blocks = filtered

        valid_blocks.sort(key=lambda c: c["center"][0])

        all_blocks = []
        if len(valid_blocks) >= 2:
            block_2 = valid_blocks[0]
            block_4 = valid_blocks[1]

            x2 = block_2["center"][0]
            x4 = block_4["center"][0]
            x3_temp = (x2 + x4) // 2
            dx = abs(((x4 - x3_temp) + (x3_temp - x2)) / 2)
            x3 = x3_temp
            x1 = x2 - dx

            all_blocks.append(
                {
                    "id": 1,
                    "center": (int(x1), block_2["center"][1]),
                    "box": [
                        [int(x1 - block_2["width"] / 2), int(block_2["center"][1] - block_2["height"] / 2)],
                        [int(x1 + block_2["width"] / 2), int(block_2["center"][1] - block_2["height"] / 2)],
                        [int(x1 + block_2["width"] / 2), int(block_2["center"][1] + block_2["height"] / 2)],
                        [int(x1 - block_2["width"] / 2), int(block_2["center"][1] + block_2["height"] / 2)],
                    ],
                    "width": block_2["width"],
                    "height": block_2["height"],
                    "mean_value": 0.0,
                    "rectangularity": block_2.get("rectangularity", 0.0),
                    "classification": "Calculated",
                    "type": "calculated",
                }
            )

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

            all_blocks.append(
                {
                    "id": 3,
                    "center": (int(x3), block_2["center"][1]),
                    "box": [
                        [int(x3 - block_2["width"] / 2), int(block_2["center"][1] - block_2["height"] / 2)],
                        [int(x3 + block_2["width"] / 2), int(block_2["center"][1] - block_2["height"] / 2)],
                        [int(x3 + block_2["width"] / 2), int(block_2["center"][1] + block_2["height"] / 2)],
                        [int(x3 - block_2["width"] / 2), int(block_2["center"][1] + block_2["height"] / 2)],
                    ],
                    "width": block_2["width"],
                    "height": block_2["height"],
                    "mean_value": 0.0,
                    "rectangularity": block_2.get("rectangularity", 0.0),
                    "classification": "Calculated",
                    "type": "calculated",
                }
            )

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

        for block in all_blocks:
            mask_sampling = np.zeros_like(img_16bit, dtype=np.uint8)
            box_array = np.array(block["box"], dtype=np.int32)
            cv2.drawContours(mask_sampling, [box_array], 0, 255, -1)
            block["mean_value"] = float(cv2.mean(img_16bit, mask=mask_sampling)[0])

        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        debug_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

        for block in all_blocks:
            color = (255, 0, 255) if block["type"] == "calculated" else (0, 255, 0)
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

        results = []
        for i, item in enumerate(valid_blocks):
            mask_sampling = np.zeros_like(img_16bit, dtype=np.uint8)
            box_array = np.array(item["box"], dtype=np.int32)
            cv2.drawContours(mask_sampling, [box_array], 0, 255, -1)
            mean_val = cv2.mean(img_16bit, mask=mask_sampling)[0]
            results.append(
                {
                    "id": i + 1,
                    "center": item["center"],
                    "box": item["box"],
                    "width": item["width"],
                    "height": item["height"],
                    "longest_side": item["longest_side"],
                    "shortest_side": item["shortest_side"],
                    "mean_value": float(mean_val),
                    "rectangularity": item["rectangularity"],
                    "solidity": item["solidity"],
                }
            )

        return {
            "blocks": results,
            "all_blocks": all_blocks,
            "detection_image": _numpy_to_base64(img_rgb),
            "mask_image": _numpy_to_base64(debug_mask_rgb),
            "count": len(results),
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in process_blocks: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


def analyze_block_histograms(file_bytes, all_blocks):
    """Keep existing block histogram output for UI compatibility."""
    try:
        img_16bit = _load_image(file_bytes)
        if img_16bit is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        fig.suptitle("Histogram Analysis for Each Block", fontsize=14, fontweight="bold")

        for idx, block in enumerate(all_blocks):
            box = np.array(block["box"], dtype=np.int32)
            mask = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.drawContours(mask, [box], 0, 255, -1)
            pixel_values = img_16bit[mask == 255]
            if len(pixel_values) == 0:
                continue

            mean_val = np.mean(pixel_values)
            median_val = np.median(pixel_values)
            std_val = np.std(pixel_values)
            min_val = np.min(pixel_values)
            max_val = np.max(pixel_values)

            ax = axes[idx]
            ax.hist(pixel_values, bins=40, color="steelblue", alpha=0.7, edgecolor="black")
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.0f}")
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.0f}",
            )
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

            stats_text = f"Min: {min_val:.0f}\\nMax: {max_val:.0f}\\nStd: {std_val:.0f}"
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
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        histogram_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {"histogram_image": histogram_base64}

    except Exception as e:
        if DEBUG:
            print(f"Error in analyze_block_histograms: {str(e)}")
        return None


def subdivide_blocks(file_bytes, all_blocks, num_subdivisions=10, scale_factor=2 / 3):
    """Subdivide each block into 10 step-wedge subdivisions."""
    try:
        img_16bit = _load_image(file_bytes)
        if img_16bit is None:
            return None

        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

        all_subdivisions = []
        for block in all_blocks:
            block_id = block["id"]
            width = block["width"]
            height = block["height"]
            actual_length = height
            actual_width = width
            subdivision_length = actual_length / num_subdivisions
            subdivision_width = actual_width
            small_length = subdivision_length * scale_factor
            small_width = subdivision_width * scale_factor

            box = np.array(block["box"])
            box_sorted_y = sorted(box, key=lambda p: p[1])
            top_point = np.mean([box_sorted_y[0], box_sorted_y[1]], axis=0)
            bottom_point = np.mean([box_sorted_y[2], box_sorted_y[3]], axis=0)

            direction_vector = bottom_point - top_point
            direction_unit = direction_vector / np.linalg.norm(direction_vector)
            perp_vector = np.array([-direction_unit[1], direction_unit[0]])
            start_point = top_point

            for i in range(num_subdivisions):
                offset_along_length = (i + 0.5) * subdivision_length
                sub_center = start_point + direction_unit * offset_along_length
                sub_center = (int(sub_center[0]), int(sub_center[1]))

                half_length = small_length / 2
                half_width = small_width / 2
                corners = [
                    sub_center + direction_unit * half_length + perp_vector * half_width,
                    sub_center + direction_unit * half_length - perp_vector * half_width,
                    sub_center - direction_unit * half_length - perp_vector * half_width,
                    sub_center - direction_unit * half_length + perp_vector * half_width,
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

                cv2.drawContours(img_rgb, [small_box], 0, (0, 255, 255), 2)
                cv2.circle(img_rgb, sub_center, 3, (255, 0, 0), -1)
                cv2.putText(
                    img_rgb,
                    f"{block_id}.{i + 1}",
                    (sub_center[0] - 15, sub_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return {
            "subdivisions": all_subdivisions,
            "subdivision_image": _numpy_to_base64(img_rgb),
            "num_subdivisions": num_subdivisions,
            "total_count": len(all_subdivisions),
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in subdivide_blocks: {str(e)}")
        return None


def analyze_subdivision_histograms(file_bytes, subdivisions, block_number=1):
    """Keep existing subdivision histogram output for UI compatibility."""
    try:
        img_16bit = _load_image(file_bytes)
        if img_16bit is None:
            return None

        subdivision_data = subdivisions.get("subdivisions", [])
        block_subs = [s for s in subdivision_data if s["parent_block"] == block_number]
        if len(block_subs) == 0:
            return None

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        fig.suptitle(
            f"Histogram Subdivisions for Block {block_number} - {len(block_subs)} Grids",
            fontsize=16,
            fontweight="bold",
        )

        for idx, sub in enumerate(block_subs):
            box = np.array(sub["box"], dtype=np.int32)
            mask = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.drawContours(mask, [box], 0, 255, -1)
            pixel_values = img_16bit[mask == 255]
            if len(pixel_values) == 0:
                continue

            mean_val = np.mean(pixel_values)
            median_val = np.median(pixel_values)
            std_val = np.std(pixel_values)
            min_val = np.min(pixel_values)
            max_val = np.max(pixel_values)

            ax = axes[idx]
            ax.hist(pixel_values, bins=40, color="steelblue", alpha=0.7, edgecolor="black")
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.0f}")
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=f"Median: {median_val:.0f}",
            )
            ax.set_title(f'Sub {sub["subdivision_id"]}', fontsize=10, fontweight="bold")
            ax.set_xlabel("Pixel Value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

            stats_text = f"Min: {min_val:.0f}\\nMax: {max_val:.0f}\\nStd: {std_val:.0f}"
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
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        histogram_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {"subdivision_histogram_image": histogram_base64, "block_number": block_number}

    except Exception as e:
        if DEBUG:
            print(f"Error in analyze_subdivision_histograms: {str(e)}")
        return None


def compare_blocks_1_vs_3(file_bytes, subdivisions):
    """
    Beer-Lambert analysis for step-wedge blocks.

    Linear model:
      -ln(It / I0) = μ_coal * x_coal + (μ_acrylic * 4)

    We fit y = m*x + b using numpy.polyfit(x, y, 1), where m is μ_coal.
    """
    try:
        img_16bit = _load_image(file_bytes)
        if img_16bit is None:
            return None

        subdivision_data = subdivisions["subdivisions"]

        def _stats_for_block(block_id):
            block_subs = [s for s in subdivision_data if s["parent_block"] == block_id]
            block_subs = sorted(block_subs, key=lambda s: s["subdivision_id"])
            stats = []
            for sub in block_subs:
                mask = np.zeros_like(img_16bit, dtype=np.uint8)
                box_array = np.array(sub["box"], dtype=np.int32)
                cv2.drawContours(mask, [box_array], 0, 255, -1)
                pixel_values = img_16bit[mask == 255]
                if len(pixel_values) == 0:
                    continue
                stats.append(
                    {
                        "subdivision_id": sub["subdivision_id"],
                        "mean": float(np.mean(pixel_values)),
                        "median": float(np.median(pixel_values)),
                        "std": float(np.std(pixel_values)),
                    }
                )
            return stats

        block1_stats = _stats_for_block(1)
        block2_stats = _stats_for_block(2)
        block3_stats = _stats_for_block(3)
        block4_stats = _stats_for_block(4)

        # Preserve orientation checking logic by inferring whether the input is swapped.
        # Candidate A: subdivision_id 1..10 maps to thickness 10..1 mm.
        # Candidate B: subdivision_id 1..10 maps to thickness 1..10 mm.
        def _orientation_score(air_stats, thicknesses):
            if len(air_stats) < 2:
                return -np.inf
            y = np.array([s["mean"] for s in air_stats], dtype=float)
            x = np.array(thicknesses[: len(y)], dtype=float)
            slope = np.polyfit(x, y, 1)[0]
            # For air, larger thickness should produce lower intensity => negative slope preferred.
            return float(-slope)

        normal_thickness = list(range(10, 0, -1))
        reversed_thickness = list(range(1, 11))

        score_normal = np.mean(
            [
                _orientation_score(block2_stats, normal_thickness),
                _orientation_score(block4_stats, normal_thickness),
            ]
        )
        score_reversed = np.mean(
            [
                _orientation_score(block2_stats, reversed_thickness),
                _orientation_score(block4_stats, reversed_thickness),
            ]
        )

        if score_reversed > score_normal:
            thickness_by_index = reversed_thickness
            orientation = "reversed"
        else:
            thickness_by_index = normal_thickness
            orientation = "normal"

        def _attach_thickness(stats):
            out = []
            for i, s in enumerate(stats):
                t = thickness_by_index[i] if i < len(thickness_by_index) else None
                row = dict(s)
                row["x_coal_mm"] = t
                out.append(row)
            return out

        block1_stats = _attach_thickness(block1_stats)
        block2_stats = _attach_thickness(block2_stats)
        block3_stats = _attach_thickness(block3_stats)
        block4_stats = _attach_thickness(block4_stats)

        # I0 definition from AIR blocks (block 2/4) at x=10 mm, using the darkest (minimum) intensity.
        air_x10_values = [s["mean"] for s in (block2_stats + block4_stats) if s.get("x_coal_mm") == 10]
        if len(air_x10_values) == 0:
            return None
        i0_air = float(np.min(air_x10_values))

        eps = 1e-9

        def _compute_mu_series(coal_stats):
            filtered = [s for s in coal_stats if s.get("x_coal_mm") is not None]
            filtered = sorted(filtered, key=lambda s: s["x_coal_mm"], reverse=True)

            x = np.array([float(s["x_coal_mm"]) for s in filtered], dtype=float)
            i_t = np.array([max(float(s["mean"]), eps) for s in filtered], dtype=float)
            ratio = np.clip(i_t / max(i0_air, eps), eps, None)

            # Beer-Lambert transform to linear form.
            y = -np.log(ratio)

            # Linear regression using numpy.polyfit(x, y, 1): slope = μ_coal.
            slope, intercept = np.polyfit(x, y, 1)

            # Per-step apparent μ for line visualization requested by UI.
            mu_point = y / np.clip(x, eps, None)

            return {
                "x": x,
                "intensity": i_t,
                "y": y,
                "mu_point": mu_point,
                "mu_coal": float(slope),
                "intercept": float(intercept),
            }

        block1_model = _compute_mu_series(block1_stats)
        block3_model = _compute_mu_series(block3_stats)

        # Air reference curve (average of block 2 and 4 means by thickness).
        air_by_t = {}
        for s in block2_stats + block4_stats:
            t = s.get("x_coal_mm")
            if t is None:
                continue
            air_by_t.setdefault(t, []).append(float(s["mean"]))

        air_t = sorted(air_by_t.keys(), reverse=True)
        air_mean = [float(np.mean(air_by_t[t])) for t in air_t]

        # Plot 1: Intensity vs subdivisions (Block 1, Block 3, Air)
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(block1_model["x"], block1_model["intensity"], marker="o", linewidth=2, label="Block 1 (Coal)")
        ax1.plot(block3_model["x"], block3_model["intensity"], marker="s", linewidth=2, label="Block 3 (Coal)")
        ax1.plot(air_t, air_mean, marker="^", linewidth=2, label="Air (Blocks 2 & 4 avg)")
        ax1.set_xlabel("Thickness x (mm)", fontweight="bold")
        ax1.set_ylabel("Mean Intensity (I)", fontweight="bold")
        ax1.set_title("Intensity vs Subdivision Thickness", fontweight="bold")
        ax1.set_xticks(list(range(10, 0, -1)))
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()

        buf1 = io.BytesIO()
        plt.savefig(buf1, format="png", dpi=100, bbox_inches="tight")
        buf1.seek(0)
        intensity_plot_image = base64.b64encode(buf1.getvalue()).decode("utf-8")
        plt.close(fig1)

        # Plot 2: μ vs thickness (Block 1 and Block 3)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(block1_model["x"], block1_model["mu_point"], marker="o", linewidth=2, label=f"Block 1 μ(x), fit μ={block1_model['mu_coal']:.5f}")
        ax2.plot(block3_model["x"], block3_model["mu_point"], marker="s", linewidth=2, label=f"Block 3 μ(x), fit μ={block3_model['mu_coal']:.5f}")
        ax2.set_xlabel("Thickness x (mm)", fontweight="bold")
        ax2.set_ylabel("μ (1/mm)", fontweight="bold")
        ax2.set_title("Coal Linear Attenuation Coefficient (μ) vs Thickness", fontweight="bold")
        ax2.set_xticks(list(range(10, 0, -1)))
        ax2.invert_xaxis()
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
        buf2.seek(0)
        mu_plot_image = base64.b64encode(buf2.getvalue()).decode("utf-8")
        plt.close(fig2)

        summary = {
            "orientation": orientation,
            "i0_air_x10": i0_air,
            "mu_block1": block1_model["mu_coal"],
            "mu_block3": block3_model["mu_coal"],
            "intercept_block1": block1_model["intercept"],
            "intercept_block3": block3_model["intercept"],
        }

        return {
            "block1_stats": block1_stats,
            "block2_stats": block2_stats,
            "block3_stats": block3_stats,
            "block4_stats": block4_stats,
            "block1_model": {
                "x": block1_model["x"].tolist(),
                "y": block1_model["y"].tolist(),
                "mu_point": block1_model["mu_point"].tolist(),
                "mu_coal": block1_model["mu_coal"],
                "intercept": block1_model["intercept"],
            },
            "block3_model": {
                "x": block3_model["x"].tolist(),
                "y": block3_model["y"].tolist(),
                "mu_point": block3_model["mu_point"].tolist(),
                "mu_coal": block3_model["mu_coal"],
                "intercept": block3_model["intercept"],
            },
            "intensity_plot_image": intensity_plot_image,
            "mu_plot_image": mu_plot_image,
            # Backward compatibility with current UI key.
            "comparison_image": mu_plot_image,
            "summary": summary,
        }

    except Exception as e:
        if DEBUG:
            print(f"Error in compare_blocks_1_vs_3: {str(e)}")
            import traceback

            traceback.print_exc()
        return None


__all__ = [
    "process_blocks",
    "analyze_block_histograms",
    "subdivide_blocks",
    "analyze_subdivision_histograms",
    "compare_blocks_1_vs_3",
]

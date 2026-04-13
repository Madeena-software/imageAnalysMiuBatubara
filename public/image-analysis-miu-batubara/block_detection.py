"""Block detection and Beer-Lambert regression analysis utilities for PyScript."""

import base64
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEBUG = True
# Shrink each ROI by scaling polygon dimensions to 88% of original (12% total shrink from center)
# before intensity sampling to avoid bright wall contamination.
ROI_SHRINK_RATIO = 0.12
AIR_GRADIENT_MIN_SCORE = 1.6
AIR_STEP_MAX_REL_DIFF = 0.20
AIR_STEP_MEAN_REL_DIFF = 0.10
SPIKE_CURVATURE_SIGMA_MULTIPLIER = 4.0
AIR_BLOCK_VALIDATION_ERROR = (
    "Validation Failed: The Air reference blocks (Block 1 & Block 3) captured the physical container walls "
    "or are incorrectly oriented. Expected arrangement: Block 1 leftmost and Block 3 rightmost. "
    "The calculated ROI is invalid. Please adjust the Block Threshold or check image alignment."
)


def _correct_intensity(value, offset=0.0, scale=1.0, eps=1e-9):
    corrected = (float(value) + float(offset)) * float(scale)
    return float(max(corrected, eps))


def _load_image(file_bytes):
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


def _shrink_box(box, ratio=ROI_SHRINK_RATIO):
    box_arr = np.array(box, dtype=np.float32)
    center = np.mean(box_arr, axis=0, keepdims=True)
    shrunk = center + (box_arr - center) * (1.0 - float(ratio))
    return shrunk.astype(np.int32)


def _mean_intensity_in_box(img_16bit, box, shrink_ratio=ROI_SHRINK_RATIO):
    mask = np.zeros_like(img_16bit, dtype=np.uint8)
    shrunk_box = _shrink_box(box, shrink_ratio)
    cv2.drawContours(mask, [np.array(shrunk_box, dtype=np.int32)], 0, 255, -1)
    pixel_values = img_16bit[mask == 255]
    if len(pixel_values) == 0:
        raise ValueError("ROI validation failed: no pixels inside shrunken ROI.")
    return float(np.mean(pixel_values)), pixel_values


def _validate_block_orientation(img_16bit, all_blocks):
    """Validate SOP orientation: darkest side (10 mm) must be at block bottom."""
    if len(all_blocks) == 0:
        raise ValueError("Orientation validation failed: no blocks available for orientation check.")

    candidate_blocks = [b for b in all_blocks if b.get("id") in (2, 4)]
    if len(candidate_blocks) == 0:
        candidate_blocks = all_blocks

    inverted_votes = 0
    checked = 0
    h, w = img_16bit.shape[:2]

    for block in candidate_blocks:
        box = np.array(block["box"], dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(box)
        if bw <= 0 or bh <= 0:
            continue

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + bw)
        y1 = min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            continue

        roi = img_16bit[y0:y1, x0:x1]
        roi_mask = np.zeros(roi.shape, dtype=np.uint8)
        shifted = box - np.array([x0, y0], dtype=np.int32)
        cv2.fillPoly(roi_mask, [shifted], 255)

        band = max(1, int((y1 - y0) * 0.2))
        top_band = np.zeros_like(roi_mask)
        bottom_band = np.zeros_like(roi_mask)
        top_band[:band, :] = 255
        bottom_band[-band:, :] = 255

        top_mask = cv2.bitwise_and(roi_mask, top_band)
        bottom_mask = cv2.bitwise_and(roi_mask, bottom_band)
        top_vals = roi[top_mask == 255]
        bottom_vals = roi[bottom_mask == 255]
        if len(top_vals) == 0 or len(bottom_vals) == 0:
            continue

        checked += 1
        top_mean = float(np.mean(top_vals))
        bottom_mean = float(np.mean(bottom_vals))
        if bottom_mean >= top_mean:
            inverted_votes += 1

    if checked == 0:
        raise ValueError("Orientation validation failed: unable to sample top/bottom intensity from detected blocks.")
    if inverted_votes > (checked / 2.0):
        raise ValueError(
            "Orientation validation failed: darkest 10 mm side is not at the bottom. "
            "Upload image with correct orientation (10 mm darkest side at bottom) and retry."
        )


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
        img_16bit = _load_and_validate_image(file_bytes)

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
        if len(valid_blocks) < 3:
            raise ValueError(
                "Block detection failed: unable to establish Block 1/2/3 sequence. "
                f"Detected valid contours: {len(valid_blocks)}."
            )

        # New spatial sequence:
        # 1) detect Block 1 and Block 3 (leftmost and rightmost)
        block_1 = valid_blocks[0]
        block_3 = valid_blocks[-1]
        x1, y1 = block_1["center"]
        x3, y3 = block_3["center"]
        if x3 <= x1:
            raise ValueError("Block geometry validation failed: Block 3 must be to the right of Block 1.")

        # 2) detect Block 2 in between Block 1 and Block 3
        between = [b for b in valid_blocks[1:-1] if x1 < b["center"][0] < x3]
        if len(between) == 0:
            raise ValueError(
                "Block detection failed: unable to detect Block 2 between Block 1 and Block 3. "
                "Adjust block detection parameters in the UI and retry."
            )
        midpoint = (x1 + x3) / 2.0
        block_2 = min(between, key=lambda b: abs(b["center"][0] - midpoint))
        x2, y2 = block_2["center"]
        if not (x1 < x2 < x3):
            raise ValueError("Block geometry validation failed: Block 2 is not spatially between Block 1 and Block 3.")

        # 3) extrapolate Block 4 using spacing delta
        dx12 = x2 - x1
        dx23 = x3 - x2
        dy12 = y2 - y1
        dy23 = y3 - y2
        dx_ref = int(round((dx12 + dx23) / 2.0))
        dy_ref = int(round((dy12 + dy23) / 2.0))
        if dx_ref <= 0:
            raise ValueError("Block geometry validation failed: invalid x-spacing for Block 4 extrapolation.")

        box3 = np.array(block_3["box"], dtype=np.int32)
        box4 = box3 + np.array([dx_ref, dy_ref], dtype=np.int32)
        center4 = (int(block_3["center"][0] + dx_ref), int(block_3["center"][1] + dy_ref))

        def _create_block_dict(block_id, detection_type, block_dict=None, center=None, box=None):
            if block_dict is not None:
                b_center = block_dict["center"]
                b_box = block_dict["box"]
                width = block_dict["width"]
                height = block_dict["height"]
                rectangularity = block_dict.get("rectangularity", 0.0)
            else:
                b_center = center
                b_box = box
                width = block_3["width"]
                height = block_3["height"]
                rectangularity = block_3.get("rectangularity", 0.0)
            return {
                "id": block_id,
                "center": (int(b_center[0]), int(b_center[1])),
                "box": np.array(b_box, dtype=np.int32).tolist(),
                "width": float(width),
                "height": float(height),
                "mean_value": 0.0,
                "rectangularity": float(rectangularity),
                "classification": "Detected" if detection_type == "detected" else "Calculated",
                "type": detection_type,
            }

        all_blocks = [
            _create_block_dict(1, "detected", block_dict=block_1),
            _create_block_dict(2, "detected", block_dict=block_2),
            _create_block_dict(3, "detected", block_dict=block_3),
            _create_block_dict(4, "calculated", center=center4, box=box4),
        ]

        h, w = img_16bit.shape[:2]
        for block in all_blocks:
            cx, cy = block["center"]
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                raise ValueError(
                    f"Block geometry validation failed: Block {block['id']} center {block['center']} out of image bounds."
                )
            mean_value, _ = _mean_intensity_in_box(img_16bit, block["box"], ROI_SHRINK_RATIO)
            block["mean_value"] = mean_value

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
        for block in all_blocks:
            results.append(
                {
                    "id": block["id"],
                    "center": block["center"],
                    "box": block["box"],
                    "width": block["width"],
                    "height": block["height"],
                    "longest_side": max(block["width"], block["height"]),
                    "shortest_side": min(block["width"], block["height"]),
                    "mean_value": block["mean_value"],
                    "rectangularity": block["rectangularity"],
                    "solidity": None,
                    "type": block["type"],
                }
            )

        if len(all_blocks) != 4:
            raise ValueError(
                "Block count validation failed: expected exactly 4 blocks for physics pipeline, "
                f"found {len(all_blocks)} (detected contours used in sequence: {len(valid_blocks)}). "
                "Adjust block detection parameters in the UI and retry."
            )

        _validate_block_orientation(img_16bit, all_blocks)

        return {
            "blocks": results,
            "all_blocks": all_blocks,
            "detection_image": _numpy_to_base64(img_rgb),
            "mask_image": _numpy_to_base64(debug_mask_rgb),
            "count": len(results),
        }

    except Exception as e:
        raise ValueError(f"Block detection failed: {str(e)}") from e


def analyze_block_histograms(file_bytes, all_blocks):
    """Keep existing block histogram output for UI compatibility."""
    try:
        img_16bit = _load_image(file_bytes)
        if img_16bit is None:
            return None

        # First pass: collect all block pixel distributions to establish one shared
        # x-axis range for all block histogram subplots.
        all_pixel_values = []
        global_min = None
        global_max = None
        for block in all_blocks:
            box = np.array(block["box"], dtype=np.int32)
            mask = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.drawContours(mask, [box], 0, 255, -1)
            pixel_values = img_16bit[mask == 255]
            all_pixel_values.append(pixel_values)
            if len(pixel_values) == 0:
                continue
            local_min = float(np.min(pixel_values))
            local_max = float(np.max(pixel_values))
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        fig.suptitle("Histogram Analysis for Each Block", fontsize=14, fontweight="bold")

        for idx, block in enumerate(all_blocks):
            pixel_values = all_pixel_values[idx] if idx < len(all_pixel_values) else np.array([])
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
            if global_min is not None and global_max is not None:
                if global_max == global_min:
                    ax.set_xlim([global_min - 0.5, global_max + 0.5])
                else:
                    pad = (global_max - global_min) * 0.02
                    ax.set_xlim([global_min - pad, global_max + pad])
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

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

        # First pass: collect subdivision pixel distributions and compute one shared
        # x-axis range so all subplot histograms are directly comparable.
        all_pixel_values = []
        global_min = None
        global_max = None
        for sub in block_subs:
            box = np.array(sub["box"], dtype=np.int32)
            mask = np.zeros_like(img_16bit, dtype=np.uint8)
            cv2.drawContours(mask, [box], 0, 255, -1)
            pixel_values = img_16bit[mask == 255]
            all_pixel_values.append(pixel_values)
            if len(pixel_values) == 0:
                continue
            local_min = float(np.min(pixel_values))
            local_max = float(np.max(pixel_values))
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        fig.suptitle(
            f"Histogram Subdivisions for Block {block_number} - {len(block_subs)} Grids",
            fontsize=16,
            fontweight="bold",
        )

        for idx, sub in enumerate(block_subs):
            pixel_values = all_pixel_values[idx] if idx < len(all_pixel_values) else np.array([])
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
            if global_min is not None and global_max is not None:
                if global_max == global_min:
                    ax.set_xlim([global_min - 0.5, global_max + 0.5])
                else:
                    pad = (global_max - global_min) * 0.02
                    ax.set_xlim([global_min - pad, global_max + pad])
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

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


def visualize_block_invalid_roi(file_bytes, subdivisions):
    """Build guidance overlay highlighting Air reference ROIs (Block 1 & 3)."""
    try:
        img_16bit = _load_and_validate_image(file_bytes)
        subdivision_data = subdivisions.get("subdivisions", [])
        img_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

        for sub in subdivision_data:
            parent = int(sub.get("parent_block", 0))
            box = np.array(sub["box"], dtype=np.int32)
            if parent in (1, 3):
                cv2.drawContours(img_rgb, [box], 0, (255, 0, 0), 2)
                shrunk = _shrink_box(box, ROI_SHRINK_RATIO)
                cv2.drawContours(img_rgb, [np.array(shrunk, dtype=np.int32)], 0, (0, 255, 255), 2)
                center = tuple(map(int, sub["center"]))
                cv2.putText(
                    img_rgb,
                    f"AIR {parent}.{sub['subdivision_id']}",
                    (center[0] - 45, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                )

        return {
            "invalid_roi_image": _numpy_to_base64(img_rgb),
            "hint": "Blue=air reference ROI, yellow=shrunken sampling ROI. Tune block threshold so ROIs avoid bright container walls.",
        }
    except Exception as e:
        raise ValueError(f"Block invalid ROI visualization failed: {str(e)}") from e


def compare_blocks_1_vs_3(file_bytes, subdivisions, params=None):
    """
    Beer-Lambert analysis for step-wedge blocks.

    Linear model:
      -ln(It / I0) = μ_coal*x_coal + μ_acrylic*(14 - x_coal)

    Rearranged:
      y = m*x + c
      m = μ_coal - μ_acrylic
      c = 14*μ_acrylic

    We fit y = m*x + c with numpy.polyfit(x, y, 1), then recover:
      μ_acrylic = c / 14
      μ_coal = m + μ_acrylic
    """
    try:
        img_16bit = _load_and_validate_image(file_bytes)
        params = params or {}
        intensity_offset = float(params.get("intensity_offset", 0.0))
        intensity_scale = float(params.get("intensity_scale", 1.0))
        air_gradient_min_score = float(params.get("air_gradient_min_score", AIR_GRADIENT_MIN_SCORE))
        air_step_max_rel_diff = float(params.get("air_step_max_rel_diff", AIR_STEP_MAX_REL_DIFF))
        air_step_mean_rel_diff = float(params.get("air_step_mean_rel_diff", AIR_STEP_MEAN_REL_DIFF))
        subdivision_data = subdivisions["subdivisions"]
        expected_steps = 10
        eps = 1e-9

        def _stats_for_block(block_id):
            block_subs = [s for s in subdivision_data if s["parent_block"] == block_id]
            block_subs = sorted(block_subs, key=lambda s: s["subdivision_id"])
            stats = []
            for sub in block_subs:
                mean_val, pixel_values = _mean_intensity_in_box(img_16bit, sub["box"], ROI_SHRINK_RATIO)
                stats.append(
                    {
                        "subdivision_id": sub["subdivision_id"],
                        "mean": float(mean_val),
                        "median": float(np.median(pixel_values)),
                        "std": float(np.std(pixel_values)),
                    }
                )
            return stats

        block1_raw = _stats_for_block(1)  # Air
        block2_raw = _stats_for_block(2)  # Coal
        block3_raw = _stats_for_block(3)  # Air
        block4_raw = _stats_for_block(4)  # Coal

        for block_id, stats in ((1, block1_raw), (2, block2_raw), (3, block3_raw), (4, block4_raw)):
            if len(stats) != expected_steps:
                raise ValueError(
                    f"Block subdivision validation failed: expected {expected_steps} steps in Block {block_id}, "
                    f"found {len(stats)}."
                )

        def _monotonic_decrease_score(values):
            """Return 0..2 score where higher means stronger top->bottom intensity decrease."""
            if len(values) < 2:
                return -np.inf
            diffs = np.diff(values)
            return float(np.mean(diffs <= 0) + (1.0 if values[0] > values[-1] else 0.0))

        b1_normal = np.array([s["mean"] for s in block1_raw], dtype=float)
        b3_normal = np.array([s["mean"] for s in block3_raw], dtype=float)
        score_normal = (_monotonic_decrease_score(b1_normal) + _monotonic_decrease_score(b3_normal)) / 2.0
        b1_reversed = b1_normal[::-1]
        b3_reversed = b3_normal[::-1]
        score_reversed = (_monotonic_decrease_score(b1_reversed) + _monotonic_decrease_score(b3_reversed)) / 2.0

        reverse_orientation = score_reversed > score_normal
        orientation = "reversed" if reverse_orientation else "normal"

        def _orient_stats(stats):
            ordered = list(reversed(stats)) if reverse_orientation else list(stats)
            thicknesses = list(range(10, 0, -1))
            out = []
            for idx, row in enumerate(ordered):
                row_out = dict(row)
                row_out["x_coal_mm"] = thicknesses[idx]
                out.append(row_out)
            return out

        block1_stats = _orient_stats(block1_raw)
        block2_stats = _orient_stats(block2_raw)
        block3_stats = _orient_stats(block3_raw)
        block4_stats = _orient_stats(block4_raw)

        air1 = np.array(
            [_correct_intensity(s["mean"], intensity_offset, intensity_scale, eps) for s in block1_stats],
            dtype=float,
        )
        air3 = np.array(
            [_correct_intensity(s["mean"], intensity_offset, intensity_scale, eps) for s in block3_stats],
            dtype=float,
        )

        air1_decrease_score = _monotonic_decrease_score(air1)
        air3_decrease_score = _monotonic_decrease_score(air3)
        if air1_decrease_score < air_gradient_min_score or air3_decrease_score < air_gradient_min_score:
            raise ValueError(AIR_BLOCK_VALIDATION_ERROR)

        rel_diff = np.abs(air1 - air3) / np.maximum((air1 + air3) / 2.0, eps)
        if float(np.max(rel_diff)) > air_step_max_rel_diff or float(np.mean(rel_diff)) > air_step_mean_rel_diff:
            raise ValueError(AIR_BLOCK_VALIDATION_ERROR)

        def _has_spike(values):
            diffs = np.diff(values)
            if len(diffs) < 3:
                return False
            curvature = np.abs(np.diff(diffs))
            scale = np.std(diffs) + eps
            return bool(np.max(curvature) > SPIKE_CURVATURE_SIGMA_MULTIPLIER * scale)

        if _has_spike(air1) or _has_spike(air3):
            raise ValueError(AIR_BLOCK_VALIDATION_ERROR)

        # Step-wise air reference from Block 1 and Block 3.
        air_ref = (air1 + air3) / 2.0
        i0_air_x10 = float(air_ref[0])
        if i0_air_x10 <= eps:
            raise ValueError("Invalid I0 reference: air intensity at x=10 is too small.")

        def _compute_mu_series(coal_stats):
            x = np.array([float(s["x_coal_mm"]) for s in coal_stats], dtype=float)
            i_t = np.array(
                [_correct_intensity(float(s["mean"]), intensity_offset, intensity_scale, eps) for s in coal_stats],
                dtype=float,
            )
            i0 = np.clip(air_ref, eps, None)
            ratio = np.clip(i_t / i0, eps, None)
            y = -np.log(ratio)
            slope, intercept = np.polyfit(x, y, 1)
            mu_acrylic = intercept / 14.0
            mu_coal = slope + mu_acrylic
            mu_point = (y - (mu_acrylic * (14.0 - x))) / np.clip(x, eps, None)
            return {
                "x": x,
                "intensity": i_t,
                "y": y,
                "mu_point": mu_point,
                "mu_coal": float(mu_coal),
                "mu_acrylic": float(mu_acrylic),
                "slope": float(slope),
                "intercept": float(intercept),
            }

        block2_model = _compute_mu_series(block2_stats)
        block4_model = _compute_mu_series(block4_stats)

        air_t = np.array([s["x_coal_mm"] for s in block1_stats], dtype=float)
        air_mean = air_ref.tolist()

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(block2_model["x"], block2_model["intensity"], marker="o", linewidth=2, label="Block 2 (Coal)")
        ax1.plot(block4_model["x"], block4_model["intensity"], marker="s", linewidth=2, label="Block 4 (Coal)")
        ax1.plot(air_t, air_mean, marker="^", linewidth=2, label="Air Ref (Block 1 & 3 avg)")
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

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(block2_model["x"], block2_model["mu_point"], marker="o", linewidth=2, label="Block 2 μ(x)")
        ax2.plot(block4_model["x"], block4_model["mu_point"], marker="s", linewidth=2, label="Block 4 μ(x)")
        ax2.set_xlabel("Thickness x (mm)", fontweight="bold")
        ax2.set_ylabel("μ (1/mm)", fontweight="bold")
        ax2.set_title(
            f"Coal μ vs Thickness (fit μ2={block2_model['mu_coal']:.5f}, μ4={block4_model['mu_coal']:.5f})",
            fontweight="bold",
        )
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
            "i0_air_x10": i0_air_x10,
            "intensity_offset": intensity_offset,
            "intensity_scale": intensity_scale,
            "air_gradient_min_score": air_gradient_min_score,
            "air_step_max_rel_diff": air_step_max_rel_diff,
            "air_step_mean_rel_diff": air_step_mean_rel_diff,
            "mu_block2": block2_model["mu_coal"],
            "mu_block4": block4_model["mu_coal"],
            "mu_acrylic_block2": block2_model["mu_acrylic"],
            "mu_acrylic_block4": block4_model["mu_acrylic"],
            "slope_block2": block2_model["slope"],
            "slope_block4": block4_model["slope"],
            "intercept_block2": block2_model["intercept"],
            "intercept_block4": block4_model["intercept"],
            # Backward compatibility aliases for callers that still expect
            # the old numbering where coal curves were exposed as block1/block3.
            "mu_block1": block2_model["mu_coal"],
            "mu_block3": block4_model["mu_coal"],
        }

        return {
            "block1_stats": block1_stats,
            "block2_stats": block2_stats,
            "block3_stats": block3_stats,
            "block4_stats": block4_stats,
            "block2_model": {
                "x": block2_model["x"].tolist(),
                "y": block2_model["y"].tolist(),
                "mu_point": block2_model["mu_point"].tolist(),
                "mu_coal": block2_model["mu_coal"],
                "mu_acrylic": block2_model["mu_acrylic"],
                "slope": block2_model["slope"],
                "intercept": block2_model["intercept"],
            },
            "block4_model": {
                "x": block4_model["x"].tolist(),
                "y": block4_model["y"].tolist(),
                "mu_point": block4_model["mu_point"].tolist(),
                "mu_coal": block4_model["mu_coal"],
                "mu_acrylic": block4_model["mu_acrylic"],
                "slope": block4_model["slope"],
                "intercept": block4_model["intercept"],
            },
            # Compatibility aliases for callers expecting old keys.
            "block1_model": {
                "x": block2_model["x"].tolist(),
                "y": block2_model["y"].tolist(),
                "mu_point": block2_model["mu_point"].tolist(),
                "mu_coal": block2_model["mu_coal"],
                "mu_acrylic": block2_model["mu_acrylic"],
                "slope": block2_model["slope"],
                "intercept": block2_model["intercept"],
            },
            "block3_model": {
                "x": block4_model["x"].tolist(),
                "y": block4_model["y"].tolist(),
                "mu_point": block4_model["mu_point"].tolist(),
                "mu_coal": block4_model["mu_coal"],
                "mu_acrylic": block4_model["mu_acrylic"],
                "slope": block4_model["slope"],
                "intercept": block4_model["intercept"],
            },
            "intensity_plot_image": intensity_plot_image,
            "mu_plot_image": mu_plot_image,
            "comparison_image": mu_plot_image,
            "summary": summary,
        }

    except Exception as e:
        raise ValueError(f"Block Beer-Lambert analysis failed: {str(e)}") from e


__all__ = [
    "process_blocks",
    "analyze_block_histograms",
    "subdivide_blocks",
    "analyze_subdivision_histograms",
    "visualize_block_invalid_roi",
    "compare_blocks_1_vs_3",
]

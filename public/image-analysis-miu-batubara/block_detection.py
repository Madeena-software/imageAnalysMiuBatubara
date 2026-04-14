"""Block detection and pre-log FFC differential regression analysis utilities for PyScript."""

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
AIR_STEP_MAX_REL_DIFF = 0.35
# Backward-compatible tunables used by integration tests.
AIR_GRADIENT_MIN_SCORE = 0.0
AIR_STEP_MEAN_REL_DIFF = AIR_STEP_MAX_REL_DIFF
SPIKE_CURVATURE_SIGMA_MULTIPLIER = 3.0
AIR_BLOCK_VALIDATION_CODE = "E_BLOCK_AIR_ROI"
AIR_BLOCK_VALIDATION_ERROR = (
    f"{AIR_BLOCK_VALIDATION_CODE}: Validation Failed: The Air reference blocks (Block 1 & Block 3) captured the physical container walls "
    "or are incorrectly oriented. Expected arrangement: Block 1 leftmost and Block 3 rightmost. "
    "The calculated ROI is invalid. Please adjust the Block Threshold or check image alignment."
)


def _load_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_16bit = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_16bit is None:
        pil_img = Image.open(io.BytesIO(file_bytes))
        img_16bit = np.array(pil_img)
    return img_16bit


def _load_and_validate_image(file_bytes):
    """Validate TIFF grayscale input and normalize any 16-bit NumPy dtype to uint16."""
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
    if img.dtype.kind != "u" or img.dtype.itemsize != 2:
        raise ValueError(
            f"Image format validation failed: expected 16-bit grayscale TIFF, got dtype '{img.dtype}'."
        )
    return img.astype(np.uint16, copy=False)


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


def _block_band_means(img_16bit, box, band_ratio=0.2):
    box_arr = np.array(box, dtype=np.int32)
    h, w = img_16bit.shape[:2]
    x, y, bw, bh = cv2.boundingRect(box_arr)

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        return None, None

    roi = img_16bit[y0:y1, x0:x1]
    roi_mask = np.zeros(roi.shape, dtype=np.uint8)
    shifted = box_arr - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(roi_mask, [shifted], 255)

    band = max(1, int((y1 - y0) * float(band_ratio)))
    top_band = np.zeros_like(roi_mask)
    bottom_band = np.zeros_like(roi_mask)
    top_band[:band, :] = 255
    bottom_band[-band:, :] = 255

    top_vals = roi[(roi_mask == 255) & (top_band == 255)]
    bottom_vals = roi[(roi_mask == 255) & (bottom_band == 255)]
    if len(top_vals) == 0 or len(bottom_vals) == 0:
        return None, None

    return float(np.mean(top_vals)), float(np.mean(bottom_vals))


def _extract_block_candidate(img_16bit, cnt):
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
    if perimeter == 0 or longest_side <= 0:
        return None

    rect_area = width * height
    if rect_area == 0:
        return None
    rectangularity = area / rect_area

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return None
    solidity = area / hull_area

    mean_value, _ = _mean_intensity_in_box(img_16bit, box, ROI_SHRINK_RATIO)
    top_mean, bottom_mean = _block_band_means(img_16bit, box, band_ratio=0.2)

    return {
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
        "mean_value": float(mean_value),
        "top_mean": None if top_mean is None else float(top_mean),
        "bottom_mean": None if bottom_mean is None else float(bottom_mean),
    }


def _build_axis_aligned_box(center, width, height):
    """Build rectangular ROI around center using locked width/height dimensions."""
    cx, cy = float(center[0]), float(center[1])
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    x0 = int(round(cx - half_w))
    x1 = int(round(cx + half_w))
    y0 = int(round(cy - half_h))
    y1 = int(round(cy + half_h))
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=np.int32,
    )


def _validate_block_orientation(img_16bit, all_blocks):
    """Validate SOP orientation: darkest side (10 mm) must be at block bottom."""
    if len(all_blocks) == 0:
        raise ValueError("Orientation validation failed: no blocks available for orientation check.")

    candidate_blocks = [b for b in all_blocks if b.get("id") in (1, 3)]
    if len(candidate_blocks) == 0:
        raise ValueError("Orientation validation failed: no Air reference blocks (Block 1 & 3) available for orientation check.")

    inverted_votes = 0
    checked = 0
    h, w = img_16bit.shape[:2]

    for block in candidate_blocks:
        top_mean, bottom_mean = _block_band_means(img_16bit, block["box"], band_ratio=0.2)
        if top_mean is None or bottom_mean is None:
            continue

        checked += 1
        if bottom_mean >= top_mean:
            inverted_votes += 1

    if checked == 0:
        raise ValueError("Orientation validation failed: unable to sample top/bottom intensity from detected blocks.")
    if inverted_votes > (checked / 2.0):
        raise ValueError(
            "Orientation validation failed: darkest 10 mm side is not at the bottom. "
            "Upload image with correct orientation (10 mm darkest side at bottom) and retry."
        )


def _select_air_reference_blocks(img_16bit, candidate_blocks):
    """Select the best matching air-reference pair from detected block contours."""
    if len(candidate_blocks) < 2:
        raise ValueError("Block detection failed: unable to establish the Block 1/3 reference pair.")

    enriched_blocks = []
    for block in candidate_blocks:
        enriched = dict(block)
        if enriched.get("mean_value") is None:
            mean_value, _ = _mean_intensity_in_box(img_16bit, enriched["box"], ROI_SHRINK_RATIO)
            enriched["mean_value"] = float(mean_value)
        if enriched.get("top_mean") is None or enriched.get("bottom_mean") is None:
            top_mean, bottom_mean = _block_band_means(img_16bit, enriched["box"], band_ratio=0.2)
            enriched["top_mean"] = None if top_mean is None else float(top_mean)
            enriched["bottom_mean"] = None if bottom_mean is None else float(bottom_mean)
        if enriched.get("top_mean") is None or enriched.get("bottom_mean") is None:
            continue
        enriched_blocks.append(enriched)

    if len(enriched_blocks) < 2:
        raise ValueError("Block detection failed: unable to establish the Block 1/3 reference pair.")

    eps = 1e-9
    image_width = float(img_16bit.shape[1]) if img_16bit.ndim >= 2 else 1.0
    best_pair = None
    best_score = -np.inf

    for left_index, left_block in enumerate(enriched_blocks):
        for right_index in range(left_index + 1, len(enriched_blocks)):
            right_block = enriched_blocks[right_index]
            if right_block["center"][0] <= left_block["center"][0]:
                continue

            pair_mean = (left_block["mean_value"] + right_block["mean_value"]) / 2.0
            mean_diff = abs(left_block["mean_value"] - right_block["mean_value"]) / max(pair_mean, eps)

            pair_top_mean = (left_block["top_mean"] + right_block["top_mean"]) / 2.0
            top_diff = abs(left_block["top_mean"] - right_block["top_mean"]) / max(pair_top_mean, eps)

            pair_bottom_mean = (left_block["bottom_mean"] + right_block["bottom_mean"]) / 2.0
            bottom_diff = abs(left_block["bottom_mean"] - right_block["bottom_mean"]) / max(pair_bottom_mean, eps)

            separation = (right_block["center"][0] - left_block["center"][0]) / max(image_width, eps)
            rectangularity = (
                float(left_block.get("rectangularity", 0.0)) + float(right_block.get("rectangularity", 0.0))
            ) / 2.0
            solidity = (float(left_block.get("solidity", 0.0)) + float(right_block.get("solidity", 0.0))) / 2.0
            darkness = 1.0 - (pair_mean / 65535.0)

            score = (
                (3.0 * darkness)
                + (0.25 * separation)
                + (0.20 * rectangularity)
                + (0.15 * solidity)
                - (5.0 * mean_diff)
                - (4.5 * bottom_diff)
                - (1.0 * top_diff)
            )

            if score > best_score:
                best_score = score
                best_pair = (left_block, right_block)

    if best_pair is None:
        raise ValueError("Block detection failed: unable to establish the Block 1/3 reference pair.")

    return [best_pair[0], best_pair[1]]


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

        candidate_blocks = []
        valid_blocks = []
        for cnt in contours:
            candidate = _extract_block_candidate(img_16bit, cnt)
            if candidate is None:
                continue

            candidate_blocks.append(candidate)

            if candidate["longest_side"] < min_length_rectangular or candidate["longest_side"] > max_length_rectangular:
                continue
            if candidate["rectangularity"] < min_rectangularity:
                continue
            if candidate["solidity"] < min_solidity:
                continue

            valid_blocks.append(candidate)

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
        if len(valid_blocks) < 2:
            raise ValueError(
                "Block detection failed: unable to establish the Block 1/3 reference pair. "
                f"Detected valid contours: {len(valid_blocks)}."
            )

        relaxed_rectangularity = max(0.85, float(min_rectangularity) - 0.05)
        reference_candidates = [
            block
            for block in candidate_blocks
            if block["longest_side"] >= min_length_rectangular
            and block["longest_side"] <= max_length_rectangular
            and block["rectangularity"] >= relaxed_rectangularity
            and block["solidity"] >= min_solidity
        ]
        if len(reference_candidates) < 2:
            reference_candidates = list(valid_blocks)

        reference_candidates.sort(key=lambda c: c["center"][0])

        # Anchor-and-Project spatial sequence:
        # 1) detect the best matching Air Block 1 and Air Block 3 pair.
        block_1, block_3 = _select_air_reference_blocks(img_16bit, reference_candidates)
        x1, y1 = block_1["center"]
        x3, y3 = block_3["center"]
        if x3 <= x1:
            raise ValueError("Block geometry validation failed: Block 3 must be to the right of Block 1.")

        # 2) choose anchor from Air blocks by best rectangular quality (cleanest ROI).
        def _anchor_score(block):
            """Rank anchor by rectangularity (primary), then solidity and area as tie-breakers."""
            return (
                float(block.get("rectangularity", 0.0)),
                float(block.get("solidity", 0.0)),
                float(block.get("area", 0.0)),
            )

        anchor_block = max((block_1, block_3), key=_anchor_score)
        locked_width = float(anchor_block["width"])
        locked_height = float(anchor_block["height"])
        if locked_width <= 0 or locked_height <= 0:
            raise ValueError("Block geometry validation failed: anchor block has invalid dimensions.")

        # 3) locate Block 2 by interpolation (exact midpoint of centers).
        center1 = (float(x1), float(y1))
        center3 = (float(x3), float(y3))
        center2 = (
            (center1[0] + center3[0]) / 2.0,
            (center1[1] + center3[1]) / 2.0,
        )
        if not (center1[0] < center2[0] < center3[0]):
            raise ValueError("Block geometry validation failed: interpolated Block 2 is not between Block 1 and Block 3.")

        # 4) locate Block 4 by extrapolation from the average spacing of Block 1 -> 2 and 2 -> 3.
        dx_12 = center2[0] - center1[0]
        dy_12 = center2[1] - center1[1]
        dx_23 = center3[0] - center2[0]
        dy_23 = center3[1] - center2[1]
        avg_dx = (dx_12 + dx_23) / 2.0
        avg_dy = (dy_12 + dy_23) / 2.0
        if avg_dx <= 0:
            raise ValueError("Block geometry validation failed: invalid x-spacing for Block 4 extrapolation.")
        center4 = (
            center3[0] + avg_dx,
            center3[1] + avg_dy,
        )

        # Apply locked anchor dimensions to ALL four ROIs.
        box1 = _build_axis_aligned_box(center1, locked_width, locked_height)
        box2 = _build_axis_aligned_box(center2, locked_width, locked_height)
        box3 = _build_axis_aligned_box(center3, locked_width, locked_height)
        box4 = _build_axis_aligned_box(center4, locked_width, locked_height)

        def _create_block_dict(block_id, detection_type, center, box, rectangularity):
            return {
                "id": block_id,
                "center": (int(round(center[0])), int(round(center[1]))),
                "box": np.array(box, dtype=np.int32).tolist(),
                "width": float(locked_width),
                "height": float(locked_height),
                "mean_value": 0.0,
                "rectangularity": float(rectangularity),
                "classification": "Detected" if detection_type == "detected" else "Calculated",
                "type": detection_type,
            }

        all_blocks = [
            _create_block_dict(1, "detected", center1, box1, block_1.get("rectangularity", 0.0)),
            _create_block_dict(2, "calculated", center2, box2, anchor_block.get("rectangularity", 0.0)),
            _create_block_dict(3, "detected", center3, box3, block_3.get("rectangularity", 0.0)),
            _create_block_dict(4, "calculated", center4, box4, anchor_block.get("rectangularity", 0.0)),
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
            "hint": "Red=air reference ROI, yellow=shrunken sampling ROI. Tune block threshold so ROIs avoid bright container walls.",
        }
    except Exception as e:
        raise ValueError(f"Block invalid ROI visualization failed: {str(e)}") from e


def compare_blocks_1_vs_3(file_bytes, subdivisions, params=None):
    """
    Differential pre-log FFC analysis for step-wedge blocks.

    For each step n, strict spatial pairing is used:
      ΔP_n_block2 = P_coal_n_block2 - P_air_n_block1
      ΔP_n_block4 = P_coal_n_block4 - P_air_n_block3

    Then fit ΔP = μ_coal * x_coal + c independently for Block 2 and Block 4,
    with x_coal in millimeters from 10 down to 1.
    """
    try:
        img_16bit = _load_and_validate_image(file_bytes)
        params = params or {}
        air_step_max_rel_diff = float(params.get("air_step_max_rel_diff", AIR_STEP_MAX_REL_DIFF))
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
            thicknesses = list(range(1, 11))
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

        air1 = np.array([s["mean"] for s in block1_stats], dtype=float)
        air3 = np.array([s["mean"] for s in block3_stats], dtype=float)
        coal2 = np.array([s["mean"] for s in block2_stats], dtype=float)
        coal4 = np.array([s["mean"] for s in block4_stats], dtype=float)

        # Validate only the bottom (max thickness) air step from Block 1 and Block 3.
        bottom_step_b1 = block1_stats[-1]
        bottom_step_b3 = block3_stats[-1]
        bottom_air1 = float(bottom_step_b1["mean"])
        bottom_air3 = float(bottom_step_b3["mean"])
        bottom_rel_diff = abs(bottom_air1 - bottom_air3) / max((bottom_air1 + bottom_air3) / 2.0, eps)
        air_validation_warning = None
        if bottom_rel_diff > air_step_max_rel_diff:
            air_validation_warning = AIR_BLOCK_VALIDATION_ERROR

        def _compute_mu_series(air_step_stats, air_values, coal_values):
            step_order = np.array([float(s["x_coal_mm"]) for s in air_step_stats], dtype=float)
            coal_thickness = 11.0 - step_order  # step_order is 1 -> 10, so coal_thickness is 10 -> 1 mm
            p_air = np.asarray(air_values, dtype=float)
            p_coal = np.asarray(coal_values, dtype=float)
            delta_p = p_coal - p_air
            fit_coeffs, fit_cov = np.polyfit(coal_thickness, delta_p, 1, cov=True)
            mu_coal = float(fit_coeffs[0])
            intercept = float(fit_coeffs[1])
            mu_coal_stderr = float(np.sqrt(max(float(fit_cov[0, 0]), 0.0)))
            y_fit = (mu_coal * coal_thickness) + intercept
            residual = delta_p - y_fit
            y_n = np.divide(delta_p, coal_thickness, out=np.zeros_like(delta_p), where=coal_thickness > eps)

            return {
                "x": coal_thickness,
                "coal_thickness": coal_thickness,
                "step_order": step_order,
                "p_air": p_air,
                "p_coal": p_coal,
                "delta_p": delta_p,
                "y_fit": y_fit,
                "residual": residual,
                "y_n": y_n,
                "mu_coal": mu_coal,
                "mu_coal_stderr": mu_coal_stderr,
                "mu_coal_pm": f"{mu_coal:.3f} ± {mu_coal_stderr:.3f}",
                "slope": mu_coal,
                "intercept": intercept,
            }

        block2_model = _compute_mu_series(block1_stats, air1, coal2)
        block4_model = _compute_mu_series(block3_stats, air3, coal4)

        def _build_attenuation_rows(sample_label, model):
            rows = []
            for idx, (step_value, coal_value, p_air_value, p_coal_value, delta_p_value, y_n_value, y_fit_value, residual_value) in enumerate(
                zip(
                    model["step_order"],
                    model["coal_thickness"],
                    model["p_air"],
                    model["p_coal"],
                    model["delta_p"],
                    model["y_n"],
                    model["y_fit"],
                    model["residual"],
                ),
                start=1,
            ):
                rows.append(
                    {
                        "sample": sample_label,
                        "step": int(idx),
                        "coal_mm": float(coal_value),
                        "p_air": float(p_air_value),
                        "p_coal": float(p_coal_value),
                        "delta_p": float(delta_p_value),
                        "y_n": float(y_n_value),
                        "y_fit": float(y_fit_value),
                        "residual": float(residual_value),
                    }
                )
            return rows

        attenuation_matrix_rows = _build_attenuation_rows("Block 2 (Coal)", block2_model)
        attenuation_matrix_rows.extend(_build_attenuation_rows("Block 4 (Coal)", block4_model))

        step_axis = np.array([s["x_coal_mm"] for s in block1_stats], dtype=float)
        coal_thickness_axis = 11.0 - step_axis

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(coal_thickness_axis, air1, marker="^", linewidth=1.5, linestyle="--", label="Block 1 (Air)")
        ax1.plot(coal_thickness_axis, coal2, marker="o", linewidth=2, label="Block 2 (Coal)")
        ax1.plot(coal_thickness_axis, air3, marker="v", linewidth=1.5, linestyle="--", label="Block 3 (Air)")
        ax1.plot(coal_thickness_axis, coal4, marker="s", linewidth=2, label="Block 4 (Coal)")
        ax1.set_xlabel("Coal Thickness (mm)", fontweight="bold")
        ax1.set_ylabel("Pixel Value (P)", fontweight="bold")
        ax1.set_title("Paired Pixel Values by Coal Thickness", fontweight="bold")
        ax1.set_xticks(list(range(1, 11)))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()

        buf1 = io.BytesIO()
        plt.savefig(buf1, format="png", dpi=100, bbox_inches="tight")
        buf1.seek(0)
        intensity_plot_image = base64.b64encode(buf1.getvalue()).decode("utf-8")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(block2_model["coal_thickness"], block2_model["y_n"], marker="o", linewidth=2, label="Block 2 Y_n")
        ax2.plot(block4_model["coal_thickness"], block4_model["y_n"], marker="s", linewidth=2, label="Block 4 Y_n")
        ax2.set_xlabel("Coal Thickness (mm)", fontweight="bold")
        ax2.set_ylabel("Y_n (Point-wise Attenuation)", fontweight="bold")
        ax2.set_title(
            f"Point-wise Attenuation Y_n by Step (μ2={block2_model['mu_coal']:.5f}, μ4={block4_model['mu_coal']:.5f})",
            fontweight="bold",
        )
        ax2.set_xticks(list(range(1, 11)))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
        buf2.seek(0)
        mu_plot_image = base64.b64encode(buf2.getvalue()).decode("utf-8")
        plt.close(fig2)

        def _r_squared(actual, predicted):
            actual = np.asarray(actual, dtype=float)
            predicted = np.asarray(predicted, dtype=float)
            ss_res = float(np.sum((actual - predicted) ** 2))
            ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
            if ss_tot <= eps:
                return float("nan")
            return float(1.0 - (ss_res / ss_tot))

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        x_fit = np.linspace(1.0, 10.0, 200)

        block2_y_fit = block2_model["y_fit"]
        block4_y_fit = block4_model["y_fit"]
        block2_r2 = _r_squared(block2_model["delta_p"], block2_y_fit)
        block4_r2 = _r_squared(block4_model["delta_p"], block4_y_fit)

        attenuation_fit_rows = [
            {
                "sample": "Block 2 (Coal)",
                "slope": float(block2_model["slope"]),
                "intercept": float(block2_model["intercept"]),
                "mu_coal": float(block2_model["mu_coal"]),
                "delta_mu": float(block2_model["mu_coal_stderr"]),
                "mu_pm": block2_model["mu_coal_pm"],
                "r2": float(block2_r2),
            },
            {
                "sample": "Block 4 (Coal)",
                "slope": float(block4_model["slope"]),
                "intercept": float(block4_model["intercept"]),
                "mu_coal": float(block4_model["mu_coal"]),
                "delta_mu": float(block4_model["mu_coal_stderr"]),
                "mu_pm": block4_model["mu_coal_pm"],
                "r2": float(block4_r2),
            },
        ]

        ax3.scatter(block2_model["coal_thickness"], block2_model["delta_p"], color="#4c78a8", marker="o", s=45, label="Block 2 data")
        ax3.plot(
            x_fit,
            block2_model["mu_coal"] * x_fit + block2_model["intercept"],
            color="#4c78a8",
            linestyle="--",
            linewidth=2,
            label=f"Block 2 fit (μ ± Δμ: {block2_model['mu_coal_pm']}, R²={block2_r2:.4f})",
        )
        ax3.scatter(block4_model["coal_thickness"], block4_model["delta_p"], color="#f58518", marker="s", s=45, label="Block 4 data")
        ax3.plot(
            x_fit,
            block4_model["mu_coal"] * x_fit + block4_model["intercept"],
            color="#f58518",
            linestyle="--",
            linewidth=2,
            label=f"Block 4 fit (μ ± Δμ: {block4_model['mu_coal_pm']}, R²={block4_r2:.4f})",
        )
        ax3.set_xlabel("Coal Thickness (mm)", fontweight="bold")
        ax3.set_ylabel("ΔP", fontweight="bold")
        ax3.set_title("Differential Linear Regression: ΔP = μ_coal·x + c", fontweight="bold")
        ax3.set_xticks(list(range(1, 11)))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.tight_layout()

        buf3 = io.BytesIO()
        plt.savefig(buf3, format="png", dpi=100, bbox_inches="tight")
        buf3.seek(0)
        linear_regression_plot_image = base64.b64encode(buf3.getvalue()).decode("utf-8")
        plt.close(fig3)

        summary = {
            "orientation": orientation,
            "air_step_max_rel_diff": air_step_max_rel_diff,
            "air_bottom_rel_diff": float(bottom_rel_diff),
            "mu_block2": block2_model["mu_coal"],
            "mu_block4": block4_model["mu_coal"],
            "delta_mu_block2": block2_model["mu_coal_stderr"],
            "delta_mu_block4": block4_model["mu_coal_stderr"],
            "mu_pm_block2": block2_model["mu_coal_pm"],
            "mu_pm_block4": block4_model["mu_coal_pm"],
            "slope_block2": block2_model["slope"],
            "slope_block4": block4_model["slope"],
            "intercept_block2": block2_model["intercept"],
            "intercept_block4": block4_model["intercept"],
            "air_validation_warning": air_validation_warning,
            "block1_mean_avg": float(np.mean([s["mean"] for s in block1_stats])),
            "block1_mean_std": float(np.std([s["mean"] for s in block1_stats])),
            "block2_mean_avg": float(np.mean([s["mean"] for s in block2_stats])),
            "block2_mean_std": float(np.std([s["mean"] for s in block2_stats])),
            "block3_mean_avg": float(np.mean([s["mean"] for s in block3_stats])),
            "block3_mean_std": float(np.std([s["mean"] for s in block3_stats])),
            "block4_mean_avg": float(np.mean([s["mean"] for s in block4_stats])),
            "block4_mean_std": float(np.std([s["mean"] for s in block4_stats])),
            "mean_difference_avg": float(np.mean(np.array([b1["mean"] - b3["mean"] for b1, b3 in zip(block1_stats, block3_stats)]))),
            "mean_difference_std": float(np.std(np.array([b1["mean"] - b3["mean"] for b1, b3 in zip(block1_stats, block3_stats)]))),
            "coal_difference_avg": float(np.mean(np.array([b2["mean"] - b4["mean"] for b2, b4 in zip(block2_stats, block4_stats)]))),
            "coal_difference_std": float(np.std(np.array([b2["mean"] - b4["mean"] for b2, b4 in zip(block2_stats, block4_stats)]))),
            "r2_block2": block2_r2,
            "r2_block4": block4_r2,
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
                "p_air": block2_model["p_air"].tolist(),
                "p_coal": block2_model["p_coal"].tolist(),
                "delta_p": block2_model["delta_p"].tolist(),
                "y_n": block2_model["y_n"].tolist(),
                "y_fit": block2_model["y_fit"].tolist(),
                "residual": block2_model["residual"].tolist(),
                "mu_coal": block2_model["mu_coal"],
                "delta_mu": block2_model["mu_coal_stderr"],
                "mu_pm": block2_model["mu_coal_pm"],
                "slope": float(block2_model["slope"]),
                "intercept": float(block2_model["intercept"]),
            },
            "block4_model": {
                "x": block4_model["x"].tolist(),
                "p_air": block4_model["p_air"].tolist(),
                "p_coal": block4_model["p_coal"].tolist(),
                "delta_p": block4_model["delta_p"].tolist(),
                "y_n": block4_model["y_n"].tolist(),
                "y_fit": block4_model["y_fit"].tolist(),
                "residual": block4_model["residual"].tolist(),
                "mu_coal": block4_model["mu_coal"],
                "delta_mu": block4_model["mu_coal_stderr"],
                "mu_pm": block4_model["mu_coal_pm"],
                "slope": float(block4_model["slope"]),
                "intercept": float(block4_model["intercept"]),
            },
            # Compatibility aliases for callers expecting old keys.
            "block1_model": {
                "x": block2_model["x"].tolist(),
                "p_air": block2_model["p_air"].tolist(),
                "p_coal": block2_model["p_coal"].tolist(),
                "delta_p": block2_model["delta_p"].tolist(),
                "y_n": block2_model["y_n"].tolist(),
                "y_fit": block2_model["y_fit"].tolist(),
                "residual": block2_model["residual"].tolist(),
                "mu_coal": block2_model["mu_coal"],
                "delta_mu": block2_model["mu_coal_stderr"],
                "mu_pm": block2_model["mu_coal_pm"],
                "slope": float(block2_model["slope"]),
                "intercept": float(block2_model["intercept"]),
            },
            "block3_model": {
                "x": block4_model["x"].tolist(),
                "p_air": block4_model["p_air"].tolist(),
                "p_coal": block4_model["p_coal"].tolist(),
                "delta_p": block4_model["delta_p"].tolist(),
                "y_n": block4_model["y_n"].tolist(),
                "y_fit": block4_model["y_fit"].tolist(),
                "residual": block4_model["residual"].tolist(),
                "mu_coal": block4_model["mu_coal"],
                "delta_mu": block4_model["mu_coal_stderr"],
                "mu_pm": block4_model["mu_coal_pm"],
                "slope": float(block4_model["slope"]),
                "intercept": float(block4_model["intercept"]),
            },
            "intensity_plot_image": intensity_plot_image,
            "mu_plot_image": mu_plot_image,
            "linear_regression_plot_image": linear_regression_plot_image,
            "attenuation_matrix_rows": attenuation_matrix_rows,
            "attenuation_fit_rows": attenuation_fit_rows,
            "comparison_image": mu_plot_image,
            "summary": summary,
        }

    except Exception as e:
        raise ValueError(f"Block differential attenuation analysis failed: {str(e)}") from e


__all__ = [
    "process_blocks",
    "analyze_block_histograms",
    "subdivide_blocks",
    "analyze_subdivision_histograms",
    "visualize_block_invalid_roi",
    "compare_blocks_1_vs_3",
]

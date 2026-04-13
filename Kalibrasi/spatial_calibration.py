# Distortion Correction + Perspective Rectification
# Automatic dot-grid detection → calibration → straightening

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ── 1. Robust blob detection ──────────────────────────────────────
def _detect_blobs(gray):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 100000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)

    kps = detector.detect(gray)
    kps_inv = detector.detect(255 - gray)
    if len(kps_inv) > len(kps):
        kps = kps_inv

    if len(kps) > 4:
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        sizes = np.array([kp.size for kp in kps])
        med_size = np.median(sizes)
        mask = (sizes > med_size * 0.3) & (sizes < med_size * 3.0)
        pts = pts[mask]
        print(f"  Blob detector: {len(kps)} raw -> {pts.shape[0]} after size filter (median size {med_size:.1f})")
        return pts

    # Fallback: adaptive threshold + contours
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers, areas = [], []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 10:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        areas.append(a)
    if not centers:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.array(centers, dtype=np.float32)
    areas = np.array(areas)
    med_a = np.median(areas)
    mask = (areas > med_a * 0.2) & (areas < med_a * 5.0)
    pts = pts[mask]
    print(f"  Contour fallback: {len(centers)} raw -> {pts.shape[0]} after area filter")
    return pts


# ── 2. Grid shape via histogram peaks ─────────────────────────────
def _infer_grid_shape(pts):
    N = pts.shape[0]
    if N < 4:
        return (0, 0)

    tree = cKDTree(pts)
    dd, _ = tree.query(pts, k=2)
    nn_dists = dd[:, 1]
    grid_spacing = np.median(nn_dists)
    print(f"  Grid spacing ~ {grid_spacing:.1f} px")

    bw = grid_spacing * 0.4

    def count_peaks(vals, bw):
        lo, hi = vals.min() - bw, vals.max() + bw
        bins = np.arange(lo, hi + bw, bw)
        hist, _ = np.histogram(vals, bins=bins)
        expected = N / max(len(bins), 1)
        threshold = max(expected * 0.2, 2)
        peaks = 0
        in_peak = False
        for h in hist:
            if h >= threshold and not in_peak:
                peaks += 1
                in_peak = True
            elif h < threshold:
                in_peak = False
        return peaks

    cols = count_peaks(pts[:, 0], bw)
    rows = count_peaks(pts[:, 1], bw)
    print(f"  Histogram peaks -> cols={cols}, rows={rows}")
    return (cols, rows)


# ── 3. Order raw points into a grid (fallback) ───────────────────
def _order_points_as_grid(pts, cols, rows):
    grid_spacing = np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])
    thresh = grid_spacing * 0.5

    order = np.argsort(pts[:, 1])
    row_groups = []
    cur = [order[0]]
    mean_y = pts[order[0], 1]
    for i in range(1, len(pts)):
        idx = order[i]
        if abs(pts[idx, 1] - mean_y) <= thresh:
            cur.append(idx)
            mean_y = np.mean(pts[np.array(cur), 1])
        else:
            row_groups.append(cur)
            cur = [idx]
            mean_y = pts[idx, 1]
    row_groups.append(cur)

    good_rows = [r for r in row_groups if len(r) == cols]
    if len(good_rows) < rows:
        row_groups.sort(key=lambda r: abs(len(r) - cols))
        good_rows = row_groups[:rows]
    good_rows = good_rows[:rows]
    good_rows.sort(key=lambda r: np.mean(pts[np.array(r), 1]))

    ordered = []
    for r in good_rows:
        r_pts = pts[np.array(r)]
        x_order = np.argsort(r_pts[:, 0])
        for xi in x_order[:cols]:
            ordered.append(r_pts[xi])

    ordered = np.array(ordered, dtype=np.float32)
    return ordered.reshape(-1, 1, 2)


# ── 4. Main calibration + rectification ──────────────────────────
def distortion_params(dot_matrix_image_path, output_path="distortion.npz"):
    img_color = cv2.imread(dot_matrix_image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Cannot read: {dot_matrix_image_path}")
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Detect dots
    pts = _detect_blobs(gray)
    N = pts.shape[0]
    print(f"Total detected dots: {N}")
    if N < 9:
        print("Too few dots detected - aborting")
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.show()
        return

    # Infer grid shape
    cols, rows_n = _infer_grid_shape(pts)
    print(f"Pattern size: ({cols}, {rows_n})  ({cols * rows_n} dots expected)")
    pattern_size = (int(cols), int(rows_n))

    # Try findCirclesGrid
    found, centers = False, None
    if pattern_size[0] >= 3 and pattern_size[1] >= 3:
        for ps in [pattern_size, (pattern_size[1], pattern_size[0])]:
            for img_try in [gray, 255 - gray]:
                ret, c = cv2.findCirclesGrid(img_try, ps,
                                             flags=cv2.CALIB_CB_SYMMETRIC_GRID)
                if ret:
                    pattern_size = ps
                    centers = c
                    found = True
                    break
            if found:
                break

    if found:
        print(f"findCirclesGrid succeeded  pattern={pattern_size}")
    else:
        print("findCirclesGrid failed - falling back to raw-point ordering")
        centers = _order_points_as_grid(pts, cols, rows_n)
        print(f"  Ordered {centers.shape[0]} points into {cols}x{rows_n} grid")

    # Camera calibration
    n_expected = pattern_size[0] * pattern_size[1]
    objp = np.zeros((n_expected, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    if centers.shape[0] != n_expected:
        print(f"  WARNING: expected {n_expected} centres but got {centers.shape[0]}; trimming")
        n_use = min(centers.shape[0], n_expected)
        centers = centers[:n_use]
        objp = objp[:n_use]

    ret_cal, mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [centers], (w, h), None, None
    )
    print(f"Calibration RMS reprojection error: {ret_cal:.4f}")

    # Undistort without cropping
    undist = cv2.undistort(img_color, mtx, dist_coeffs, None, mtx)

    # Homography: straighten the grid without cropping
    pts_undist = cv2.undistortPoints(centers, mtx, dist_coeffs, P=mtx)
    pts_undist = pts_undist.reshape(-1, 2)

    # Use same centroid and spacing as detected dots
    det_cx = np.mean(pts_undist[:, 0])
    det_cy = np.mean(pts_undist[:, 1])
    tree_u = cKDTree(pts_undist)
    dd_u, _ = tree_u.query(pts_undist, k=2)
    spacing = np.median(dd_u[:, 1])

    x0 = det_cx - spacing * (pattern_size[0] - 1) / 2
    y0 = det_cy - spacing * (pattern_size[1] - 1) / 2
    ideal = np.zeros((pts_undist.shape[0], 2), np.float32)
    for j in range(pattern_size[1]):
        for i in range(pattern_size[0]):
            idx = j * pattern_size[0] + i
            if idx < ideal.shape[0]:
                ideal[idx] = [x0 + i * spacing, y0 + j * spacing]

    n_use = min(pts_undist.shape[0], ideal.shape[0])
    homography, _ = cv2.findHomography(pts_undist[:n_use], ideal[:n_use])
    rectified = cv2.warpPerspective(undist, homography, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)
    print("Homography applied - grid straightened (no crop)")

    # Save parameters
    np.savez(
        output_path,
        mtx=mtx,
        dist=dist_coeffs,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        homography=homography,
        pattern_size=np.array(pattern_size),
    )
    print(f"Parameters saved -> {output_path}")

    # Visualization (2x3)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.ravel()

    axs[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    axs[0].set_title("1 - Raw Image")
    axs[0].axis("off")

    vis1 = img_color.copy()
    for x, y in pts:
        cv2.circle(vis1, (int(x), int(y)), 5, (0, 255, 0), 2)
    axs[1].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"2 - Detected Dots ({N})")
    axs[1].axis("off")

    vis2 = img_color.copy()
    for c in centers.reshape(-1, 2):
        cv2.circle(vis2, (int(c[0]), int(c[1])), 5, (255, 0, 0), 2)
    axs[2].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axs[2].set_title(f"3 - Ordered Grid ({pattern_size[0]}x{pattern_size[1]})")
    axs[2].axis("off")

    axs[3].imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    axs[3].set_title("4 - Lens Undistorted")
    axs[3].axis("off")

    axs[4].imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
    axs[4].set_title("5 - Rectified (straight grid)")
    axs[4].axis("off")

    vis3 = rectified.copy()
    for j in range(pattern_size[1]):
        yl = int(y0 + j * spacing)
        cv2.line(vis3, (int(x0), yl),
                 (int(x0 + (pattern_size[0] - 1) * spacing), yl), (0, 0, 255), 1)
    for i in range(pattern_size[0]):
        xl = int(x0 + i * spacing)
        cv2.line(vis3, (xl, int(y0)),
                 (xl, int(y0 + (pattern_size[1] - 1) * spacing)), (0, 0, 255), 1)
    axs[5].imshow(cv2.cvtColor(vis3, cv2.COLOR_BGR2RGB))
    axs[5].set_title("6 - Rectified + Grid Lines")
    axs[5].axis("off")

    plt.tight_layout()
    plt.show()
    return {"pattern_size": pattern_size, "dots": N, "rms": ret_cal}


# ── Apply distortion correction to image ──────────────────────────

def correct_image(img_path, cal_path, out_path=None,
                  crop=(70, 70, 70, 50), show=False):
    """
    Apply lens-distortion correction + perspective rectification + cropping.

    Parameters
    ----------
    img_path : str
        Path to the input image.
    cal_path : str
        Path to the .npz calibration file (must contain mtx, dist, homography).
    out_path : str or None
        Where to save the corrected image. If None, not saved.
    crop : tuple (top, left, right, bottom)
        Pixels to crop from each side. Set to (0,0,0,0) to skip cropping.
    show : bool
        If True, display before/after comparison.

    Returns
    -------
    corrected : np.ndarray
        The corrected (and optionally cropped) image.
    """
    # Load calibration parameters
    cal = np.load(cal_path, allow_pickle=True)
    mtx = cal["mtx"]
    dist = cal["dist"]
    homography = cal["homography"]

    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    h, w = img.shape[:2]

    # Step 1: Undistort (remove lens distortion)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # Step 2: Apply homography (perspective rectification)
    corrected = cv2.warpPerspective(undistorted, homography, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)

    # Step 3: Crop
    ct, cl, cr, cb = crop
    if any(v > 0 for v in crop):
        cropped = corrected[ct:h - cb, cl:w - cr]
    else:
        cropped = corrected

    # Save
    if out_path is not None:
        cv2.imwrite(out_path, cropped)
        print(f"Saved -> {out_path}  ({cropped.shape[1]}x{cropped.shape[0]})")

    # Visualization
    if show:
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))

        def _show(ax, image, title):
            if image.ndim == 2:
                ax.imshow(image, cmap="gray")
            elif image.dtype == np.uint16:
                disp = (image / 256).astype(np.uint8)
                ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")

        _show(axs[0, 0], img, "1 - Original")
        _show(axs[0, 1], undistorted, "2 - Undistorted")
        _show(axs[1, 0], corrected, "3 - Corrected (undistort + rectify)")
        _show(axs[1, 1], cropped, f"4 - Cropped ({cropped.shape[1]}x{cropped.shape[0]})")
        plt.tight_layout()
        plt.show()

    return cropped


# ── Run ───────────────────────────────────────────────────────────
# dot_matrix_image_path = r"C:\Users\arief\Downloads\Processed kalibrator.tiff"
# output_path = r"C:\Users\arief\Downloads\Processed kalibrator_distortion.npz"

# result = distortion_params(dot_matrix_image_path, output_path)
# print(result)

# ── Example usage ─────────────────────────────────────────────────
cal_path = r"Kalibrasi/calibration parameter.npz"
img_path = r"1771914199828_processedimage.tiff"
out_path = r"1771914199828_processedimage_corrected.tiff"

corrected = correct_image(img_path, cal_path, out_path)

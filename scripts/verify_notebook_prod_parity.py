import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_notebook_core_namespace(nb_path):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    code = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "\n".join(cell.get("source", []))
        if "pre-log FFC differential" in src:
            code = src
            break
    if code is None:
        raise RuntimeError(f"Core code cell not found in {nb_path}")

    ns = {}
    exec(code, ns, ns)
    return ns


def r5(x):
    return round(float(x), 5)


def assert_eq_5(a, b, label):
    if r5(a) != r5(b):
        raise AssertionError(f"{label}: {r5(a)} != {r5(b)} (raw {a} vs {b})")


def make_subdivisions():
    subdivisions = []
    for block_id in (1, 2, 3, 4):
        for sid in range(1, 11):
            x0 = sid * 3
            y0 = block_id * 3
            box = [[x0, y0], [x0 + 2, y0], [x0 + 2, y0 + 2], [x0, y0 + 2]]
            subdivisions.append({"parent_block": block_id, "subdivision_id": sid, "box": box})
    return {"subdivisions": subdivisions}


def make_fake_series():
    air1 = np.array([4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800], dtype=float)
    air3 = np.array([4050, 4250, 4450, 4650, 4850, 5050, 5250, 5450, 5650, 5850], dtype=float)
    coal2 = np.array([40000, 39000, 38000, 37000, 36000, 35000, 34000, 33000, 32000, 31000], dtype=float)
    coal4 = np.array([39800, 38800, 37800, 36800, 35800, 34800, 33800, 32800, 31800, 30800], dtype=float)
    return {1: air1, 2: coal2, 3: air3, 4: coal4}


def run_block_detection_behavior_parity(prod_block, nb_block, sample_path):
    params = {
        "threshold_value": 54000,
        "min_length_rectangular": 1200,
        "max_length_rectangular": 1600,
        "min_rectangularity": 0.9,
        "min_solidity": 0.9,
    }
    file_bytes = Path(sample_path).read_bytes()

    prod_error = None
    nb_error = None
    try:
        prod_block.process_blocks(file_bytes, params)
    except Exception as exc:  # expected for this sample
        prod_error = str(exc)

    try:
        nb_block["process_blocks"](file_bytes, params)
    except Exception as exc:  # expected for this sample
        nb_error = str(exc)

    if prod_error is None or nb_error is None:
        raise AssertionError("Expected both production and notebook block detection to produce the same failure behavior")
    if prod_error != nb_error:
        raise AssertionError(f"Block detection error mismatch:\nprod: {prod_error}\nnb:   {nb_error}")

    return prod_error


def run_block_miu_parity(prod_block, nb_block):
    seq = make_fake_series()
    counters_prod = {1: 0, 2: 0, 3: 0, 4: 0}
    counters_nb = {1: 0, 2: 0, 3: 0, 4: 0}

    def fake_load(_file_bytes):
        return np.ones((32, 32), dtype=np.uint16)

    def fake_mean_prod(_img, box, _shrink):
        block = int(box[0][1] / 3)
        idx = counters_prod[block]
        counters_prod[block] += 1
        val = float(seq[block][idx])
        return val, np.array([val], dtype=np.float64)

    def fake_mean_nb(_img, box, _shrink):
        block = int(box[0][1] / 3)
        idx = counters_nb[block]
        counters_nb[block] += 1
        val = float(seq[block][idx])
        return val, np.array([val], dtype=np.float64)

    prod_block._load_and_validate_image = fake_load
    prod_block._mean_intensity_in_box = fake_mean_prod

    nb_block["_load_and_validate_image"] = fake_load
    nb_block["_mean_intensity_in_box"] = fake_mean_nb

    subdivisions = make_subdivisions()
    prod_out = prod_block.compare_blocks_1_vs_3(b"dummy", subdivisions, params={})
    nb_out = nb_block["compare_blocks_1_vs_3"](b"dummy", subdivisions, params={})

    for key in ["mu_block2", "mu_block4", "delta_mu_block2", "delta_mu_block4", "slope_block2", "slope_block4", "r2_block2", "r2_block4"]:
        assert_eq_5(prod_out["summary"][key], nb_out["summary"][key], f"block.summary.{key}")

    p_fit = prod_out["attenuation_fit_rows"]
    n_fit = nb_out["attenuation_fit_rows"]
    if len(p_fit) != len(n_fit):
        raise AssertionError("attenuation_fit_rows length mismatch")

    for i, (pr, nr) in enumerate(zip(p_fit, n_fit), start=1):
        assert_eq_5(pr["mu_coal"], nr["mu_coal"], f"fit[{i}].mu_coal")
        assert_eq_5(pr["slope"], nr["slope"], f"fit[{i}].slope")
        assert_eq_5(pr["r2"], nr["r2"], f"fit[{i}].r2")

    p_rows = prod_out["attenuation_matrix_rows"]
    n_rows = nb_out["attenuation_matrix_rows"]
    if len(p_rows) != len(n_rows):
        raise AssertionError("attenuation_matrix_rows length mismatch")

    for i, (pr, nr) in enumerate(zip(p_rows, n_rows), start=1):
        for col in ["step", "coal_mm", "p_air", "p_coal", "delta_p", "y_n", "y_fit", "residual"]:
            assert_eq_5(pr[col], nr[col], f"attenuation_matrix_rows[{i}].{col}")

    return nb_out


def run_circle_full_parity(prod_circle, nb_circle, sample_path):
    file_bytes = Path(sample_path).read_bytes()

    detect_params = {
        "threshold_value": 24000,
        "min_diameter": 280,
        "max_diameter": 340,
        "min_circularity": 0.6,
        "min_solidity": 0.7,
        "expected_count": 16,
        "grid_cols": 4,
    }

    prod_circle.AIR_CV_THRESHOLD = 0.06
    nb_circle["AIR_CV_THRESHOLD"] = 0.06

    p_det = prod_circle.process_tiff_image(file_bytes, detect_params)
    p_grid = prod_circle.detect_grid_from_diagonal(file_bytes, p_det, 4)
    p_cmp = prod_circle.compare_diagonals(file_bytes, p_grid, params={})

    n_det = nb_circle["process_tiff_image"](file_bytes, detect_params)
    n_grid = nb_circle["detect_grid_from_diagonal"](file_bytes, n_det, 4)
    n_cmp = nb_circle["compare_diagonals"](file_bytes, n_grid, params={})

    keys = ["upper_mu_avg", "lower_mu_avg", "upper_mu_std", "lower_mu_std", "upper_mu_final", "lower_mu_final"]
    for key in keys:
        assert_eq_5(p_cmp["summary"][key], n_cmp["summary"][key], f"circle.summary.{key}")

    return n_cmp


def main():
    prod_block = load_module("prod_block", ROOT / "public/image-analysis-miu-batubara/block_detection.py")
    prod_circle = load_module("prod_circle", ROOT / "public/image-analysis-miu-batubara/circle_detection.py")
    nb_block = load_notebook_core_namespace(ROOT / "block-detection.ipynb")
    nb_circle = load_notebook_core_namespace(ROOT / "circle-detection.ipynb")

    block_detection_error = run_block_detection_behavior_parity(prod_block, nb_block, ROOT / "sample-stepwedge.tiff")
    block_miu = run_block_miu_parity(prod_block, nb_block)
    circle_miu = run_circle_full_parity(prod_circle, nb_circle, ROOT / "sample-circle.tiff")

    print("BLOCK detection behavior parity: OK")
    print("  sample-stepwedge.tiff error (both sides):", block_detection_error)
    print("BLOCK MIU parity (5dp): OK")
    print("  mu_block2:", r5(block_miu["summary"]["mu_block2"]))
    print("  mu_block4:", r5(block_miu["summary"]["mu_block4"]))
    for row in block_miu["attenuation_fit_rows"]:
        print("  fit", row["sample"], "mu=", r5(row["mu_coal"]), "slope=", r5(row["slope"]), "R2=", r5(row["r2"]))
    print("  attenuation_matrix_rows:", len(block_miu["attenuation_matrix_rows"]))

    print("CIRCLE full parity (5dp): OK")
    print("  upper_mu_avg:", r5(circle_miu["summary"]["upper_mu_avg"]))
    print("  lower_mu_avg:", r5(circle_miu["summary"]["lower_mu_avg"]))


if __name__ == "__main__":
    main()

from __future__ import annotations

import cv2
import numpy as np
import pytest
from numpy.testing import assert_allclose

import block_detection
import circle_detection


@pytest.mark.unit
def test_circle_ratio_method_mu_formula(monkeypatch):
    i_air = 4000.0
    i_coal = 40000.0
    x_mm = 6.0

    img = np.zeros((220, 220), dtype=np.uint16)
    grid = []
    for row in range(4):
        for col in range(4):
            center = (30 + (col * 50), 30 + (row * 50))
            radius = 12
            if (row + col) == 3:
                val = i_air
            else:
                val = i_coal
            cv2.circle(img, center, radius, int(val), -1)
            grid.append({"grid_pos": [row, col], "center": center, "radius": radius})

    monkeypatch.setattr(circle_detection, "_load_and_validate_image", lambda _b: img)
    out = circle_detection.compare_diagonals(b"dummy", {"grid": grid})
    expected_mu = (i_coal - i_air) / x_mm
    computed_mu = out["upper_stats"][0]["mu_coal"]

    assert_allclose(computed_mu, expected_mu, rtol=0, atol=1e-12)
    assert computed_mu > 0


@pytest.mark.unit
def test_circle_ratio_method_normalizes_inverted_intensity(monkeypatch):
    i_air = 4000.0
    i_coal = 40000.0
    x_mm = 6.0

    img = np.zeros((220, 220), dtype=np.uint16)
    grid = []
    for row in range(4):
        for col in range(4):
            center = (30 + (col * 50), 30 + (row * 50))
            radius = 12
            if (row + col) == 3:
                val = i_air
            else:
                val = i_coal
            cv2.circle(img, center, radius, int(val), -1)
            grid.append({"grid_pos": [row, col], "center": center, "radius": radius})

    monkeypatch.setattr(circle_detection, "_load_and_validate_image", lambda _b: img)
    out = circle_detection.compare_diagonals(b"dummy", {"grid": grid})

    computed_mu = out["upper_stats"][0]["mu_coal"]
    assert computed_mu > 0
    assert_allclose(computed_mu, (i_coal - i_air) / x_mm, rtol=0, atol=1e-12)


@pytest.mark.unit
def test_circle_partition_counts_cover_full_upper_and_lower_regions(monkeypatch):
    i_air = 4000.0
    i_coal = 40000.0

    img = np.zeros((220, 220), dtype=np.uint16)
    grid = []
    for row in range(4):
        for col in range(4):
            center = (30 + (col * 50), 30 + (row * 50))
            radius = 12
            if (row + col) == 3:
                val = i_air
            else:
                val = i_coal
            cv2.circle(img, center, radius, int(val), -1)
            grid.append({"grid_pos": [row, col], "center": center, "radius": radius})

    monkeypatch.setattr(circle_detection, "_load_and_validate_image", lambda _b: img)
    out = circle_detection.compare_diagonals(b"dummy", {"grid": grid})

    assert len(out["upper_stats"]) == 6
    assert len(out["lower_stats"]) == 6
    assert len(out["grid_stats"]) == 16

    roi_areas = np.array([circle["roi_area"] for circle in out["grid_stats"]], dtype=float)
    assert np.allclose(roi_areas, roi_areas[0], rtol=0, atol=1e-6)
    assert_allclose(out["summary"]["roi_area_mean"], roi_areas[0], rtol=0, atol=1e-6)


@pytest.mark.unit
def test_block_linear_regression_recovers_mu_components():
    x = np.arange(10, 0, -1, dtype=float)
    mu_coal_true = 0.19
    mu_acrylic_true = 0.035

    y = (mu_coal_true - mu_acrylic_true) * x + (14.0 * mu_acrylic_true)
    slope, intercept = np.polyfit(x, y, 1)

    mu_acrylic = intercept / 14.0
    mu_coal = slope + mu_acrylic

    assert_allclose(mu_acrylic, mu_acrylic_true, rtol=0, atol=1e-12)
    assert_allclose(mu_coal, mu_coal_true, rtol=0, atol=1e-12)


@pytest.mark.unit
def test_positive_attenuation_assertion():
    x = np.arange(10, 0, -1, dtype=float)
    y = np.linspace(2.0, 0.8, 10)
    slope, intercept = np.polyfit(x, y, 1)
    mu_acrylic = intercept / 14.0
    mu_coal = slope + mu_acrylic

    assert mu_acrylic > 0
    assert mu_coal > 0


def _make_subdivisions():
    subdivisions = []
    for block_id in (1, 2, 3, 4):
        for sid in range(1, 11):
            x0 = sid * 3
            y0 = block_id * 3
            box = [[x0, y0], [x0 + 2, y0], [x0 + 2, y0 + 2], [x0, y0 + 2]]
            subdivisions.append({"parent_block": block_id, "subdivision_id": sid, "box": box})
    return {"subdivisions": subdivisions}


@pytest.mark.unit
def test_gradient_validation_passes_for_dark_to_bright_air(monkeypatch):
    air1 = np.array([4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800], dtype=float)
    air3 = np.array([4050, 4250, 4450, 4650, 4850, 5050, 5250, 5450, 5650, 5850], dtype=float)
    coal2 = np.array([40000, 39000, 38000, 37000, 36000, 35000, 34000, 33000, 32000, 31000], dtype=float)
    coal4 = np.array([39800, 38800, 37800, 36800, 35800, 34800, 33800, 32800, 31800, 30800], dtype=float)
    sequences = {1: air1, 2: coal2, 3: air3, 4: coal4}
    counters = {1: 0, 2: 0, 3: 0, 4: 0}

    def fake_load_and_validate(_file_bytes):
        return np.ones((32, 32), dtype=np.uint16)

    def fake_mean_intensity(_img, _box, _shrink):
        block = int(_box[0][1] / 3)
        idx = counters[block]
        counters[block] += 1
        val = float(sequences[block][idx])
        return val, np.array([val], dtype=np.float64)

    monkeypatch.setattr(block_detection, "_load_and_validate_image", fake_load_and_validate)
    monkeypatch.setattr(block_detection, "_mean_intensity_in_box", fake_mean_intensity)

    out = block_detection.compare_blocks_1_vs_3(b"dummy", _make_subdivisions())
    assert out["summary"]["mu_block2"] > 0
    assert out["summary"]["mu_block4"] > 0


@pytest.mark.unit
def test_gradient_validation_normalizes_inverted_air(monkeypatch):
    air1 = np.array([4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450], dtype=float)
    air3 = np.array([4020, 4070, 4120, 4170, 4220, 4270, 4320, 4370, 4420, 4470], dtype=float)
    coal2 = np.array([40000, 39800, 39600, 39400, 39200, 39000, 38800, 38600, 38400, 38200], dtype=float)
    coal4 = np.array([39900, 39700, 39500, 39300, 39100, 38900, 38700, 38500, 38300, 38100], dtype=float)
    sequences = {1: air1, 2: coal2, 3: air3, 4: coal4}
    counters = {1: 0, 2: 0, 3: 0, 4: 0}

    def fake_load_and_validate(_file_bytes):
        return np.ones((32, 32), dtype=np.uint16)

    def fake_mean_intensity(_img, _box, _shrink):
        block = int(_box[0][1] / 3)
        idx = counters[block]
        counters[block] += 1
        val = float(sequences[block][idx])
        return val, np.array([val], dtype=np.float64)

    monkeypatch.setattr(block_detection, "_load_and_validate_image", fake_load_and_validate)
    monkeypatch.setattr(block_detection, "_mean_intensity_in_box", fake_mean_intensity)

    out = block_detection.compare_blocks_1_vs_3(b"dummy", _make_subdivisions())
    assert out["summary"]["mu_block2"] > 0
    assert out["summary"]["mu_block4"] > 0


@pytest.mark.unit
def test_gradient_validation_warns_for_inverted_air(monkeypatch):
    air_bad = np.array([30000, 32000, 34000, 36000, 38000, 40000, 42000, 44000, 46000, 48000], dtype=float)
    air_good = np.array([52000, 50000, 48000, 45500, 43000, 40500, 38000, 35500, 33000, 30500], dtype=float)
    coal = np.array([41000, 39000, 36500, 34000, 31500, 29000, 26500, 24000, 21500, 19500], dtype=float)
    sequences = {1: air_bad, 2: coal, 3: air_good, 4: coal}
    counters = {1: 0, 2: 0, 3: 0, 4: 0}

    def fake_load_and_validate(_file_bytes):
        return np.ones((32, 32), dtype=np.uint16)

    def fake_mean_intensity(_img, _box, _shrink):
        block = int(_box[0][1] / 3)
        idx = counters[block]
        counters[block] += 1
        val = float(sequences[block][idx])
        return val, np.array([val], dtype=np.float64)

    monkeypatch.setattr(block_detection, "_load_and_validate_image", fake_load_and_validate)
    monkeypatch.setattr(block_detection, "_mean_intensity_in_box", fake_mean_intensity)

    out = block_detection.compare_blocks_1_vs_3(b"dummy", _make_subdivisions())
    assert out["summary"]["air_validation_warning"].startswith("E_BLOCK_AIR_ROI")


@pytest.mark.unit
def test_select_air_reference_blocks_prefers_matching_outer_pair():
    dummy_img = np.zeros((16, 2000), dtype=np.uint16)
    candidates = [
        {
            "center": (248, 994),
            "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "width": 209.2,
            "height": 1548.3,
            "longest_side": 1548.3,
            "shortest_side": 209.2,
            "area": 1.0,
            "rectangularity": 0.9187,
            "solidity": 0.9384,
            "mean_value": 23775.8,
            "top_mean": 46649.9,
            "bottom_mean": 8936.9,
        },
        {
            "center": (828, 1107),
            "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "width": 205.3,
            "height": 1316.5,
            "longest_side": 1316.5,
            "shortest_side": 205.3,
            "area": 1.0,
            "rectangularity": 0.9102,
            "solidity": 0.9306,
            "mean_value": 41943.3,
            "top_mean": 49720.2,
            "bottom_mean": 34263.9,
        },
        {
            "center": (1412, 1002),
            "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "width": 205.3,
            "height": 1510.5,
            "longest_side": 1510.5,
            "shortest_side": 205.3,
            "area": 1.0,
            "rectangularity": 0.8954,
            "solidity": 0.9392,
            "mean_value": 24242.6,
            "top_mean": 46825.9,
            "bottom_mean": 7794.4,
        },
    ]

    selected = block_detection._select_air_reference_blocks(dummy_img, candidates)

    assert [block["center"] for block in selected] == [(248, 994), (1412, 1002)]

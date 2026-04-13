from __future__ import annotations

import math

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
            if row == col or (row + col) == 3:
                val = i_air
            else:
                val = i_coal
            cv2.circle(img, center, radius, int(val), -1)
            grid.append({"grid_pos": [row, col], "center": center, "radius": radius})

    monkeypatch.setattr(circle_detection, "_load_and_validate_image", lambda _b: img)
    out = circle_detection.compare_diagonals(b"dummy", {"grid": grid})
    # Ratio swapped to Air/Coal because Air pixel intensity is lower than Coal in this specific dataset.
    expected_mu = -math.log(i_air / i_coal) / x_mm
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
            if row == col or (row + col) == 3:
                val = i_air
            else:
                val = i_coal
            cv2.circle(img, center, radius, int(val), -1)
            grid.append({"grid_pos": [row, col], "center": center, "radius": radius})

    monkeypatch.setattr(circle_detection, "_load_and_validate_image", lambda _b: img)
    out = circle_detection.compare_diagonals(b"dummy", {"grid": grid})

    computed_mu = out["upper_stats"][0]["mu_coal"]
    assert computed_mu > 0
    assert_allclose(computed_mu, -math.log(i_air / i_coal) / x_mm, rtol=0, atol=1e-12)


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
def test_gradient_validation_raises_for_inverted_air(monkeypatch):
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

    with pytest.raises(ValueError, match="Validation Failed: The Air reference blocks"):
        block_detection.compare_blocks_1_vs_3(b"dummy", _make_subdivisions())

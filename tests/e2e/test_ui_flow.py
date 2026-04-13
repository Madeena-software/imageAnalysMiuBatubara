from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


pytestmark = [pytest.mark.e2e]


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_URL_PATH = "/public/image-analysis-miu-batubara/index.html"
SERVER_STARTUP_TIMEOUT_SECONDS = 20


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="session")
def e2e_base_url():
    if os.getenv("RUN_E2E") != "1":
        pytest.skip("Set RUN_E2E=1 to run Playwright E2E tests.")

    port = _free_port()
    cmd = ["python", "-m", "http.server", str(port), "--bind", "127.0.0.1"]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + SERVER_STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.2)
        else:
            raise RuntimeError("Local HTTP server failed to start for E2E tests.")
        yield f"http://127.0.0.1:{port}{APP_URL_PATH}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture()
def browser_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        messages = {"console_errors": [], "page_errors": []}

        def on_console(msg):
            text = msg.text or ""
            if msg.type == "error" or ("Traceback" in text or "Error processing image" in text):
                messages["console_errors"].append(text)

        page.on("console", on_console)
        page.on("pageerror", lambda exc: messages["page_errors"].append(str(exc)))
        yield page, messages
        browser.close()


def _wait_pyscript_ready(page):
    page.wait_for_selector("#initLoadingOverlay.hidden", timeout=180000)
    page.wait_for_selector("#processBtn", timeout=30000)


def _assert_no_runtime_errors(messages):
    assert messages["page_errors"] == []
    assert messages["console_errors"] == []


def _assert_error_container_empty(page):
    err = page.locator("#errorMessage")
    assert (err.text_content() or "").strip() == ""
    assert "active" not in (err.get_attribute("class") or "")


def _assert_data_img_visible(page, selector):
    loc = page.locator(selector)
    loc.wait_for(timeout=120000)
    src = loc.get_attribute("src")
    assert src is not None and src.startswith("data:image/png;base64,")
    assert loc.is_visible()


def _assert_not_empty(page, selector):
    loc = page.locator(selector)
    loc.wait_for(timeout=120000)
    text = (loc.text_content() or "").strip()
    has_table = loc.locator("table").count() > 0
    assert text != "" or has_table
    assert loc.is_visible()


def _assert_parent_details_open(page, target_selector):
    is_open = page.eval_on_selector(target_selector, "el => !!el.closest('details')?.open")
    assert is_open is True


def test_circle_detection_ui_flow(e2e_base_url, browser_page):
    page, messages = browser_page
    page.goto(e2e_base_url, wait_until="domcontentloaded", timeout=120000)
    _wait_pyscript_ready(page)

    page.click("#circleModeBtn")
    assert "active" in (page.get_attribute("#circleModeBtn", "class") or "")

    file_path = REPO_ROOT / "Wadah Silinder Baru_1770011538520_processedimage.tiff"
    page.set_input_files("#fileInput", str(file_path))
    page.wait_for_timeout(1000)
    page.click("#processBtn")

    page.wait_for_selector("#resultsSection.active", timeout=180000)
    _assert_error_container_empty(page)

    _assert_data_img_visible(page, "#histogramImage img")
    _assert_data_img_visible(page, "#muPlotImage")
    _assert_not_empty(page, "#diagonalSummary")
    page.locator("#resultsSection .export-section").wait_for(timeout=120000)
    assert page.locator("#resultsSection .export-section").is_visible()
    assert page.locator("#exportCirclePdfBtn").is_visible()
    assert page.locator("#exportCircleImagesBtn").is_visible()
    _assert_parent_details_open(page, "#histogramImage")
    _assert_parent_details_open(page, "#circleFinalMuPlot")

    _assert_no_runtime_errors(messages)


def test_block_detection_ui_flow(e2e_base_url, browser_page):
    page, messages = browser_page
    page.goto(e2e_base_url, wait_until="domcontentloaded", timeout=120000)
    _wait_pyscript_ready(page)

    page.click("#blockModeBtn")
    assert "active" in (page.get_attribute("#blockModeBtn", "class") or "")

    file_path = REPO_ROOT / "1771914199828_processedimage.tiff"
    page.set_input_files("#fileInput", str(file_path))
    page.wait_for_timeout(1000)
    page.click("#processBlockBtn")

    page.wait_for_selector("#blockResultsSection.active", timeout=180000)
    _assert_error_container_empty(page)

    _assert_data_img_visible(page, "#subdivisionImage img")
    _assert_data_img_visible(page, "#blockComparisonImage img")
    _assert_parent_details_open(page, "#blockComparisonImage")

    _assert_no_runtime_errors(messages)

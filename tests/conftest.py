from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "public" / "image-analysis-miu-batubara"
CIRCLE_GOLDEN = REPO_ROOT / "Wadah Silinder Baru_1770011538520_processedimage.tiff"
BLOCK_GOLDEN = REPO_ROOT / "1771914199828_processedimage_corrected.tiff"


if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def circle_golden_path() -> Path:
    return CIRCLE_GOLDEN


@pytest.fixture(scope="session")
def block_golden_path() -> Path:
    return BLOCK_GOLDEN


@pytest.fixture(scope="session")
def circle_golden_bytes(circle_golden_path: Path) -> bytes:
    return circle_golden_path.read_bytes()


@pytest.fixture(scope="session")
def block_golden_bytes(block_golden_path: Path) -> bytes:
    return block_golden_path.read_bytes()


@pytest.fixture(scope="session")
def circle_golden_base64(circle_golden_bytes: bytes) -> str:
    return base64.b64encode(circle_golden_bytes).decode("ascii")


@pytest.fixture(scope="session")
def block_golden_base64(block_golden_bytes: bytes) -> str:
    return base64.b64encode(block_golden_bytes).decode("ascii")

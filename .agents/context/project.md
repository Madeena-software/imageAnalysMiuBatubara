<!-- antigravity-code-agent-template:managed -->
# Project Context

**Status:** Initialized
**Last verified:** 2026-07-21
**Repository checkpoint:** Unknown

Run `onboard-repository` before relying on this file. Every durable claim must identify repository evidence or a successful command. Preserve the distinction between verified current behavior, proposed behavior, superseded facts, and unknowns.

## Purpose

A web-based TIFF image analyzer with circle detection and grid analysis, powered by PyScript to run Python directly in the browser without any backend server. Provides drag & drop file upload and interactive visual adjustments. (Evidence: `README.md`)

## Intended users

Laboratory sample analysis, quality control inspection, pattern recognition research, automated measurement systems, and object counting/classification. (Evidence: `README.md` -> Use Cases)

## Current capabilities and flows

- Drag and drop or browse to upload TIFF files.
- Adjustable parameters for detection (threshold, min/max area, circularity, solidity, grid columns).
- Client-side image processing utilizing PyScript and OpenCV.
- Visualizing results (detected circles, threshold mask, statistics table).
- Generating grid representations of the detected items.
- Works entirely within a modern browser without server-side processing dependencies.
(Evidence: `README.md` and `public/image-analysis-miu-batubara/index.html`)

## Technology stack

- **Frontend:** HTML5, CSS3, JavaScript (pdf_exporter.js)
- **Runtime:** PyScript (Python running in browser)
- **Dependencies (Python):** OpenCV, NumPy, Matplotlib, Pillow, fpdf2, scipy, pytest (Evidence: `requirements.txt`)
- **Testing:** Pytest, Playwright for E2E tests.

## Architecture and entry points

- `public/image-analysis-miu-batubara/index.html`: Main web application UI.
- `public/image-analysis-miu-batubara/processor.py`: PyScript processing module linking UI and logic.
- `public/image-analysis-miu-batubara/circle_detection.py` & `block_detection.py`: Core logic for detections.
- `run.py`: Simple local HTTP server for local development, serves the `public/` directory on port 8000 by default.

## Commands

| Purpose | Command | Evidence | Verification status |
|---|---|---|---|
| Install | `bash scripts/setup_venv.sh` | `README.md` | Not run (modifies environment) |
| Develop | `python3 run.py 8000` | `README.md`, `run.py` | Not run (long-running) |
| Test (Unit) | `pytest -m unit -q` | `README.md`, `pytest.ini`, `tests/` | Failed (pytest missing from environment) |
| Test (Integration) | `pytest -m integration -q` | `README.md`, `pytest.ini`, `tests/` | Not run (needs pytest) |
| Test (E2E) | `RUN_E2E=1 pytest -m e2e -q` | `README.md`, `tests/e2e/` | Not run (needs pytest & Playwright) |
| Lint or format | `python3 -m compileall public/image-analysis-miu-batubara` | `README.md` | Success |
| Build | None required (client-side PyScript) | `README.md` | Verified |

## Data and integrations

No external backend integrations or databases are required. All image processing occurs within the client browser. No secrets are documented or used. (Evidence: `README.md` -> Privacy & Security)

## Repository conventions

- Testing follows a testing pyramid under `tests/unit/`, `tests/integration/`, and `tests/e2e/`.
- Logic is kept mostly in PyScript modules in the `public/image-analysis-miu-batubara/` directory.

## Constraints and hazards

- Performance depends strictly on the user's browser, as the application processes images client-side via WebAssembly/PyScript.
- Needs internet connection for initial PyScript library loading. (Evidence: `README.md`)
- `scripts/setup_venv.sh` sets up an isolated testing environment for development.

## Evidence provenance

- `README.md`: Provided project purpose, architecture, parameters, use cases, and testing commands.
- `requirements.txt`: Provided list of Python dependencies.
- `run.py`: Local development HTTP server script.
- `tests/` and `pytest.ini`: Provided evidence of the testing pyramid (unit, integration, e2e).
- `public/image-analysis-miu-batubara/`: Core implementation files (`processor.py`, `circle_detection.py`, etc.).

## Proposed behavior

None verified. Record only explicit product direction and keep it separate from current behavior.

## Superseded facts

None. Move stale claims here with their replacement evidence instead of silently preserving them.

## Known gaps

- Tests could not be fully run locally since `pytest` is not installed by default in the agent's global environment.

## Open questions

None.

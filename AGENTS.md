# Repository Guidelines

This project automates Zoom attendance by coordinating screen capture, face detection, scheduling, and logging modules.

## Project Structure & Module Organization
- `main.py` orchestrates the capture loop, scheduling, and logging; treat it as the integration point for new work.
- `desktop_app.py` hosts the PyQt5 interface; related GUI helpers live in `monitor_selector.py` and `notification_system.py`.
- Core services live in single-purpose modules: `screen_capture.py`, `face_detector.py`, `scheduler.py`, and `logger.py`.
- `build.py` manages the PyInstaller packaging flow; artefacts land in `dist/` and `release/` when present.
- Runtime artefacts such as `captures/`, `zoom_attendance.log`, and `attendance_log.csv` are generated at run time—keep them gitignored.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` — provision the virtual environment.
- `pip install -r requirements.txt` — install PyQt5, OpenCV, MSS, and other dependencies.
- `python desktop_app.py` — launch the GUI for validation on a dual-monitor setup.
- `python main.py --test` — trigger headless test mode; confirm captures and CSV logging.
- `python build.py` — run the PyInstaller pipeline, producing artefacts under `dist/` and `release/`.

## Coding Style & Naming Conventions
Adhere to PEP 8 with 4-space indentation, module-level docstrings, and descriptive logging messages. Keep module and file names in `snake_case`, classes in `PascalCase`, and methods/functions in `snake_case`. Prefer explicit type hints for new public APIs and reuse the existing logging framework instead of `print`.

## Testing Guidelines
No automated test suite ships today; rely on `python main.py --test` and GUI walk-throughs to verify detection accuracy, scheduling triggers, and CSV output. When adding automated coverage, collect new tests under `tests/` (pytest-friendly) and mirror the period/capture scenarios seen in production. Document any manual QA steps in the PR description.

## Commit & Pull Request Guidelines
Follow the current Git history: short, imperative, English subject lines (`Fix UI label safety issues`). Group related changes per commit, reference issue IDs where available, and note user-facing impact. Pull requests should include a concise summary, verification notes (commands run, screenshots of the GUI when applicable), and highlight any environment or dependency changes. Update README/DEPLOYMENT docs concurrently when behaviors shift.

## Security & Configuration Tips
Do not commit CSV logs or screenshots; they may contain student data. When debugging PyInstaller builds, scrub personally identifiable information from artifacts before sharing. Credentials are not stored in this repo—use OS-level keychains if future integrations require tokens.

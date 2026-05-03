"""
B1.3: Run FastAPI app with watch mode (auto-reload on file changes).

Usage:
  python scripts/run_with_watch.py

Features:
- Uses uvicorn --reload to auto-restart on app/*.py changes
- File watcher monitors app/ directory
- On SIGTERM (from watcher), uvicorn gracefully reloads

Alternative (watchdog-based, no uvicorn --reload needed):
  ENABLE_WATCH_MODE=true python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

Dependencies:
  pip install watchdog
"""

import subprocess
import sys
from pathlib import Path

# Default: use uvicorn --reload (built-in)
def run_uvicorn_reload():
    """Run uvicorn with --reload flag."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ]
    print("[B1.3] Starting uvicorn with --reload...")
    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_watchdog_mode():
    """Run with watchdog-based file watcher (app-level, no --reload)."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    import os
    os.environ["ENABLE_WATCH_MODE"] = "true"

    print("[B1.3] Starting app with ENABLE_WATCH_MODE=true...")
    subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="B1.3: Run FastAPI with watch mode")
    parser.add_argument("--mode", choices=["uvicorn", "watchdog"], default="uvicorn",
                      help="Reload mode: uvicorn (--reload flag) or watchdog (in-app thread)")
    args = parser.parse_args()

    if args.mode == "uvicorn":
        run_uvicorn_reload()
    else:
        run_watchdog_mode()

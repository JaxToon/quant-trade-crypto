"""
Launch the TraderStuff monitoring dashboard.
Run this in a separate terminal alongside main.py.

Usage:
    python run_dashboard.py

Then open: http://localhost:8080
"""
import subprocess, sys

try:
    import fastapi, uvicorn
except ImportError:
    print("Installing dashboard dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])

import uvicorn
uvicorn.run("dashboard.api:app", host="0.0.0.0", port=8080, reload=False, log_level="warning")

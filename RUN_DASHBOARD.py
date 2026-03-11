#!/usr/bin/env python
"""
One-click startup for web dashboard.
Just run this file!
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   🔬 CRI BENCHMARK WEB DASHBOARD SETUP                     ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# Check Python version
print(f"✅ Python {sys.version.split()[0]}")

# Check dependencies
print("\n📦 Checking dependencies...")
deps_needed = []
try:
    import flask; print("  ✓ flask")
except: deps_needed.append("flask")

try:
    import flask_cors; print("  ✓ flask-cors")
except: deps_needed.append("flask-cors")

# Install if needed
if deps_needed:
    print(f"\n⬇️  Installing {', '.join(deps_needed)}...")
    for pkg in deps_needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("  ✓ Installation complete")

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                          🚀 READY TO LAUNCH                                ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 Dashboard Features:
  ✓ Real-time metric updates (every 500ms)
  ✓ Live isolation score tracking (R-tree vs LCHI)
  ✓ Cross-CRI evolution visualization
  ✓ Query latency distribution histogram
  ✓ Interactive comparison table
  ✓ Progress bar + statistics

🎮 How to Use:
  1. Open browser: http://localhost:5000
  2. Click "▶ Start Benchmark" button
  3. Watch real-time metrics update
  4. Charts animate as benchmark runs
  5. Click "↻ Reset" to clear and restart

⏱️  Expected Runtime:
  • 500 operations: ~30-40 seconds
  • Full analysis + visualization: ~50 seconds

📍 Endpoints:
  GET  /api/metrics  → Current benchmark state (JSON)
  POST /api/start    → Launch benchmark
  POST /api/reset    → Clear metrics

🌐 Web Interface:
  Dashboard: http://localhost:5000
  API only:  http://localhost:5000/api/metrics

⏹️  Stop Server:
  Press Ctrl+C in this terminal

╔════════════════════════════════════════════════════════════════════════════╗
""")

print("\n✅ Starting Flask server on port 5000...\n")

# Change to scripts directory
import os
os.chdir(Path(__file__).parent)

# Import and run Flask app
from web_dashboard import app

class CustomFormatter:
    def format(self, record):
        return f"[{record.levelname}] {record.getMessage()}"

if __name__ == "__main__":
    # Disable Flask's startup banner
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   🌐 FLASK SERVER RUNNING                                  ║
├════════════════════════════════════════════════════════════════════════════┤
║                                                                              ║
║  🔗 Open this URL in your browser:                                        ║
║                                                                              ║
║      👉  http://localhost:5000  👈                                         ║
║                                                                              ║
║  API for custom integrations:                                             ║
║      http://localhost:5000/api/metrics                                    ║
║                                                                              ║
║  Status: READY ✓                                                          ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    # Run Flask (blocking)
    app.run(
        debug=False,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
    )

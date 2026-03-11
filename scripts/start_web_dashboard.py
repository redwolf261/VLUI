#!/usr/bin/env python
"""
Quick start script for the web dashboard.

Install requirements:
    pip install -r requirements-web.txt

Then run:
    python start_web_dashboard.py

Open in browser: http://localhost:5000
"""

import subprocess
import sys
import os
import webbrowser
import time

# Check if Flask is installed
try:
    import flask
    from flask_cors import CORS
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-web.txt"])

# Change to scripts directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   🔬 CRI BENCHMARK WEB DASHBOARD                           ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 Starting Flask server...
🌐 Opening browser in 3 seconds...
💻 Dashboard: http://localhost:5000

Press Ctrl+C to stop the server.
""")

# Start Flask app
from web_dashboard import app

# Open browser after delay
def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

import threading
browser_thread = threading.Thread(target=open_browser, daemon=True)
browser_thread.start()

# Run Flask
app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)

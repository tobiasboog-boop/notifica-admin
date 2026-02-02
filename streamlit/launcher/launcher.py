"""
Streamlit Launcher Service
Kleine Flask server die Streamlit apps kan starten via HTTP requests
Start deze launcher eenmalig bij opstarten van Windows
"""
from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import psutil
import os

app = Flask(__name__)
CORS(app)  # Allow CORS for admin portal

# Track running processes
running_apps = {}

def is_port_in_use(port):
    """Check if a port is already in use"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == 'LISTEN':
            return True
    return False

def start_streamlit_app(app_name, app_path, port):
    """Start a Streamlit app if not already running"""
    if is_port_in_use(port):
        return {"status": "already_running", "port": port}

    try:
        # Start Streamlit in background
        process = subprocess.Popen(
            ['python', '-m', 'streamlit', 'run', app_path, '--server.port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        running_apps[app_name] = {
            'process': process,
            'port': port,
            'path': app_path
        }

        return {"status": "started", "port": port, "pid": process.pid}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "running_apps": list(running_apps.keys())})

@app.route('/start/liquiditeit')
def start_liquiditeit():
    """Start liquiditeit dashboard"""
    result = start_streamlit_app(
        'liquiditeit',
        'streamlit/liquiditeit/app.py',
        8501
    )
    return jsonify(result)

@app.route('/start/voorraad')
def start_voorraad():
    """Start voorraad dashboard"""
    result = start_streamlit_app(
        'voorraad',
        'streamlit/voorraad/app.py',
        8503
    )
    return jsonify(result)

@app.route('/start/mappingtool')
def start_mappingtool():
    """Start mapping tool"""
    result = start_streamlit_app(
        'mappingtool',
        'streamlit/mappingtool/app.py',
        8504
    )
    return jsonify(result)

@app.route('/status/<app_name>')
def status(app_name):
    """Check if an app is running"""
    port_map = {
        'liquiditeit': 8501,
        'voorraad': 8503,
        'mappingtool': 8504
    }

    if app_name not in port_map:
        return jsonify({"status": "unknown_app"})

    port = port_map[app_name]
    is_running = is_port_in_use(port)

    return jsonify({
        "app": app_name,
        "port": port,
        "running": is_running
    })

if __name__ == '__main__':
    # Start launcher on port 8500
    print("Streamlit Launcher gestart op http://localhost:8500")
    print("Liquiditeit: http://localhost:8500/start/liquiditeit")
    print("Voorraad: http://localhost:8500/start/voorraad")
    print("Mappingtool: http://localhost:8500/start/mappingtool")
    app.run(host='localhost', port=8500, debug=False)

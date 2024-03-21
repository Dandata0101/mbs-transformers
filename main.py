from flask import Flask, render_template
import os
import subprocess
import signal
import sys
from waitress import serve
from dotenv import load_dotenv

# Set matplotlib to use 'Agg' backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

app = Flask(__name__)
tensorboard_process = None

def start_tensorboard(logdir, port=6006):
    global tensorboard_process
    command = ['tensorboard', '--logdir', logdir, '--port', str(port), '--bind_all']
    tensorboard_process = subprocess.Popen(command)
    return port

@app.route('/')
def show_tensorboard():
    tensorboard_port = 6006  # This should be set to the actual dynamic port if changed
    tensorboard_url = f'http://localhost:{tensorboard_port}'
    return render_template('tensor.html', tensorboard_url=tensorboard_url)

def graceful_exit(signum, frame):
    if tensorboard_process:
        tensorboard_process.terminate()
        tensorboard_process.wait()
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_exit)
signal.signal(signal.SIGINT, graceful_exit)

if __name__ == '__main__':
    logdir = 'logs/fit'
    tensorboard_port = start_tensorboard(logdir, 6006)  # This can be made dynamic

    # Configure the Flask app's debug mode based on the FLASK_DEBUG environment variable
    app.debug = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1']

    # Get the port number from the PORT environment variable for the Flask app
    port = int(os.getenv('PORT', 8000))

    # Print the local URL where the Flask app will be accessible
    print(f"Running on http://localhost:{port}")

    # Serve the Flask app with Waitress on the specified host and port
    try:
        serve(app, host="0.0.0.0", port=port)
    finally:
        graceful_exit(None, None)

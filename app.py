from flask import Flask
import subprocess
import os

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/start-detection', methods=['POST'])
def start_detection():
    try:
        print("[INFO] Button clicked! Trying to run TestBasic.py...")

        # ✅ Option 1: Simplest way if python is working directly
        result = subprocess.Popen(["python", "TestBasic.py"], shell=True)

        # ✅ Option 2 (if using conda env path directly) — try this if Option 1 fails
        # result = subprocess.Popen(r'"C:\Users\wankh\anaconda3\Scripts\activate.bat tf_env && python TestBasic.py"', shell=True)

        print("[INFO] Script started.")
        return "Detection started!", 200

    except Exception as e:
        print(f"[ERROR] {e}")
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/start-detection', methods=['POST'])
def start_detection():
    # Here, you would start the webcam-based hand sign detection logic
    # For demonstration, we return a success message.
    return "Hand sign detection started! Please show a sign."

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# Define the paths to your JSON files using absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_RESULTS_PATH = os.path.join(BASE_DIR, 'daily_results.json')
TRAINING_HISTORY_PATH = os.path.join(BASE_DIR, 'training_history.json')

@app.route('/api/training-data', methods=['GET'])
def get_training_data():
    try:
        print(f"Attempting to read from: {DAILY_RESULTS_PATH}")
        if not os.path.exists(DAILY_RESULTS_PATH):
            print(f"File not found at: {DAILY_RESULTS_PATH}")
            return jsonify({
                "message": "No training data yet",
                "data": {}
            })
            
        with open(DAILY_RESULTS_PATH, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded data: {data}")
        return jsonify(data)
    except Exception as e:
        print(f"Error reading training data: {str(e)}")
        return jsonify({
            "error": "Failed to read training data",
            "details": str(e)
        }), 500

@app.route('/api/current-status', methods=['GET'])
def get_current_status():
    try:
        if not os.path.exists(TRAINING_HISTORY_PATH):
            # Return default status if file doesn't exist
            return jsonify({
                "weeks_trained": 0,
                "last_trained_week": None,
                "message": "Training hasn't started yet"
            })
            
        with open(TRAINING_HISTORY_PATH, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        print(f"Error reading training history: {str(e)}")
        return jsonify({
            "error": "Failed to read training history",
            "details": str(e)
        }), 500

# Add a test route to verify the server is running
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"status": "Server is running!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
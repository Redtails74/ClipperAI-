from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
import requests

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'gpt2'  # or whatever model you want to use
inference = InferenceClient(model_name, token=API_KEY)

# Base URL for Hugging Face API
BASE_URL = "https://api-inference.huggingface.co/models/"

# Function to query Hugging Face Inference API
def query_huggingface_api(model, payload):
    url = f"{BASE_URL}{model}"
    response = requests.post(url, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload)
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Serve index.html from the main directory

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        # Using the call method correctly
        response = inference([user_message])  # Pass as a list
        response_text = response[0]['generated_text']  # Extract the generated text
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
import requests

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Set up Hugging Face API key and model
API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
gpt2_model_name = 'gpt2'  # Model for text generation
inference = InferenceClient(model=gpt2_model_name, token=API_KEY)

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
    return render_template('index.html')  # Serves the HTML file

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        response = inference.predict(user_message)
        response_text = response.get('generated_text', 'Error: No response from model')
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

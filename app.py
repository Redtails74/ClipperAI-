from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://redtails74.github.io"}})

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
    response.raise_for_status()
    return response.json()

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route for generating text with GPT-2 (via Hugging Face API)
@app.route('/generate-text', methods=['POST'])
def generate_text():
    input_text = request.json.get('text', '')
    generated = query_huggingface_api(gpt2_model_name, {"inputs": input_text})
    return jsonify(generated)

# Route for the chat API
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
        print(f"Error in /api/chat: {e}")  # Log the error for debugging
        return jsonify({'error': str(e)}), 500

# Route for initial data
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    app.run(debug=True)

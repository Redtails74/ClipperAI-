from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
import requests

# Initialize Flask app
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

# Route for generating text with GPT-2 (via Hugging Face API)
@app.route('/generate-text', methods=['POST'])
def generate_text():
    input_text = request.json.get('text', '')
    generated = query_huggingface_api(gpt2_model_name, {"inputs": input_text})
    return jsonify(generated)

# Route for question answering with RoBERTa (via Hugging Face API)
@app.route('/question-answer', methods=['POST'])
def answer_question():
    question = request.json.get('question', '')
    context = request.json.get('context', '')
    result = query_huggingface_api("deepset/roberta-base-squad2", {"inputs": {"question": question, "context": context}})
    return jsonify(result)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({'message': 'Hello, World!'})

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
    app.run(debug=True)

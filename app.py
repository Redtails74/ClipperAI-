from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Set up Hugging Face API key
API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')

# Load the Inference API for a model (e.g., GPT-2)
model_name = 'gpt2'  # Change this to a different model if desired
inference = InferenceClient(model=model_name, token=API_KEY)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')  # If index.html is in the same directory as app.py

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({'message': 'Hello, World!'})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    # Call Hugging Face API
    try:
        response = inference(user_message)
        response_text = response.get('generated_text', 'Error: No response from model')
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Set up Hugging Face API key and model
API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'gpt2'  # Change this to a different model if desired
inference = InferenceClient(model=model_name, token=API_KEY)

# Redirect to GitHub Pages
@app.route('/')
def index():
    return redirect('https://Redtails74.github.io/ClipperAI-/index.html')

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
        response = inference.predict(user_message)  # Use the correct method for inference
        response_text = response.get('generated_text', 'Error: No response from model')
        return jsonify({'response': response_text})
    except Exception as e:
        print(f"Error in /api/chat: {e}")  # Log the error for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

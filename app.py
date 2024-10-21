import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'gpt2'  # Specify the model you want to use

# Check available devices
pipeline_device = pipeline('text-generation').device
logger.info(f"Available device: {pipeline_device}")

# Use the available device
if pipeline_device.type == 'cuda':
    device = 0
else:
    device = -1

# Initialize the text generation pipeline
generator = pipeline(
    'text-generation', 
    model=model_name, 
    tokenizer=model_name, 
    device=device, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=1,
    pad_token_id=50256  # Set the pad token id
)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Serve index.html from the main directory

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        # Generate response using the model
        response = generator(user_message, max_length=100, do_sample=True, num_return_sequences=1, truncation=True)
        response_text = response[0]['generated_text']  # Access the generated text
        return jsonify({'response': response_text})
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/Code-Clipper-Logo-01.jpg')
def serve_logo():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'Code Clipper Logo-01.jpg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import os

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load environment variables
API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'distilgpt2'
device = -1  # Use CPU; set to 0 for GPU if available

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/logo')
def serve_logo():
    return send_from_directory('static', 'Code Clipper Logo-01.jpg')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        # Frame the input message for better context
        prompt = f"User: {user_message}\nAI:"
        response = generator(
            prompt,
            max_length=150,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            truncation=True
        )
        response_text = response[0]['generated_text'].strip()
        return jsonify({'response': response_text})
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return jsonify({'error': str(e)}), 500

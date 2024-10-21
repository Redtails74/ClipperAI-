import os
import json
import random
import string
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from urllib.parse import quote

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # More specific CORS if possible

app.config['PREFERRED_URL_SCHEME'] = 'https'

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
if API_KEY.startswith('hf_'):
    try:
        tokenizer = AutoTokenizer.from_pretrained(API_KEY.replace('hf_', ''))
        model = AutoModelForSeq2SeqLM.from_pretrained(API_KEY.replace('hf_', ''))
        burta_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1, top_k=50, top_p=0.95, num_return_sequences=1, truncation=True)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        burta_generator = None

    try:
        tokenizer = AutoTokenizer.from_pretrained('Burta')
        model = AutoModelForSeq2SeqLM.from_pretrained('Burta')
        gpt2_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1, top_k=50, top_p=0.95, num_return_sequences=1, truncation=True)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        gpt2_generator = None
else:
    logger.error("Invalid API key format")
    burta_generator = None
    gpt2_generator = None

def generate_response(model, user_message):
    if model == 'burta':
        response = burta_generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
    elif model == 'gpt2':
        response = gpt2_generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
    else:
        raise ValueError(f"Invalid model: {model}")
    response_text = response[0]['generated_text']
    return response_text

@app.route('/')
def home():
    return 'Flask app is running'

@app.route('/api/data')
def get_data():
    data = {
        'models': ['burta', 'gpt2'],
        'example_prompts': ['Write a short story about a dragon.', 'Translate this English text to French: Hello, how are you?']
    }
    return jsonify(data)

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    model = request.json.get('model', 'burta')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400
    if not model or model not in ['burta', 'gpt2']:
        return jsonify({'error': 'Invalid model provided.'}), 400

    try:
        response_text = generate_response(model, user_message)
        return jsonify({'text': response_text})
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
``


from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from transformers import pipeline
from urllib.parse import quote
import logging

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # More specific CORS if possible

app.config['PREFERRED_URL_SCHEME'] = 'https'

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
if API_KEY.startswith('hf_'):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

@app.route('/')
def home():
    return 'Flask app is running'

@app.route('/api/data')
def get_data():
    return jsonify({'message': 'Data fetched successfully.'})

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    model = request.json.get('model', 'burta')
    if model == 'burta':
        if burta_generator:
            try:
                # Your existing logic for text generation
                response = burta_generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
                response_text = response[0]['generated_text']
                return jsonify({'text': response_text})
            except Exception as e:
                logger.error(f"Error in chat route: {e}")
                return jsonify({'error': 'An internal server error occurred'}), 500
        else:
            return jsonify({'error': 'Invalid API key'}), 400
    elif model == 'gpt2':
        if gpt2_generator:
            try:
                # Your existing logic for text generation
                response = gpt2_generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
                response_text = response[0]['generated_text']
                return jsonify({'text': response_text})
            except Exception as e:
                logger.error(f"Error in chat route: {e}")
                return jsonify({'error': 'An internal server error occurred'}), 500
        else:
            return jsonify({'error': 'Invalid API key'}), 

import os
import json
import random
import string
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['PREFERRED_URL_SCHEME'] = 'https'

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
if API_KEY.startswith('hf_'):
    try:
        tokenizer = AutoTokenizer.from_pretrained(API_KEY.replace('hf_', ''))
        model = AutoModelForSequenceClassification.from_pretrained(API_KEY.replace('hf_', ''))
        burta_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1, top_k=50, top_p=0.95, num_return_sequences=1, truncation=True)
    except Exception as e:
        logger.error(f"Error initializing Burta model: {e}")
        burta_generator = None

    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForSeq2SeqLM.from_pretrained('gpt2')
        gpt2_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1, top_k=50, top_p=0.95, num_return_sequences=1, truncation=True)
    except Exception as e:
        logger.error(f"Error initializing GPT-2 model: {e}")
        gpt2_generator = None

    try:
        tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
        model_roberta = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)  # Adjust num_labels as needed
        roberta_classifier = pipeline('text-classification', model=model_roberta, tokenizer=tokenizer_roberta, device=-1)
    except Exception as e:
        logger.error(f"Error initializing RoBERTa model: {e}")
        roberta_classifier = None
else:
    logger.error("Invalid API key format")
    burta_generator = None
    gpt2_generator = None
    roberta_classifier = None

def generate_response(model, user_message):
    if model == 'burta':
        if burta_generator is None:
            raise ValueError("Burta generator is not initialized.")
        response = burta_generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
    elif model == 'gpt2':
        if gpt2_generator is None:
            raise ValueError("GPT-2 generator is not initialized.")
        response = gpt2_generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
    elif model == 'roberta':
        if roberta_classifier is None:
            raise ValueError("RoBERTa classifier is not initialized.")
        response = roberta_classifier(user_message)
        return response  # Return classification results directly
    else:
        raise ValueError(f"Invalid model: {model}")

    response_text = response[0]['generated_text'] if 'generated_text' in response[0] else response[0]
    return response_text

@app.route('/')
def home():
    # Serve the index.html file directly from the root directory
    with open('index.html') as f:
        return f.read()
        
@app.route('/api/data')
def get_data():
    data = {
        'models': ['burta', 'gpt2', 'roberta'],
        'example_prompts': ['Write a short story about a dragon.', 'Translate this English text to French: Hello, how are you?', 'Classify the sentiment of this text: "I love this!"']
    }
    return jsonify(data)

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    model = request.json.get('model', 'burta')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400
    if not model or model not in ['burta', 'gpt2', 'roberta']:
        return jsonify({'error': 'Invalid model provided.'}), 400

    try:
        response_text = generate_response(model, user_message)
        return jsonify({'text': response_text})
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

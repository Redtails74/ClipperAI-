import random
import string
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import logging
import os
from collections import deque
import torch
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

class Config:
    MAX_HISTORY = 10
    MODELS = {
        'grok1': 'allenai/grok',  # Ensure this path is correct
        'DialoGPT': 'microsoft/DialoGPT-small',
        'FlanT5': 'google/flan-t5-small',
    }
    HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "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")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

# Configuration
app = Flask(__name__, 
            static_url_path='/static',
            static_folder='../static',
            template_folder='./templates')

app.config.from_object(Config)

# Database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models storage
models = {}

def load_model(model_name, model_path):
    """Load model and tokenizer synchronously."""
    try:
        logger.info(f"Loading model {model_name} from {model_path}...")
        if model_name in ['FlanT5', 'grok1']:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=Config.HUGGINGFACE_API_KEY)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=Config.HUGGINGFACE_API_KEY)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=Config.HUGGINGFACE_API_KEY)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        if model_name == 'grok1':
            logger.info(f"Falling back to DialoGPT for {model_name}")
            return load_model("DialoGPT", "microsoft/DialoGPT-small")
        return None, None

def initialize_app():
    """Initialize application by loading models."""
    global models
    if not models:
        try:
            for model_name, model_path in Config.MODELS.items():
                model, tokenizer = load_model(model_name, model_path)
                if model and tokenizer:
                    models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'pipeline': pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
                    }
                    logger.info(f"Model pipeline for {model_name} initialized successfully.")
                else:
                    logger.error(f"Failed to load model {model_name}.")
            logger.info("All models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

# Initialize models once the app starts
initialize_app()

@app.before_request
def before_request():
    g.conversation_memory = deque(maxlen=Config.MAX_HISTORY)

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

def is_repeating(response, user_message, previous_responses):
    """Check if the generated response is too similar to previous responses."""
    return response in previous_responses or response == user_message

def regenerate_response(model, user_message, tokenizer, model_name):
    """Try regenerating the response in case of repetition with adjusted parameters."""
    inputs = tokenizer(user_message, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].cuda() if torch.cuda.is_available() else inputs['input_ids']
    attention_mask = inputs['attention_mask'].cuda() if torch.cuda.is_available() else inputs['attention_mask']

    result = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        temperature=random.uniform(0.7, 1.2),
        top_p=random.uniform(0.8, 0.95),
        top_k=random.randint(30, 70),
    )

    return tokenizer.decode(result[0], skip_special_tokens=True)

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    g.conversation_memory.append(f"user: {user_message}")

    responses = {}
    try:
        for model_name, model_data in models.items():
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            
            inputs = tokenizer(user_message, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                input_ids = inputs['input_ids'].cuda() if torch.cuda.is_available() else inputs['input_ids']
                attention_mask = inputs['attention_mask'].cuda() if torch.cuda.is_available() else inputs['attention_mask']

                result = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50
                )

            response = tokenizer.decode(result[0], skip_special_tokens=True)

            previous_responses = [msg.split(": ")[1].strip() for msg in list(g.conversation_memory)[-5:]]
            if is_repeating(response, user_message, previous_responses):
                response = regenerate_response(model, user_message, tokenizer, model_name)

            response = filter_inappropriate_words(response)
            responses[model_name] = response
            g.conversation_memory.append(f"{model_name}: {response}")

        return jsonify({
            'responses': responses,
            'conversation': list(g.conversation_memory)
        })
    
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def filter_inappropriate_words(text):
    """Filters inappropriate words from the generated text."""
    bad_words = ["badword1", "badword2"]  # Replace with actual bad words
    for word in bad_words:
        text = text.replace(word, "[REDACTED]")
    return text

if __name__ == '__main__':
    app.run(debug=True)

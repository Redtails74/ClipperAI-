from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import os
from collections import deque
import re
import torch
import asyncio
from blinker import signal

# Set up Flask app configuration
class Config:
    MAX_HISTORY = 10
    MODELS = {
        'grok1': 'Grok-1',  # Assuming Grok-1 is available in Hugging Face
        'DialoGPT': 'microsoft/DialoGPT-small',
        'FlanT5': 'google/flan-t5-small',
        # Add more models here if you want
    }
    HUGGINGFACE_API_KEY = "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"  # Add your Hugging Face key here

# Setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Dictionary to hold models and pipelines
models = {}

async def load_model(model_name, model_path):
    """Asynchronous model loading."""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=Config.HUGGINGFACE_API_KEY)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=Config.HUGGINGFACE_API_KEY)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or '<pad>'

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        return None, None

got_first_request = signal('got_first_request')

@got_first_request.connect
def load_models(sender, **extra):
    global models
    if not models:
        try:
            # Load each model asynchronously
            for model_name, model_path in Config.MODELS.items():
                model, tokenizer = asyncio.run(load_model(model_name, model_path))
                if model and tokenizer:
                    models[model_name] = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
            logger.info("All models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

# Use deque for efficient memory management of conversation history
conversation_memory = deque(maxlen=Config.MAX_HISTORY)

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

def is_repeating(generated_text, user_message):
    """Check if the generated text is a repetition of the user's message."""
    last_user_input = "user: " + user_message
    generated_response = generated_text.split('\n')[-1].strip()
    return last_user_input.lower() in generated_response.lower()

@app.route('/api/chat', methods=['POST'])
async def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    conversation_memory.append(f"user: {user_message}")
    
    try:
        responses = {}
        for model_name, generator in models.items():
            response = generator(user_message, max_length=100, num_return_sequences=1)[0]['generated_text']
            
            # Check for repetition and regenerate response if necessary
            if is_repeating(response, user_message):
                for _ in range(5):  # Try generating again up to 5 times if repetition occurs
                    response = generator(user_message, max_length=100, num_return_sequences=1)[0]['generated_text']
                    if not is_repeating(response, user_message):
                        break
                else:
                    response = "I'm sorry, I'm having trouble generating a response. Please try again later."
            
            # Filter inappropriate content from the response
            response = filter_inappropriate_words(response)
            
            # Store the response in the dictionary
            responses[model_name] = response
            
            # Append the response to the conversation history
            conversation_memory.append(f"{model_name}: {response}")
        
        # Return the responses from all models and the conversation history
        return jsonify({
            'responses': responses,
            'conversation': list(conversation_memory)
        })
    
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def filter_inappropriate_words(text):
    """Filters inappropriate words from the generated text."""
    bad_words = ["badword1", "badword2"]  # Replace with actual list of bad words
    for word in bad_words:
        text = re.sub(r'\b' + re.escape(word) + r'\b', lambda m: '*' * len(m.group()), text, flags=re.IGNORECASE)
    return text

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')

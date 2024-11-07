from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import os
from collections import deque
import re
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
class Config:
    MAX_HISTORY = 10
    MODEL_NAME = 'microsoft/DialoGPT-large'
    # Use environment variable if available, otherwise fallback to hardcoded API key
    API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')

# Setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_url_path='/static',
            static_folder='../static',
            template_folder='./templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load model and tokenizer
model = None
tokenizer = None
generator = None

# Use before_request to load model on first request, as before_first_request is deprecated
@app.before_request
def load_model_on_first_request():
    global model, tokenizer, generator
    if model is None:  # Check if model is already loaded to avoid reloading
        try:
            # Load the model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, token=Config.API_KEY)
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, token=Config.API_KEY)
            
            # Ensure pad_token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or '<pad>'

            model.eval()  # Set model to evaluation mode
            if torch.cuda.is_available():
                model = model.cuda()

            # Initialize the text generation pipeline
            generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

            logger.info(f"Model and tokenizer loaded successfully: {Config.MODEL_NAME}")

        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
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
def chat():
    # The rest of your code here...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')

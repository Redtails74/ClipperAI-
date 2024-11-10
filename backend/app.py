import os
import random
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
from collections import deque
import torch

class Config:
    MAX_HISTORY = 10
    # Update the MODEL_PATH to the latest Grok model identifier
    MODEL_PATH = 'xai-org/grok-2'
    HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "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")

# Configuration
app = Flask(__name__, 
            static_url_path='/static',
            static_folder='../static',
            template_folder='./templates')

app.config.from_object(Config)

# CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model storage
model_info = {
    'model': None,
    'tokenizer': None,
    'pipeline': None
}

def load_model():
    """Load the Grok-2 model and tokenizer."""
    try:
        logger.info(f"Loading Grok-2 model from {Config.MODEL_PATH}...")
        # Use the appropriate model class if Grok-2 has a different one; here we assume it's still CausalLM
        model = AutoModelForCausalLM.from_pretrained(Config.MODEL_PATH, token=Config.HUGGINGFACE_API_KEY)
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH, token=Config.HUGGINGFACE_API_KEY)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        # Create pipeline
        model_info['model'] = model
        model_info['tokenizer'] = tokenizer
        # Ensure the pipeline is configured for your model's capabilities, here we're using text-generation
        model_info['pipeline'] = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        logger.info("Grok-2 model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Grok-2 model: {e}")
        raise

# Load model when the app starts
load_model()

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

def regenerate_response(user_message):
    """Try regenerating the response in case of repetition with adjusted parameters."""
    model = model_info['model']
    tokenizer = model_info['tokenizer']

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

def generate_response(user_message):
    """Generate a response from Grok-2."""
    pipeline = model_info['pipeline']
    response = pipeline(user_message, max_length=150)[0]['generated_text']
    return response

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    g.conversation_memory.append(f"user: {user_message}")

    try:
        response = generate_response(user_message)
        
        previous_responses = [msg.split(": ")[1].strip() for msg in list(g.conversation_memory)[-5:]]
        if is_repeating(response, user_message, previous_responses):
            response = regenerate_response(user_message)

        # Simplified response filtering
        response = response.replace("badword1", "[REDACTED]").replace("badword2", "[REDACTED]")

        g.conversation_memory.append(f"AI: {response}")

        return jsonify({
            'response': response,
            'conversation': list(g.conversation_memory)
        })
    
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)

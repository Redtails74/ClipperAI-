from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
import time
import os
from collections import Counter
import re

# Configuration class
class Config:
    MAX_HISTORY = 5
    MODEL_NAME = 'EleutherAI/gpt-neo-125M'
    API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_url_path='/static', 
            static_folder='./static', 
            template_folder='./templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load model and tokenizer
device = -1  # Use CPU; set to 0 for GPU if available
model_name = Config.MODEL_NAME

try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

# Simple conversation memory
conversation_memory = []

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip().lower()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    # Add user message to conversation memory
    conversation_memory.append({"role": "user", "content": user_message})

    # Limit conversation history to the last N messages
    while len(conversation_memory) > Config.MAX_HISTORY:
        conversation_memory.pop(0)

    try:
        # Detect repetition
        user_history = [msg['content'].lower() for msg in conversation_memory if msg['role'] == 'user']
        most_common = Counter(user_history).most_common(1)
        
        if most_common[0][1] > 1:  # If a message is repeated more than once
            response_text = handle_repetition(conversation_memory)
        else:
            # Construct prompt with context
            prompt = construct_prompt(conversation_memory, user_message)

            # Generate response with optimized parameters
            response = generator(
                prompt,
                max_length=200,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                num_beams=3,
                early_stopping=True,
                truncation=True
            )

            generated_text = response[0].get('generated_text', '').strip()
            response_text = extract_response(generated_text)

        # Filter out inappropriate language
        response_text = filter_inappropriate_words(response_text)

        # Add AI's response to conversation memory
        conversation_memory.append({"role": "assistant", "content": response_text})

        return jsonify({'response': response_text, 'conversation': conversation_memory})

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def handle_repetition(conversation_memory):
    user_last_message = conversation_memory[-1]['content']
    if re.search(r'\btry\b', user_last_message, re.IGNORECASE):
        return "Are you finding what you're looking for? Maybe we can switch topics or try something new."
    else:
        return "It seems we might be stuck in a loop. Can we explore something else or do you have a different question?"

def construct_prompt(conversation_memory, user_message):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_memory] + [f"user: {user_message}"]) + "\nassistant:"

def extract_response(generated_text):
    return generated_text.split("\nassistant:")[-1].strip()

def filter_inappropriate_words(text):
    profane_words = ['ass', 'damn', 'hell']  # Expand this list as needed
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in profane_words:
            words[i] = '*' * len(word)
    return ' '.join(words)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

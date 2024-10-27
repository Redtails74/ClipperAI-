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
    MODEL_NAME = 'google/gemma-7b-it'  # Changed to Google's Gemma model
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
    # Ensure you're using the correct pipeline for instruction tuning if necessary
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    
    # Note: Depending on the model's capabilities, you might need to adjust the pipeline parameters like 
    # 'conversation' or 'text2text-generation' for instruction tuning models.
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
        # Construct prompt with context
        prompt = construct_prompt(conversation_memory, user_message)

        # Generate response
        response = generator(
            prompt,
            max_length=200,  # You might want to adjust this based on the model's capabilities
            # Ensure these parameters are appropriate for the Gemma model. Gemma might not support all of these.
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
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

# The rest of your functions (handle_repetition, construct_prompt, extract_response, filter_inappropriate_words) 
# can remain the same unless you need to adjust for any new model-specific requirements or prompt formats.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

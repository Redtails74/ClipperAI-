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
    # Using EleutherAI's GPT-Neo 125M for lower memory consumption
    MODEL_NAME = 'EleutherAI/gpt-neo-125M'
    API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')  # This might not be necessary for public models

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_url_path='/static',
            static_folder='../static',
            template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load model and tokenizer
device = -1  # Use CPU
model_name = Config.MODEL_NAME

try:
    # Load the model. Note: EleutherAI models might not require .from_pretrained to specify model type directly
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

# Conversation memory
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
    # Construct prompt from conversation memory
    prompt = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_memory]) + "\nassistant:"

    # Generate response
    response = generator(
        prompt,
        max_length=200,  # Increase length to allow for more detailed responses
        do_sample=True,
        temperature=0.7,  # Lower temperature for more focused responses but still allowing for some creativity
        top_k=40,  # Narrow down the token selection to make the output more consistent with the context
        top_p=0.90,  # Slightly lower for more focused output
        repetition_penalty=1.2,  # Increase slightly to reduce repetition
        num_return_sequences=1,  # Generate one response
        # Add these for more control:
        num_beams=4,  # Use beam search for better quality
        early_stopping=True,  # Stop beam search when at least num_beams complete sequences are found
        no_repeat_ngram_size=2  # Prevent repetitions of 2-gram sequences
    )

    # Extract the generated text and clean it up
    generated_text = response[0]['generated_text']
    response_text = generated_text.split('assistant:')[-1].strip()

        # Filter out inappropriate language (You'll need to implement or import this function)
        response_text = filter_inappropriate_words(response_text)

        # Add AI's response to conversation memory
        conversation_memory.append({"role": "assistant", "content": response_text})

        return jsonify({'response': response_text, 'conversation': conversation_memory})

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Implement these functions if they are not already in your code or need adjustments:

def construct_prompt(conversation, message):
    # Assuming the prompt construction is simple. Adjust if needed for context.
    return "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation]) + f"\nuser: {message}"

def extract_response(text):
    # Simple function to extract the response from the generated text
    return text.split('assistant:')[-1].strip()

def filter_inappropriate_words(text):
    # Implement or adjust this function to filter out inappropriate language
    # Example implementation:
    bad_words = ["badword1", "badword2"]  # Replace with actual bad words list
    for word in bad_words:
        text = text.replace(word, '*' * len(word))
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

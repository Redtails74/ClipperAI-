from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
import time
import os

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
    """Chat API endpoint with context retention and relevance enhancement."""
    user_message = request.json.get('message', '')
    if not isinstance(user_message, str):
        return jsonify({'error': 'Invalid input message'}), 400

    if not user_message.strip():
        return jsonify({'error': 'No input message provided'}), 400

    # Add user message to conversation memory
    conversation_memory.append({"role": "user", "content": user_message})

    # Limit conversation history to the last N messages
    while len(conversation_memory) > Config.MAX_HISTORY:
        conversation_memory.pop(0)

    try:
        # Construct prompt with context
        prompt = construct_prompt(conversation_memory, user_message)

        # Start timing the response generation
        start_time = time.time()

        # Generate response with optimized parameters
        response = generator(
            prompt,
            max_length=200,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.6,  # Slightly increased for some creativity
            top_k=40,         # Increased for more sampling options
            top_p=0.90,       # Slightly more diverse token choices
            repetition_penalty=1.6,  # Increased to further discourage repetition
            num_beams=5,      # Added for potentially better quality
            early_stopping=True,
            truncation=True
        )

        # Log the time taken for generation
        logger.info(f"Time taken for generation: {time.time() - start_time:.2f} seconds")

        generated_text = response[0].get('generated_text', '').strip()
        if not generated_text:
            return jsonify({'error': 'No response generated'}), 500

        logger.info(f"Generated Text: {generated_text}")
        response_text = extract_response(generated_text)
        logger.info(f"Extracted Response: {response_text}")

        # Add AI's response to conversation memory
        conversation_memory.append({"role": "assistant", "content": response_text})

        return jsonify({'response': response_text, 'conversation': conversation_memory})

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def construct_prompt(conversation_memory, user_message):
    """Construct a prompt based on conversation history."""
    # Joining messages with newlines for better context separation
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_memory] + [f"user: {user_message}"]) + "\nassistant:"

def extract_response(generated_text):
    """Extract the assistant's reply from the generated text."""
    # Assuming the response starts after the last \nassistant:
    return generated_text.split("\nassistant:")[-1].strip()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

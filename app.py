from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
import time
import os

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_url_path='/static', 
            static_folder='./static', 
            template_folder='./templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load environment variables
API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'EleutherAI/gpt-neo-1.3B'
device = -1  # Use CPU; set to 0 for GPU if available

# Load model and tokenizer
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
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    # Add user message to conversation memory
    conversation_memory.append({"role": "user", "content": user_message})

    # Limit conversation history to the last N messages
    MAX_HISTORY = 5  # Adjust this value as needed
    if len(conversation_memory) > MAX_HISTORY:
        conversation_memory.pop(0)

    try:
        # Construct prompt with context
        prompt = construct_prompt(conversation_memory, user_message)

        # Start timing the response generation
        start_time = time.time()
        
        # Generate response with optimized parameters
        response = generator(
            prompt,
            max_length=200,  # Keep this reduced for faster responses
            do_sample=True,
            num_return_sequences=1,
            temperature=0.5,  # Focused response
            top_k=30,         # Faster sampling
            top_p=0.85,       # Limit token choices
            repetition_penalty=1.5,  # Moderate repetition penalty
            truncation=True
        )

        # Log the time taken for generation
        logger.info(f"Time taken for generation: {time.time() - start_time:.2f} seconds")

        generated_text = response[0].get('generated_text', '').strip()
        logger.info(f"Generated Text: {generated_text}")
        response_text = extract_response(generated_text)
        logger.info(f"Extracted Response: {response_text}")

        # Add AI's response to conversation memory
        conversation_memory.append({"role": "assistant", "content": response_text})

        return jsonify({'response': response_text, 'conversation': conversation_memory})

    except Exception as e:
        logger.error(f"Error processing response: {e}")
        return jsonify({'error': str(e)}), 500

def construct_prompt(conversation, current_message):
    """Construct a prompt with conversation history."""
    prompt = "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation])
    prompt += f"\nUser: {current_message}\nAI:"
    return prompt

def extract_response(text):
    """Extract AI response from generated text."""
    if "AI:" in text:
        return text.split('AI:')[-1].strip()
    return text

# For template reloading
@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True, extra_files=['templates/'])  # Watch for changes in templates

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
import os

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
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
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/logo')
def serve_logo():
    """Serve the logo image."""
    return send_from_directory('static', 'code_clipper_logo.jpg')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat API endpoint with context retention and relevance enhancement."""
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    # Add user message to conversation memory
    conversation_memory.append({"role": "user", "content": user_message})

    try:
        # Construct prompt with context
        prompt = construct_prompt(conversation_memory, user_message)
        
        response = generator(
            prompt,
            max_length=200,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.6,
            top_k=40,
            top_p=0.9,
            repetition_penalty=2.5,
            truncation=True
        )

        generated_text = response[0].get('generated_text', '').strip()
        response_text = extract_response(generated_text)
        
        # Check for generic response and regenerate if necessary
        if is_generic_response(response_text):
            response_text = regenerate_response(prompt, user_message, generator, max_length=150)

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

def is_generic_response(response):
    """Check if response is too generic."""
    generic_phrases = ["discuss", "Let's talk", "topic further", "Let's explore"]
    return any(phrase in response.lower() for phrase in generic_phrases)

def regenerate_response(prompt, user_message, generator, max_length):
    """Regenerate response with a more specific prompt."""
    specific_prompt = f"{prompt} Provide detailed information or a solution regarding '{user_message}'."
    response = generator(
        specific_prompt,
        max_length=max_length,
        do_sample=True,
        num_return_sequences=1,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        repetition_penalty=2.0,
        truncation=True
    )
    return extract_response(response[0].get('generated_text', '').strip())

# For template reloading
@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True)

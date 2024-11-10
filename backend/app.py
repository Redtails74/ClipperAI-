import os
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import logging
from collections import deque
import torch

class Config:
    MAX_HISTORY = 10
    MODEL_PATH = 'google/flan-t5-large'  # Using FLAN-T5 model
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
    """Load the FLAN-T5 model and tokenizer."""
    try:
        logger.info(f"Loading FLAN-T5 model from {Config.MODEL_PATH}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_PATH, token=Config.HUGGINGFACE_API_KEY)
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH, token=Config.HUGGINGFACE_API_KEY)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos token if missing

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        # Create pipeline for text generation
        model_info['model'] = model
        model_info['tokenizer'] = tokenizer
        model_info['pipeline'] = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

        logger.info("FLAN-T5 model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading FLAN-T5 model: {e}")
        raise

# Load model when the app starts
load_model()

@app.before_request
def before_request():
    """Initialize conversation memory before each request."""
    g.conversation_memory = deque(maxlen=Config.MAX_HISTORY)

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat interactions via POST request."""
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Generate a response based on the user's message
        response = generate_response(user_message)
        
        # Send the conversation memory back with the response
        return jsonify({
            "response": response,
            "conversation": list(g.conversation_memory)
        })
    except Exception as e:
        logger.error(f"Error in /api/chat: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

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
        temperature=0.6,  # Reduced temperature for more deterministic responses
        top_p=0.85,       # Reduced top-p for more focused responses
        top_k=30,         # Reduced top-k for less randomness
        no_repeat_ngram_size=3,  # Avoid repetition
        early_stopping=True  # Stop when a stopping criterion is met
    )

    return tokenizer.decode(result[0], skip_special_tokens=True)

def generate_response(user_message):
    """Generate a response from FLAN-T5 with improved logic."""
    try:
        pipeline = model_info['pipeline']
        
        # Include conversation context to improve response coherence
        conversation_history = ' '.join([msg.split(": ")[1].strip() for msg in list(g.conversation_memory)])
        prompt = f"Given the conversation so far: {conversation_history} Answer the following question logically: {user_message}"
        
        # Generate response using the pipeline
        response = pipeline(prompt, max_length=150)[0]['generated_text']

        # Handle cases where the model might produce suboptimal responses like "Loading..." or empty responses
        if response.strip().lower() == "loading..." or not response.strip():
            logger.warning(f"Model returned an empty or loading response: {response}")
            response = "I'm having trouble generating a response. Please try again later."
        
        # Ensure response coherence by trimming and removing problematic patterns
        response = response.strip()

        # Optional: Store the response in conversation memory
        g.conversation_memory.append(f"User: {user_message}")
        g.conversation_memory.append(f"Bot: {response}")

        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "An error occurred while generating the response. Please try again later."

if __name__ == '__main__':
    app.run(debug=True)

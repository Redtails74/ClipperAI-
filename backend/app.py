from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import os
from collections import deque
import re
import torch
import openai  # OpenAI API client

# Set up Flask app configuration
class Config:
    MAX_HISTORY = 10
    HUGGINGFACE_MODEL_NAME = 'gpt2'  # Switching to GPT-2 for text generation
    OPENAI_API_KEY = "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"  # Add your OpenAI key here
    HUGGINGFACE_API_KEY = "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"  # Add your Hugging Face key here

# Set up OpenAI API Key
openai.api_key = Config.OPENAI_API_KEY

# Setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_url_path='/static',
            static_folder='../static',
            template_folder='./templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load model and tokenizer for Hugging Face and OpenAI
huggingface_model = None
huggingface_tokenizer = None
huggingface_generator = None

@app.before_request
def load_models_on_first_request():
    global huggingface_model, huggingface_tokenizer, huggingface_generator
    if huggingface_model is None:  # Check if model is already loaded to avoid reloading
        try:
            # Initialize Hugging Face model and tokenizer for GPT-2
            huggingface_model = AutoModelForCausalLM.from_pretrained(Config.HUGGINGFACE_MODEL_NAME)
            huggingface_tokenizer = AutoTokenizer.from_pretrained(Config.HUGGINGFACE_MODEL_NAME)
            
            # Ensure pad_token is set
            if huggingface_tokenizer.pad_token is None:
                huggingface_tokenizer.pad_token = huggingface_tokenizer.eos_token or '<pad>'

            huggingface_model.eval()  # Set model to evaluation mode
            if torch.cuda.is_available():
                huggingface_model = huggingface_model.cuda()

            # Set up Hugging Face's GPT-2 pipeline for text generation
            huggingface_generator = pipeline('text-generation', model=huggingface_model, tokenizer=huggingface_tokenizer, device=0 if torch.cuda.is_available() else -1)

            logger.info(f"Hugging Face model and tokenizer loaded successfully: {Config.HUGGINGFACE_MODEL_NAME}")

        except Exception as e:
            logger.error(f"Error loading Hugging Face model or tokenizer: {e}")
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
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    # Add user message to conversation history
    conversation_memory.append(f"user: {user_message}")
    
    try:
        # Build a conversation context with the last few exchanges
        conversation_context = "\n".join([entry for entry in conversation_memory if entry.startswith("user:") or entry.startswith("assistant:")])
        
        # Construct prompt based on the conversation context
        prompt = f"Assistant: Here's my response:\n{conversation_context}"

        # Use OpenAI API with the new interface
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",  # Or another model if needed
            messages=[{"role": "user", "content": user_message}],
            max_tokens=150,
            temperature=0.7
        )
        openai_response_text = openai_response['choices'][0]['message']['content'].strip()

        # Use Hugging Face GPT-2 for text generation
        huggingface_response = huggingface_generator(user_message, max_length=100, num_return_sequences=1)[0]['generated_text']

        # Retry generating response if repetition is detected
        if is_repeating(huggingface_response, user_message):
            for _ in range(5):  # Try generating again up to 5 times if repetition occurs
                huggingface_response = huggingface_generator(user_message, max_length=100, num_return_sequences=1)[0]['generated_text']
                if not is_repeating(huggingface_response, user_message):
                    break
            else:
                huggingface_response = "I'm sorry, I'm having trouble generating a response. Please try again later."

        # Filter inappropriate content from both models' responses
        openai_response_text = filter_inappropriate_words(openai_response_text)
        huggingface_response = filter_inappropriate_words(huggingface_response)
        
        # Append both responses to conversation history
        conversation_memory.append(f"openai: {openai_response_text}")
        conversation_memory.append(f"huggingface: {huggingface_response}")

        return jsonify({
            'openai_response': openai_response_text,
            'huggingface_response': huggingface_response,
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

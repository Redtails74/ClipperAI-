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

@app.before_first_request
def load_model():
    """Load model and tokenizer before first request."""
    global model, tokenizer, generator
    try:
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, use_auth_token=Config.API_KEY)
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_auth_token=Config.API_KEY)
        
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
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    # Add user message to conversation history
    conversation_memory.append(f"user: {user_message}")
    
    try:
        # Construct prompt based on the conversation history
        prompt = "Assistant: Here's my response:\n" + "\n".join([entry for entry in conversation_memory if entry.startswith("user:")])
        logger.info(f"Generated prompt for model: {prompt}")
        
        if not prompt:
            logger.error("Error: Prompt is empty!")
            return jsonify({'error': 'Prompt is empty'}), 500

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024, padding=True)
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}
        
        # Generate a response from the model
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                do_sample=True,
                temperature=1.2,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                repetition_penalty=0.9
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Retry generating response if repetition is detected
        if is_repeating(response_text, user_message):
            for _ in range(5):
                outputs = model.generate(
                    **inputs,
                    max_length=150,
                    do_sample=True,
                    temperature=1.2,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=0.9
                )
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if not is_repeating(response_text, user_message):
                    break
            else:
                response_text = "I'm sorry, I'm having trouble generating a response. Please try again later."
        else:
            # Ensure the assistant response starts after the "Assistant:" part
            response_text = response_text.split('Assistant:')[-1].strip() if 'Assistant:' in response_text else response_text.split('\n')[-1].strip()

        # Filter inappropriate content from the response
        response_text = filter_inappropriate_words(response_text)
        
        # Append the assistant's response to conversation memory
        conversation_memory.append(f"assistant: {response_text}")

        return jsonify({'response': response_text, 'conversation': list(conversation_memory)})

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def filter_inappropriate_words(text):
    """Filters inappropriate words from the generated text."""
    bad_words = ["badword1", "badword2"]  # Replace with actual list
    for word in bad_words:
        text = re.sub(r'\b' + re.escape(word) + r'\b', lambda m: '*' * len(m.group()), text, flags=re.IGNORECASE)
    return text

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')

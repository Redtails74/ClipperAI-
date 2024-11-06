from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import os
from collections import deque
import re

# Configuration
class Config:
    MAX_HISTORY = 5  # Limit conversation history to a few exchanges
    MODEL_NAME = 'microsoft/DialoGPT-medium'  # Better suited for conversational AI
    API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')  # Hugging Face token

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_url_path='/static',
            static_folder='../static',
            template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load model and tokenizer
try:
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, use_auth_token=Config.API_KEY)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_auth_token=Config.API_KEY)
    
    # Set pad_token to eos_token (common workaround for models without a pad_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else -1)
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

# Use deque for efficient memory management of conversation history
conversation_memory = deque(maxlen=Config.MAX_HISTORY)

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    # Add user message to conversation history
    conversation_memory.append(f"user: {user_message}")
    
    try:
        # Construct prompt from conversation history (without the latest user input)
        # Only include the previous assistant's response and the previous user message.
        # This prevents repeating the user message in the prompt.

        prompt = "\n".join([entry for entry in conversation_memory if not entry.startswith("user:")])

        logger.info(f"Generated prompt for model: {prompt}")

        # Tokenize input with truncation to avoid long prompts
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024, padding=True)

        # Manually set the attention mask (required because pad_token == eos_token)
        inputs['attention_mask'] = inputs['attention_mask'].fill_(1)

        # Generate response
        response = generator(
            prompt,
            max_length=150,  # Adjust length to allow for more diverse responses
            do_sample=True,
            temperature=0.7,  # Moderate temperature for better variety in answers
            top_k=50,  # Experiment with smaller values of top_k
            top_p=0.9,  # Use higher value for more randomness
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2  # Less penalty to allow for some repetition
        )

        # Check if response is empty or invalid
        if not response or 'generated_text' not in response[0]:
            logger.error("Error: Model did not return a valid response.")
            return jsonify({'error': 'Model did not return a valid response'}), 500
        
        # Extract the generated text
        generated_text = response[0]['generated_text']
        logger.info(f"Model generated text: {generated_text}")
        
        # Clean and extract the AI response from the generated text
        response_text = generated_text.split('assistant:')[-1].strip() if 'assistant:' in generated_text else generated_text.split('\n')[-1].strip()

        # Filter out inappropriate language
        response_text = filter_inappropriate_words(response_text)

        # Add AI's response to conversation history
        conversation_memory.append(f"assistant: {response_text}")

        return jsonify({'response': response_text, 'conversation': list(conversation_memory)})

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def filter_inappropriate_words(text):
    """Filters out inappropriate language from the response."""
    bad_words = ["badword1", "badword2"]  # Replace with actual bad words list
    pattern = r'\b(?:' + '|'.join(map(re.escape, bad_words)) + r')\b'
    return re.sub(pattern, '*' * 8, text, flags=re.IGNORECASE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

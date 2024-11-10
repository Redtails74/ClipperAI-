from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import logging
import os
import torch
from collections import deque

# Set up Flask app configuration
class Config:
    MAX_HISTORY = 10
    MODELS = {
        'grok1': 'allenai/grok',  # Assuming this is the correct path for Grok
        'DialoGPT': 'microsoft/DialoGPT-small',
        'FlanT5': 'google/flan-t5-small',
    }
    HUGGINGFACE_API_KEY = "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"  # Replace with your Hugging Face API key

# Setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app with template folder, static folder, and CORS configuration
app = Flask(
    __name__,
    static_url_path='/static',
    static_folder='../static',
    template_folder='./templates'
)

# Enable CORS for the API routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Dictionary to hold models, tokenizers, and pipelines
models = {}

def load_model(model_name, model_path):
    """Load model and tokenizer synchronously."""
    try:
        logger.info(f"Loading model {model_name} from {model_path}...")
        if model_name in ['FlanT5', 'grok1']:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=Config.HUGGINGFACE_API_KEY)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=Config.HUGGINGFACE_API_KEY)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=Config.HUGGINGFACE_API_KEY)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        logger.info(f"Model {model_name} and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        # Try loading a fallback model (e.g., DialoGPT) if Grok fails
        if model_name == 'grok1':
            logger.info(f"Falling back to DialoGPT for {model_name}")
            return load_model("DialoGPT", "microsoft/DialoGPT-small")
        return None, None

# Use deque for efficient memory management of conversation history
conversation_memory = deque(maxlen=Config.MAX_HISTORY)

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

def is_repeating(response, user_message, previous_responses):
    """Check if the generated response is too similar to previous responses."""
    return response in previous_responses or response == user_message

def regenerate_response(model, user_message, tokenizer, model_name):
    """Try regenerating the response in case of repetition."""
    logger.info(f"Regenerating response for {model_name}...")

    # Tokenize the input
    inputs = tokenizer(user_message, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].cuda() if torch.cuda.is_available() else inputs['input_ids']
    attention_mask = inputs['attention_mask'].cuda() if torch.cuda.is_available() else inputs['attention_mask']

    # Set diverse parameters for regeneration (e.g., temperature, top_p)
    result = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.9,  # Higher temperature for more randomness
        top_p=0.9,        # Top-p (nucleus sampling) to get more diverse responses
        top_k=50,         # Restrict token sampling to top-k
    )

    response = tokenizer.decode(result[0], skip_special_tokens=True)
    return response

# Initialize application by loading models
def initialize_app():
    """Initialize application by loading models."""
    global models
    if not models:
        try:
            # Load each model synchronously
            for model_name, model_path in Config.MODELS.items():
                logger.info(f"Loading model pipeline for: {model_name}")
                model, tokenizer = load_model(model_name, model_path)
                if model and tokenizer:
                    # Store both model and tokenizer separately
                    models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'pipeline': pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
                    }
                    logger.info(f"Model pipeline for {model_name} initialized successfully.")
                else:
                    logger.error(f"Failed to load model {model_name}.")
            logger.info("All models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

# Initialize models once the app starts
initialize_app()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat request."""
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    conversation_memory.append(f"user: {user_message}")
    
    try:
        responses = {}
        for model_name, model_data in models.items():
            logger.info(f"Generating response with model: {model_name}")
            
            # Extract model, tokenizer, and pipeline from the model_data dictionary
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            generator = model_data['pipeline']

            # Tokenize the user input and ensure truncation
            inputs = tokenizer(user_message, return_tensors='pt', truncation=True, padding=True, max_length=512)
            logger.info(f"Tokenized input: {inputs}")

            # Generate the response
            with torch.no_grad():
                input_ids = inputs['input_ids'].cuda() if torch.cuda.is_available() else inputs['input_ids']
                attention_mask = inputs['attention_mask'].cuda() if torch.cuda.is_available() else inputs['attention_mask']

                result = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=150,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50
                )

            # Decode the generated response
            response = tokenizer.decode(result[0], skip_special_tokens=True)
            logger.info(f"Generated response: {response}")
            
            # Check for repetition and regenerate response if necessary
            previous_responses = [msg.split(": ")[1].strip() for msg in list(conversation_memory)[-5:]]  # Get the last 5 responses
            if is_repeating(response, user_message, previous_responses):
                response = regenerate_response(model, user_message, tokenizer, model_name)
            
            # Filter inappropriate content from the response
            response = filter_inappropriate_words(response)
            
            # Store the response in the dictionary
            responses[model_name] = response
            
            # Append the response to the conversation history
            conversation_memory.append(f"{model_name}: {response}")
        
        # Return the responses from all models and the conversation history
        return jsonify({
            'responses': responses,
            'conversation': list(conversation_memory)
        })
    
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def filter_inappropriate_words(text):
    """Filters inappropriate words from the generated text."""
    bad_words = ["badword1", "badword2"]  # Replace with your actual list of bad words
    for word in bad_words:
        text = text.replace(word, "***")
    return text

# Running the app
if __name__ == '__main__':
    app.run(debug=True)

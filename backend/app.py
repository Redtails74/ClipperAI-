from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import logging
import torch
from collections import deque

# Set up Flask app configuration
class Config:
    MAX_HISTORY = 10
    MODELS = {
        'grok1': 'Grok-1',  # Assuming Grok-1 is available in Hugging Face
        'DialoGPT': 'microsoft/DialoGPT-small',
        'FlanT5': 'google/flan-t5-small',
        # Add more models here if you want
    }
    HUGGINGFACE_API_KEY = "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"  # Add your Hugging Face key here

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
        if model_name == 'FlanT5':
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
        return None, None

# Use deque for efficient memory management of conversation history
conversation_memory = deque(maxlen=Config.MAX_HISTORY)

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

def is_repeating(generated_text, user_message, previous_responses):
    """Check if the generated text is a repetition of the user's message or past responses."""
    last_user_input = "user: " + user_message
    # Check against the last few AI responses as well
    for past_response in previous_responses:
        if last_user_input.lower() in past_response.lower() or generated_text.lower() in past_response.lower():
            return True
    return False

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
                    if model_name == 'FlanT5':
                        models[model_name] = {
                            'model': model,
                            'tokenizer': tokenizer,
                            'pipeline': pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
                        }
                    else:
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

            # Generate the response with temperature and top_p for better variability
            with torch.no_grad():
                input_ids = inputs['input_ids'].cuda() if torch.cuda.is_available() else inputs['input_ids']
                attention_mask = inputs['attention_mask'].cuda() if torch.cuda.is_available() else inputs['attention_mask']

                result = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=150,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=0.7,  # More randomness
                    top_p=0.9,  # Use nucleus sampling
                    top_k=50  # Further restrict token sampling
                )

            # Decode the generated response
            response = tokenizer.decode(result[0], skip_special_tokens=True)
            logger.info(f"Generated response: {response}")
            
            # Check if the response is valid
            if not response or response.strip() == "":
                logger.error(f"Failed to generate response with model {model_name}")
                responses[model_name] = "Error generating response."
                continue

            # Check for repetition and regenerate response if necessary
            previous_responses = [msg.split(": ")[1].strip() for msg in list(conversation_memory)[-5:]]  # Get the last 5 responses
            if is_repeating(response, user_message, previous_responses):
                for _ in range(5):  # Try regenerating up to 5 times
                    logger.info(f"Repeating detected for {model_name}, regenerating response...")
                    result = generator(user_message, max_length=150, num_return_sequences=1)
                    if result:
                        response = result[0]['generated_text']
                    if not is_repeating(response, user_message, previous_responses):
                        break
                else:
                    response = "I'm sorry, I'm having trouble generating a response. Please try again later."
            
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
    bad_words = ["badword1", "badword2"]  # Replace with actual bad words to filter
    for word in bad_words:
        text = text.replace(word, "[REDACTED]")
    return text

if __name__ == '__main__':
    app.run(debug=True)

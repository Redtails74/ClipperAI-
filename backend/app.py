from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import os
from collections import deque
import re
import torch
import openai  # OpenAI API client
from dotenv import load_dotenv

# Set up OpenAI and Hugging Face API keys directly
openai.api_key = "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"
HUGGINGFACE_API_KEY = "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"

# Set up Flask app configuration
class Config:
    MAX_HISTORY = 10
    HUGGINGFACE_MODEL_NAME = 'distilbert-base-uncased'  # Use your model of choice

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

# Set up OpenAI API Key
openai.api_key = Config.OPENAI_API_KEY

@app.before_request
def load_models_on_first_request():
    global huggingface_model, huggingface_tokenizer, huggingface_generator
    if huggingface_model is None:  # Check if model is already loaded to avoid reloading
        try:
            # Initialize Hugging Face model and tokenizer
            huggingface_model = AutoModelForCausalLM.from_pretrained(Config.HUGGINGFACE_MODEL_NAME)
            huggingface_tokenizer = AutoTokenizer.from_pretrained(Config.HUGGINGFACE_MODEL_NAME)
            
            # Ensure pad_token is set
            if huggingface_tokenizer.pad_token is None:
                huggingface_tokenizer.pad_token = huggingface_tokenizer.eos_token or '<pad>'

            huggingface_model.eval()  # Set model to evaluation mode
            if torch.cuda.is_available():
                huggingface_model = huggingface_model.cuda()

            # Initialize Hugging Face text generation pipeline
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

        # Prepare response from OpenAI's GPT model
        openai_response = openai.Completion.create(
            engine="text-davinci-003",  # Or you can use "gpt-4" or other available models
            prompt=user_message,
            max_tokens=150,
            temperature=0.7
        )
        openai_response_text = openai_response.choices[0].text.strip()

        # Prepare response from Hugging Face's DistilBERT (or other Hugging Face models)
        huggingface_inputs = huggingface_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024, padding=True)
        if torch.cuda.is_available():
            huggingface_inputs = {key: value.cuda() for key, value in huggingface_inputs.items()}

        with torch.no_grad():
            huggingface_outputs = huggingface_model.generate(
                **huggingface_inputs,
                max_length=150,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                repetition_penalty=0.9
            )

        huggingface_response_text = huggingface_tokenizer.decode(huggingface_outputs[0], skip_special_tokens=True)

        # Retry generating response if repetition is detected
        if is_repeating(huggingface_response_text, user_message):
            for _ in range(5):  # Try generating again up to 5 times if repetition occurs
                huggingface_outputs = huggingface_model.generate(
                    **huggingface_inputs,
                    max_length=150,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=0.9
                )
                huggingface_response_text = huggingface_tokenizer.decode(huggingface_outputs[0], skip_special_tokens=True)
                if not is_repeating(huggingface_response_text, user_message):
                    break
            else:
                huggingface_response_text = "I'm sorry, I'm having trouble generating a response. Please try again later."

        # Filter inappropriate content from both models' responses
        openai_response_text = filter_inappropriate_words(openai_response_text)
        huggingface_response_text = filter_inappropriate_words(huggingface_response_text)
        
        # Append both responses to conversation history
        conversation_memory.append(f"openai: {openai_response_text}")
        conversation_memory.append(f"huggingface: {huggingface_response_text}")

        return jsonify({
            'openai_response': openai_response_text,
            'huggingface_response': huggingface_response_text,
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

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import logging  # Import the logging module
import os

# Setting up logger
logging.basicConfig(level=logging.INFO)  # Configure basic logging level
logger = logging.getLogger(__name__)  # Create a logger with the current module's name

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'distilgpt2'  # Example model, consider using environment variable for flexibility
device = 0 if torch.cuda.is_available() else -1

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Serve index.html from the main directory

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

   try:
        # Generate response using the model
        response = generator(user_message, max_length=100, do_sample=True, num_return_sequences=1, truncation=True)
        response_text = response[0]['generated_text']  # Access the generated text
        return jsonify({'response': response_text})
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

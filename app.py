from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from transformers import pipeline
from urllib.parse import quote
import logging

# Setting up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # More specific CORS if possible

app.config['PREFERRED_URL_SCHEME'] = 'https'

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'gpt2'

# Check available device and use the appropriate one
pipeline_device = pipeline('text-generation').device
device = 0 if pipeline_device.type == 'cuda' else -1
generator = pipeline('text-generation', model=model_name, tokenizer=model_name, device=device, top_k=50, top_p=0.95, num_return_sequences=1, truncation=True)

@app.route('/')
def home():
    return 'Flask app is running'

@app.route('/api/data')
def get_data():
    return jsonify({'message': 'Data fetched successfully.'})

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        # Your existing logic for text generation
        response = generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
        response_text = response[0]['generated_text']
        encoded_response = quote(response_text)
        return jsonify({'url': f'https://Redtails74.github.io/ClipperAI-/?data={encoded_response}'})
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

# DNS query handler
@app.route('/dns-query', methods=['GET'])
def dns_query():
    name = request.args.get('name')
    query_type = request.args.get('type', 'A')  # Default to A record
    if name:
        # Example response
        return jsonify({
            'name': name,
            'type': query_type,
            'address': '1.2.3.4'  # Mock IP address
        })
    else:
        return jsonify({'error': 'Name parameter is required'}), 400

if __name__ == '__main__':
    # Read host and port from environment variables or use defaults
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 5000))  # Default to 5000 if not set
    
    app.run(debug=False, host=host, port=port)  # Debug mode disabled for production

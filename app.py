from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import os
from transformers import pipeline
from werkzeug.middleware.proxy_fix import ProxyFix  # Add this line

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['PREFERRED_URL_SCHEME'] = 'https'
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'gpt2'

# Check available device and use the appropriate one
pipeline_device = pipeline('text-generation').device
device = 0 if pipeline_device.type == 'cuda' else -1
generator = pipeline('text-generation', model=model_name, tokenizer=model_name, device=device, top_k=50, top_p=0.95, num_return_sequences=1)

@app.route('/')
def home():
    return 'Flask app is running'

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        response = generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
        response_text = response[0]['generated_text']
        return redirect(f'https://Redtails74.github.io/response?data={response_text}')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

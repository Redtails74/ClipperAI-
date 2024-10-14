from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from transformers import pipeline

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'gpt2'  # Specify the model you want to use
generator = pipeline('text-generation', model=model_name, tokenizer=model_name, device=0, top_k=50, top_p=0.95, num_return_sequences=1)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Serve index.html from the main directory

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        # Using the generator correctly
        response = generator(user_message, max_length=100, do_sample=True, num_return_sequences=1)
        response_text = response[0]['generated_text']  # Access the generated text
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)


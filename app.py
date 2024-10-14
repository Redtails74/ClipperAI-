from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'gpt2'  # Specify the model you want to use
inference = InferenceClient(model_name, token=API_KEY)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Serve index.html from the main directory

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        # Use the inference client to generate a response
        response = inference({"input_text": user_message})  # Use a dictionary with the correct key

        # Print the response for debugging
        print(response)  # Log the response for troubleshooting
        
        response_text = response[0]['generated_text']  # Adjust based on the actual response structure
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

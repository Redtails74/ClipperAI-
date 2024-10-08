from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)  # Enable CORS

# Set your Hugging Face API key
API_KEY = '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'
# Load the Inference API for a model (e.g., GPT-2)
model_name = 'gpt2'  # Change this to a different model if desired
inference = InferenceClient(model=model_name, token=API_KEY)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(message="Hello from Flask!")

@app.route('/api/chat', methods=['POST'])
def chat():
    input_text = request.json.get('message')
    
    # Get response from Hugging Face Inference API
    response = inference(input_text)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

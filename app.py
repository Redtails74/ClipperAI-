from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)  # Enable CORS

arameters = {}  # Set your parameters
inference = InferenceClient(parameters)

# Print available methods
print(dir(inference))

# Set your Hugging Face API key (consider using environment variables for security)
API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')  # Fallback to hardcoded value (not recommended for production)

# Load the Inference API for a model (e.g., GPT-2)
model_name = 'gpt2'  # Change this to a different model if desired
inference = InferenceClient(model=model_name, token=API_KEY)

# Example usage with the predict method
input_text = "Your input text here"
response = inference.predict(input_text)  # Use predict instead of calling inference directly
print(response)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(message="Hello from Flask!")

@app.route('/api/chat', methods=['POST'])
def chat():
    input_text = request.json.get('message')
    
    if not input_text:
        return jsonify({'error': 'No input message provided.'}), 400

    # Get response from Hugging Face Inference API
    try:
        response = inference(input_text)
        response_text = response['generated_text']  # Adjust based on actual response structure
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

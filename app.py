from flask import Flask, request, jsonify
from huggingface_hub import InferenceApi

app = Flask(__name__)

# Set your Hugging Face API key
API_KEY = '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'

# Load the Inference API for a model (e.g., GPT-2)
model_name = 'gpt2'  # Change this to a different model if desired
inference = InferenceApi(model=model_name, token=API_KEY)

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.json.get('message')
    
    # Get response from Hugging Face Inference API
    response = inference(input_text)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

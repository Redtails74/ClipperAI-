from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Dynamic model loading
model_name = 'distilgpt2'  # Example model, consider using environment variable for flexibility
device = 0 if torch.cuda.is_available() else -1

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No input message provided.'}), 400

    try:
        # Example of dynamic parameters
        max_length = data.get('max_length', 100)
        do_sample = data.get('do_sample', True)
        num_return_sequences = data.get('num_return_sequences', 1)
        
        response = generator(user_message, max_length=max_length, do_sample=do_sample, 
                              num_return_sequences=num_return_sequences, top_k=50, top_p=0.95)
        return jsonify({'response': response[0]['generated_text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ... rest of your code remains similar

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False) # threaded=False for async if going that route in future

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Configuration
class Config:
    MAX_HISTORY = 10
    MODEL_NAME = 'microsoft/DialoGPT-medium'
     API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')  

app = Flask(__name__,
            static_url_path='/static',
            static_folder='../static',
            template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_auth_token=Config.API_KEY)
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, use_auth_token=Config.API_KEY)
    model.eval()
    # Model might be on CPU by default, move it to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

# Conversation history
conversation_memory = []

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No input message provided'}), 400

    # Add user message to conversation history
    conversation_memory.append(f"user: {user_message}")

    # Construct prompt from conversation history
    prompt = "\n".join(conversation_memory)

    # Tokenize the input
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=1000,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3
        )

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response.split('assistant:')[-1].strip() if 'assistant:' in response else response.split('\n')[-1].strip()

    # Add AI's response to conversation history
    conversation_memory.append(f"assistant: {response_text}")

    # Keep history to the last N messages
    while len(conversation_memory) > Config.MAX_HISTORY:
        conversation_memory.pop(0)

    return jsonify({'response': response_text, 'conversation': list(conversation_memory)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

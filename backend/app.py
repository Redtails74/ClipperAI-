from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForQuestionAnswering
import logging
import os
from collections import deque
import torch
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    MAX_HISTORY = 10
    QA_MODEL_NAME = 'distilbert-base-uncased-distilled-squad'  # QA Model
    DIALOGUE_MODEL_NAME = 'microsoft/DialoGPT-medium'  # DialoGPT Model for dialogue
    API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')

# Setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_url_path='/static',
            static_folder='../static',
            template_folder='./templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load models and tokenizers
qa_model = None
qa_tokenizer = None
dialogue_model = None
dialogue_tokenizer = None
generator = None

@app.before_request
def load_models_on_first_request():
    global qa_model, qa_tokenizer, dialogue_model, dialogue_tokenizer, generator
    if qa_model is None or dialogue_model is None:  # Check if models are already loaded to avoid reloading
        try:
            # Load QA model and tokenizer
            qa_model = AutoModelForQuestionAnswering.from_pretrained(Config.QA_MODEL_NAME)
            qa_tokenizer = AutoTokenizer.from_pretrained(Config.QA_MODEL_NAME)
            
            # Load Dialogue model and tokenizer
            dialogue_model = AutoModelForCausalLM.from_pretrained(Config.DIALOGUE_MODEL_NAME)
            dialogue_tokenizer = AutoTokenizer.from_pretrained(Config.DIALOGUE_MODEL_NAME)
            
            dialogue_model.eval()  # Set dialogue model to evaluation mode
            qa_model.eval()  # Set QA model to evaluation mode
            if torch.cuda.is_available():
                qa_model = qa_model.cuda()
                dialogue_model = dialogue_model.cuda()

            # Initialize the text generation pipeline for DialoGPT
            generator = pipeline('text-generation', model=dialogue_model, tokenizer=dialogue_tokenizer, device=0 if torch.cuda.is_available() else -1)

            logger.info("Models and tokenizers loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models or tokenizers: {e}")
            raise

# Use deque for efficient memory management of conversation history
conversation_memory = deque(maxlen=Config.MAX_HISTORY)

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
    
    try:
        # Check if the message is a question
        if is_question(user_message):
            # Construct context for the QA model
            context = "\n".join([entry for entry in conversation_memory])

            # Tokenize the context and user message
            inputs = qa_tokenizer(context, user_message, return_tensors='pt', truncation=True, max_length=1024, padding=True)
            if torch.cuda.is_available():
                inputs = {key: value.cuda() for key, value in inputs.items()}

            # Generate answer using the QA model
            with torch.no_grad():
                outputs = qa_model(**inputs)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits

            # Decode the answer
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            answer_ids = inputs.input_ids[0][start_idx:end_idx + 1]
            response_text = qa_tokenizer.decode(answer_ids, skip_special_tokens=True)
        else:
            # For conversational responses, use DialoGPT
            prompt = "\n".join([entry for entry in conversation_memory if entry.startswith("user:")])
            response = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
            response_text = response.split("Assistant:")[-1].strip()

        # Append the assistant's response to conversation memory
        conversation_memory.append(f"assistant: {response_text}")

        return jsonify({'response': response_text, 'conversation': list(conversation_memory)})

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def is_question(text):
    """Determine if the user input is a question based on simple heuristics."""
    question_keywords = ['what', 'how', 'why', 'where', 'who', 'when']
    if any(text.lower().startswith(keyword) for keyword in question_keywords) or '?' in text:
        return True
    return False

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')

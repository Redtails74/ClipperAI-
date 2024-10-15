from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import os

if os.getenv('WERKZEUG_RUN_MAIN') == 'true':
    os.environ['WERKZEUG_SERVER_FD'] = str(3)

from transformers import pipeline

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['PREFERRED_URL_SCHEME'] = 'https'
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')
model_name = 'gpt2'  # Specify the model you want to use

# Check available devices
pipeline_device = pipeline('text-generation').device
print(f"Available device: {pipeline_device}")

# Use the available device
if pipeline_device.type == 'cuda':
    device = 0
else:
    device = -1

generator = pipeline('text-generation', model=model_name, tokenizer=model_name, device=device, top_k=50, top_p=0.95, num_return_sequences=1)

@app.route('/')
def home():
    return 'Flask app is running'

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    # Retrieve the query data from the request
    query_data = request.form['query']

    # Process the query, including communication with the Hugging Face model
    response_data = process_query(query_data)

    # Redirect the response back to the GitHub Pages HTML
    return redirect(f'https://Redtails74.github.io/response?data={response_data}')

def process_query(query_data):
    # Implement your logic to process the query and get the response
    # This could include calling the Hugging Face model
    response_data = 'This is a sample response.'
    return response_data

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

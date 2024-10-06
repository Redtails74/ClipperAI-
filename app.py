from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = 'gpt2'  # You can change this to a different model if desired
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.json.get('message')
    
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate a response from the model
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode the output and return the response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from model.llama2_model import LLaMA2Model

app = Flask(__name__)

# Initialize the LLaMA 2 model
model = LLaMA2Model(model_path="model/llama2/")

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# API route to handle chat interactions
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Get the response from the LLaMA 2 model
    response = model.get_response(user_input)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

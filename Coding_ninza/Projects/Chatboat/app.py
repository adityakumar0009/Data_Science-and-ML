from flask import Flask, render_template, request, jsonify
import torch
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import json

app = Flask(__name__)

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load("data.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Function to generate chatbot responses
def get_response(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Get model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=0)
    tag = tags[predicted.item()]

    # Find response for the predicted tag
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    
    return "I'm not sure how to respond to that."

@app.route("/")
def home():
    return render_template("index.html")  # Renders the frontend UI

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_reply = get_response(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)

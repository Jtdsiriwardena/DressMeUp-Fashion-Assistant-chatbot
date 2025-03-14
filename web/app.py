from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load necessary files
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())  # Load intents
model = tf.keras.models.load_model('chatbot_model.h5')  # Load trained model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to clean and process sentence input
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into a bag of words (binary vector)
def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

# Predict class for the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.4  # Confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "unknown"

# Get response for the predicted class
def get_response(intent_tag):
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"]) 
    return "I'm not sure how to respond. Could you ask differently?"


def suggest_outfit(occasion, season, preference):
    outfits = {
        "casual": {
            "summer": {
                "comfortable": "A loose t-shirt with shorts and sandals.",
                "fashionable": "A stylish tank top, high-waisted shorts, and trendy sneakers."
            },
            "winter": {
                "comfortable": "A cozy sweater with jeans and boots.",
                "fashionable": "A chic cardigan, high-waisted trousers, and ankle boots."
            }
        },
        "formal": {
            "summer": {
                "comfortable": "A light, breathable suit with loafers.",
                "fashionable": "A slim-fit suit with a tie, a pocket square, and stylish shoes."
            },
            "winter": {
                "comfortable": "A wool coat with a smart blazer and scarf.",
                "fashionable": "A tailored coat, dark trousers, and leather shoes."
            }
        }
    }
    
    return outfits.get(occasion, {}).get(season, {}).get(preference, "Sorry, no outfit suggestions available.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["messageText"]
    
    # Predict intent
    intent_tag = predict_class(user_input)
    
    if intent_tag == "outfit_suggestion":
     
        occasion = "casual"
        season = "summer"
        preference = "comfortable"
        
      
        response = suggest_outfit(occasion, season, preference)
    else:
       
        response = get_response(intent_tag)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

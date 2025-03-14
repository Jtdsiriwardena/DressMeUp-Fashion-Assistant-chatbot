import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())  # Load intents

# Load trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.7  # Confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > threshold]

    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "unknown"

def get_response(intent_tag):
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond. Could you ask differently?"


def suggest_outfit(occasion, season, preference):
    # Sample outfit suggestions
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

print("Fashion Assistant Chatbot is ready! Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Have a stylish day!")
        break
    
    # Predict intent of user input
    intent_tag = predict_class(user_input)
    
    if intent_tag == "outfit_suggestion":
        occasion = "casual"  # Example default value
        season = "summer"    # Example default value
        preference = "comfortable"  # Example default value
        
        # Suggest an outfit
        response = suggest_outfit(occasion, season, preference)
        print(f"Chatbot: {response}")
    else:
        response = get_response(intent_tag)  # Default response based on intent
        print(f"Chatbot: {response}")

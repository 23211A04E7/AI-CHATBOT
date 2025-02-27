import nltk
import random
import json
import string
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# Predefined responses
responses = {
    "hello": "Hi there! How can I assist you?",
    "how are you": "I'm just a chatbot, but I'm doing great! How about you?",
    "what is your name": "I'm your AI chatbot!",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're welcome!"
}

# Text Preprocessing
lemmatizer = nltk.WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in tokens]

# Training Data Preparation
corpus = list(responses.keys())
vectorizer = TfidfVectorizer(tokenizer=preprocess)
X = vectorizer.fit_transform(corpus)

# Chatbot Response Function
def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_input = " ".join(user_input)
    
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    best_match_index = np.argmax(similarity)
    confidence = similarity[0][best_match_index]

    if confidence > 0.3:  # Response threshold
        return responses[corpus[best_match_index]]
    else:
        return "I'm sorry, I don't understand. Can you rephrase?"

# Flask API for Web Deployment
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    print("Chatbot: Hi! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print(f"Chatbot: {chatbot_response(user_input)}")

    # Run Flask App
    app.run(debug=True)

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
import os
import requests

# Sample training data
training_data = [
   ("What is your purpose?", "My purpose is to aid and serve you on the battlefield."),
    ("Warlock", "Ready."),
    ("What is your name?", "My name is Warlock, it stands for (Warfare-Automated-Response-and-Logistics-Operations-Command-Kit)"),
]

class Chatbot:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_model()

    def train_model(self, training_data):
        X, y = zip(*training_data)
        self.vectorizer = TfidfVectorizer()
        X_vectorized = self.vectorizer.fit_transform(X)
        self.model = SGDClassifier()
        self.model.fit(X_vectorized, y)

    def save_model(self):
        with open("chatbot_model.pkl", "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load_model(self):
        if os.path.exists("chatbot_model.pkl"):
            with open("chatbot_model.pkl", "rb") as f:
                self.vectorizer, self.model = pickle.load(f)
        else:
            self.train_model(training_data)
            self.save_model()

    def get_response(self, user_input):
        user_input_vectorized = self.vectorizer.transform([user_input])
        response = self.model.predict(user_input_vectorized)[0]
        return response

    def add_training_data(self, new_data):
        global training_data
        training_data.append(new_data)
        self.train_model(training_data)
        self.save_model()
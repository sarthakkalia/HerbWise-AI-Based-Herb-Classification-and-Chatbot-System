import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json
import random

app = Flask(__name__)

# Load your trained image classification model
MODEL_PATH = 'model/saved_model.keras'
model = load_model(MODEL_PATH)

# Define class names for image classification
class_names = ['Aloevera', 'Ashwagandha', 'Neem', 'Tulasi']  # Add your class names

# Load the trained chatbot model and vectorizer
with open('model/chatbot_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the intents data for the chatbot
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

# Preprocess the uploaded image for classification
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image
    return img_array

# Predict the class of the uploaded image
def predict_image(model, image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Generate chatbot response
def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break

    return response

# Route for the main page (image upload and chatbot interface)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)  # Create the uploads folder if it doesn't exist

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        predicted_class_name = predict_image(model, filepath)

        return render_template('result.html', prediction=predicted_class_name, image_file=filepath)

# Route to handle chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)

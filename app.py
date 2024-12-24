import cv2
import pytesseract
import numpy as np
import os
import torch
from flask import Flask, render_template, Response, jsonify, redirect, url_for, request
from transformers import MarianMTModel, MarianTokenizer
import joblib
from sentence_transformers import SentenceTransformer
import time
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Charger le modèle de traduction MarianMT
model_name = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Charger le modèle de classification
classifier = joblib.load('random_forest_model.pkl')  # Mettez à jour le nom du fichier

# Charger le modèle d'embedding
embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Variable pour stocker le genre prédit
predicted_genre = None

# Fonction pour traduire le texte du français à l'anglais
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text[0] if translated_text else ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    image_data = request.form['image']
    # Décoder l'image base64
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # Convertir les bytes en image
    image = Image.open(BytesIO(image_bytes))
    image.save('captured_image.jpg')  # Sauvegarder l'image capturée
    
    # Rediriger vers la page d'attente pendant le traitement
    return redirect(url_for('processing'))

@app.route('/processing')
def processing():
    # Simuler un délai de traitement avec un GIF d'attente
    return render_template('processing.html')

@app.route('/process_data')
def process_data():
    global predicted_genre  # Utiliser la variable globale pour stocker le genre

    # Charger l'image capturée
    frame = cv2.imread('captured_image.jpg')

    if frame is None:
        return jsonify({'error': 'Could not load captured image'})

    # Prétraitement de l'image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # OCR
    text = pytesseract.image_to_string(thresh, lang='fra')
    print("Texte extrait:", text)

    # Traduire le texte extrait
    translated_text = translate(text)
    print("Texte traduit:", translated_text)

    # Vectoriser le texte traduit
    vectorized_text = embed.encode(translated_text)

    # Prédire le genre du livre
    predicted_genre = classifier.predict([vectorized_text])[0]
    print("Genre prédit:", predicted_genre)

    # Simuler un délai pour voir le GIF
    time.sleep(5)  # Temps pour le traitement (à ajuster selon besoin)

    # Rediriger vers la page de résultat
    return redirect(url_for('result'))

@app.route('/result')
def result():
    global predicted_genre
    if predicted_genre is None:
        return redirect(url_for('index'))  # Si pas de genre prédit, retourner à la page d'accueil

    return render_template('result.html', genre=predicted_genre)

if __name__ == '__main__':
    app.run(debug=True)

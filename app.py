import os
import cv2
import dlib
import numpy as np
from keras.models import load_model
import streamlit as st

# Constants
IMAGE_SIZE = 48
NORMALIZATION_FACTOR = 255.0

# Paths and model setup
folder_path = st.file_uploader("Select a folder:", type=["jpeg", "png", "img"])
model_path = "model_optimal_v3.h5"
detector = dlib.get_frontal_face_detector()
model = load_model(model_path)
emotion_labels = ['angry', 'disgust', 'afraid', 'happy', 'neutral', 'sad', 'surprise']
stressed_emotions = ['sad', 'afraid', 'disgust', 'angry']
non_stressed_emotions = ['happy', 'neutral', 'surprise']

def preprocess_image(gray: np.ndarray) -> np.ndarray:
    """Preprocess an image for emotion detection"""
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized / NORMALIZATION_FACTOR
    processed_image = np.expand_dims(normalized, axis=-1)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def predict_emotion(processed_image: np.ndarray) -> np.ndarray:
    """Predict emotions from a preprocessed image"""
    return model.predict(processed_image)

def classify_emotions(probabilities: np.ndarray) -> float:
    """Classify emotions into stressed and non-stressed categories"""
    stressed_prob = 0
    non_stressed_prob = 0

    for i, label in enumerate(emotion_labels):
        if label in stressed_emotions:
            stressed_prob += probabilities[0][i]
        elif label in non_stressed_emotions:
            non_stressed_prob += probabilities[0][i]

    total = stressed_prob + non_stressed_prob
    if total == 0:  # Avoid division by zero
        return 0
    stress_percentage = (stressed_prob / total) * 100
    return stress_percentage

def analyze_image(image_path: str) -> float:
    """Analyze an image and return the average stress level"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    stress_levels = []

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cropped_face = gray[y:y+h, x:x+w]
        processed_image = preprocess_image(cropped_face)
        probabilities = predict_emotion(processed_image)
        stress_level = classify_emotions(probabilities)
        stress_levels.append(stress_level)

    if not stress_levels:  # No faces detected
        return None
    return np.mean(stress_levels)

def process_folder(folder_path: str) -> None:
    """Process all images in a folder and display stress analysis results"""
    stress_results = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            average_stress = analyze_image(image_path)
            stress_results[filename] = average_stress

    st.write("Stress Analysis Results:")
    for filename, stress in stress_results.items():
        if stress is None:
            st.write(f"{filename}: No faces detected.")
        else:
            st.write(f"{filename}: Average Stress Level = {stress:.2f}%")

def main() -> None:
    """Main entry point"""
    if folder_path is not None:
        try:
            process_folder(folder_path)
        except Exception as e:
            st.write(f"An error occurred while processing the folder:\n\n```\n{str(e)}\n```")

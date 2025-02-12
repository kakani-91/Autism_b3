import os
import gdown
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

app = Flask(__name__)

# Google Drive model URL (replace with your actual file ID)
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1Ee4UtNvwhkPBsNjjppcSRrWGYPPrRs3f/view?usp=sharing"
MODEL_PATH = "vgg19_fine_tuned_model.keras"

# Download model if it does not exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to detect if the uploaded image contains a face
def detect_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return len(faces) > 0  # Returns True if at least one face is detected

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to match VGG19 input
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded. Please upload an image.")

        file = request.files["file"]
        
        if file.filename == "":
            return render_template("index.html", message="No file selected. Please choose an image.")

        # Save the uploaded file
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)

        # Check if the uploaded image contains a face
        if not detect_face(file_path):
            return render_template("index.html", message="Please upload a valid face image for prediction.")

        # Preprocess and make prediction
        image = preprocess_image(file_path)
        prediction = model.predict(image)[0][0]  # Get prediction score
        
        # Determine prediction label
        result = "Autism Detected" if prediction > 0.5 else "No Autism Detected"

        return render_template("index.html", prediction=result, image_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)  # Ensure upload directory exists
    app.run(debug=True)

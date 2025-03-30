from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load Best Dyslexia Model
DYSLEXIA_MODEL_PATH = "dyslexia_model.keras"
if not os.path.exists(DYSLEXIA_MODEL_PATH):
    raise FileNotFoundError(f"❌ Dyslexia model not found: {DYSLEXIA_MODEL_PATH}")

print("🟡 Loading Dyslexia model...")
dyslexia_model = tf.keras.models.load_model(DYSLEXIA_MODEL_PATH, compile=False)

# Recompile the model with Adam optimiser
dyslexia_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss="binary_crossentropy",
                       metrics=["accuracy"])
print("✅ Dyslexia model loaded and compiled!")


# Load Dysgraphia Model
DYS_GRAPHIA_MODEL_PATH = "best_svm_model.pkl"
if not os.path.exists(DYS_GRAPHIA_MODEL_PATH):
    raise FileNotFoundError(f"❌ Dysgraphia model not found: {DYS_GRAPHIA_MODEL_PATH}")
print("🟡 Loading Dysgraphia model...")
dysgraphia_model = joblib.load(DYS_GRAPHIA_MODEL_PATH)
print("✅ Dysgraphia model loaded and compiled!")


# Load Optimal Threshold from File
THRESHOLD_PATH = "optimal_threshold.txt"
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as f:
        optimal_threshold = float(f.read().strip())
    print(f"✅ Using Optimal Threshold: {optimal_threshold:.4f}")
else:
    raise FileNotFoundError("❌ Threshold file not found! Run `threshold_finder.py` first.")

# Feature Extraction for Dysgraphia
def extract_features(image):
    image_resized = cv2.resize(image, (64, 64))  # Resize to match training data size
    return image_resized.flatten() / 255.0  # Normalize

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        print("📌 Received file:", file.filename)

        # Read image as grayscale
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("🚨 Error: Image could not be decoded.")
            return jsonify({"error": "Invalid image format"}), 400

        print("✅ Image successfully decoded")

        # Preprocess Image for Dyslexia Model
        image_resized = cv2.resize(image, (96, 96))  # Resize to match model input
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        image_reshaped = image_rgb.reshape(1, 96, 96, 3) / 255.0  # Normalize

        # Dyslexia Prediction (Apply dynamic threshold)
        dyslexia_pred = dyslexia_model.predict(image_reshaped)[0, 0]  # Extract scalar probability
        dyslexia_result = "Potential Dyslexia" if dyslexia_pred < optimal_threshold else "Normal"

        print(f"✅ Dyslexia Prediction: {dyslexia_result} (Prob: {dyslexia_pred:.4f}, Threshold: {optimal_threshold:.4f})")

        # Dysgraphia Prediction
        features = extract_features(image)
        dysgraphia_pred = dysgraphia_model.predict([features])[0]
        dysgraphia_result = "Potential Dysgraphia" if dysgraphia_pred == 0 else "Normal"

        print(f"✅ Dysgraphia prediction result: {dysgraphia_result}")

        return jsonify({
            "dyslexia": dyslexia_result,
            "dysgraphia": dysgraphia_result
        })

    except Exception as e:
        print(f"🚨 Error processing image: {str(e)}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

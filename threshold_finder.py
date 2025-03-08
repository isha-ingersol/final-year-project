import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Load best trained model
MODEL_PATH = "dyslexia_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Define training dataset path
TRAIN_DATASET_PATH = "/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting/Train"

# ✅ Image Preprocessing (same as training)
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2  # Use 20% of training data for validation

datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=VALIDATION_SPLIT)

# ✅ Load Validation Dataset (from Train split)
val_dataset = datagen.flow_from_directory(
    TRAIN_DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",  # Ensures labels are numerical (0, 1, 2)
    subset="validation",
    shuffle=False
)

# ✅ Get raw probability predictions on validation set
val_probs = model.predict(val_dataset)  # Shape: (num_samples, 1)
val_probs = val_probs.flatten()  # Convert to 1D array

# ✅ Get true labels
y_true = val_dataset.classes  # Ground truth labels

# ✅ Convert Multi-Class Labels to Binary
# Assuming "Reversal" = 2 (Dyslexic), and "Normal"/"Corrected" = 0,1 (Non-Dyslexic)
y_true = np.where(y_true == 2, 1, 0)  # Convert to binary: Dyslexic (1) vs Normal (0)

# ✅ Plot probability distribution
plt.hist(val_probs, bins=30, edgecolor='black')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Dyslexia Model Probability Distribution on Validation Data")
plt.show()

# ✅ Compute optimal threshold using Youden’s J statistic
fpr, tpr, thresholds = roc_curve(y_true, val_probs)  # Binary labels
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"🔍 Optimal Classification Threshold Found: {optimal_threshold:.4f}")

# ✅ Save the threshold value for app.py
with open("optimal_threshold.txt", "w") as f:
    f.write(str(optimal_threshold))
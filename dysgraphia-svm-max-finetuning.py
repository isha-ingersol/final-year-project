import os
import warnings
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import albumentations as A
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define dataset paths
dysgraphia_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dysgraphia-handwriting'
dyslexia_normal_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting/Test/Normal'
categories = ['dysgraphia', 'normal']

# Define Augmentation
transform = A.Compose([
    A.GaussNoise(var_limit=(10, 50), p=0.5),  
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5)
])

# Load dataset
def load_dataset():
    data, labels = [], []
    
    # Merge dysgraphia categories
    for category in ['low-potential-dysgraphia', 'potential-dysgraphia']:
        category_path = os.path.join(dysgraphia_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Smaller size for SVM
                img = img / 255.0  # Normalize to range [0, 1]

                # Convert the image to uint8 before augmentation
                img_uint8 = (img * 255).astype(np.uint8)
                img_aug = transform(image=img_uint8)['image']  # Apply Augmentation

                # Convert back to float32 if necessary
                img_aug = img_aug.astype(np.float32) / 255.0

                img_flat = img_aug.flatten()  # Flatten into a 1D vector
                data.append(img_flat)
                labels.append(0)  # 0 for dysgraphia
    
    # Randomly sample 250 normal handwriting images from dyslexia dataset
    normal_images = [f for f in os.listdir(dyslexia_normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(normal_images) == 0:
        print("⚠️ Warning: No normal handwriting images found. Check the dataset path!")
    else:
        sampled_normal_images = random.sample(normal_images, min(250, len(normal_images)))
        for img_name in sampled_normal_images:
            img_path = os.path.join(dyslexia_normal_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img / 255.0  # Normalize
                
                # Convert the image to uint8 before augmentation
                img_uint8 = (img * 255).astype(np.uint8)
                img_aug = transform(image=img_uint8)['image']  # Apply Augmentation
                
                # Convert back to float32 if necessary
                img_aug = img_aug.astype(np.float32) / 255.0
                
                img_flat = img_aug.flatten()
                data.append(img_flat)
                labels.append(1)  # 1 for normal-writing
    
    data = np.array(data)
    labels = np.array(labels)
    print(f"Total Samples: {len(data)}")
    print("Class Distribution:", dict(zip(*np.unique(labels, return_counts=True))))
    return data, labels

# Load data
data, labels = load_dataset()

# Split dataset (70% Train, 15% Validation, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Apply Noise to Training Data
noise_factor = 0.05
X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)

# Check dataset distribution
print("Training Set Distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Validation Set Distribution:", dict(zip(*np.unique(y_val, return_counts=True))))
print("Test Set Distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

# Check for data leakage
train_files = set([img.tobytes() for img in X_train])
test_files = set([img.tobytes() for img in X_test])
print("Overlap Found (Train-Test):", len(train_files.intersection(test_files)))

# Check Train-Test Similarity
train_sample = X_train[:100]  # Random subset
test_sample = X_test[:100]
similarity_matrix = cosine_similarity(train_sample, test_sample)
print(f"Max Similarity Score (Train-Test): {similarity_matrix.max()}")

# Define and fine-tune SVM model
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))

# Train model
svm_model.fit(X_train_noisy, y_train)

# Evaluate model
val_preds = svm_model.predict(X_val)
test_preds = svm_model.predict(X_test)

val_acc = accuracy_score(y_val, val_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f'Validation Accuracy: {val_acc * 100:.2f}%')
print(f'Test Accuracy: {test_acc * 100:.2f}%')
print("Classification Report:")
print(classification_report(y_test, test_preds, target_names=categories))
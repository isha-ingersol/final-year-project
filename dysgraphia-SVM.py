import os
import warnings
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=UserWarning)

# Define dataset paths
dysgraphia_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dysgraphia-handwriting'
dyslexia_normal_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting/Test/Normal'
categories = ['dysgraphia', 'normal']

# Augmentation
transform = A.Compose([
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5)
])

def load_dataset():
    data, labels = [], []

    # --- Dysgraphia: Load exactly 249 images ---
    dysgraphia_images = []
    for category in ['low-potential-dysgraphia', 'potential-dysgraphia']:
        category_path = os.path.join(dysgraphia_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            dysgraphia_images.append(img_path)
    sampled_dysgraphia = random.sample(dysgraphia_images, 249)

    for img_path in sampled_dysgraphia:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            img_uint8 = (img * 255).astype(np.uint8)
            img_aug = transform(image=img_uint8)['image']
            img_aug = img_aug.astype(np.float32) / 255.0
            img_flat = img_aug.flatten()
            data.append(img_flat)
            labels.append(0)  # dysgraphia

    # --- Normal: Load exactly 250 images ---
    normal_images = [f for f in os.listdir(dyslexia_normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sampled_normal = random.sample(normal_images, 250)

    for img_name in sampled_normal:
        img_path = os.path.join(dyslexia_normal_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            img_uint8 = (img * 255).astype(np.uint8)
            img_aug = transform(image=img_uint8)['image']
            img_aug = img_aug.astype(np.float32) / 255.0
            img_flat = img_aug.flatten()
            data.append(img_flat)
            labels.append(1)  # normal

    data = np.array(data)
    labels = np.array(labels)
    print(f"✅ Total Samples: {len(data)}")
    print("Class Distribution:", dict(zip(*np.unique(labels, return_counts=True))))
    return data, labels

# Load and split data
data, labels = load_dataset()
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Add noise to training data
noise_factor = 0.05
X_train_noisy = np.clip(X_train + noise_factor * np.random.normal(size=X_train.shape), 0., 1.)

# Sanity checks
print("Training Set Distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Validation Set Distribution:", dict(zip(*np.unique(y_val, return_counts=True))))
print("Test Set Distribution:", dict(zip(*np.unique(y_test, return_counts=True))))
similarity_matrix = cosine_similarity(X_train[:100], X_test[:100])
print(f"Max Similarity Score (Train-Test): {similarity_matrix.max():.4f}")

# Train SVM model
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))
svm_model.fit(X_train_noisy, y_train)

# Predict
test_preds = svm_model.predict(X_test)
y_score = svm_model.decision_function(X_test)

# Metrics
test_acc = accuracy_score(y_test, test_preds)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print("📋 Classification Report:")
print(classification_report(y_test, test_preds, target_names=categories))

# Confusion Matrix
cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix – Dysgraphia SVM")
plt.grid(False)
plt.tight_layout()
plt.show()

# ROC & AUC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(f"🎯 AUC Score: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve – Dysgraphia SVM')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
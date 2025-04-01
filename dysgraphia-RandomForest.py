import os
import warnings
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import seaborn as sns
import albumentations as A

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Dataset paths
dysgraphia_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dysgraphia-handwriting'
dyslexia_normal_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting/Test/Normal'
categories = ['dysgraphia', 'normal']

# Augmentation
transform = A.Compose([
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5)
])

# Load dataset
def load_dataset():
    data, labels = [], []

    # Dysgraphia: merge both folders
    for category in ['low-potential-dysgraphia', 'potential-dysgraphia']:
        category_path = os.path.join(dysgraphia_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img / 255.0

                img_uint8 = (img * 255).astype(np.uint8)
                img_aug = transform(image=img_uint8)['image']
                img_aug = img_aug.astype(np.float32) / 255.0

                data.append(img_aug.flatten())
                labels.append(0)  # dysgraphia

    # Normal: randomly sample from dyslexia Test/Normal
    normal_images = [f for f in os.listdir(dyslexia_normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sampled_images = random.sample(normal_images, min(250, len(normal_images)))
    
    for img_name in sampled_images:
        img_path = os.path.join(dyslexia_normal_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img / 255.0

            img_uint8 = (img * 255).astype(np.uint8)
            img_aug = transform(image=img_uint8)['image']
            img_aug = img_aug.astype(np.float32) / 255.0

            data.append(img_aug.flatten())
            labels.append(1)  # normal

    data = np.array(data)
    labels = np.array(labels)

    print(f"Total Samples: {len(data)}")
    print("Class Distribution:", dict(zip(*np.unique(labels, return_counts=True))))
    return data, labels

# Load and split
data, labels = load_dataset()
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Define Random Forest pipeline
rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))

# Train
rf_model.fit(X_train, y_train)

# Predict
val_preds = rf_model.predict(X_val)
test_preds = rf_model.predict(X_test)
test_probs = rf_model.predict_proba(X_test)[:, 1]

# Scores
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds) * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_test, test_preds) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, test_preds, target_names=categories))

# Confusion Matrix
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, test_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
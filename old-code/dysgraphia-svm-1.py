import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Define dataset paths
data_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dysgraphia-handwriting'
categories = ['low-potential-dysgraphia', 'potential-dysgraphia']

# Function to apply SWT and extract stroke width features
def apply_swt(image):
    edges = cv2.Canny(image, 50, 150)  # Edge detection
    swt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)  # Apply distance transform as SWT approximation
    mean_stroke_width = np.mean(swt[swt > 0])  # Compute mean stroke width
    std_stroke_width = np.std(swt[swt > 0])    # Compute stroke width variance
    contrast = np.max(swt) - np.min(swt)  # Stroke contrast
    return mean_stroke_width, std_stroke_width, contrast

# Load dataset and extract SWT features
def load_dataset():
    data, labels = [], []
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                mean_swt, std_swt, contrast = apply_swt(img)  # Apply SWT
                data.append([mean_swt, std_swt, contrast])
                labels.append(label)
    return np.array(data), np.array(labels)

# Load data
data, labels = load_dataset()

# Split dataset (70% Train, 15% Validation, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='linear', probability=True)  # Enable probability for ROC
svm.fit(X_train, y_train)

# Evaluate model on validation set
y_val_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Evaluate model on test set
y_test_pred = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=categories))

# Function to plot ROC curve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

# Generate ROC curve
y_test_scores = svm.predict_proba(X_test)[:, 1]  # Get probability scores for class 1
plot_roc_curve(y_test, y_test_scores)

# Generate Confusion Matrix
plot_confusion_matrix(y_test, y_test_pred, categories)
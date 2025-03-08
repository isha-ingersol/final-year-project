import os
import warnings
import tensorflow as tf
import numpy as np
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import logging

# Suppress all unnecessary TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define dataset paths
dysgraphia_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dysgraphia-handwriting'
dyslexia_normal_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting/Test/Normal'
categories = ['dysgraphia', 'normal']

# Load dataset
def load_dataset():
    data, labels = [], []
    max_height, max_width = 0, 0
    
    # First pass: Find max dimensions
    for category in ['low-potential-dysgraphia', 'potential-dysgraphia']:
        category_path = os.path.join(dysgraphia_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                max_height = max(max_height, h)
                max_width = max(max_width, w)
    
    # Merge dysgraphia categories
    for category in ['low-potential-dysgraphia', 'potential-dysgraphia']:
        category_path = os.path.join(dysgraphia_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                img = np.expand_dims(img, axis=-1)
                img = img / 255.0
                data.append(img)
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
                img = cv2.resize(img, (256, 256))
                img = np.expand_dims(img, axis=-1)
                img = img / 255.0
                data.append(img)
                labels.append(1)  # 1 for normal-writing
    
    data = np.array(data)
    labels = to_categorical(np.array(labels), num_classes=len(categories))
    print(f"Total Samples: {len(data)}")
    print("Class Distribution:", dict(zip(*np.unique(np.argmax(labels, axis=1), return_counts=True))))
    return data, labels

# Load data
data, labels = load_dataset()

# Split dataset (70% Train, 15% Validation, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Compute class weights safely
unique_classes = np.unique(np.argmax(labels, axis=1))
if len(unique_classes) > 1:
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=np.argmax(labels, axis=1))
    class_weight_dict = {i: class_weights[i] for i in range(len(categories))}
else:
    print("⚠️ Warning: Only one class detected. Skipping class weights.")
    class_weight_dict = None

# Enhanced Data Augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest',
)

# Define CNN model
def create_cnn_model():
    model = Sequential([
        Input(shape=(256, 256, 1)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.6),
        Dense(len(categories), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train CNN model
cnn_model = create_cnn_model()
history = cnn_model.fit(
    data_augmentation.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=25,
    class_weight=class_weight_dict
)

# Evaluate model on test set
test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Define dataset paths
data_dir = '/Users/ishaingersol/Desktop/fyp-dataset/dysgraphia-handwriting'
categories = ['low-potential-dysgraphia', 'potential-dysgraphia']

# Load dataset
def load_dataset():
    data, labels = [], []
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            img = np.expand_dims(img, axis=-1)  # Ensure correct shape
            img = img / 255.0  # Normalize
            data.append(img)
            labels.append(label)
    data = np.array(data)
    labels = to_categorical(np.array(labels), num_classes=len(categories))
    print(f"Total Samples: {len(data)}")
    return data, labels

# Load data
data, labels = load_dataset()

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define CNN-LSTM hybrid model (similar to Dyslexia model)
def create_model():
    model = Sequential([
        tf.keras.Input(shape=(None, None, 1)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        GlobalAveragePooling2D(),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(len(categories), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train model
model = create_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Save model
model.save('dysgraphia_model.h5')
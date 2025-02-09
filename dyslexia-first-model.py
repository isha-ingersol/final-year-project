import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras
from tensorflow.keras import layers

# Set paths
data_dir = "/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting"
train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")

##### 1. Data Loading Using Keras #####
img_size = (64, 64)
batch_size = 128  # Increased batch size to reduce total number of batches
sample_size = 20000  # Use only 10000 random samples

# Function to check if file is a valid image
def is_valid_image(file_path):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

# Function to load and preprocess images safely
def load_and_preprocess_image(file_path, label):
    def _load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)  # Ensure 3 channels (RGB)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
        return image

    image = tf.py_function(func=_load_image, inp=[file_path], Tout=tf.float32)
    image.set_shape([64, 64, 3])  # Ensure static shape is assigned
    label = tf.cast(label, tf.int32)  # Properly cast labels
    
    return image, label

# Create dataset from Train and Test directories
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)

image_paths = []
labels = []
for label, class_name in enumerate(class_names):
    class_dir = os.path.join(train_dir, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if file_name.startswith('.') or not os.path.isfile(file_path) or not is_valid_image(file_path):  # Ignore hidden/non-image files
            continue
        image_paths.append(file_path)
        labels.append(label)

image_paths = np.array(image_paths, dtype=np.string_)
labels = np.array(labels, dtype=np.int32)

# Randomly select 5000 samples
indices = np.random.choice(len(image_paths), sample_size, replace=False)
image_paths = image_paths[indices]
labels = labels[indices]

# Split dataset: 70% train, 15% validation, 15% test
train_split = int(0.7 * len(image_paths))
val_split = int(0.85 * len(image_paths))

train_image_paths, val_image_paths, test_image_paths = image_paths[:train_split], image_paths[train_split:val_split], image_paths[val_split:]
train_labels, val_labels, test_labels = labels[:train_split], labels[train_split:val_split], labels[val_split:]

# Debugging: Check if dataset is correctly loaded
print(f"Total samples: {len(image_paths)}")
print(f"Training samples: {len(train_image_paths)}")
print(f"Validation samples: {len(val_image_paths)}")
print(f"Testing samples: {len(test_image_paths)}")
if len(val_image_paths) == 0 or len(test_image_paths) == 0:
    print("ERROR: No validation or test samples! Check dataset split.")

# Create TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))

train_dataset = train_dataset.map(load_and_preprocess_image).shuffle(512).batch(batch_size)
val_dataset = val_dataset.map(load_and_preprocess_image).batch(batch_size)
test_dataset = test_dataset.map(load_and_preprocess_image).batch(batch_size)

##### 2. Model Definition Using Keras #####
def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

input_shape = (64, 64, 3)
model = build_model(input_shape, num_classes)

##### 3. Training Using Keras #####

# Debug dataset before training
for img, lbl in train_dataset.take(1):
    print(f"Image Shape: {img.shape}, Label: {lbl}")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    verbose=1
)

##### 4. Evaluation on Test Set #####
def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.2f}")

evaluate_model(model, test_dataset)
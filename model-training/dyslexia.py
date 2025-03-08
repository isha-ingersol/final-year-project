import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set paths
data_dir = "/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting"
train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")

##### 1. Data Loading & Preprocessing #####
img_size = (64, 64)
batch_size = 128
sample_size = 15000  # Limit dataset size for efficiency

# Function to load and preprocess images
def load_and_preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize
    return image, label

# Get class names
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)

# Load file paths and labels
image_paths, labels = [], []
for label, class_name in enumerate(class_names):
    class_dir = os.path.join(train_dir, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if file_name.startswith('.') or not os.path.isfile(file_path):
            continue
        image_paths.append(file_path)
        labels.append(label)

# Convert to numpy arrays
image_paths = np.array(image_paths, dtype=np.string_)
labels = np.array(labels, dtype=np.int32)

# Randomly select samples
indices = np.random.choice(len(image_paths), sample_size, replace=False)
image_paths, labels = image_paths[indices], labels[indices]

# Shuffle before splitting
shuffled_indices = np.random.permutation(len(image_paths))
image_paths, labels = image_paths[shuffled_indices], labels[shuffled_indices]

# Split dataset: 70% train, 15% validation, 15% test
train_split, val_split = int(0.7 * len(image_paths)), int(0.85 * len(image_paths))
train_image_paths, val_image_paths, test_image_paths = image_paths[:train_split], image_paths[train_split:val_split], image_paths[val_split:]
train_labels, val_labels, test_labels = labels[:train_split], labels[train_split:val_split], labels[val_split:]

# Convert to TensorFlow datasets
def create_tf_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(train_image_paths, train_labels)
val_dataset = create_tf_dataset(val_image_paths, val_labels)
test_dataset = create_tf_dataset(test_image_paths, test_labels)

print(f"Total samples: {len(image_paths)} | Training: {len(train_image_paths)}, Validation: {len(val_image_paths)}, Testing: {len(test_image_paths)}")

##### 2. Improved Model Definition with Reduced Augmentation #####
def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Apply Data Augmentation inside the model (Reduced Strength)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)  # Reduced from 0.1
    x = layers.RandomZoom(0.05)(x)  # Reduced from 0.1
    x = layers.RandomContrast(0.05)(x)  # Reduced from 0.1

    # Convolutional layers with Stronger L2 Regularization
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.Dropout(0.6)(x)  # Increased from 0.5
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Optimizer with Fixed Learning Rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

input_shape = (64, 64, 3)

# Check if model exists, else create a new one
model_path = "dyslexia_model.h5"
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = keras.models.load_model(model_path)
else:
    model = build_model(input_shape, num_classes)

##### 3. Improved Training Strategy #####
early_stopping = keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor="val_loss"
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    factor=0.5, patience=2, verbose=1, monitor="val_loss"
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

##### 4. Enhanced Evaluation #####
def evaluate_model(model, test_dataset):
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Final Accuracy
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"\n✅ Test Accuracy: {test_accuracy:.2f}")

evaluate_model(model, test_dataset)
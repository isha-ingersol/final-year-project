####### GRAD-CAM DID NOT WORK #########


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend as K
import pandas as pd
import cv2
from sklearn.metrics import roc_auc_score, roc_curve
import random


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy("mixed_float16")

##### Step 1: Data Pre-Processing and Loading #####
# Set dataset paths
data_dir = "/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting"
train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")

# Image settings
img_size = (96, 96)
batch_size = 128
sample_size = 75000  # Limit dataset size to prevent overfitting

# Define binary class mapping
class_map = {"Corrected": 0, "Reversal": 1, "Normal": 0}

# Load dataset paths & labels
image_paths, labels = [], []
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if file_name.startswith('.') or not os.path.isfile(file_path):
            continue
        image_paths.append(file_path)
        labels.append(class_map[class_name])

# Convert to NumPy arrays
image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)

# Balance dataset: Ensure equal Dyslexic & Normal cases
dyslexic_indices = np.where(labels == 1)[0]
normal_indices = np.where(labels == 0)[0]

num_dyslexic = len(dyslexic_indices)
num_normal = min(len(normal_indices), num_dyslexic)  # Keep equal class balance

# Randomly sample from both classes
balanced_dyslexic_indices = np.random.choice(dyslexic_indices, num_dyslexic, replace=False)
balanced_normal_indices = np.random.choice(normal_indices, num_normal, replace=False)

# Combine and limit to 50k total samples
selected_indices = np.concatenate([balanced_dyslexic_indices, balanced_normal_indices])
if len(selected_indices) > sample_size:
    selected_indices = np.random.choice(selected_indices, sample_size, replace=False)

# Apply selection to dataset
image_paths, labels = image_paths[selected_indices], labels[selected_indices]

# Split dataset (70% train, 15% val, 15% test)
train_split, val_split = int(0.7 * len(image_paths)), int(0.85 * len(image_paths))
train_image_paths, val_image_paths, test_image_paths = image_paths[:train_split], image_paths[train_split:val_split], image_paths[val_split:]
train_labels, val_labels, test_labels = labels[:train_split], labels[train_split:val_split], labels[val_split:]

##### Step 2: Data Augmentation #####
# Data augmentation to prevent overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.25),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
])

# Image preprocessing function
def load_and_preprocess_image(file_path, label, augment=False):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize
    if augment:
        image = data_augmentation(image)
    return image, label

# Create TensorFlow datasets with caching & prefetching
def create_tf_dataset(image_paths, labels, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(train_image_paths, train_labels, augment=True)
val_dataset = create_tf_dataset(val_image_paths, val_labels)
test_dataset = create_tf_dataset(test_image_paths, test_labels)

print(f"Total balanced samples: {len(image_paths)} | Training: {len(train_image_paths)}, Validation: {len(val_image_paths)}, Testing: {len(test_image_paths)}")

##### Step 3: Adjusted Class Weights for Dyslexia #####
class_weights = {0: 1.0, 1: 1.8}  # Increased Dyslexic class weight

# Load previous model if it exists, else throw an error
previous_model_path = "dyslexia_model.keras"
if os.path.exists(previous_model_path):
    print(f"\n🔄 Loading best saved model from {previous_model_path}...")
    model = keras.models.load_model(previous_model_path)
else:
    raise FileNotFoundError(f"\n❌ {previous_model_path} not found! Train a model first.")

##### Step 4: Define a Custom F1-score Metric #####
def f1_score(y_true, y_pred):
    precision = K.sum(K.round(y_true * y_pred)) / (K.sum(K.round(y_pred)) + K.epsilon())
    recall = K.sum(K.round(y_true * y_pred)) / (K.sum(K.round(y_true)) + K.epsilon())
    return (2 * precision * recall) / (precision + recall + K.epsilon())

##### Step 5: Fine-tuning #####
# Fine-tuning: Unfreeze last 15 layers
for layer in model.get_layer("mobilenetv2_1.00_96").layers[-15:]:
    layer.trainable = True

# Recompile with adjusted learning rate for fine-tuning
model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6),   
              loss="binary_crossentropy", 
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name="auc"), f1_score])

##### Step 6: Training with Checkpoints #####
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)

os.makedirs("models", exist_ok=True)

# Save the best model during training
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "models/dyslexia-mobilenetv2_epoch-{epoch:02d}_val-acc-{val_accuracy:.4f}_val-loss-{val_loss:.4f}_f1score-{val_f1_score:.4f}.keras",
    monitor="val_f1_score",
    save_best_only=False,
    save_weights_only=False,
    mode="max",
    verbose=1
)

history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=15,  
    class_weight=class_weights,  
    callbacks=[early_stopping, reduce_lr, checkpoint_callback]
)

print(f"\n✅ Training completed. Best model saved as {previous_model_path}")

# Get the filepath of the best model saved during training
best_model_path = checkpoint_callback.filepath.format(
    epoch=history.epoch[np.argmax(history.history["val_f1_score"])]+1,
    val_accuracy=max(history.history["val_accuracy"]),
    val_loss=min(history.history["val_loss"]),
    val_f1_score=max(history.history["val_f1_score"])
)

print(f"🏆 Best model saved as: {best_model_path}")

##### Step 7: Evaluate Model with Adjusted Threshold #####
def evaluate_model(model, test_dataset, threshold=0.55):  
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred.extend((predictions > threshold).astype(int))  
        y_true.extend(labels.numpy())

    #  Confusion Matrix:

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Dyslexic"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["Normal", "Dyslexic"], yticklabels=["Normal", "Dyslexic"], fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


    # AUC-ROC Curve
    probs = []
    for images, _ in test_dataset:
        probs.extend(model.predict(images).flatten())

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


evaluate_model(model, test_dataset)


# Add Grad-CAM Visualisation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Class 0 = Normal, 1 = Dyslexic

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4, cmap="jet"):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = heatmap_color * alpha + image
    return np.uint8(overlayed)

def show_gradcam(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    img_resized = tf.image.resize(img, img_size)
    img_normalized = tf.image.convert_image_dtype(img_resized, tf.float32)
    input_array = tf.expand_dims(img_normalized, axis=0)

    heatmap = make_gradcam_heatmap(input_array, model)
    original = img_resized.numpy()
    original_uint8 = np.uint8(original * 255)
    cam_img = overlay_heatmap(original_uint8, heatmap)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original.astype("uint8"))
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cam_img)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



# Save Training History to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history_dyslexia_mobilenetv2.csv", index=False)
print("📁 Training history saved to 'training_history_dyslexia_mobilenetv2.csv'")


# Visualise Grad-CAM on one "random" test image
random_index = random.randint(0, len(test_image_paths) - 1)
show_gradcam(test_image_paths[random_index])

# Visualise Grad-CAM on one Mr Hill's test image
show_gradcam("/Users/ishaingersol/Desktop/fyp-dataset/test-nick-hill.png")
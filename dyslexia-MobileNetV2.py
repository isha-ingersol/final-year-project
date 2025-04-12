import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import random

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy("mixed_float16")

##### Step 1: Data Preprocessing #####
data_dir = "/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting"
train_dir = os.path.join(data_dir, "Train")
img_size = (96, 96)
batch_size = 128
sample_size = 75000

class_map = {"Corrected": 0, "Reversal": 1, "Normal": 0}
image_paths, labels = [], []

for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for file in os.listdir(class_dir):
        path = os.path.join(class_dir, file)
        if not file.startswith('.') and os.path.isfile(path):
            image_paths.append(path)
            labels.append(class_map[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)

# Balance classes
dyslexic_indices = np.where(labels == 1)[0]
normal_indices = np.where(labels == 0)[0]
num_dyslexic = len(dyslexic_indices)
num_normal = min(len(normal_indices), num_dyslexic)

selected_indices = np.concatenate([
    np.random.choice(dyslexic_indices, num_dyslexic, replace=False),
    np.random.choice(normal_indices, num_normal, replace=False)
])
if len(selected_indices) > sample_size:
    selected_indices = np.random.choice(selected_indices, sample_size, replace=False)

image_paths, labels = image_paths[selected_indices], labels[selected_indices]

# Split into train/val/test
train_split, val_split = int(0.7 * len(image_paths)), int(0.85 * len(image_paths))
train_image_paths, val_image_paths, test_image_paths = image_paths[:train_split], image_paths[train_split:val_split], image_paths[val_split:]
train_labels, val_labels, test_labels = labels[:train_split], labels[train_split:val_split], labels[val_split:]

##### Step 2: Augmentation & Datasets #####
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.25),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
])

def preprocess(path, label, augment=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if augment:
        image = data_aug(image)
    return image, label

def make_dataset(paths, labels, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x, y: preprocess(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(1024).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

train_dataset = make_dataset(train_image_paths, train_labels, augment=True)
val_dataset = make_dataset(val_image_paths, val_labels)
test_dataset = make_dataset(test_image_paths, test_labels)

##### Step 3: Load Model #####
previous_model_path = "dyslexia_model.keras"
if not os.path.exists(previous_model_path):
    raise FileNotFoundError("❌ Model not found at path:", previous_model_path)
model = keras.models.load_model(previous_model_path, compile=False)

##### Step 4: Custom F1 with Threshold 0.614 #####
@register_keras_serializable()
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.614, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    precision = tp / (tf.reduce_sum(y_pred) + K.epsilon())
    recall = tp / (tf.reduce_sum(y_true) + K.epsilon())
    f1 = (2 * precision * recall) / (precision + recall + K.epsilon())
    return (f1/100)


##### Step 5: Fine-tuning #####
for layer in model.get_layer("mobilenetv2_1.00_96").layers[-15:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(5e-6),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(name="auc"), f1_score]
)

##### Step 6: Training #####
os.makedirs("models", exist_ok=True)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(
        filepath="models/dyslexia-mobilenetv2_epoch-{epoch:02d}_val-acc-{val_accuracy:.4f}_val-loss-{val_loss:.4f}_f1score-{val_f1_score:.4f}.keras",
        monitor="val_f1_score",
        save_best_only=False,
        save_weights_only=False,
        mode="max",
        verbose=1
    )
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    class_weight={0: 1.0, 1: 1.8},
    callbacks=callbacks
)

##### Step 7: Save History #####
pd.DataFrame(history.history).to_csv("training_history_dyslexia_mobilenetv2.csv", index=False)
print("📁 Training history saved.")

##### Step 8: Evaluate #####
def evaluate_model(model, test_dataset, threshold=0.614):
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        preds = model.predict(images)
        y_pred.extend((preds > threshold).astype(int))
        y_true.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Dyslexic"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Dyslexic"], yticklabels=["Normal", "Dyslexic"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    probs = []
    for images, _ in test_dataset:
        probs.extend(model.predict(images).flatten())

    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_score = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

evaluate_model(model, test_dataset)
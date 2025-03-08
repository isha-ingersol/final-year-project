import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ✅ Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ✅ Set dataset paths
data_dir = "/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting"
train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")

# ✅ Image settings
img_size = (96, 96)
batch_size = 128
sample_size = 50000  # ✅ Reduce dataset size for faster training

# ✅ Define new binary class mapping
class_map = {"Corrected": 0, "Reversal": 1, "Normal": 0}

# ✅ Load dataset paths & labels
image_paths, labels = [], []
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip non-folder items
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if file_name.startswith('.') or not os.path.isfile(file_path):
            continue
        image_paths.append(file_path)
        labels.append(class_map[class_name])  # ✅ Map labels to binary values

# ✅ Debugging: Print class mapping
print("Class Mapping Used:", class_map)
print("Unique Labels in Dataset:", np.unique(labels))

# ✅ Select 50K random samples
indices = np.random.choice(len(image_paths), sample_size, replace=False)
image_paths, labels = np.array(image_paths)[indices], np.array(labels, dtype=np.int32)[indices]

# ✅ Split dataset (70% train, 15% val, 15% test)
train_split, val_split = int(0.7 * sample_size), int(0.85 * sample_size)
train_image_paths, val_image_paths, test_image_paths = image_paths[:train_split], image_paths[train_split:val_split], image_paths[val_split:]
train_labels, val_labels, test_labels = labels[:train_split], labels[train_split:val_split], labels[val_split:]

##### 2️⃣ Data Augmentation #####
# ✅ Data augmentation to prevent overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),  # ✅ Stronger rotation
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),  # ✅ Increased contrast variation
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1)  # ✅ Added random translation
])

# ✅ Image preprocessing function
def load_and_preprocess_image(file_path, label, augment=False):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize
    if augment:
        image = data_augmentation(image)
    return image, label

# ✅ Create TensorFlow datasets with caching & prefetching
def create_tf_dataset(image_paths, labels, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(train_image_paths, train_labels, augment=True)
val_dataset = create_tf_dataset(val_image_paths, val_labels)
test_dataset = create_tf_dataset(test_image_paths, test_labels)

print(f"Total samples: {len(image_paths)} | Training: {len(train_image_paths)}, Validation: {len(val_image_paths)}, Testing: {len(test_image_paths)}")

##### 3️⃣ Adjusted Class Weights for Dyslexia #####
class_weights = {0: 1.0, 1: 1.5}  # ✅ Increase Dyslexic class weight

# ✅ Load previous model if it exists, else throw an error
previous_model_path = "dyslexia.keras"
best_model_path = "best_dyslexia.keras"

if os.path.exists(previous_model_path):
    print(f"\n🔄 Loading previously trained model from {previous_model_path}...")
    model = keras.models.load_model(previous_model_path)
else:
    raise FileNotFoundError(f"\n❌ {previous_model_path} not found! Train a model first.")

# ✅ Fine-tuning: Unfreeze fewer layers (reduce from 30 → 20)
for layer in model.get_layer("mobilenetv2_1.00_96").layers[-20:]:
    layer.trainable = True

# ✅ Recompile with adjusted learning rate for fine-tuning
model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6),  
              loss="binary_crossentropy", 
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

##### 5️⃣ Training with Checkpoints #####
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1, min_lr=1e-6)

# ✅ Save the best model during training
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    best_model_path,  # ✅ Save best model as "best_dyslexia.keras"
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    verbose=1
)

history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=10,  
    class_weight=class_weights,  # ✅ Corrected variable name
    callbacks=[early_stopping, reduce_lr, checkpoint_callback]
)

print(f"\n✅ Training completed. Best model saved as {best_model_path}")

##### 6️⃣ Evaluate Model with Adjusted Threshold #####
def evaluate_model(model, test_dataset, threshold=0.6):  # ✅ Increased threshold
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred.extend((predictions > threshold).astype(int))  # ✅ Stricter threshold
        y_true.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Dyslexic"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["Normal", "Dyslexic"], yticklabels=["Normal", "Dyslexic"], fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

evaluate_model(model, test_dataset)
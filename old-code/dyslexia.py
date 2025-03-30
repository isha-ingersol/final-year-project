import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc

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

##### Step 4: Define a Proper Custom F1-score Metric #####
def f1_score(y_true, y_pred):
    """Compute F1-score metric for Keras models"""
    y_true = K.cast(y_true, "float32")  # Ensure both are float32
    y_pred = K.cast(y_pred, "float32")
    y_pred = K.round(y_pred)  # Convert probabilities to binary (0 or 1)

    tp = K.sum(y_true * y_pred)  # True Positives
    fp = K.sum(y_pred) - tp  # False Positives
    fn = K.sum(y_true) - tp  # False Negatives

    precision = tp / (tp + fp + K.epsilon())  # Avoid division by zero
    recall = tp / (tp + fn + K.epsilon())  # Avoid division by zero

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

##### Step 5: Fine-tuning #####
# Fine-tuning: Unfreeze last 15 layers
for layer in model.get_layer("mobilenetv2_1.00_96").layers[-15:]:
    layer.trainable = True

# Recompile with adjusted learning rate for fine-tuning
model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6),   
              loss="binary_crossentropy", 
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score])

##### Step 6: Training with Checkpoints #####
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)

# Save the best model based on F1-score
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "models/dyslexia_epoch-{epoch:02d}_val-acc-{val_accuracy:.4f}_val-loss-{val_loss:.4f}_f1score-{val_f1_score:.4f}.keras",
    monitor="val_f1_score",
    save_best_only=True,
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

##### Step 7: Model Evaluation #####
def evaluate_model(model, test_dataset, threshold=0.55):
    """Evaluate the model and generate classification metrics and a confusion matrix."""
    y_true, y_pred = [], []

    # Generate predictions
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred.extend((predictions > threshold).astype(int))  # Convert probabilities to 0/1
        y_true.extend(labels.numpy())

    # Compute Precision, Recall, F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n📊 **Classification Report:**")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Dyslexic"]))

    print(f"\n🔹 **Precision:** {precision:.4f}")
    print(f"🔹 **Recall:** {recall:.4f}")
    print(f"🔹 **F1-score:** {f1:.4f}")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=["Normal", "Dyslexic"], yticklabels=["Normal", "Dyslexic"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Run the evaluation
evaluate_model(model, test_dataset)

def plot_auc_roc(model, test_dataset):
    """Compute and plot the AUC-ROC curve separately without modifying classification metrics."""
    y_true, y_pred_probs = [], []

    # Collect raw probability predictions
    for images, labels in test_dataset:
        predictions = model.predict(images)  # Raw probabilities
        y_pred_probs.extend(predictions.flatten())  # Store probabilities
        y_true.extend(labels.numpy())

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, np.array(y_pred_probs))
    roc_auc = auc(fpr, tpr)

    # Plot AUC-ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    print(f"\n🔹 **AUC-ROC Score:** {roc_auc:.4f}")

# Run the AUC-ROC evaluation separately
plot_auc_roc(model, test_dataset)
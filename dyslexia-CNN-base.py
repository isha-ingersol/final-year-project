import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import glob

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ========== CONFIG ==========
data_dir = "/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting"
train_dir = os.path.join(data_dir, "Train")
img_size = (96, 96)
batch_size = 128
sample_size = 35000
class_map = {"Corrected": 0, "Reversal": 1, "Normal": 0}

# ========== Load & Balance ==========
image_paths, labels = [], []
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if os.path.isdir(class_dir):
        for file in os.listdir(class_dir):
            path = os.path.join(class_dir, file)
            if not file.startswith('.') and os.path.isfile(path):
                image_paths.append(path)
                labels.append(class_map[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)

dys_idx = np.where(labels == 1)[0]
norm_idx = np.where(labels == 0)[0]
min_class = min(len(dys_idx), len(norm_idx))
dys_idx = np.random.choice(dys_idx, min_class, replace=False)
norm_idx = np.random.choice(norm_idx, min_class, replace=False)

indices = np.concatenate([dys_idx, norm_idx])
if len(indices) > sample_size:
    indices = np.random.choice(indices, sample_size, replace=False)

image_paths, labels = image_paths[indices], labels[indices]
train_split, val_split = int(0.7 * len(image_paths)), int(0.85 * len(image_paths))
train_image_paths, val_image_paths, test_image_paths = image_paths[:train_split], image_paths[train_split:val_split], image_paths[val_split:]
train_labels, val_labels, test_labels = labels[:train_split], labels[train_split:val_split], labels[val_split:]

# ========== Augmentation ==========
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.25),
    layers.RandomTranslation(0.1, 0.1)
])

def preprocess_image(path, label, augment_img=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if augment_img:
        img = augment(img)
    return img, label

def make_dataset(paths, labels, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x, y: preprocess_image(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(1024).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_image_paths, train_labels, augment=True)
val_ds = make_dataset(val_image_paths, val_labels)
test_ds = make_dataset(test_image_paths, test_labels)

print(f"Samples: {len(image_paths)} | Train: {len(train_image_paths)}, Val: {len(val_image_paths)}, Test: {len(test_image_paths)}")

# ========== Model ==========
def build_cnn():
    return keras.Sequential([
        layers.Input(shape=img_size + (3,)),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(), layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(), layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu'), layers.MaxPooling2D(), layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')  # Mixed precision output
    ])

def f1_metric(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    fp = K.sum(y_pred) - tp
    fn = K.sum(y_true) - tp
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), f1_metric]
)

# ========== Callbacks ==========
class_weights = {0: 1.0, 1: 1.8}
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/cnn_epoch-{epoch:02d}_val-acc-{val_accuracy:.4f}_val-loss-{val_loss:.4f}_f1score-{val_f1_metric:.4f}.keras",
        monitor="val_f1_metric",
        mode="max",
        save_best_only=False  # ✅ Save all
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

# ========== Train ==========
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=7,
    class_weight=class_weights,
    callbacks=callbacks
)

# ========== Evaluate ==========
def evaluate(model, ds, threshold=0.55):
    y_true, y_pred = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_pred.extend((p > threshold).astype(int))
        y_true.extend(y.numpy())
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Dyslexic"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Dyslexic"], yticklabels=["Normal", "Dyslexic"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return y_true, y_pred

def plot_roc(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        prob = model.predict(x, verbose=0)
        y_prob.extend(prob.flatten())
        y_true.extend(y.numpy())
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
    print(f"\n🔹 AUC-ROC Score: {auc_score:.4f}")

# ========== Best Model Detection ==========
model_files = glob.glob("models/cnn_epoch-*.keras")

def extract_f1(filename):
    for p in filename.split('_'):
        if p.startswith('f1score-'):
            return float(p.replace('f1score-', '').replace('.keras', ''))
    return 0.0

model_files.sort(key=extract_f1, reverse=True)
best_model_path = model_files[0]
print(f"\n🏆 Best Model: {best_model_path}")

best_model = keras.models.load_model(best_model_path, custom_objects={"f1_metric": f1_metric})
print("\n📌 Evaluation of Best Model on Test Set:")
evaluate(best_model, test_ds)
plot_roc(best_model, test_ds)
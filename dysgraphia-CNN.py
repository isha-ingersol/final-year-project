import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import albumentations as A

# ====== CONFIG ======
tf.keras.mixed_precision.set_global_policy("mixed_float16")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

dysgraphia_dir = "/Users/ishaingersol/Desktop/fyp-dataset/dysgraphia-handwriting"
normal_dir = "/Users/ishaingersol/Desktop/fyp-dataset/dyslexia-handwriting/Test/Normal"
img_size = (64, 64)
class_map = {"dysgraphia": 0, "normal": 1}
batch_size = 32
sample_size = 500  # Total after balancing

# ====== Albumentations Augmentation ======
transform = A.Compose([
    A.GaussNoise(mean=0, std=25.0, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5)
])

# ====== Load & Balance Dataset ======
def load_images():
    data, labels = [], []

    # Dysgraphia (both subfolders)
    for subfolder in ['low-potential-dysgraphia', 'potential-dysgraphia']:
        folder = os.path.join(dysgraphia_dir, subfolder)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = transform(image=(img * 255).astype(np.uint8))['image'].astype(np.float32) / 255.0
                data.append(img[..., np.newaxis])
                labels.append(0)

    # Normal (randomly sampled)
    normal_files = [f for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sampled = random.sample(normal_files, min(250, len(normal_files)))
    for file in sampled:
        path = os.path.join(normal_dir, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = transform(image=(img * 255).astype(np.uint8))['image'].astype(np.float32) / 255.0
            data.append(img[..., np.newaxis])
            labels.append(1)

    X = np.array(data)
    y = np.array(labels)

    # Balance classes
    dys_idx = np.where(y == 0)[0]
    norm_idx = np.where(y == 1)[0]
    min_class = min(len(dys_idx), len(norm_idx))
    dys_idx = np.random.choice(dys_idx, min_class, replace=False)
    norm_idx = np.random.choice(norm_idx, min_class, replace=False)
    idx = np.concatenate([dys_idx, norm_idx])
    np.random.shuffle(idx)

    return X[idx], y[idx]

X, y = load_images()
n_total = len(y)
train_idx = int(0.7 * n_total)
val_idx = int(0.85 * n_total)
X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
X_test, y_test = X[val_idx:], y[val_idx:]

print(f"✅ Samples: {n_total} | Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ====== Datasets ======
def make_dataset(X, y, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train, augment=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test, y_test)

# ====== Custom F1 Metric ======
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

# ====== CNN Model ======
def build_model():
    return keras.Sequential([
        layers.Input(shape=img_size + (1,)),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(), layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(), layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')  # for mixed precision
    ])

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall(), f1_metric]
)

# ====== Callbacks ======
os.makedirs("models", exist_ok=True)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/dysgraphia_epoch-{epoch:02d}_val-acc-{val_accuracy:.4f}_val-loss-{val_loss:.4f}_f1score-{val_f1_metric:.4f}.keras",
        monitor="val_f1_metric", 
        mode="max", 
        save_best_only=False),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

# ====== Train ======
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, class_weight={0: 1.0, 1: 1.8})

# ====== Evaluate Best Model ======
def evaluate(model, ds, threshold=0.5):
    y_true, y_pred = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_pred.extend((p > threshold).astype(int))
        y_true.extend(y.numpy())
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Dysgraphia", "Normal"]))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Dysgraphia", "Normal"], yticklabels=["Dysgraphia", "Normal"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("cnn_confusion_matrix.png")
    plt.show()

def plot_roc(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        prob = model.predict(x, verbose=0)
        y_prob.extend(prob.flatten())
        y_true.extend(y.numpy())
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_roc_curve.png")
    plt.show()
    print(f"\n🔹 AUC-ROC Score: {roc_auc:.4f}")

# ====== Select Best Model ======
model_files = glob.glob("models/dysgraphia_epoch-*.keras")
def extract_f1(filename):
    for p in filename.split('_'):
        if p.startswith('f1score-'):
            return float(p.replace('f1score-', '').replace('.keras', ''))
    return 0.0

model_files.sort(key=extract_f1, reverse=True)
best_model_path = model_files[0]
print(f"\n🏆 Best Model: {best_model_path}")

best_model = keras.models.load_model(best_model_path, custom_objects={"f1_metric": f1_metric})
print("\n📌 Evaluation on Test Set:")
evaluate(best_model, make_dataset(X_test, y_test))
plot_roc(best_model, make_dataset(X_test, y_test))

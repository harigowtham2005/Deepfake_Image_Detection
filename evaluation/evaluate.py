import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 16
DATASET_PATH = "dataset_faces_mtcnn"
MODEL_PATH = "models/deepfake_model.h5"

# Load dataset exactly like training
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)

val_ds = dataset.take(val_size)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Collect true labels FIRST
true_labels = []
for images, labels in val_ds:
    true_labels.extend(labels.numpy())

true_labels = np.array(true_labels)

# Recreate val_ds again (iterator reset)
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = dataset.take(val_size)

# Predict
predictions = model.predict(val_ds)
pred_labels = (predictions > 0.5).astype(int).flatten()

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))

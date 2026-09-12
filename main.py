import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing import preprocess
from models.model import build_model

# Label mapping
label_map = {
    "insomnia": 0,
    "narcolepsy": 1,
    "normal": 2,
    "sdb": 3
}

data_folder = os.path.join("data", "EEG_DATASET")

print("Loading dataset...\n")
print("Folders found:", os.listdir(data_folder))

X = []
y = []

# Load dataset
for label_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, label_name)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nProcessing folder: {label_name}")

    files = [f for f in os.listdir(folder_path) if f.endswith(".edf")]
    print(f"Found {len(files)} EDF files")

    for file in files:
        file_path = os.path.join(folder_path, file)

        try:
            data = preprocess(file_path)

            if data is None or len(data) == 0:
                continue

            X.append(data)

            label = label_map[label_name]
            labels = np.full((data.shape[0],), label)
            y.append(labels)

        except Exception as e:
            print(f"Skipping file: {file_path}")
            print(e)

# Safety check
if len(X) == 0:
    raise ValueError("No data loaded. Check dataset path.")

# Combine data
X = np.concatenate(X)
y = np.concatenate(y)

print("\nTotal data shape:", X.shape)

# Add channel dimension
X = X[..., np.newaxis]

# One-hot encoding
y = to_categorical(y, num_classes=4)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training shape:", X_train.shape)

# Build model
model = build_model(input_shape=X_train.shape[1:])

# Early stopping
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    epochs=12,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluate basic accuracy
loss, acc = model.evaluate(X_test, y_test)
print("\nFinal Accuracy:", acc)

# Save model
model.save("sleep_model.h5")

# =====================
# GRAPHS
# =====================

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =====================
# CONFUSION MATRIX
# =====================

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =====================
# FINAL EVALUATION METRICS 
# =====================

target_names = ["insomnia", "narcolepsy", "normal", "sdb"]

print("\nEVALUATION METRICS")
print("--------------------------------------------------")

print(classification_report(y_true, y_pred_classes, target_names=target_names))

accuracy = accuracy_score(y_true, y_pred_classes)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
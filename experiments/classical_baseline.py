import os
import json
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "reduced"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "classical_baseline"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD REDUCED DATASET
# ============================================================

print("=" * 60)
print("CLASSICAL IDS BASELINE")
print("=" * 60)

print("\nLoading reduced dataset...")

X_train = np.load(
    os.path.join(DATA_DIR, "X_train_reduced.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train_reduced.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test_reduced.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test_reduced.npy")
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# 3. CHECK LABELS
# ============================================================

print("\nChecking labels...")

print("Training labels:")
print("Normal :", np.sum(y_train == 0))
print("Attack :", np.sum(y_train == 1))

print("\nTesting labels:")
print("Normal :", np.sum(y_test == 0))
print("Attack :", np.sum(y_test == 1))


# ============================================================
# 4. CREATE CLASSICAL MODEL
# ============================================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining model...")

train_start = time.time()

model.fit(X_train, y_train)

train_time = time.time() - train_start

print("Training completed.")
print("Training time: {:.4f} seconds".format(train_time))


# ============================================================
# 6. PREDICTION
# ============================================================

print("\nGenerating predictions...")

test_start = time.time()

y_pred = model.predict(X_test)

prediction_time = time.time() - test_start

print("Prediction completed.")
print("Prediction time: {:.4f} seconds".format(prediction_time))


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = fp / (fp + tn)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CLASSICAL IDS RESULTS")
print("=" * 60)

print("Accuracy          :", round(accuracy, 6))
print("Precision         :", round(precision, 6))
print("Recall            :", round(recall, 6))
print("F1 Score          :", round(f1, 6))
print("False Positive Rate:", round(false_positive_rate, 6))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)


# ============================================================
# 9. SAVE MODEL
# ============================================================

model_path = os.path.join(
    RESULT_DIR,
    "model.joblib"
)

joblib.dump(model, model_path)

print("\nModel saved:")
print(model_path)


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    RESULT_DIR,
    "predictions.npy"
)

np.save(
    prediction_path,
    y_pred
)

print("Predictions saved:")
print(prediction_path)


# ============================================================
# 11. SAVE METRICS
# ============================================================

metrics = {
    "model": "RandomForestClassifier",
    "n_estimators": 100,
    "random_state": 42,

    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "features": int(X_train.shape[1]),

    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "false_positive_rate": float(false_positive_rate),

    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),

    "training_time_seconds": float(train_time),
    "prediction_time_seconds": float(prediction_time)
}

metrics_path = os.path.join(
    RESULT_DIR,
    "metrics.json"
)

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved:")
print(metrics_path)


# ============================================================
# 12. CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Classical IDS - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Normal", "Attack"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Attack"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

cm_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

print("Confusion matrix saved:")
print(cm_path)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STAGE 8 COMPLETED")
print("=" * 60)

print("Model              : Random Forest")
print("Features           :", X_train.shape[1])
print("Training samples   :", X_train.shape[0])
print("Testing samples    :", X_test.shape[0])
print("Accuracy           :", round(accuracy, 6))
print("Precision          :", round(precision, 6))
print("Recall             :", round(recall, 6))
print("F1 Score           :", round(f1, 6))
print("False Positive Rate:", round(false_positive_rate, 6))

print("\nResults directory:")
print(RESULT_DIR)

print("\nDo NOT move to Stage 9 yet.")
print("First verify all Stage 8 outputs.")
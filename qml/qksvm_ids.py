import os
import json
import time
import joblib

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "reduced"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "qml_ids"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

N_QUBITS = 8

TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 50

RANDOM_STATE = 42


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 60)
print("QUANTUM KERNEL SVM - IDS")
print("=" * 60)

print("\nLoading reduced dataset...")

X_train_full = np.load(
    os.path.join(
        DATA_DIR,
        "X_train_reduced.npy"
    )
)

y_train_full = np.load(
    os.path.join(
        DATA_DIR,
        "y_train_reduced.npy"
    )
)

X_test_full = np.load(
    os.path.join(
        DATA_DIR,
        "X_test_reduced.npy"
    )
)

y_test_full = np.load(
    os.path.join(
        DATA_DIR,
        "y_test_reduced.npy"
    )
)

print(
    "Full training data:",
    X_train_full.shape
)

print(
    "Full testing data :",
    X_test_full.shape
)


# ============================================================
# 4. CREATE BALANCED QML SUBSET
# ============================================================

print("\nCreating controlled QML subset...")

rng = np.random.default_rng(
    RANDOM_STATE
)


def balanced_subset(
    X,
    y,
    samples_per_class
):

    normal_indices = np.where(
        y == 0
    )[0]

    attack_indices = np.where(
        y == 1
    )[0]

    selected_normal = rng.choice(
        normal_indices,
        size=samples_per_class,
        replace=False
    )

    selected_attack = rng.choice(
        attack_indices,
        size=samples_per_class,
        replace=False
    )

    indices = np.concatenate(
        [
            selected_normal,
            selected_attack
        ]
    )

    rng.shuffle(indices)

    return X[indices], y[indices]


X_train, y_train = balanced_subset(
    X_train_full,
    y_train_full,
    TRAIN_PER_CLASS
)

X_test, y_test = balanced_subset(
    X_test_full,
    y_test_full,
    TEST_PER_CLASS
)


print("\nQML training subset:")
print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print("\nQML testing subset:")
print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)

print("\nTraining class counts:")
print(
    "Normal:",
    np.sum(y_train == 0)
)

print(
    "Attack:",
    np.sum(y_train == 1)
)

print("\nTesting class counts:")
print(
    "Normal:",
    np.sum(y_test == 0)
)

print(
    "Attack:",
    np.sum(y_test == 1)
)


# ============================================================
# 5. QUANTUM DEVICE
# ============================================================

print("\nCreating quantum device...")

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# 6. QUANTUM FEATURE MAP
# ============================================================

def feature_map(x):

    for i in range(N_QUBITS):

        qml.RY(
            x[i],
            wires=i
        )

    for i in range(N_QUBITS - 1):

        qml.CNOT(
            wires=[i, i + 1]
        )


# ============================================================
# 7. QUANTUM KERNEL CIRCUIT
# ============================================================

@qml.qnode(dev)
def kernel_circuit(x1, x2):

    feature_map(x1)

    for i in range(
        N_QUBITS - 2,
        -1,
        -1
    ):

        qml.CNOT(
            wires=[i, i + 1]
        )

    for i in range(
        N_QUBITS - 1,
        -1,
        -1
    ):

        qml.RY(
            -x2[i],
            wires=i
        )

    return qml.probs(
        wires=range(N_QUBITS)
    )


# ============================================================
# 8. QUANTUM KERNEL FUNCTION
# ============================================================

def quantum_kernel(x1, x2):

    probabilities = kernel_circuit(
        x1,
        x2
    )

    return float(
        probabilities[0]
    )


# ============================================================
# 9. BUILD KERNEL MATRIX
# ============================================================

def build_kernel_matrix(
    X1,
    X2,
    name
):

    matrix = np.zeros(
        (
            len(X1),
            len(X2)
        )
    )

    total = len(X1) * len(X2)

    completed = 0

    start_time = time.time()

    print(
        f"\nBuilding {name} kernel matrix..."
    )

    for i in range(len(X1)):

        for j in range(len(X2)):

            matrix[i, j] = quantum_kernel(
                X1[i],
                X2[j]
            )

            completed += 1

        if (
            i == 0
            or (i + 1) % 10 == 0
            or i == len(X1) - 1
        ):

            elapsed = (
                time.time() - start_time
            )

            print(
                f"Completed rows: "
                f"{i + 1}/{len(X1)} "
                f"| calculations: "
                f"{completed}/{total} "
                f"| time: "
                f"{elapsed:.2f}s"
            )

    return matrix


# ============================================================
# 10. BUILD TRAINING KERNEL
# ============================================================

kernel_start = time.time()

K_train = build_kernel_matrix(
    X_train,
    X_train,
    "training"
)

train_kernel_time = (
    time.time() - kernel_start
)


# ============================================================
# 11. BUILD TEST KERNEL
# ============================================================

test_kernel_start = time.time()

K_test = build_kernel_matrix(
    X_test,
    X_train,
    "testing"
)

test_kernel_time = (
    time.time() - test_kernel_start
)


# ============================================================
# 12. SAVE KERNEL MATRICES
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "K_train.npy"
    ),
    K_train
)

np.save(
    os.path.join(
        RESULT_DIR,
        "K_test.npy"
    ),
    K_test
)

print("\nKernel matrices saved.")


# ============================================================
# 13. TRAIN SVM USING PRECOMPUTED KERNEL
# ============================================================

print("\nTraining SVM...")

svm_start = time.time()

model = SVC(
    kernel="precomputed"
)

model.fit(
    K_train,
    y_train
)

svm_training_time = (
    time.time() - svm_start
)


# ============================================================
# 14. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating QKSVM predictions...")

prediction_start = time.time()

y_pred = model.predict(
    K_test
)

prediction_time = (
    time.time() - prediction_start
)


# ============================================================
# 15. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = (
    fp / (fp + tn)
)


# ============================================================
# 16. SAVE CONFUSION MATRIX IMAGE
# ============================================================

confusion_matrix_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix.png"
)

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Normal",
        "Attack"
    ],
    yticklabels=[
        "Normal",
        "Attack"
    ]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "QKSVM IDS Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    confusion_matrix_path,
    dpi=300
)

plt.close()


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("QKSVM IDS RESULTS")
print("=" * 60)

print(
    "Accuracy           :",
    round(accuracy, 6)
)

print(
    "Precision          :",
    round(precision, 6)
)

print(
    "Recall             :",
    round(recall, 6)
)

print(
    "F1 Score           :",
    round(f1, 6)
)

print(
    "False Positive Rate:",
    round(false_positive_rate, 6)
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

model_path = os.path.join(
    RESULT_DIR,
    "qksvm_model.joblib"
)

joblib.dump(
    model,
    model_path
)


# ============================================================
# 19. SAVE PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    RESULT_DIR,
    "qksvm_predictions.npy"
)

np.save(
    prediction_path,
    y_pred
)


# ============================================================
# 20. SAVE METRICS
# ============================================================

metrics = {

    "model":
        "Quantum Kernel SVM",

    "quantum_framework":
        "PennyLane",

    "pennylane_version":
        qml.__version__,

    "qubits":
        N_QUBITS,

    "features":
        int(X_train.shape[1]),

    "training_samples":
        int(len(X_train)),

    "testing_samples":
        int(len(X_test)),

    "training_samples_per_class":
        TRAIN_PER_CLASS,

    "testing_samples_per_class":
        TEST_PER_CLASS,

    "random_state":
        RANDOM_STATE,

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "false_positive_rate":
        float(false_positive_rate),

    "true_negative":
        int(tn),

    "false_positive":
        int(fp),

    "false_negative":
        int(fn),

    "true_positive":
        int(tp),

    "training_kernel_time_seconds":
        float(train_kernel_time),

    "testing_kernel_time_seconds":
        float(test_kernel_time),

    "svm_training_time_seconds":
        float(svm_training_time),

    "prediction_time_seconds":
        float(prediction_time)
}

metrics_path = os.path.join(
    RESULT_DIR,
    "qksvm_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# 21. FINAL
# ============================================================

print("\n" + "=" * 60)
print("QKSVM IDS EXPERIMENT FINISHED")
print("=" * 60)

print("\nResults directory:")
print(RESULT_DIR)

print("\nGenerated files:")

print("K_train.npy")
print("K_test.npy")
print("qksvm_model.joblib")
print("qksvm_metrics.json")
print("qksvm_predictions.npy")
print("confusion_matrix.png")

print("\nConfusion matrix saved to:")
print(confusion_matrix_path)
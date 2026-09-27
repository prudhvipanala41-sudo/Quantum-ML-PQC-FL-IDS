import os
import json
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from fl_model import FLIDSModel


# Load reduced test data
X_test = np.load(
    "data/processed/reduced/X_test_reduced.npy"
)

y_test = np.load(
    "data/processed/reduced/y_test_reduced.npy"
)

print("Test X shape:", X_test.shape)
print("Test y shape:", y_test.shape)


# Convert to PyTorch tensors
X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


# Load trained global FL model
model = FLIDSModel()

model.load_state_dict(
    torch.load(
        "results/federated/global_model.pt",
        map_location="cpu"
    )
)

model.eval()


# Prediction
with torch.no_grad():

    output = model(X_test_tensor)

    probabilities = torch.sigmoid(output)

    predictions = (
        probabilities >= 0.5
    ).int().numpy().flatten()


# Calculate metrics
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)


# Calculate False Positive Rate
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)


# Display results
print("\n==============================")
print("FEDERATED LEARNING EVALUATION")
print("==============================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("FPR      :", fpr)

print("\nConfusion Matrix:")
print(cm)


# Save results
os.makedirs(
    "results/federated",
    exist_ok=True
)

results = {
    "model": "Federated IDS Neural Network",
    "test_samples": int(len(y_test)),
    "features": int(X_test.shape[1]),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "false_positive_rate": float(fpr),
    "confusion_matrix": cm.tolist()
}


with open(
    "results/federated/evaluation_results.json",
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print("\nEvaluation results saved to:")
print("results/federated/evaluation_results.json")
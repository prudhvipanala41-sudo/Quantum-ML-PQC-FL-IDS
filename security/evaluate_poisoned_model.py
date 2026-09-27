import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.fl_model import FLIDSModel


TEST_X = PROJECT_ROOT / "data" / "processed" / "reduced" / "X_test_reduced.npy"
TEST_Y = PROJECT_ROOT / "data" / "processed" / "reduced" / "y_test_reduced.npy"

CLEAN_MODEL = PROJECT_ROOT / "results" / "federated" / "global_model.pt"
POISONED_MODEL = PROJECT_ROOT / "results" / "poisoning" / "poisoned_global_model.pt"


def load_model(model_path):

    model = FLIDSModel()

    state = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state)
    model.eval()

    return model


def evaluate_model(model, X, y):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():

        logits = model(X_tensor).squeeze(1)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).int().numpy()

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }


def main():

    print("========================================")
    print("POISONED MODEL PERFORMANCE EVALUATION")
    print("========================================")

    X = np.load(TEST_X)
    y = np.load(TEST_Y)

    print("Test samples:", X.shape[0])
    print("Test features:", X.shape[1])

    clean_model = load_model(
        CLEAN_MODEL
    )

    poisoned_model = load_model(
        POISONED_MODEL
    )

    clean_results = evaluate_model(
        clean_model,
        X,
        y
    )

    poisoned_results = evaluate_model(
        poisoned_model,
        X,
        y
    )

    print("\n==============================")
    print("CLEAN FL MODEL")
    print("==============================")

    for key, value in clean_results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    print("\n==============================")
    print("POISONED FL MODEL")
    print("==============================")

    for key, value in poisoned_results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    print("\n==============================")
    print("PERFORMANCE DIFFERENCE")
    print("==============================")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr"
    ]:

        difference = (
            poisoned_results[metric]
            - clean_results[metric]
        )

        print(
            f"{metric}: {difference:+.6f}"
        )


if __name__ == "__main__":
    main()
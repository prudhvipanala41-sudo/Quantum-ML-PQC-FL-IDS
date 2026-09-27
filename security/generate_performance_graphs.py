import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_experiments"
    / "final_model_comparison.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "graphs"
    / "performance"
)


def main():

    print("========================================")
    print("FINAL PERFORMANCE GRAPH GENERATION")
    print("========================================")

    with open(INPUT_FILE, "r") as file:
        data = json.load(file)

    models = data["models"]

    model_names = [
        "Plain FedAvg",
        "Poisoned FedAvg",
        "Robust Median"
    ]

    model_keys = [
        "plain_fedavg",
        "poisoned_fedavg",
        "robust_median"
    ]

    metrics = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "fpr": "False Positive Rate"
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for metric_key, metric_title in metrics.items():

        values = [
            models[key][metric_key]
            for key in model_keys
        ]

        plt.figure(figsize=(8, 5))

        plt.bar(
            model_names,
            values
        )

        plt.ylabel(metric_title)

        plt.title(
            f"{metric_title} Comparison"
        )

        plt.ylim(
            0,
            1
        )

        plt.xticks(
            rotation=10
        )

        plt.tight_layout()

        output_file = (
            OUTPUT_DIR
            / f"{metric_key}_comparison.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"{metric_title} graph saved:"
        )
        print(output_file)

    print("\n========================================")
    print("PERFORMANCE GRAPH GENERATION PASSED")
    print("========================================")


if __name__ == "__main__":
    main()
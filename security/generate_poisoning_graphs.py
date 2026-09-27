import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "robust_aggregation"
    / "robust_aggregation_comparison.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "graphs"
    / "poisoning"
)


def main():

    print("========================================")
    print("POISONING & ROBUSTNESS GRAPH GENERATION")
    print("========================================")

    with open(INPUT_FILE, "r") as file:
        data = json.load(file)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    clean = data["clean_fedavg"]
    poisoned = data["poisoned_fedavg"]
    robust = data["robust_median"]

    model_labels = [
        "Clean FedAvg",
        "Poisoned FedAvg",
        "Robust Median"
    ]

    model_keys = [
        clean,
        poisoned,
        robust
    ]

    metrics = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "fpr": "False Positive Rate"
    }

    for metric_key, metric_title in metrics.items():

        values = [
            model[metric_key]
            for model in model_keys
        ]

        plt.figure(figsize=(8, 5))

        plt.bar(
            model_labels,
            values
        )

        plt.ylabel(metric_title)

        plt.title(
            f"{metric_title}: Clean vs Poisoned vs Robust"
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
            / f"{metric_key}_robustness.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"{metric_title} robustness graph saved:"
        )
        print(output_file)

    poisoning_effect = {
        metric: poisoned[metric] - clean[metric]
        for metric in metrics
    }

    plt.figure(figsize=(8, 5))

    plt.bar(
        list(poisoning_effect.keys()),
        list(poisoning_effect.values())
    )

    plt.ylabel("Change from Clean FedAvg")

    plt.title(
        "Model Poisoning Impact"
    )

    plt.xticks(
        rotation=10
    )

    plt.tight_layout()

    poisoning_output = (
        OUTPUT_DIR
        / "poisoning_impact.png"
    )

    plt.savefig(
        poisoning_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Poisoning impact graph saved:")
    print(poisoning_output)

    recovery = {
        metric: robust[metric] - poisoned[metric]
        for metric in metrics
    }

    plt.figure(figsize=(8, 5))

    plt.bar(
        list(recovery.keys()),
        list(recovery.values())
    )

    plt.ylabel("Change from Poisoned FedAvg")

    plt.title(
        "Robust Aggregation Recovery"
    )

    plt.xticks(
        rotation=10
    )

    plt.tight_layout()

    recovery_output = (
        OUTPUT_DIR
        / "robust_recovery.png"
    )

    plt.savefig(
        recovery_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Robust recovery graph saved:")
    print(recovery_output)

    print("\n========================================")
    print("POISONING & ROBUSTNESS GRAPH GENERATION PASSED")
    print("========================================")


if __name__ == "__main__":
    main()
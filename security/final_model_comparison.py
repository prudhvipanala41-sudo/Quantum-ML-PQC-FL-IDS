import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_experiments"
    / "final_experiment_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_experiments"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "final_model_comparison.json"
)


def main():

    print("========================================")
    print("FINAL MODEL PERFORMANCE COMPARISON")
    print("========================================")

    with open(INPUT_FILE, "r") as file:
        data = json.load(file)

    federated = data["plain_federated_learning"]

    robust = data["robust_aggregation"]

    clean = robust["clean_fedavg"]
    poisoned = robust["poisoned_fedavg"]
    robust_model = robust["robust_median"]

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr"
    ]

    poisoning_effect = {}

    for metric in metrics:

        poisoning_effect[metric] = (
            poisoned[metric]
            - clean[metric]
        )

    comparison = {
        "project": data["project"],
        "stage": "Final Experiments",

        "models": {
            "plain_fedavg": clean,
            "poisoned_fedavg": poisoned,
            "robust_median": robust_model
        },

        "poisoning_effect": poisoning_effect,

        "robust_vs_poisoned": robust[
            "robust_vs_poisoned"
        ],

        "robust_vs_clean": robust[
            "robust_vs_clean"
        ],

        "federated_learning_artifact": federated
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            comparison,
            file,
            indent=4
        )

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    print(
        f"{'Metric':<15}"
        f"{'Clean FL':<15}"
        f"{'Poisoned FL':<15}"
        f"{'Robust Median'}"
    )

    for metric in metrics:

        print(
            f"{metric:<15}"
            f"{clean[metric]:<15.6f}"
            f"{poisoned[metric]:<15.6f}"
            f"{robust_model[metric]:.6f}"
        )

    print("\n==============================")
    print("POISONING EFFECT")
    print("==============================")

    for metric in metrics:

        print(
            f"{metric}: "
            f"{poisoning_effect[metric]:+.6f}"
        )

    print("\n==============================")
    print("ROBUST VS POISONED")
    print("==============================")

    for metric in metrics:

        print(
            f"{metric}: "
            f"{robust['robust_vs_poisoned'][metric]:+.6f}"
        )

    print("\n==============================")
    print("ROBUST VS CLEAN")
    print("==============================")

    for metric in metrics:

        print(
            f"{metric}: "
            f"{robust['robust_vs_clean'][metric]:+.6f}"
        )

    print("\nFinal comparison saved:")
    print(OUTPUT_FILE)

    print("\n========================================")
    print("FINAL MODEL COMPARISON PASSED")
    print("========================================")


if __name__ == "__main__":
    main()
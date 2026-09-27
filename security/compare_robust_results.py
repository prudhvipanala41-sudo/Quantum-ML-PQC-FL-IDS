import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

CLEAN_RESULTS = {
    "accuracy": 0.7550124201561391,
    "precision": 0.9536991062562066,
    "recall": 0.5986908750876646,
    "f1": 0.7356024701996265,
    "fpr": 0.03841005045824323
}

POISONED_RESULTS = {
    "accuracy": 0.734564,
    "precision": 0.963962,
    "recall": 0.554430,
    "f1": 0.703968,
    "fpr": 0.027392
}

ROBUST_RESULTS = {
    "accuracy": 0.764771,
    "precision": 0.954820,
    "recall": 0.615912,
    "f1": 0.748804,
    "fpr": 0.038513
}


def percentage_change(new_value, old_value):

    if old_value == 0:
        return 0.0

    return ((new_value - old_value) / old_value) * 100


def main():

    comparison = {
        "stage": "Robust Aggregation",
        "aggregation_method": "Coordinate-wise Median",
        "poisoned_client_id": 1,
        "poison_factor": -3.0,
        "rounds": 3,
        "clients_per_round": 5,

        "clean_fedavg": CLEAN_RESULTS,

        "poisoned_fedavg": POISONED_RESULTS,

        "robust_median": ROBUST_RESULTS,

        "poisoning_impact": {},

        "robust_vs_poisoned": {},

        "robust_vs_clean": {}
    }

    for metric in CLEAN_RESULTS:

        comparison["poisoning_impact"][metric] = (
            POISONED_RESULTS[metric]
            - CLEAN_RESULTS[metric]
        )

        comparison["robust_vs_poisoned"][metric] = (
            ROBUST_RESULTS[metric]
            - POISONED_RESULTS[metric]
        )

        comparison["robust_vs_clean"][metric] = (
            ROBUST_RESULTS[metric]
            - CLEAN_RESULTS[metric]
        )

    output_dir = RESULTS_DIR / "robust_aggregation"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "robust_aggregation_comparison.json"
    )

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            comparison,
            file,
            indent=4
        )

    print("========================================")
    print("ROBUST AGGREGATION COMPARISON")
    print("========================================")

    print("\nMetric                  Clean       Poisoned      Robust")

    for metric in CLEAN_RESULTS:

        print(
            f"{metric:<23}"
            f"{CLEAN_RESULTS[metric]:.6f}    "
            f"{POISONED_RESULTS[metric]:.6f}      "
            f"{ROBUST_RESULTS[metric]:.6f}"
        )

    print("\n========================================")
    print("POISONING IMPACT")
    print("========================================")

    for metric, value in comparison["poisoning_impact"].items():

        print(
            f"{metric}: {value:+.6f}"
        )

    print("\n========================================")
    print("ROBUST VS POISONED")
    print("========================================")

    for metric, value in comparison["robust_vs_poisoned"].items():

        print(
            f"{metric}: {value:+.6f}"
        )

    print("\n========================================")
    print("ROBUST VS CLEAN")
    print("========================================")

    for metric, value in comparison["robust_vs_clean"].items():

        print(
            f"{metric}: {value:+.6f}"
        )

    print("\nComparison artifact saved:")
    print(output_path)


if __name__ == "__main__":
    main()
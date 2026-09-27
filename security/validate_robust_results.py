import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "robust_aggregation"
)

COMPARISON_FILE = (
    RESULTS_DIR
    / "robust_aggregation_comparison.json"
)

MODEL_FILE = (
    RESULTS_DIR
    / "robust_global_model.pt"
)


def main():

    print("========================================")
    print("ROBUST AGGREGATION VALIDATION")
    print("========================================")

    validation_passed = True

    if COMPARISON_FILE.exists():
        print("Comparison artifact exists: PASSED")
    else:
        print("Comparison artifact exists: FAILED")
        validation_passed = False

    if MODEL_FILE.exists():
        print("Robust model exists: PASSED")
    else:
        print("Robust model exists: FAILED")
        validation_passed = False

    if not COMPARISON_FILE.exists():
        print("\nROBUST AGGREGATION VALIDATION FAILED")
        return

    with open(COMPARISON_FILE, "r") as file:
        results = json.load(file)

    required_sections = [
        "stage",
        "aggregation_method",
        "poisoned_client_id",
        "poison_factor",
        "rounds",
        "clients_per_round",
        "clean_fedavg",
        "poisoned_fedavg",
        "robust_median",
        "poisoning_impact",
        "robust_vs_poisoned",
        "robust_vs_clean"
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in results
    ]

    if not missing_sections:
        print("Required fields: PASSED")
    else:
        print("Required fields: FAILED")
        print("Missing:", missing_sections)
        validation_passed = False

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr"
    ]

    metric_validation = True

    for section in [
        "clean_fedavg",
        "poisoned_fedavg",
        "robust_median"
    ]:

        for metric in metrics:

            if metric not in results[section]:
                metric_validation = False

    if metric_validation:
        print("Metric fields: PASSED")
    else:
        print("Metric fields: FAILED")
        validation_passed = False

    consistency_passed = True

    for metric in metrics:

        clean = results["clean_fedavg"][metric]
        poisoned = results["poisoned_fedavg"][metric]
        robust = results["robust_median"][metric]

        poisoning_difference = (
            poisoned - clean
        )

        robust_poisoned_difference = (
            robust - poisoned
        )

        robust_clean_difference = (
            robust - clean
        )

        stored_poisoning = (
            results["poisoning_impact"][metric]
        )

        stored_robust_poisoned = (
            results["robust_vs_poisoned"][metric]
        )

        stored_robust_clean = (
            results["robust_vs_clean"][metric]
        )

        if abs(
            stored_poisoning - poisoning_difference
        ) > 1e-9:
            consistency_passed = False

        if abs(
            stored_robust_poisoned
            - robust_poisoned_difference
        ) > 1e-9:
            consistency_passed = False

        if abs(
            stored_robust_clean
            - robust_clean_difference
        ) > 1e-9:
            consistency_passed = False

    if consistency_passed:
        print("Metric consistency: PASSED")
    else:
        print("Metric consistency: FAILED")
        validation_passed = False

    if results["aggregation_method"] == "Coordinate-wise Median":
        print("Aggregation method: PASSED")
    else:
        print("Aggregation method: FAILED")
        validation_passed = False

    if results["poisoned_client_id"] == 1:
        print("Poisoned client configuration: PASSED")
    else:
        print("Poisoned client configuration: FAILED")
        validation_passed = False

    if results["poison_factor"] == -3.0:
        print("Poison factor configuration: PASSED")
    else:
        print("Poison factor configuration: FAILED")
        validation_passed = False

    if validation_passed:
        print("\n========================================")
        print("ROBUST AGGREGATION VALIDATION PASSED")
        print("========================================")
    else:
        print("\n========================================")
        print("ROBUST AGGREGATION VALIDATION FAILED")
        print("========================================")


if __name__ == "__main__":
    main()
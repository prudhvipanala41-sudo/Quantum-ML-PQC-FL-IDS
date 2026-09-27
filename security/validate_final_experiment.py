import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_experiments"
    / "final_experiment_results.json"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_experiments"
    / "final_model_comparison.json"
)


def main():

    print("========================================")
    print("FINAL EXPERIMENT VALIDATION")
    print("========================================")

    validation_passed = True

    if FINAL_FILE.exists():
        print("Final experiment dataset exists: PASSED")
    else:
        print("Final experiment dataset exists: FAILED")
        validation_passed = False

    if COMPARISON_FILE.exists():
        print("Final comparison artifact exists: PASSED")
    else:
        print("Final comparison artifact exists: FAILED")
        validation_passed = False

    if not validation_passed:
        print("\nFINAL EXPERIMENT VALIDATION FAILED")
        return

    with open(FINAL_FILE, "r") as file:
        final_data = json.load(file)

    with open(COMPARISON_FILE, "r") as file:
        comparison = json.load(file)

    required_final_sections = [
        "project",
        "stage",
        "plain_federated_learning",
        "secure_federated_learning",
        "benchmarking",
        "model_poisoning",
        "robust_aggregation"
    ]

    missing_final = [
        section
        for section in required_final_sections
        if section not in final_data
    ]

    if not missing_final:
        print("Final dataset sections: PASSED")
    else:
        print("Final dataset sections: FAILED")
        print("Missing:", missing_final)
        validation_passed = False

    required_comparison_sections = [
        "project",
        "stage",
        "models",
        "poisoning_effect",
        "robust_vs_poisoned",
        "robust_vs_clean"
    ]

    missing_comparison = [
        section
        for section in required_comparison_sections
        if section not in comparison
    ]

    if not missing_comparison:
        print("Comparison sections: PASSED")
    else:
        print("Comparison sections: FAILED")
        print("Missing:", missing_comparison)
        validation_passed = False

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr"
    ]

    model_names = [
        "plain_fedavg",
        "poisoned_fedavg",
        "robust_median"
    ]

    metric_fields_passed = True

    for model_name in model_names:

        if model_name not in comparison["models"]:
            metric_fields_passed = False
            continue

        for metric in metrics:

            if metric not in comparison["models"][model_name]:
                metric_fields_passed = False

    if metric_fields_passed:
        print("Model metric fields: PASSED")
    else:
        print("Model metric fields: FAILED")
        validation_passed = False

    consistency_passed = True

    for metric in metrics:

        clean = comparison["models"]["plain_fedavg"][metric]

        poisoned = comparison["models"]["poisoned_fedavg"][metric]

        robust = comparison["models"]["robust_median"][metric]

        expected_poisoning = poisoned - clean

        expected_robust_poisoned = robust - poisoned

        expected_robust_clean = robust - clean

        actual_poisoning = comparison[
            "poisoning_effect"
        ][metric]

        actual_robust_poisoned = comparison[
            "robust_vs_poisoned"
        ][metric]

        actual_robust_clean = comparison[
            "robust_vs_clean"
        ][metric]

        if abs(
            expected_poisoning - actual_poisoning
        ) > 1e-9:
            consistency_passed = False

        if abs(
            expected_robust_poisoned
            - actual_robust_poisoned
        ) > 1e-9:
            consistency_passed = False

        if abs(
            expected_robust_clean
            - actual_robust_clean
        ) > 1e-9:
            consistency_passed = False

    if consistency_passed:
        print("Comparison numerical consistency: PASSED")
    else:
        print("Comparison numerical consistency: FAILED")
        validation_passed = False

    if comparison["project"] == final_data["project"]:
        print("Project identity consistency: PASSED")
    else:
        print("Project identity consistency: FAILED")
        validation_passed = False

    if (
        comparison["stage"] == "Final Experiments"
        and final_data["stage"] == "Final Experiments"
    ):
        print("Stage identity consistency: PASSED")
    else:
        print("Stage identity consistency: FAILED")
        validation_passed = False

    if validation_passed:

        print("\n========================================")
        print("FINAL EXPERIMENT VALIDATION PASSED")
        print("========================================")

    else:

        print("\n========================================")
        print("FINAL EXPERIMENT VALIDATION FAILED")
        print("========================================")


if __name__ == "__main__":
    main()
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "poisoning"
    / "poisoning_comparison.json"
)


def main():

    print("========================================")
    print("POISONING RESULT VALIDATION")
    print("========================================")

    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Missing artifact: {RESULT_PATH}"
        )

    with open(RESULT_PATH, "r") as file:
        results = json.load(file)

    required_sections = [
        "stage",
        "poisoned_client_id",
        "poison_factor",
        "rounds",
        "clients_per_round",
        "total_samples",
        "clean_fl",
        "poisoned_fl",
        "performance_difference",
        "attack_validation"
    ]

    for section in required_sections:

        if section not in results:
            raise ValueError(
                f"Missing required field: {section}"
            )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr"
    ]

    for metric in metrics:

        clean = results["clean_fl"][metric]
        poisoned = results["poisoned_fl"][metric]
        recorded_difference = results[
            "performance_difference"
        ][metric]

        calculated_difference = poisoned - clean

        if abs(
            calculated_difference - recorded_difference
        ) > 1e-5:

            raise ValueError(
                f"Mismatch in {metric}: "
                f"calculated={calculated_difference}, "
                f"recorded={recorded_difference}"
            )

    validation = results["attack_validation"]

    if validation["global_model_changed"] is not True:
        raise ValueError(
            "Global model change validation failed."
        )

    if validation["nan_validation"] is not True:
        raise ValueError(
            "NaN validation failed."
        )

    if validation["poisoned_model_saved"] is not True:
        raise ValueError(
            "Poisoned model save validation failed."
        )

    print("Artifact exists: PASSED")
    print("Required fields: PASSED")
    print("Metric consistency: PASSED")
    print("Global model change: PASSED")
    print("NaN validation: PASSED")
    print("Poisoned model saved: PASSED")

    print()
    print("========================================")
    print("POISONING RESULT VALIDATION PASSED")
    print("========================================")


if __name__ == "__main__":
    main()
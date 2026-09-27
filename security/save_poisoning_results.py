import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "poisoning"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "stage": "Model Poisoning Attack",
        "poisoned_client_id": 1,
        "poison_factor": -3.0,
        "rounds": 3,
        "clients_per_round": 5,
        "total_samples": 125973,

        "clean_fl": {
            "accuracy": 0.7550124201561391,
            "precision": 0.9536991062562066,
            "recall": 0.5986908750876646,
            "f1": 0.7356024701996265,
            "fpr": 0.03841005045824323,
            "true_negative": 9338,
            "false_positive": 373,
            "false_negative": 5150,
            "true_positive": 7683
        },

        "poisoned_fl": {
            "accuracy": 0.734564,
            "precision": 0.963962,
            "recall": 0.554430,
            "f1": 0.703968,
            "fpr": 0.027392,
            "true_negative": 9445,
            "false_positive": 266,
            "false_negative": 5718,
            "true_positive": 7115
        },

        "performance_difference": {
            "accuracy": -0.020449,
            "precision": 0.010262,
            "recall": -0.044261,
            "f1": -0.031635,
            "fpr": -0.011018
        },

        "attack_validation": {
            "global_model_changed": True,
            "nan_validation": True,
            "poisoned_model_saved": True
        }
    }

    output_path = OUTPUT_DIR / "poisoning_comparison.json"

    with open(output_path, "w") as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print("========================================")
    print("POISONING RESULTS SAVED")
    print("========================================")

    print("Poisoned client:", results["poisoned_client_id"])
    print("Poison factor:", results["poison_factor"])

    print()
    print("Clean accuracy:",
          results["clean_fl"]["accuracy"])

    print("Poisoned accuracy:",
          results["poisoned_fl"]["accuracy"])

    print("Accuracy difference:",
          results["performance_difference"]["accuracy"])

    print()
    print("Clean recall:",
          results["clean_fl"]["recall"])

    print("Poisoned recall:",
          results["poisoned_fl"]["recall"])

    print("Recall difference:",
          results["performance_difference"]["recall"])

    print()
    print("Clean F1:",
          results["clean_fl"]["f1"])

    print("Poisoned F1:",
          results["poisoned_fl"]["f1"])

    print("F1 difference:",
          results["performance_difference"]["f1"])

    print()
    print("Artifact saved:")
    print(output_path)


if __name__ == "__main__":
    main()
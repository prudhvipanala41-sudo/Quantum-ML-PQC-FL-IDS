import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS = PROJECT_ROOT / "results"

FEDERATED_FILE = (
    RESULTS
    / "federated"
    / "evaluation_results.json"
)

SECURE_FL_FILE = (
    RESULTS
    / "secure_federated"
    / "secure_fl_validation.json"
)

BENCHMARK_FILE = (
    RESULTS
    / "benchmarking"
    / "benchmark_summary.json"
)

POISONING_FILE = (
    RESULTS
    / "poisoning"
    / "poisoning_comparison.json"
)

ROBUST_FILE = (
    RESULTS
    / "robust_aggregation"
    / "robust_aggregation_comparison.json"
)


OUTPUT_DIR = RESULTS / "final_experiments"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "final_experiment_results.json"
)


def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    with open(path, "r") as file:
        return json.load(file)


def main():

    print("========================================")
    print("BUILDING FINAL EXPERIMENT DATASET")
    print("========================================")

    federated = load_json(FEDERATED_FILE)

    secure_fl = load_json(SECURE_FL_FILE)

    benchmark = load_json(BENCHMARK_FILE)

    poisoning = load_json(POISONING_FILE)

    robust = load_json(ROBUST_FILE)

    final_results = {
        "project": (
            "Quantum ML-Based Intrusion Detection "
            "with PQC-Secured Federated Model Updates"
        ),

        "stage": "Final Experiments",

        "plain_federated_learning": federated,

        "secure_federated_learning": secure_fl,

        "benchmarking": benchmark,

        "model_poisoning": poisoning,

        "robust_aggregation": robust
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
            final_results,
            file,
            indent=4
        )

    print("\nArtifacts loaded successfully:")
    print("Plain FL:", FEDERATED_FILE)
    print("Secure FL:", SECURE_FL_FILE)
    print("Benchmarking:", BENCHMARK_FILE)
    print("Model Poisoning:", POISONING_FILE)
    print("Robust Aggregation:", ROBUST_FILE)

    print("\nFinal experiment dataset created:")
    print(OUTPUT_FILE)

    print("\n========================================")
    print("FINAL EXPERIMENT DATASET BUILD PASSED")
    print("========================================")


if __name__ == "__main__":
    main()
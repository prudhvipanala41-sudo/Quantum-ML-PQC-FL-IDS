from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRAPH_DIR = (
    PROJECT_ROOT
    / "results"
    / "graphs"
)

EXPECTED_GRAPHS = [
    "performance/accuracy_comparison.png",
    "performance/precision_comparison.png",
    "performance/recall_comparison.png",
    "performance/f1_comparison.png",
    "performance/fpr_comparison.png",
    "security/crypto_overhead.png",
    "security/plain_vs_secure_runtime.png",
    "security/communication_overhead.png",
    "poisoning/accuracy_robustness.png",
    "poisoning/precision_robustness.png",
    "poisoning/recall_robustness.png",
    "poisoning/f1_robustness.png",
    "poisoning/fpr_robustness.png",
    "poisoning/poisoning_impact.png",
    "poisoning/robust_recovery.png"
]


def validate_png(path):

    if not path.exists():
        return False

    if path.stat().st_size == 0:
        return False

    with open(path, "rb") as file:
        header = file.read(8)

    return header == b"\x89PNG\r\n\x1a\n"


def main():

    print("========================================")
    print("GRAPH ARTIFACT VALIDATION")
    print("========================================")

    passed = 0

    for relative_path in EXPECTED_GRAPHS:

        graph_path = GRAPH_DIR / relative_path

        if validate_png(graph_path):
            print(f"PASSED: {relative_path}")
            passed += 1
        else:
            print(f"FAILED: {relative_path}")

    total = len(EXPECTED_GRAPHS)

    print()
    print(f"Graphs validated: {passed}/{total}")

    if passed != total:
        raise RuntimeError(
            "GRAPH ARTIFACT VALIDATION FAILED"
        )

    print()
    print("========================================")
    print("GRAPH ARTIFACT VALIDATION PASSED")
    print("========================================")


if __name__ == "__main__":
    main()
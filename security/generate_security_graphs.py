import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmarking"
    / "secure_fl_crypto_benchmark.json"
)

RUNTIME_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmarking"
    / "plain_vs_secure_fl_runtime.json"
)

COMMUNICATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmarking"
    / "model_communication_overhead.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "graphs"
    / "security"
)


def main():

    print("========================================")
    print("SECURITY GRAPH GENERATION")
    print("========================================")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(BENCHMARK_FILE, "r") as file:
        benchmark = json.load(file)

    with open(RUNTIME_FILE, "r") as file:
        runtime = json.load(file)

    with open(COMMUNICATION_FILE, "r") as file:
        communication = json.load(file)

    crypto_labels = [
        "Key Establishment",
        "AES Encryption",
        "AES Decryption"
    ]

    crypto_values = [
    benchmark["keygen_encapsulation_mean_ms"],
    benchmark["encryption_mean_ms"],
    benchmark["decryption_mean_ms"]
   ]

    plt.figure(figsize=(8, 5))
    plt.bar(
        crypto_labels,
        crypto_values
    )
    plt.ylabel("Time (ms)")
    plt.title("Secure FL Cryptographic Overhead")
    plt.tight_layout()

    crypto_output = (
        OUTPUT_DIR
        / "crypto_overhead.png"
    )

    plt.savefig(
        crypto_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Cryptographic overhead graph saved:")
    print(crypto_output)

    runtime_labels = [
        "Plain FL",
        "Secure FL"
    ]

    runtime_values = [
        runtime["plain_fl_runtime_seconds"],
        runtime["secure_fl_runtime_seconds"]
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(
        runtime_labels,
        runtime_values
    )
    plt.ylabel("Runtime (seconds)")
    plt.title("Plain FL vs Secure FL Runtime")
    plt.tight_layout()

    runtime_output = (
        OUTPUT_DIR
        / "plain_vs_secure_runtime.png"
    )

    plt.savefig(
        runtime_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Runtime comparison graph saved:")
    print(runtime_output)

    communication_labels = [
        "Plain Update",
        "Secure Payload"
    ]

    communication_values = [
    communication["plain_model_update_bytes"],
    communication["secure_communication_payload_bytes"]
   ] 

    plt.figure(figsize=(8, 5))
    plt.bar(
        communication_labels,
        communication_values
    )
    plt.ylabel("Communication Size (bytes)")
    plt.title("Plain vs Secure FL Communication Payload")
    plt.tight_layout()

    communication_output = (
        OUTPUT_DIR
        / "communication_overhead.png"
    )

    plt.savefig(
        communication_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Communication overhead graph saved:")
    print(communication_output)

    print("\n========================================")
    print("SECURITY GRAPH GENERATION PASSED")
    print("========================================")


if __name__ == "__main__":
    main()
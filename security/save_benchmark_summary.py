import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarking"


def load_json(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark artifact: {path}")

    with open(path, "r") as file:
        return json.load(file)


def main():
    print("=== Benchmark Summary ===")
    print()

    crypto = load_json("secure_fl_crypto_benchmark.json")
    runtime = load_json("plain_vs_secure_fl_runtime.json")
    communication = load_json("model_communication_overhead.json")

    summary = {
        "benchmark_stage": "Stage 14 - Benchmarking",

        "secure_fl_crypto_overhead": {
            "iterations": crypto["iterations"],
            "successful_runs": crypto["successful_runs"],
            "model_update_size_bytes": crypto["model_update_size_bytes"],
            "kem_ciphertext_size_bytes": crypto["kem_ciphertext_size_bytes"],
            "aes_nonce_size_bytes": crypto["aes_nonce_size_bytes"],
            "encrypted_update_size_bytes": crypto["encrypted_update_size_bytes"],
            "keygen_encapsulation_mean_ms": crypto[
                "keygen_encapsulation_mean_ms"
            ],
            "encryption_mean_ms": crypto["encryption_mean_ms"],
            "decryption_mean_ms": crypto["decryption_mean_ms"],
            "total_crypto_mean_ms": crypto["total_crypto_mean_ms"],
            "all_integrity_checks_passed": crypto[
                "all_integrity_checks_passed"
            ]
        },

        "plain_vs_secure_fl_runtime": {
            "clients_per_round": runtime["clients_per_round"],
            "rounds": runtime["rounds"],
            "plain_fl_runtime_seconds": runtime[
                "plain_fl_runtime_seconds"
            ],
            "secure_fl_runtime_seconds": runtime[
                "secure_fl_runtime_seconds"
            ],
            "additional_runtime_seconds": runtime[
                "additional_runtime_seconds"
            ],
            "runtime_overhead_percentage": runtime[
                "runtime_overhead_percentage"
            ]
        },

        "model_communication_overhead": {
            "model": communication["model"],
            "plain_model_update_bytes": communication[
                "plain_model_update_bytes"
            ],
            "ml_kem_ciphertext_bytes": communication[
                "ml_kem_ciphertext_bytes"
            ],
            "aes_gcm_encrypted_update_bytes": communication[
                "aes_gcm_encrypted_update_bytes"
            ],
            "secure_communication_payload_bytes": communication[
                "secure_communication_payload_bytes"
            ],
            "additional_communication_bytes": communication[
                "additional_communication_bytes"
            ],
            "communication_overhead_percentage": communication[
                "communication_overhead_percentage"
            ]
        }
    }

    output_path = RESULTS_DIR / "benchmark_summary.json"

    with open(output_path, "w") as file:
        json.dump(summary, file, indent=4)

    print("=== Benchmark Summary Created ===")
    print()

    print("Cryptographic overhead:")
    print(
        f"  Key establishment: "
        f"{crypto['keygen_encapsulation_mean_ms']:.4f} ms"
    )
    print(
        f"  AES encryption: "
        f"{crypto['encryption_mean_ms']:.4f} ms"
    )
    print(
        f"  AES decryption: "
        f"{crypto['decryption_mean_ms']:.4f} ms"
    )
    print(
        f"  Total crypto: "
        f"{crypto['total_crypto_mean_ms']:.4f} ms"
    )
    print(
        f"  Successful runs: "
        f"{crypto['successful_runs']}/{crypto['iterations']}"
    )
    print(
        f"  Integrity checks passed: "
        f"{crypto['all_integrity_checks_passed']}"
    )

    print()
    print("Runtime:")
    print(
        f"  Plain FL: "
        f"{runtime['plain_fl_runtime_seconds']:.4f} seconds"
    )
    print(
        f"  Secure FL: "
        f"{runtime['secure_fl_runtime_seconds']:.4f} seconds"
    )
    print(
        f"  Measured difference: "
        f"{runtime['runtime_overhead_percentage']:.2f}%"
    )

    print()
    print("Communication:")
    print(
        f"  Plain update: "
        f"{communication['plain_model_update_bytes']} bytes"
    )
    print(
        f"  Secure payload: "
        f"{communication['secure_communication_payload_bytes']} bytes"
    )
    print(
        f"  Communication overhead: "
        f"{communication['communication_overhead_percentage']:.2f}%"
    )

    print()
    print("Benchmark summary artifact saved:")
    print(output_path)


if __name__ == "__main__":
    main()